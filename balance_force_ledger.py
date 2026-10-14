"""Current Balance forces, with an explicit work-preserving spatial lift.

The rigid generalized force is exact. The differential cell field is a reduced
approximation, not a resolved membrane stress. Boundary/contact ports are lifted
to nearby *owned* controls with area-weighted finite-cell rotational metrics.
If transport leaves a plate with no dominant-owner controls, its genuine
positive fractional footprint supplies support-weighted controls instead.
This does not create a categorical plate domain or an eligible breakup cut.
The adjoint of that same lift defines port velocities when evaluating a virtual
daughter mode; velocity-dependent forces are never frozen as dead loads.

Fractional basal support is retained even where its dominant control owner is
another plate. Such a contribution is explicitly lifted, never dropped or
replaced by an equilibrium residual. Unsupported assembly terms fail closed.
This module reads one frozen geometry and does not change any physical law.
"""
from dataclasses import dataclass
import math
import numpy as np
import plate_balance as pb

_IDENTITY = np.eye(3)[None, :, :]
_TORQUE = pb.RADIUS_M/pb.CM_YR_M_S
DEFAULT_BAND_KM = 400.
MAX_LIFT_CONDITION = 1e12
AGREEMENT_TOLERANCE = 2e-8
UNIFORM_RESTRICTION_TOLERANCE = 1e-9


class _ControlTree:
    """Small fixed-radius Cartesian bucket index; no optional SciPy runtime."""
    def __init__(self, points, width):
        self.points = points
        self.width = width
        buckets = {}
        for index, key in enumerate(np.floor(points/width).astype(int)):
            buckets.setdefault(tuple(key), []).append(index)
        self.buckets = {key: np.asarray(value, int) for key, value in buckets.items()}

    def query_ball_point(self, point, radius):
        key = np.floor(point/self.width).astype(int)
        rows = []
        reach = int(math.ceil(radius/self.width))
        for a in range(-reach, reach+1):
            for b in range(-reach, reach+1):
                for c in range(-reach, reach+1):
                    found = self.buckets.get(tuple(key+np.array([a, b, c])))
                    if found is not None:
                        rows.append(found)
        if not rows:
            return np.zeros(0, int)
        selected = np.concatenate(rows)
        squared = np.sum((self.points[selected]-point)**2, axis=1)
        return selected[squared <= radius*radius]

    def query(self, point, k):
        squared = np.sum((self.points-point)**2, axis=1)
        selected = np.argpartition(squared, k-1)[:k]
        return np.sqrt(squared[selected]), selected


class UnsupportedForceLedger(ValueError):
    """The current force law cannot be represented without inventing a load."""


@dataclass(slots=True)
class Port:
    owner: int
    cells: np.ndarray
    matrices: np.ndarray
    condition: float = 1.
    source_cell: int = -1
    off_owned: bool = False
    nearest_fallback: bool = False
    distance_km: float = 0.

    def uniform(self):
        """Qualify the lift before using its analytic partition of unity."""
        if (self.matrices.shape != (len(self.cells), 3, 3) or not len(self.cells)
                or not np.isfinite(self.matrices).all()
                or np.linalg.norm(self.matrices.sum(axis=0)-np.eye(3))/math.sqrt(3.)
                > UNIFORM_RESTRICTION_TOLERANCE):
            raise UnsupportedForceLedger('A common force port does not preserve uniform motion.')
        return np.eye(3)

    def motion(self, field):
        self.uniform()
        values = field[self.cells]
        anchor = values[0]
        return anchor+np.einsum('cij,ci->j', self.matrices, values-anchor)

    def lift(self, force):
        """Adjoint of the anchored velocity map, independent of equilibrium."""
        self.uniform()
        lifted = np.einsum('cij,j->ci', self.matrices, force)
        lifted[0] = force-lifted[1:].sum(axis=0)
        return lifted

    def basis(self, field, index=None):
        cells = self.cells if index is None else index[self.cells]
        if np.any(cells < 0):
            raise UnsupportedForceLedger('Finite mode omits a retained owned spatial port.')
        self.uniform()
        values = field[cells]
        anchor = values[0]
        # Exact constant fields prevent an ulp in a port sum from activating
        # a unilateral law. The same linear map defines motion and its adjoint.
        return anchor+np.einsum('cij,cik->jk', self.matrices, values-anchor)


@dataclass(slots=True)
class LocalTerm:
    kind: str
    owners: tuple
    ports: tuple
    matrix: np.ndarray | None = None
    drive: np.ndarray | None = None
    shape: str = 'quadratic'
    rows: tuple = ()
    coefficient: float = 0.
    scale: float = 1.
    bounds: tuple = ()
    version: int = 0

    def coordinates(self, x, slots):
        return np.concatenate([x[3*slots[p]:3*slots[p]+3] for p in self.owners])

    def evaluate(self, x, delta):
        """Original port potential, gradient and Hessian in scaled Euler units."""
        if self.shape == 'quadratic':
            return (.5*x@self.matrix@x-self.drive@x,
                    self.matrix@x-self.drive, self.matrix)
        if self.shape == 'hinge':
            row = self.rows[0]; rate = float(row@x)
            closing = min(rate, 0.)
            return (.5*self.coefficient*closing*closing,
                    self.coefficient*closing*row,
                    self.coefficient*np.outer(row, row)*(rate < 0.))
        if self.shape == 'arc_opening':
            import weld_geometry
            value, first, second, _ = weld_geometry.integrate(
                self.rows[0], x, *self.bounds, delta)
            return self.coefficient*value, self.coefficient*first, self.coefficient*second
        width = delta*self.scale
        if self.shape == 'cone':
            normal, tangent = self.rows
            opening, tangential = normal@x, tangent@x
            closed, first, second = pb._cone_closing_terms(opening, width, self.version)
            magnitude = math.sqrt(closed*closed+tangential*tangential+width*width)
            value = ((closed*closed+tangential*tangential)/(magnitude+width)
                     if self.version == 1 else magnitude-width)
            jn = first*normal
            direction = (closed*jn+tangential*tangent)/magnitude
            hessian = (np.outer(jn, jn)+np.outer(tangent, tangent)
                +closed*second*np.outer(normal, normal)-np.outer(direction, direction))
            return self.coefficient*value, self.coefficient*direction, self.coefficient/magnitude*hessian
        law = (pb._abs_terms if self.shape == 'abs' else
               pb._passive_negative_terms if self.version == 1 else pb._negative_terms)
        row = self.rows[0]
        value, first, second = law(row@x, width)
        return self.coefficient*value, self.coefficient*first*row, self.coefficient*second*np.outer(row, row)


class _Spatial:
    def __init__(self, balance, band_km):
        self.s = balance.s
        self.points = np.asarray(self.s.xyz, float)
        self.area = np.asarray(self.s.cell_area, float)
        self.owner = np.asarray(self.s.plate, int)
        if (self.points.shape != (len(self.area), 3) or self.owner.shape != self.area.shape
                or not np.isfinite(self.points).all() or not np.isfinite(self.area).all()
                or np.any(self.area <= 0.)
                or np.max(np.abs(np.linalg.norm(self.points, axis=1)-1.), initial=0.) > 1e-7):
            raise UnsupportedForceLedger('Spatial force ledger requires aligned finite positive unit-sphere controls.')
        self.cells = {p: np.flatnonzero(self.owner == p) for p in balance.plates}
        self.fractional_only = set()
        self.fraction = {}
        for p in balance.plates:
            if not len(self.cells[p]):
                # Transport retains fractional lithosphere even when another
                # owner wins every control. Keep its force and virtual work on
                # that actual footprint; never assign it a neighbour's cells.
                fraction = balance._support(p)
                self.cells[p] = np.flatnonzero(fraction > 0.)
                if len(self.cells[p]):
                    self.fractional_only.add(p)
                    self.fraction[p] = fraction.copy()
        chord = 2.*math.sin(min(band_km/pb.RADIUS_KM, math.pi)*.5)
        self.trees = {p: _ControlTree(self.points[c], chord) for p, c in self.cells.items() if len(c)}
        self.metrics = (pb._cell_rotation_metric(self.s.native_mesh)
            if hasattr(self.s, 'native_mesh') else
            np.eye(3)[None]-np.einsum('ci,cj->cij', self.points, self.points))
        if self.metrics.shape != (len(self.points), 3, 3):
            raise UnsupportedForceLedger('Native finite-cell quadrature is not aligned with controls.')
        self.band_km = band_km
        self.cache = {}

    def port(self, owner, point, *, cell=None, direct=False):
        owner = int(owner)
        if owner not in self.trees:
            raise UnsupportedForceLedger(f'Plate {owner} has a force port but no owned native controls '
                                         'or positive fractional support.')
        if direct and cell is not None and self.owner[cell] == owner:
            key = (owner, int(cell), 'direct')
            if key not in self.cache:
                self.cache[key] = Port(owner, np.array([cell], int), _IDENTITY, source_cell=int(cell))
            return self.cache[key]
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all() or np.linalg.norm(point) <= 0.:
            raise UnsupportedForceLedger('A force port requires a finite nonzero spherical location.')
        point = point/np.linalg.norm(point)
        key = (owner, point.tobytes(), int(cell) if cell is not None else -1)
        if key in self.cache:
            return self.cache[key]
        tree = self.trees[owner]; owned = self.cells[owner]
        chord = 2.*math.sin(min(self.band_km/pb.RADIUS_KM, math.pi)*.5)
        selected = np.asarray(tree.query_ball_point(point, chord), int)
        fallback = len(selected) < min(3, len(owned))
        if fallback:
            _, selected = tree.query(point, k=min(6, len(owned)))
            selected = np.atleast_1d(selected).astype(int)
        cells = np.sort(owned[selected])
        area = self.area[cells]
        if owner in self.fractional_only:
            area = area*self.fraction[owner][cells]
        if not np.isfinite(area).all() or not np.isfinite(area.sum()) or area.sum() <= 0.:
            raise UnsupportedForceLedger(f'Plate {owner} spatial lift lacks finite positive support area.')
        weights = area/area.sum()
        metric = np.einsum('c,cij->ij', weights, self.metrics[cells])
        values = np.linalg.eigvalsh(.5*(metric+metric.T))
        condition = float(values[-1]/values[0]) if values[0] > 0. else math.inf
        if not np.isfinite(condition) or condition > MAX_LIFT_CONDITION:
            raise UnsupportedForceLedger(f'Plate {owner} spatial lift is singular/ill-conditioned ({condition:g}).')
        matrices = (weights[:, None, None]*self.metrics[cells])@np.linalg.inv(metric)
        if np.linalg.norm(matrices.sum(axis=0)-np.eye(3)) > 1e-7:
            raise UnsupportedForceLedger(f'Plate {owner} spatial lift cannot preserve rigid virtual work.')
        distance = float(np.max(np.arccos(np.clip(self.points[cells]@point, -1., 1.)))*pb.RADIUS_KM)
        port = Port(owner, cells, matrices, condition,
                    source_cell=int(cell) if cell is not None else -1,
                    off_owned=cell is not None and self.owner[cell] != owner,
                    nearest_fallback=fallback, distance_km=distance)
        self.cache[key] = port
        return port


def _indices(balance, owners):
    return np.concatenate([np.arange(3*balance.slot[p], 3*balance.slot[p]+3) for p in owners])


def _owners(balance, *arrays):
    return tuple(p for p in balance.plates if any(
        np.any(np.asarray(a)[..., 3*balance.slot[p]:3*balance.slot[p]+3] != 0.) for a in arrays))


def _relative_error(actual, expected, scale=None):
    return float(np.max(np.abs(actual-expected), initial=0.)/
                 max(float(np.max(np.abs(expected), initial=0.)), float(scale or 0.), 1.))


def export(balance, x=None, *, band_km=None, delta=None):
    """Export every current force, plus reusable law and spatial port operators.

    ``plates[p]['torques_n_m']`` aligns with ``cells``: dominant-owner controls,
    or positive fractional controls only when that owner has no dominant cells.
    ``localcomponents_n_m`` contains separate contributions at those controls.
    ``generalized_force`` is in Balance's W/(cm/year) units. No equilibrium
    correction is applied. ``local_terms`` can be projected by ``compile_mode``.
    The returned diagnostics contain only portable finite JSON values.
    """
    if getattr(balance, '_continental_entry_applied', False):
        raise UnsupportedForceLedger('Persistent continental-entry force ports are not implemented in this ledger.')
    if getattr(balance, 'effective_subduction', False):
        import effective_subduction_carrier_traction as carrier
        current_enabled = carrier.enabled(balance.s)
        if (current_enabled != getattr(balance, 'effective_carrier_enabled', False)
                or carrier.alpha(balance.s) != getattr(balance, 'effective_carrier_alpha', 0.)):
            raise UnsupportedForceLedger('Carrier-traction policy changed after the frozen balance assembly.')
    unknown_drivers = set(balance.drivers)-{'slab', 'ridge', 'collision'}
    if unknown_drivers:
        raise UnsupportedForceLedger(f'Unsupported current driving terms: {sorted(unknown_drivers)!r}.')
    if x is None:
        x = getattr(balance, 'x', None)
    x = np.asarray(x, float)
    if x.shape != (balance.size,) or not np.isfinite(x).all():
        raise ValueError('Force ledger requires finite current Balance coordinates.')
    if band_km is None:
        band_km = getattr(balance.s, 'config', {}).get('deformation_width_km', DEFAULT_BAND_KM)
    try:
        band_km = float(band_km)
    except (TypeError, ValueError) as error:
        raise ValueError('Force ledger lift band must be finite and positive.') from error
    if not np.isfinite(band_km) or band_km <= 0.:
        raise ValueError('Force ledger lift band must be finite and positive.')
    if delta is None:
        delta = pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR
    if not np.isfinite(delta) or delta <= 0.:
        raise ValueError('Force ledger plastic width must be finite and positive.')
    spatial = _Spatial(balance, float(band_km))
    terms = []
    known_matrix = np.zeros_like(balance.stiffness)
    known_drive = np.zeros_like(balance.torque)

    def quadratic(kind, owners, matrix, drive, ports):
        if not owners:
            return
        ids = _indices(balance, owners)
        known_matrix[np.ix_(ids, ids)] += matrix
        known_drive[ids] += drive
        terms.append(LocalTerm(kind, owners, ports, matrix=matrix, drive=drive))

    def full_quadratic(kind, matrix, drive, point):
        owners = _owners(balance, matrix, drive)
        if not owners:
            return
        ids = _indices(balance, owners)
        quadratic(kind, owners, matrix[np.ix_(ids, ids)], drive[ids],
                  tuple(spatial.port(p, point) for p in owners))

    # Same fractional support and finite-cell quadrature as the actual solve.
    keel = np.asarray(pb.KEEL)[np.clip(np.asarray(balance.s.crust, int), 0, len(pb.KEEL)-1)]
    basal_weight = pb.ASTHENOSPHERE_DRAG_PA_S_M*keel*spatial.area*1e6*pb.CM_YR_M_S**2
    off_owned_count, off_owned_fraction, off_owned_force = {}, {}, {}
    for p in balance.plates:
        fraction = balance._support(p)
        selected = np.flatnonzero(fraction > 0.)
        off_owned_count[p] = int(np.count_nonzero(spatial.owner[selected] != p))
        off_owned_fraction[p] = float(np.sum(fraction[selected][spatial.owner[selected] != p]))
        off_owned_force[p] = np.zeros(3)
        for cell in selected:
            matrix = basal_weight[cell]*fraction[cell]*spatial.metrics[cell]
            port = spatial.port(p, spatial.points[cell], cell=int(cell), direct=True)
            quadratic('basal', (p,), matrix, np.zeros(3), (port,))
            if port.off_owned:
                off_owned_force[p] -= matrix@x[3*balance.slot[p]:3*balance.slot[p]+3]*_TORQUE

    # Export the condensed attached law edge by edge, not raw slab weight.
    zero = np.zeros_like(balance.torque)
    if getattr(balance, 'effective_subduction', False):
        original_edge_forces = None
        for edge in balance.trench_edges:
            owner = int(balance.slab_owner[edge])
            sign = 1. if owner == balance.bp[edge] else -1.
            drive = np.zeros(balance.size)
            value = (pb.CM_YR_M_S*balance.length_m[edge]*balance.effective_line_force[edge]
                     *sign*balance.an[edge])
            balance._block(owner, value, drive)
            if getattr(balance, 'effective_carrier_alpha', 0.) > 0.:
                other = int(balance.effective_carrier_owner[edge])
                balance._block(other, -balance.effective_carrier_alpha*value, drive)
            full_quadratic('effective_subduction', np.zeros_like(balance.stiffness), drive,
                           balance.radial[edge])
    elif balance.slab_tethers:
        import slab_tether_forces
        for channel in balance.tether_assembly['channels']:
            cs, cn, weight = (channel[k] for k in ('mantle_drag_n_s_m', 'neck_drag_n_s_m', 'weight_n'))
            c, sine = channel['dip_cosine'], channel['dip_sine']
            horizontal = channel.get('horizontal_intake', False)
            denominator = cs+cn*(c*c if horizontal else 1.)
            drag = cs*cn/denominator
            for edge, share, q, h in zip(channel['edges'], channel['edge_shares'],
                                       channel['inlet_rows'], channel['hinge_rows']):
                coupled = q+(c*c if horizontal else c)*h
                matrix = share*pb.CM_YR_M_S**2*(drag*np.outer(coupled, coupled)
                    +cs*sine*sine*np.outer(h, h))
                drive = share*pb.CM_YR_M_S*weight*(cn*(c if horizontal else 1.)/denominator*q
                                                -cs*c/denominator*h)
                full_quadratic('attached_slab', matrix, drive, balance.radial[edge])
        original_edge_forces = slab_tether_forces.edge_force_ledger(balance.tether_assembly, x)
    else:
        original_edge_forces = None
        for i, edge in enumerate(balance.trench_edges):
            for rows in (balance.slab_horizontal_rows, balance.slab_vertical_rows):
                full_quadratic('slab_viscous', balance.slab_stokes_coefficient[i]*np.outer(rows[i], rows[i]),
                               zero, balance.radial[edge])
            owner = int(balance.slab_owner[edge])
            sign = 1. if owner == balance.bp[edge] else -1.
            drive = np.zeros(balance.size)
            value = (pb.CM_YR_M_S*balance.length_m[edge]*pb.GRAVITY_M_S2*balance.slab_load[edge]
                     *math.sin(math.radians(pb.slab_memory.SUBDUCTION_DIP_DEG))*sign*balance.an[edge])
            balance._block(owner, value, drive)
            if balance.subduction_response_version == 1:
                other = int(balance.bq[edge] if owner == balance.bp[edge] else balance.bp[edge])
                balance._block(other, -value, drive)
            full_quadratic('legacy_slab_drive', np.zeros_like(balance.stiffness), drive, balance.radial[edge])
    for i, edge in enumerate(balance.trench_edges):
        if not getattr(balance, 'effective_subduction', False):
            row = balance.slab_anchor_rows[i]
            full_quadratic('upper_anchor', balance.slab_anchor_coefficient[i]*np.outer(row, row), zero, balance.radial[edge])
        row = balance.hinge[i]; owners = _owners(balance, row)
        ids = _indices(balance, owners)
        terms.append(LocalTerm('hinge_bending', owners, tuple(spatial.port(p, balance.radial[edge]) for p in owners),
            shape='hinge', rows=(row[ids],), coefficient=float(balance.hinge_coefficient[i])))

    import collision_interface
    if getattr(balance.s, 'collision_interface_version', 0):
        values = collision_interface.parameters(balance.s)
        factor = pb.CM_YR_M_S*float(balance.s.material_surface.get('radius_km', pb.RADIUS_KM))/pb.RADIUS_KM
        for region in collision_interface.regions(balance.s):
            owners = (int(region['top_owner']), int(region['under_owner']))
            block = values['viscosity_pa_s']/values['thickness_m']*factor**2*region['metric_m2']
            matrix = np.block([[block, -block], [-block, block]])
            mesh = balance.s.material_surface
            point = np.asarray(mesh['vertices'])[np.asarray(mesh['faces'])[region['top_face']]].sum(axis=0)
            quadratic('collision_interface', owners, matrix, np.zeros(6), tuple(spatial.port(p, point) for p in owners))

    force = pb.ridge_force_density(balance.s)
    if force is not None:
        drive = pb.CM_YR_M_S*np.cross(spatial.points, force)*(spatial.area*1e6)[:, None]
        for p in balance.plates:
            # The assembled ridge law is categorical, unlike fractional basal
            # resistance. A broader port stencil must not create extra sources.
            for cell in np.flatnonzero(spatial.owner == p):
                if np.any(drive[cell]):
                    quadratic('ridge_driver', (p,), np.zeros((3, 3)), drive[cell],
                        (spatial.port(p, spatial.points[cell], cell=int(cell), direct=True),))

    # Rigid share only: reusing the full unsplit material gradient here would
    # spend the internal deformation potential for a second time.
    mesh = getattr(balance.s, 'material_surface', None)
    if isinstance(mesh, dict) and len(mesh.get('faces', ())) >= 2 and len(np.unique(mesh['vertex_owner'])) >= 2:
        import native_material_evolution
        import column_density
        import gravitational_relaxation as gr
        _, gradient, _ = gr.energy_gradient(mesh['vertices'], mesh['faces'], balance.s.structure['thickness_km'],
            native_material_evolution.gravitational_reference(balance.s), balance.s.parcel_collision_sheet,
            radius=mesh.get('radius_km', pb.RADIUS_KM), overlap=getattr(balance.s, '_collision_overlap', None),
            **column_density.options(balance.s))
        partition = gr.rigid_mode_partition(mesh['vertices'], mesh['faces'], mesh['area_km2'], gradient, mesh['vertex_owner'])
        rigid = gradient-partition['residual']
        coefficient = partition['diagnostics']['density_coefficient_n_m3']
        local = -coefficient*1e12*np.cross(mesh['vertices'], rigid)/_TORQUE
        for vertex, (p, point) in enumerate(zip(mesh['vertex_owner'], mesh['vertices'])):
            if int(p) not in balance.slot:
                if np.any(local[vertex]):
                    raise UnsupportedForceLedger('Collision gravitational driver has an inactive material owner.')
                continue
            if np.any(local[vertex]):
                quadratic('collision_gpe_driver', (int(p),), np.zeros((3, 3)), local[vertex],
                          (spatial.port(int(p), point),))

    # Keep constitutive rows intact, so finite daughter modes reevaluate the
    # same normal/tangential/weld laws with the same continuation width.
    for element in balance.elements:
        shape = element['shape']
        if shape not in ('abs', 'negative', 'cone', 'arc_opening'):
            raise UnsupportedForceLedger(f'Unsupported {element["kind"]} force shape {shape!r}.')
        for i, coefficient in enumerate(element['coefficient']):
            rows = tuple(np.asarray(r[i]) for r in element['rows'])
            owners = _owners(balance, *rows)
            if not owners:
                continue
            ids = _indices(balance, owners)
            if element.get('edges') is not None:
                point = balance.radial[int(element['edges'][i])]
            elif element['kind'] == 'suture_weld' and i < len(balance.weld_rows):
                point = balance.weld_rows[i]['front_center']
            else:
                raise UnsupportedForceLedger(f'Unsupported nonlocal {element["kind"]} force port; no physical geometry.')
            terms.append(LocalTerm(element['kind'], owners, tuple(spatial.port(p, point) for p in owners),
                shape=shape, rows=tuple(r[..., ids] for r in rows), coefficient=float(coefficient),
                scale=float(element['scale'][i]), bounds=tuple(element['bounds'][i]) if shape == 'arc_opening' else (),
                version=balance.resistance_version))

    matrix_error = max((_relative_error(known_matrix[3*i:3*i+3], balance.stiffness[3*i:3*i+3])
                        for i in range(balance.count)), default=0.)
    drive_error = _relative_error(known_drive, balance.torque)
    if max(matrix_error, drive_error) > AGREEMENT_TOLERANCE:
        entry = (' (persistent continental entry is unsupported)'
                 if getattr(balance.s, 'continental_entry_regions', None) is not None else '')
        raise UnsupportedForceLedger(f'Force decomposition does not cover current assembly{entry}: '
            f'stiffness relative error {matrix_error:g}, drive relative error {drive_error:g}.')
    expected_value, expected_gradient, _, _, _ = balance._evaluate(x, delta)
    actual_force = np.zeros(balance.size)
    component_generalized = {}
    local = {p: {} for p in balance.plates}
    local_gross = {p: 0. for p in balance.plates}
    lookup = {p: {int(c): i for i, c in enumerate(spatial.cells[p])} for p in balance.plates}
    actual_value = 0.
    for term in terms:
        coordinates = term.coordinates(x, balance.slot)
        value, gradient, _ = term.evaluate(coordinates, delta)
        actual_value += float(value)
        ids = _indices(balance, term.owners)
        actual_force[ids] -= gradient
        component_generalized.setdefault(term.kind, np.zeros(balance.size))[ids] -= gradient
        for j, port in enumerate(term.ports):
            arrays = local[port.owner]
            if term.kind not in arrays:
                arrays[term.kind] = np.zeros((len(spatial.cells[port.owner]), 3))
            positions = np.fromiter((lookup[port.owner][int(c)] for c in port.cells), int, count=len(port.cells))
            lifted = -port.lift(gradient[3*j:3*j+3])*_TORQUE
            arrays[term.kind][positions] += lifted
            local_gross[port.owner] += float(np.abs(lifted).sum())
    gross = float(np.max(balance._force_reference, initial=0.))
    gradient_error = _relative_error(actual_force, -expected_gradient, gross)
    if gradient_error > AGREEMENT_TOLERANCE:
        raise UnsupportedForceLedger(f'Constitutive force ledger disagrees with current potential gradient ({gradient_error:g}).')
    all_ports = list(spatial.cache.values())
    plates = {}
    closure = 0.
    for p in balance.plates:
        arrays = local[p]
        torques = sum(arrays.values(), np.zeros((len(spatial.cells[p]), 3)))
        index = 3*balance.slot[p]
        expected = -expected_gradient[index:index+3]*_TORQUE
        gross_local = local_gross[p]
        error = _relative_error(torques.sum(axis=0), expected, gross_local)
        force_error = _relative_error(actual_force[index:index+3], -expected_gradient[index:index+3],
                                      float(np.max(balance._force_reference[index:index+3], initial=0.)))
        gradient_error = max(gradient_error, force_error)
        closure = max(closure, error)
        owned_ports = [port for port in all_ports if port.owner == p]
        diagnostics = dict(gradient_agreement=force_error, spatial_sum_agreement=error,
            net_torque_n_m=torques.sum(axis=0).tolist(),
            unresolved_torque_n_m=torques.sum(axis=0).tolist(),
            gross_component_torque_n_m=gross_local, off_owned_basal_contributions=off_owned_count[p],
            off_owned_basal_support_weight=off_owned_fraction[p], off_owned_basal_torque_n_m=off_owned_force[p].tolist(),
            maximum_lift_condition=max((port.condition for port in owned_ports), default=1.),
            nearest_owned_fallback_ports=sum(port.nearest_fallback for port in owned_ports),
            maximum_lift_distance_km=max((port.distance_km for port in owned_ports), default=0.),
            spatial_field='local work-preserving reduced lift; differential field is approximate',
            added_equilibrium_reaction=False,
            spatial_control_kind=('positive fractional footprint' if p in spatial.fractional_only
                                  else 'dominant owner controls'),
            categorical_owned_controls=int(np.count_nonzero(spatial.owner == p)),
            fractional_only_controls=len(spatial.cells[p]) if p in spatial.fractional_only else 0)
        plates[p] = dict(cells=spatial.cells[p].copy(), torques_n_m=torques,
                         cell_generalized_force=torques/_TORQUE,
                         localcomponents_n_m=arrays, components_n_m=arrays, diagnostics=diagnostics)
    if gradient_error > AGREEMENT_TOLERANCE:
        raise UnsupportedForceLedger(f'Constitutive force ledger disagrees with a current plate gradient ({gradient_error:g}).')
    if closure > AGREEMENT_TOLERANCE:
        raise UnsupportedForceLedger(f'Spatial lift cannot preserve assembled generalized force ({closure:g}).')
    attached_error = 0.
    if original_edge_forces is not None:
        attached_error = _relative_error(component_generalized.get('attached_slab', zero),
            original_edge_forces['net_generalized_forces'].sum(axis=0), gross)
        if attached_error > AGREEMENT_TOLERANCE:
            raise UnsupportedForceLedger('Attached force ledger disagrees with exact expanded-channel edge forces.')
    return dict(plates=plates, generalized_force=actual_force, expected_generalized_force=-expected_gradient,
        motion_coordinates=x.copy(),
        local_terms=terms, source_balance=balance, delta=float(delta),
        diagnostics=dict(version=1, source='current assembled Balance', complete=True, unsupported_terms=[],
            stiffness_agreement=matrix_error, drive_agreement=drive_error, gradient_agreement=gradient_error,
            spatial_sum_agreement=closure, attached_edge_agreement=attached_error,
            plate_diagnostics={str(p): entry['diagnostics'] for p, entry in plates.items()},
            maximum_lift_condition=max((port.condition for port in all_ports), default=1.),
            fractional_only_plate_slots=sorted(spatial.fractional_only),
            off_owned_basal_contributions=sum(off_owned_count.values()),
            potential_agreement=_relative_error(np.array([actual_value]), np.array([expected_value])),
            spatial_approximation='owned finite-control minimum metric lift; off-owned support retained; '
                'zero-dominant-owner plates use only their positive fractional footprint; '
                'contact region and weld arc represented by their physical port center',
            band_km=float(band_km), physical_laws_changed=False, equilibrium_reaction_added=False))


class CompiledMode(pb.Balance):
    """Balance-compatible potential on a fixed geometry's reduced velocity mode."""
    def _row_rates(self, operator, x):
        common = getattr(self, 'common_size', 0)
        if not common:
            return operator@x
        # Adding daughter columns must not change the native common dot
        # product's reduction at a unilateral activation surface. Split every
        # motion, including nonzero daughter rates, by the same linear map.
        return (np.ascontiguousarray(operator[..., :common])@x[:common]
                +np.ascontiguousarray(operator[..., common:])@x[common:])

    def evaluate(self, z, delta=None):
        return self._evaluate(np.asarray(z, float), self.delta if delta is None else delta)[:3]


def compile_mode(ledger, basis):
    """Project the complete force law with the adjoint of its spatial lift.

    ``basis[p]`` has shape (all_native_cells,3,m), or (ledger_cells,3,m).
    Ledger cells are dominant-owner controls, with a positive fractional
    footprint fallback only for owners having no dominant controls.
    Coordinates and widths retain Balance's cm/year-scaled Euler units; the
    potential is W, gradient W per scaled coordinate. The returned object also
    supports ``_evaluate`` and ``_resistance_work`` with ordinary Balance APIs.
    This is an instantaneous reduced mode, not a changed-geometry full step.
    """
    source = ledger['source_balance']
    n = len(source.s.xyz)
    prepared = {}
    inverse = {}
    m = None
    for p, entry in ledger['plates'].items():
        values = np.asarray(basis[p], float)
        if values.ndim != 3 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError('Finite mode requires aligned finite native-cell Euler basis arrays.')
        if m is None:
            m = values.shape[2]
        if values.shape[2] != m:
            raise ValueError('All finite-mode basis arrays must have the same reduced size.')
        if values.shape[0] == len(entry['cells']) and values.shape[0] != n:
            index = np.full(n, -1, int)
            index[entry['cells']] = np.arange(len(entry['cells']))
            inverse[p] = index
        elif values.shape[0] == n:
            inverse[p] = None
        else:
            raise ValueError('Finite-mode basis length disagrees with native/owned controls.')
        prepared[p] = values
    if not m:
        raise ValueError('Finite-mode basis has no coordinates.')
    result = object.__new__(CompiledMode)
    common = m >= source.size
    if common:
        for p, entry in ledger['plates'].items():
            values = prepared[p] if inverse[p] is not None else prepared[p][entry['cells']]
            expected = np.zeros((3, source.size))
            expected[:, 3*source.slot[p]:3*source.slot[p]+3] = np.eye(3)
            if not np.all(values[:, :, :source.size] == expected):
                common = False
                break
    result.common_size = source.size if common else 0
    result.size = m; result.stiffness = np.zeros((m, m)); result.torque = np.zeros(m)
    result.elements = []; result.hinge = []; result.hinge_coefficient = []
    result.resistance_version = source.resistance_version
    result.delta = ledger['delta']; result.notes = {}; result.partition = None
    groups = {}
    for term in ledger['local_terms']:
        projection = np.vstack([port.basis(prepared[port.owner], inverse[port.owner]) for port in term.ports])
        if term.shape == 'quadratic':
            # Common-plate bases are sparse by block. Avoid a dense m*m multiply
            # for every one of the full-resolution basal quadrature cells.
            columns = np.flatnonzero(np.any(projection != 0., axis=0))
            small = projection[:, columns]
            result.stiffness[np.ix_(columns, columns)] += small.T@term.matrix@small
            result.torque[columns] += small.T@term.drive
        elif term.shape == 'hinge':
            result.hinge.append(term.rows[0]@projection)
            result.hinge_coefficient.append(term.coefficient)
        else:
            key = (term.kind, term.shape)
            if key not in groups:
                groups[key] = dict(kind=term.kind, shape=term.shape, rows=[[] for _ in term.rows],
                                   coefficient=[], scale=[], bounds=[], edges=None)
            group = groups[key]
            for dest, row in zip(group['rows'], term.rows):
                dest.append(row@projection)
            group['coefficient'].append(term.coefficient); group['scale'].append(term.scale)
            if term.shape == 'arc_opening':
                group['bounds'].append(term.bounds)
    result.hinge = np.asarray(result.hinge).reshape(-1, m)
    result.hinge_coefficient = np.asarray(result.hinge_coefficient)
    for group in groups.values():
        group['rows'] = tuple(np.asarray(rows) for rows in group['rows'])
        group['coefficient'] = np.asarray(group['coefficient']); group['scale'] = np.asarray(group['scale'])
        if group['shape'] == 'arc_opening':
            group['bounds'] = np.asarray(group['bounds'])
        result.elements.append(group)
    result.metadata = dict(complete=True, unsupported_terms=[], units='W potential; cm/year-scaled Euler coordinates',
        spatial_approximation=ledger['diagnostics']['spatial_approximation'], changed_geometry=False,
        velocity_dependent_laws_reevaluated=True, band_km=ledger['diagnostics']['band_km'])
    return result
