"""Area-based viscous shear between adjacent, overlapping continental sheets.

Reduced thin-layer Couette closure: traction = (eta / h) * relative velocity.
Both horizontal components contribute, including spin wholly tangent to a
contained perimeter. Only adjacent layers touch: a third layer screens direct
coupling of the outer two. Equal/opposite torques dissipate mechanical work.

The default eta=1e20 Pa s and h=13.2 km reproduce the weak-zone values used in
Bottrill, van Hunen & Allen (2012), section 2.1, doi:10.5194/se-3-387-2012.
They are explicit reference parameters, not a calibration of this world or a
temperature/fluids solver. This Newtonian law does not impose plastic yield.
"""
from copy import deepcopy
import math
import numpy as np

import burial_depth
import collision_surface
import eclogite_sink
import mesh_coverage
from exact_polygon import ExactPolygon

VERSION = 1
VISCOSITY_PA_S = 1e20
THICKNESS_M = 13200.


def parameters(s):
    values = getattr(s, 'collision_interface_parameters', None)
    if not isinstance(values, dict) or set(values) != {'viscosity_pa_s', 'thickness_m'}:
        raise ValueError('Area-based contact shear requires explicit interface parameters.')
    if any(isinstance(v, (bool, np.bool_)) or not np.isscalar(v) or not np.isfinite(v) or v <= 0.
           for v in values.values()):
        raise ValueError('Interface viscosity and thickness must be finite and positive.')
    return values


def upgrade(s, *, viscosity_pa_s=VISCOSITY_PA_S, thickness_m=THICKNESS_M):
    """Explicit model boundary, without moving any material or changing config."""
    version = getattr(s, 'collision_interface_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, VERSION):
        raise ValueError('Unsupported collision interface version.')
    if version:
        parameters(s)
        return deepcopy(getattr(s, 'collision_interface_migration', {}))
    from copy import copy
    staged = copy(s)
    staged.collision_interface_parameters = dict(viscosity_pa_s=viscosity_pa_s, thickness_m=thickness_m)
    parameters(staged)
    staged.collision_interface_parameters = {name: float(value) for name, value in staged.collision_interface_parameters.items()}
    # Require valid physical stacks before selecting their mechanical closure.
    interfaces = regions(staged)
    report = dict(from_version=0, to_version=VERSION, time_myr=float(s.t),
                  area_km2=sum(row['area_km2'] for row in interfaces),
                  parameters=deepcopy(staged.collision_interface_parameters),
                  calibration='reference weak-zone parameters; not calibrated to this world')
    s.collision_interface_parameters = staged.collision_interface_parameters
    s.collision_interface_version = VERSION
    s.collision_interface_migration = report
    return deepcopy(report)


def _precise_rotation_integral(polygon):
    """Same analytic integral with guard digits for narrow clipped regions.

    Edge terms scale with perimeter while their sum scales with area. At a
    thin overlap those terms cancel far below double precision, even when the
    polygon is valid. Integrate the supplied rays with guard digits, retaining
    exact clipped rays when binary64 vertices cannot represent the footprint.
    No eigenvalues or areas are corrected after integration.
    """
    from decimal import Decimal, localcontext

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    def atan(value):
        # Half-angle reduction bounds the alternating series geometrically.
        multiplier = 1
        while abs(value) > Decimal('.1'):
            value /= 1+(1+value*value).sqrt()
            multiplier *= 2
        power = total = value
        square = -value*value
        denominator = 1
        while True:
            power *= square
            denominator += 2
            updated = total+power/denominator
            if updated == total:
                return total*multiplier
            total = updated

    exact = isinstance(polygon, ExactPolygon)
    with localcontext() as context:
        context.prec = 80 if exact else 64
        points = []
        lengths = []
        rays = [point[:3] for point in polygon.homogeneous] if exact else polygon
        for point in rays:
            # A positive homogeneous denominator cancels on normalization.
            row = ([Decimal(value) for value in point] if exact else
                   [Decimal.from_float(float(value)) for value in point])
            norm = dot(row, row).sqrt()
            points.append([value/norm for value in row])
            lengths.append(norm)
        area = Decimal(0)
        a = points[0]
        signs = []
        for index, (b, c) in enumerate(zip(points[1:-1], points[2:]), 1):
            ab, ac = ([y-x for x, y in zip(a, other)] for other in (b, c))
            if exact:
                determinant = dot(rays[0], cross(rays[index], rays[index+1]))
                if determinant < 0:
                    raise ValueError('Interface polygon requires positive convex winding.')
                if determinant == 0:
                    continue
                numerator = Decimal(determinant)/(lengths[0]*lengths[index]*lengths[index+1])
            else:
                numerator = dot(a, cross(ab, ac))
            denominator = 1+dot(a, b)+dot(b, c)+dot(c, a)
            if denominator <= 0:
                raise ValueError('Interface polygon must occupy a minor convex spherical patch.')
            signs.append(numerator)
            area += 2*atan(numerator/denominator)
        if any(value > 0 for value in signs) and any(value < 0 for value in signs):
            raise ValueError('Interface polygon has inconsistent convex winding.')
        if area < 0:
            raise ValueError('Interface polygon requires positive convex winding.')
        if exact and area == 0:
            raise ValueError('Interface polygon requires positive convex winding.')
        boundary = [[Decimal(0) for _ in range(3)] for _ in range(3)]
        for index, (a, b) in enumerate(zip(points, points[1:]+points[:1])):
            if exact:
                following = (index+1) % len(points)
                normal = [Decimal(value)/(lengths[index]*lengths[following])
                          for value in cross(rays[index], rays[following])]
            else:
                normal = cross(a, b)
            midpoint = [x+y for x, y in zip(a, b)]
            denominator = 1+dot(a, b)
            if denominator <= 0:
                raise ValueError('Interface polygon has an antipodal edge.')
            for i in range(3):
                for j in range(3):
                    boundary[i][j] -= (normal[i]*midpoint[j]+midpoint[i]*normal[j])/denominator
        metric = np.array([[float((2*area/3 if i == j else 0)+boundary[i][j]/6)
                            for j in range(3)] for i in range(3)])
    return metric, float(area)


def _rotation_metric_uncached(polygon, radius_km):
    """Exact integral of (I-r*r.T) dA, in m2, on a convex unit-sphere polygon.

    On the unit sphere, Laplacian(r_i*r_j)=2*delta_ij-6*r_i*r_j.
    The divergence theorem converts its second moment to boundary integrals.
    Each edge's outward normal is constant and its integral of r is analytic.
    """
    if isinstance(polygon, ExactPolygon):
        if not np.isfinite(radius_km) or radius_km <= 0:
            raise ValueError('Interface integration requires a finite unit-sphere polygon and positive radius.')
        metric, area = _precise_rotation_integral(polygon)
        if np.linalg.eigvalsh(metric).min(initial=0.) < -2e-11*max(area, 1e-30):
            raise ValueError('Interface rotation metric has negative resisting work.')
        return metric*(radius_km*1000.)**2
    polygon = np.asarray(polygon, float)
    if (polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3
            or not np.isfinite(polygon).all() or not np.isfinite(radius_km) or radius_km <= 0
            or not np.allclose(np.linalg.norm(polygon, axis=1), 1., rtol=0., atol=1e-12)):
        raise ValueError('Interface integration requires a finite unit-sphere polygon and positive radius.')
    a = polygon[0]
    angles = [2*math.atan2(float(a@np.cross(b-a, c-a)), float(1+a@b+b@c+c@a))
              for b, c in zip(polygon[1:-1], polygon[2:])]
    signed_area = math.fsum(angles)
    area = abs(signed_area)
    terms = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        normal = np.cross(a, b-a)
        denominator = 1+float(a@b)
        if denominator <= 0:
            raise ValueError('Interface polygon has an antipodal edge.')
        # tan(theta/2)/sin(theta) = 1/(1+cos(theta)): no division by
        # the tiny norm of a clipped edge, nor a round trip through angles.
        terms.append(-(np.outer(normal, a+b)+np.outer(a+b, normal))/denominator)
    boundary = np.array([[math.fsum(term[i, j] for term in terms) for j in range(3)] for i in range(3)])
    metric = np.eye(3)*(2*area/3.)+boundary/6.
    metric = .5*(metric+metric.T)
    cancellation_bound = 32*np.finfo(float).eps*sum(float(np.max(np.abs(term))) for term in terms)
    # A bad positive answer can also pass an eigenvalue test. Recompute when
    # the forward roundoff bound threatens the requested area-relative accuracy.
    if (cancellation_bound > 2e-12*area or any(value < 0 for value in angles)
            or np.linalg.eigvalsh(metric).min(initial=0.) < 0.):
        metric, area = _precise_rotation_integral(polygon)
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues.min(initial=0.) < -2e-11*max(area, 1e-30):
        raise ValueError('Interface rotation metric has negative resisting work.')
    # No eigenvalue projection: retain the analytically integrated operator.
    return metric*(radius_km*1000.)**2


def rotation_metric(polygon, radius_km):
    """Exact unchanged moment, with detached bounded pure-geometry reuse."""
    return burial_depth.reuse_rotation_metric(polygon, radius_km, _rotation_metric_uncached)


def regions(s):
    """One geometry contribution per region of adjacent material interface."""
    surface = s.material_surface
    radius = float(surface.get('radius_km', 6371.))
    triangles = np.asarray(surface['vertices'])[np.asarray(surface['faces'])]
    area = np.asarray(surface['area_km2'], float)
    sheets = np.asarray(s.parcel_collision_sheet)
    owners = np.asarray(s.parcel_plate)
    graph = collision_surface.descendants(s.collision_contacts)
    upper, lower, pair_area, contact = eclogite_sink._depth_pairs(s)
    order = np.lexsort((upper, lower))
    faces, starts = np.unique(lower[order], return_index=True)
    result = []
    geometry_session = burial_depth.partition_geometry_session(triangles, upper)
    spans = list(zip(faces, starts, np.r_[starts[1:], len(order)]))
    partitions = geometry_session.partition_faces(
        (face, order[start:end], area[face], radius) for face, start, end in spans)
    for (face, start, end), (pieces, areas, _) in zip(spans, partitions):
        selected = order[start:end]
        represented = {int(pair): 0. for pair in selected}
        for (polygon, cover), weight in zip(pieces, areas):
            if not cover:
                continue
            for pair in cover:
                represented[pair] += weight
            candidates = [pair for pair in cover if not any(
                sheets[upper[other]] in graph.get(int(sheets[upper[pair]]), ())
                for other in cover if other != pair)]
            if len(candidates) != 1:
                if weight <= area[face]*2e-10:
                    continue  # zero-width internal-edge overlap at clipping precision
                raise ValueError('Contact shear requires a unique adjacent upper layer.')
            chosen = candidates[0]
            top = int(upper[chosen]); one, two = int(owners[top]), int(owners[face])
            if one == two:
                continue
            result.append(dict(top_owner=one, under_owner=two, top_face=top, under_face=int(face),
                contact_id=int(contact[chosen]), area_km2=float(weight), metric_m2=rotation_metric(polygon, radius)))
        if not np.allclose([represented[int(pair)] for pair in selected], pair_area[selected],
                           rtol=2e-9, atol=area[face]*2e-11):
            raise ValueError('Interface regions disagree with cached overlap areas.')
    return result


def assemble(model):
    """Add symmetric plate-pair blocks to the same viscous force solve."""
    s = model.s
    model.interface_stiffness = np.zeros_like(model.stiffness)
    version = getattr(s, 'collision_interface_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, VERSION):
        raise ValueError('Unsupported collision interface version.')
    if not version:
        return
    values = parameters(s)
    coefficient = values['viscosity_pa_s']/values['thickness_m']
    # x is scaled to cm/yr at the model's reference Earth radius.
    from plate_balance import CM_YR_M_S, RADIUS_KM
    factor = CM_YR_M_S*float(s.material_surface.get('radius_km', RADIUS_KM))/RADIUS_KM
    area = 0.
    for row in regions(s):
        top, under = row['top_owner'], row['under_owner']
        if top not in model.slot or under not in model.slot:
            raise ValueError('A retained material interface has an inactive plate owner.')
        a, b = 3*model.slot[top], 3*model.slot[under]
        block = coefficient*factor**2*row['metric_m2']
        for i, j, sign in ((a, a, 1.), (b, b, 1.), (a, b, -1.), (b, a, -1.)):
            model.interface_stiffness[i:i+3, j:j+3] += sign*block
        area += row['area_km2']
    model.stiffness += model.interface_stiffness
    model.notes['collision_interface'] = dict(version=VERSION, adjacent_area_km2=area,
        viscosity_pa_s=values['viscosity_pa_s'], thickness_m=values['thickness_m'],
        temperature_resolved=False, frictional_yield_resolved=False,
        model='Newtonian shear through a finite weak interface layer')


def snapshot(s):
    if not getattr(s, 'collision_interface_version', 0):
        return {}
    return dict(collision_interface_version=s.collision_interface_version,
                collision_interface_parameters=deepcopy(parameters(s)),
                collision_interface_migration=deepcopy(getattr(s, 'collision_interface_migration', {})))


def validate_frame(frame):
    from types import SimpleNamespace
    version = frame.get('collision_interface_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, VERSION):
        raise ValueError('Unsupported saved contact shear version.')
    if version:
        if frame.get('collision_contact_version', 0) != 1:
            raise ValueError('Contact shear requires persistent material layers.')
        parameters(SimpleNamespace(**frame))
    elif 'collision_interface_parameters' in frame or 'collision_interface_migration' in frame:
        raise ValueError('Contact shear state requires its version.')
