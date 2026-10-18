"""Optional artist evolution: persistent provinces, contact belts and material coordinates.

Plate IDs identify moving frames. Province IDs and their reference coordinates identify rock
through ownership changes. Contact histories record intensive strain and age, not a second
inventory of crust. Actual crust volumes remain in RegionWorld's material ledger.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from poly import sphere
from poly.topology import segment_motion
from poly.transfer import segment_distances, transfer_sutures

VERSION = 1
RIFT_HALF_WIDTH_KM = 100.
BREAKUP_THICKNESS_KM = 12.


def enabled(world):
    return getattr(world, 'geology', None) is not None


def activate(world, seed=1):
    """Declare a fresh scenario with sustained collisions and persistent rock identity."""
    if world.time_myr != 0. or world.ledger or enabled(world):
        raise ValueError('Artist geology requires a fresh, unadvanced world.')
    world.geology = dict(version=VERSION, seed=int(seed), next_province=1, next_feature=1,
                         features=[], sea_level_km=0., water_depth_km=None)
    rng = np.random.default_rng(seed)
    for piece in world.crust:
        if piece.kind == 'continent':
            initialize_piece(world, piece, 'inherited', -float(rng.uniform(1000., 3500.)))
    return world


def initialize_piece(world, piece, origin='arc', formed_myr=None):
    if not enabled(world) or piece.kind != 'continent' or piece.geology is not None:
        return
    ident = world.geology['next_province']
    world.geology['next_province'] += 1
    piece.geology = dict(province_id=ident, reference_from_plate=np.eye(3), origin=str(origin),
                         formed_myr=world.time_myr if formed_myr is None else float(formed_myr))


def carry(properties, rotation):
    """The same rock reference after old plate coordinates become R @ old coordinates."""
    if properties is None:
        return None
    result = deepcopy(properties)
    result['reference_from_plate'] = np.asarray(properties['reference_from_plate']) @ rotation.T
    return result


def reference_points(piece, local_points):
    transform = np.eye(3) if piece.geology is None else piece.geology['reference_from_plate']
    return np.asarray(local_points) @ np.asarray(transform).T


def maximum_thickness(world):
    """An upper bound over every continental field, including unmeshed islands.

    A sparse set of map or equal-area samples can miss a narrow high remnant.
    Linear face interpolation is bounded by its nodal values.
    """
    return max((float(c.mesh.thickness.max()) if c.mesh is not None else float(c.thickness_km)
                for c in world.crust if c.kind == 'continent'), default=0.)


def _small_contact(world, piece):
    """The larger body sharing the most actual coast with a uniform remnant."""
    import spherely
    from poly.build import densify
    rings = [densify(r, 25., closed=True) for poly in sphere.polygons(piece.shape) for r in poly]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None
    mids = np.vstack([sphere.unit(r + np.roll(r, -1, axis=0)) for r in rings])
    lengths = np.concatenate([sphere.arc_km(r, np.roll(r, -1, axis=0)) for r in rings])
    world_mids = sphere.rotate(mids, world.frame(piece.plate))
    area, local = piece.area_km2(), {}
    best = None
    for candidate in world.crust:
        if candidate is piece or candidate.kind != 'continent' or candidate.area_km2() <= area:
            continue
        plate = candidate.plate
        if plate not in local:
            xyz = sphere.rotate(world_mids, world.frame(plate).T)
            lon, lat = sphere.to_lonlat(xyz)
            local[plate] = xyz, spherely.points(lon, lat)
        xyz, points = local[plate]
        # A one-metre geometric tolerance recognizes the shared edge after
        # frame conversion; separated islands do not exchange rock across sea.
        hit = np.asarray(spherely.distance(candidate.shape, points)) <= 1.
        length = float(lengths[hit].sum())
        if length > 1e-3 and (best is None or length > best[0]):
            best = length, candidate, xyz[hit]
    return best


def _relax_uniform_contact(piece, receiver, contact, length_km, dt):
    """Implicit finite-volume flow between a small cell and nearby crust.

    The star-shaped coupling conserves volume and has the same yielding
    mobility and 300 km reach as continental spreading. It connects fields
    across a material boundary without merging their province coordinates.
    """
    from scipy.spatial import cKDTree
    from poly.mesh import REACH_KM, REFERENCE_KM, SPREADING_KM2_PER_MYR
    area = piece.area_km2()
    if receiver.mesh is None:
        weights = np.ones(1)
        areas = np.array([receiver.area_km2()])
        thickness = np.array([receiver.thickness_km])
    else:
        field = receiver.mesh
        distance = cKDTree(contact).query(field.nodes, k=1)[0] * sphere.RADIUS_KM
        weights = np.exp(-.5 * (distance / (REACH_KM / 2.)) ** 2)
        weights[distance > REACH_KM] = 0.
        areas = field.node_areas_km2() * receiver.area_km2() / field.area_km2()
        thickness = field.thickness.copy()
    old = float(piece.thickness_km)
    excess = np.maximum((old + thickness) / 2. - REFERENCE_KM, 0.)
    mobility = 1e-3 + (excess / 10.) ** 3
    coupling = (dt * SPREADING_KM2_PER_MYR / REACH_KM ** 2 * length_km / np.sqrt(area)
                * weights * areas * mobility)
    ratio = np.divide(coupling, areas + coupling, out=np.zeros_like(coupling), where=areas > 0.)
    mass = areas * ratio
    shared = (area * old + float(mass @ thickness)) / (area + float(mass.sum()))
    changed = thickness + ratio * (shared - thickness)
    # Close the floating-point residual on the scalar cell. This is an
    # internal exchange, not a source/sink or a chosen thickness ceiling.
    piece.thickness_km = old - float(areas @ (changed - thickness)) / area
    if receiver.mesh is None:
        receiver.thickness_km = float(changed[0])
    else:
        receiver.mesh.thickness = changed
        receiver.thickness_km = receiver.mesh.volume_km3(receiver.area_km2()) / receiver.area_km2()
    return abs(area * (piece.thickness_km - old))


def relax_small_bodies(world, dt, entry):
    """Accrete contacting unresolved bodies after every plate has resolved its losses.

    Doing this during one plate's update would let a later plate cut away the
    newly docked body with its obsolete loss mask, leaving a geometry gap.
    """
    from poly.mesh import MINIMUM_KM2
    small = sorted((c for c in world.crust if c.kind == 'continent' and c.mesh is None
                    and sphere.DETAIL_KM2 <= c.area_km2() < MINIMUM_KM2), key=lambda c: c.area_km2())
    for piece in small:
        found = _small_contact(world, piece)
        if found is None:
            continue
        length, receiver, contact = found
        if piece.plate != receiver.plate:
            owner = piece.plate
            others = [c for c in world.crust if c.plate != owner]
            if not world._dock(piece, owner, receiver.plate, others, entry):
                continue
        if piece.mesh is not None or piece.area_km2() < sphere.DETAIL_KM2:
            continue
        moved = _relax_uniform_contact(piece, receiver, contact, length, dt)
        entry['contact_flow_km3'] = entry.get('contact_flow_km3', 0.) + moved


def separate_bodies(world, entry=None):
    """Disconnected continents are separate material bodies, including for terrane rules.

    A plate can own pieces of several continents. Treating their union as one body made
    the small-terrane test see a continent-sized object and put shortening into tiny remote
    remnants. Preserve the existing local field and total volume while separating ownership.
    """
    import spherely
    from poly.mesh import ContinentMesh, MINIMUM_KM2
    from poly.world import Crust
    if not enabled(world):
        return
    result = []
    for piece in world.crust:
        rings = sphere.polygons(piece.shape) if piece.kind == 'continent' else []
        resolved = sum(abs(sphere.loop_area_km2(r[0])) >= sphere.DETAIL_KM2 for r in rings)
        if resolved <= 1:
            result.append(piece)
            continue
        parts, _ = sphere.parts(piece.shape, minimum_km2=sphere.DETAIL_KM2)
        if len(parts) <= 1:
            result.append(piece)
            continue
        area, volume = piece.area_km2(), piece.volume_km3()
        mesh = piece.mesh
        if mesh is not None:
            centres = sphere.unit(mesh.nodes[mesh.triangles].sum(axis=1))
            lon, lat = sphere.to_lonlat(centres)
            probes = spherely.points(lon, lat)
            used = np.zeros(len(centres), bool)
        children = []
        for shape in parts:
            child = Crust(piece.plate, 'continent', shape, piece.born_myr, piece.thickness_km,
                          geology=deepcopy(piece.geology))
            if mesh is not None:
                spherely.prepare(shape)
                hit = np.asarray(spherely.contains(shape, probes)) & ~used
                used |= hit
                faces = mesh.triangles[hit]
                if len(faces):
                    nodes, remap = np.unique(faces, return_inverse=True)
                    field = ContinentMesh(mesh.nodes[nodes].copy(), remap.reshape(-1, 3),
                                          mesh.thickness[nodes].copy())
                    child.thickness_km = field.volume_km3(child.area_km2()) / child.area_km2()
                    if (child.area_km2() >= MINIMUM_KM2
                            and abs(field.area_km2() - child.area_km2()) > max(1e-5, child.area_km2() * 1e-9)):
                        field = field.recut(shape, child.area_km2(), child.volume_km3())
                    child.mesh = field if child.area_km2() >= MINIMUM_KM2 else None
            children.append(child)
        remapped = sum(c.volume_km3() for c in children)
        factor = volume / remapped if remapped > 0. else 1.
        total_area = sum(c.area_km2() for c in children)
        for child in children:
            child.thickness_km *= factor
            if child.mesh is not None:
                child.mesh.thickness *= factor
            if getattr(piece, 'escape_km2', 0.):
                child.escape_km2 = piece.escape_km2 * child.area_km2() / total_area
        if entry is not None:
            change = total_area - area
            if change >= 0.:
                entry['numerical_area_added_km2'] += change
                entry['gained']['junction'] += change
            else:
                entry['numerical_area_removed_km2'] -= change
                entry['lost']['sliver'] -= change
            entry['partitioned_continental_bodies'] = entry.get('partitioned_continental_bodies', 0) + len(children) - 1
            entry['partition_field_remap_km3'] = entry.get('partition_field_remap_km3', 0.) + remapped - volume
        result.extend(children)
    world.crust = result


def extend_rift(world, pieces, gain, entry):
    """Stretch adjacent continental rock into a spreading gap before oceanization.

    The added footprint comes from thinning the same material within a finite band.
    It is not new continental magma. Once the edge is too thin, subsequent opening
    creates ocean in the usual way; the previously stretched margins remain continental.
    All arguments and fields are in the owning plate's local frame.
    """
    import spherely
    from scipy.spatial import cKDTree
    from poly.mesh import ContinentMesh
    remaining = gain
    for piece in pieces:
        if (piece.kind != 'continent' or piece.mesh is None
                or sphere.area_km2(remaining) < sphere.DETAIL_KM2
                or spherely.distance(piece.shape, remaining) > 1000.):
            continue
        rings = [r for domain in sphere.polygons(piece.shape) for r in domain]
        coast = sphere.union(*[sphere.strip(np.vstack((r, r[:1])), RIFT_HALF_WIDTH_KM) for r in rings])
        candidate = sphere.intersection(remaining, coast)
        if sphere.area_km2(candidate) < sphere.DETAIL_KM2:
            continue
        # Retain only patches actually touching this continental body, not remote
        # oceanic ridge segments that merely share a boundary run with it.
        domains, _ = sphere.parts(candidate)
        domains = [g for g in domains if spherely.distance(piece.shape, g) <= 1000.]
        if not domains:
            continue
        candidate = sphere.intersection(candidate, sphere.union(*domains))
        outline = np.vstack([r for domain in sphere.polygons(candidate) for r in domain])
        edge_thickness = piece.mesh.sample(outline)
        if float(np.mean(edge_thickness)) <= BREAKUP_THICKNESS_KM:
            continue
        area = sphere.area_km2(candidate)
        if area < sphere.DETAIL_KM2:
            continue
        volume = piece.volume_km3()
        shape = sphere.union(piece.shape, candidate)
        grown = sphere.area_km2(shape)
        field = ContinentMesh.build(shape, 0., field=piece.mesh.sample)
        if field is None:
            continue
        areas = field.node_areas_km2() * grown / field.area_km2()
        seed = field.thickness.copy()
        initial = float(areas @ seed)
        if initial > volume:
            distance = cKDTree(outline).query(field.nodes)[0] * sphere.RADIUS_KM
            weight = np.exp(-.5 * (distance / RIFT_HALF_WIDTH_KM) ** 2)
            def inventory(strain):
                return float(areas @ (seed / (1. + strain * weight)))
            lower, upper = 0., 1.
            while inventory(upper) > volume and upper < 1e12:
                upper *= 2.
            if inventory(upper) > volume:
                raise ValueError('The local rift band cannot fund its new continental area.')
            for _ in range(60):
                middle = .5 * (lower + upper)
                if inventory(middle) > volume:
                    lower = middle
                else:
                    upper = middle
            field.thickness = seed / (1. + .5 * (lower + upper) * weight)
        elif initial > 0.:
            # Correct interpolation roundoff without creating or deleting rock.
            field.thickness *= volume / initial
        else:
            raise ValueError('An extending continental field has no material.')
        piece.shape, piece.mesh, piece.thickness_km = shape, field, volume / grown
        remaining = sphere.difference(remaining, candidate)
        entry['continental_extension_km2'] = entry.get('continental_extension_km2', 0.) + area
    return remaining


def transfer_features(world, plate, to, domain, source_frame, target_frame, whole=False):
    if enabled(world):
        world.geology['features'] = transfer_sutures(world.geology['features'], plate, to, domain,
                                                    source_frame, target_frame, whole=whole)


def weak_lines(world):
    if enabled(world):
        for feature in world.geology['features']:
            if feature['kind'] == 'collision' and feature['plate'] in world.regions:
                yield sphere.rotate(feature['points'], world.frame(feature['plate'])), feature['time_myr']


def _runs(kinds):
    start = 0
    for k in range(1, len(kinds) + 1):
        if k == len(kinds) or kinds[k] != kinds[start]:
            yield start, k, kinds[start]
            start = k


def _record(world, boundary, lo, hi, kind, plate, strain_km, dt):
    state = world.geology
    points = sphere.rotate(boundary.points[lo:hi + 1], world.frame(plate).T)
    idents = set(int(x) for x in boundary.idents[lo:hi])
    candidates = []
    midpoint = sphere.unit(points.sum(axis=0))[None]
    for feature in state['features']:
        if feature['plate'] != plate or feature['kind'] != kind:
            continue
        if not idents.intersection(feature['boundary_idents']):
            continue
        edges = np.stack((feature['points'][:-1], feature['points'][1:]), axis=1)
        distance = float(segment_distances(midpoint, edges).min())
        if distance <= 300.:
            candidates.append((distance, feature['uid'], feature))
    if candidates:
        feature = min(candidates, key=lambda item: item[:2])[2]
        feature['points'] = points
        feature['boundary_idents'] = sorted(idents)
    else:
        feature = dict(uid=state['next_feature'], kind=kind, plate=int(plate), points=points,
                       time_myr=float(world.time_myr - dt), last_active_myr=float(world.time_myr),
                       cumulative_strain_km=0., width_km=150. if kind == 'collision' else 100.,
                       boundary_idents=sorted(idents))
        state['next_feature'] += 1
        state['features'].append(feature)
    feature['last_active_myr'] = float(world.time_myr)
    feature['cumulative_strain_km'] += float(strain_km)
    feature['width_km'] = min(600., (150. if kind == 'collision' else 100.)
                              + .3 * feature['cumulative_strain_km'])


def advance(world, rotations, dt, entry=None):
    """Keep each side of collision and continental-opening belts in its own moving frame."""
    for boundary in world.boundaries:
        opening, _ = segment_motion(boundary.points, rotations[boundary.left], rotations[boundary.right])
        for lo, hi, kind in _runs(boundary.kinds):
            if kind not in ('collision', 'ridge'):
                continue
            if kind == 'ridge':
                middle = (lo + hi) // 2
                sides = world._sides(boundary, middle)
                if 'continent' not in sides:
                    continue
                strain = float(np.maximum(opening[lo:hi], 0.).mean()) * dt
                feature_kind = 'rift'
            else:
                strain = float(np.maximum(-opening[lo:hi], 0.).mean()) * dt
                feature_kind = 'collision'
            if strain <= 0.:
                continue
            for plate in (boundary.left, boundary.right):
                _record(world, boundary, lo, hi, feature_kind, plate, strain, dt)
    # Feature geometry must never be left on a retired owner.
    active = set(world.regions)
    if any(f['plate'] not in active for f in world.geology['features']):
        raise ValueError('A geological feature lost its moving plate frame.')
    if world.geology['water_depth_km'] is not None:
        # Water is model state: output cadence must never alter the next erosion step.
        from poly.artist import update_water
        update_water(world, diagnostics=entry)


def validate_piece(properties):
    if properties is None:
        return
    expected = {'province_id', 'reference_from_plate', 'origin', 'formed_myr'}
    if not isinstance(properties, dict) or set(properties) != expected:
        raise ValueError('Invalid province state.')
    ident = properties['province_id']
    matrix = np.asarray(properties['reference_from_plate'])
    if type(ident) is not int or ident < 1 or not isinstance(properties['origin'], str):
        raise ValueError('Invalid province identity.')
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9, rtol=0.)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-9, rtol=0.)):
        raise ValueError('Invalid material reference rotation.')
    if not np.isfinite(properties['formed_myr']):
        raise ValueError('Invalid province formation time.')


def validate(world):
    if not enabled(world):
        if any(piece.geology is not None for piece in world.crust):
            raise ValueError('Rock provinces require their world geology state.')
        return
    state = world.geology
    expected = {'version', 'seed', 'next_province', 'next_feature', 'features',
                'sea_level_km', 'water_depth_km'}
    if not isinstance(state, dict) or set(state) != expected or state['version'] != VERSION:
        raise ValueError('Unsupported artist geology state.')
    for key in ('seed', 'next_province', 'next_feature'):
        if type(state[key]) is not int or (key != 'seed' and state[key] < 1):
            raise ValueError('Invalid geology counter.')
    if not np.isfinite(state['sea_level_km']):
        raise ValueError('Invalid geological sea level.')
    water = state['water_depth_km']
    if water is not None and (not np.isfinite(water) or water < 0.):
        raise ValueError('Invalid initial water inventory.')
    if not isinstance(state['features'], list):
        raise ValueError('Invalid geological features.')
    fields = {'uid', 'kind', 'plate', 'points', 'time_myr', 'last_active_myr',
              'cumulative_strain_km', 'width_km', 'boundary_idents'}
    for feature in state['features']:
        if not isinstance(feature, dict) or set(feature) != fields:
            raise ValueError('Unsupported geological feature.')
        points = np.asarray(feature['points'])
        if (feature['kind'] not in ('rift', 'collision') or feature['plate'] not in world.regions
                or type(feature['uid']) is not int or not 0 < feature['uid'] < state['next_feature']
                or points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()
                or not np.allclose(np.linalg.norm(points, axis=1), 1., atol=1e-9, rtol=0.)):
            raise ValueError('Invalid geological feature geometry or owner.')
        values = [feature[k] for k in ('time_myr', 'last_active_myr', 'cumulative_strain_km', 'width_km')]
        if (not np.isfinite(values).all() or values[0] > values[1]
                or values[1] > world.time_myr + 1e-9 or values[2] < 0. or values[3] <= 0.):
            raise ValueError('Invalid geological feature history.')
        if (not isinstance(feature['boundary_idents'], list)
                or any(type(x) is not int or x < 0 for x in feature['boundary_idents'])):
            raise ValueError('Invalid source boundary identities.')
    for piece in world.crust:
        validate_piece(piece.geology)
        if piece.geology is not None and piece.geology['province_id'] >= state['next_province']:
            raise ValueError('Province counter precedes a recorded rock identity.')
