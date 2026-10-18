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


def advance(world, rotations, dt):
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
        update_water(world)


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
