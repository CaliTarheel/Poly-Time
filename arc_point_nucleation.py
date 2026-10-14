"""Pending volcanic points and conservative compact foundation proposals.

Points expose the existing pending magma inventory; they are not a second
crust reservoir and exert no mechanical force. Seeded compact footprints are
centred on an actual source. Cohort connectivity, contributor containment,
column capacity and geographic exclusion remain the caller's admission gates.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real
import numpy as np
import arc_birth_footprint
import arc_source_cohorts
import material_surface

VERSION = 1
SOURCE_COLUMN_KM = 25.
RAYS = 8
POLICY = dict(version=VERSION, source_column_km=SOURCE_COLUMN_KM,
              minimum_column_km=8., maximum_column_km=75.,
              maximum_constructive_slope_deg=20.,
              representation='references_to_pending_source_origins',
              promotion='funded_compact_footprint_with_connected_contained_sources',
              density='conserved_volume_per_physical_footprint_not_point_count',
              seed='stateless_seed_and_persistent_origin_ids',
              relocation_km=0.)


def _integer(value, minimum=None):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and (minimum is None or value >= minimum))


def _scalar(value, minimum=None):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and (minimum is None or value >= minimum))


def version(s):
    value = getattr(s, 'native_arc_point_version', 0)
    if not _integer(value) or value not in (0, VERSION):
        raise ValueError('Unsupported volcanic point nucleation policy.')
    # Explicit old-policy fixtures disable future-only extensions, exactly as
    # the preceding arc footprint/deposition policies do.
    if value and (getattr(s, 'native_arc_birth_profile_version', 0) != 1
                  or getattr(s, 'native_arc_footprint_version', 0) != 1):
        return 0
    if value and getattr(s, 'native_arc_emplacement_version', 0) != 1:
        raise ValueError('Volcanic points require physical profiles, funded footprints and geographic admission.')
    return int(value)


def rank(seed, origin_ids):
    """Stateless ordering/shape key; observation order cannot change a choice."""
    if not _integer(seed):
        raise ValueError('Volcanic point randomness requires an integer world seed.')
    values = list(origin_ids)
    if not values or any(not _integer(value, 1) for value in values):
        raise ValueError('Volcanic point randomness requires persistent origin IDs.')
    payload = json.dumps([int(seed), sorted(set(map(int, values)))], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('ascii')).digest()[:8], 'big')


def seed_key(s, provenance_rows):
    rows = list(provenance_rows)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Volcanic point seeds require source provenance rows.')
    return rank(s.config['seed'], [row.get('origin_id') for row in rows])


def _feature(point, owner, area, provenance):
    return dict(id=int(provenance['origin_id']), owner=int(owner),
                owner_uid=int(provenance['advection_host_plate_uid']),
                geometry_xyz=[np.asarray(point, float).tolist()],
                volume_km3=float(area)*SOURCE_COLUMN_KM,
                source_time_myr=float(provenance['source_time_myr']),
                status='pending', reason=provenance['reason'])


def snapshot_fields(s):
    if not version(s):
        return {}
    import arc_emplacement_geometry
    pending = getattr(s, 'native_arc_pending', dict(
        xyz=np.empty((0, 3)), owner=np.empty(0, np.int16), area=np.empty(0), source_provenance=[]))
    arc_emplacement_geometry.validate_pending(pending)
    rows = pending.get('source_provenance', [])
    arc_source_cohorts.validate(rows, len(pending['area']))
    features = [_feature(point, owner, area, row)
                for point, owner, area, row in zip(pending['xyz'], pending['owner'], pending['area'], rows)
                if area > 0.]
    return dict(arc_point_version=VERSION, arc_point_policy=dict(POLICY),
                volcanic_point_features=features)


def validate_frame(frame):
    """Every visible point is exactly one already-budgeted pending origin."""
    tag = frame.get('arc_point_version', 0)
    if not _integer(tag) or tag not in (0, VERSION):
        raise ValueError('Unsupported saved volcanic point nucleation policy.')
    if not tag:
        _validate_promotions(frame, enabled=False)
        if any(key in frame for key in ('arc_point_policy', 'volcanic_point_features')):
            raise ValueError('Saved volcanic points require their explicit policy version.')
        return 0
    if any(not _integer(frame.get(key)) or frame[key] != 1 for key in
           ('arc_birth_profile_version', 'arc_footprint_version', 'arc_emplacement_version')):
        raise ValueError('Saved volcanic points require physical profiles and geographic footprints.')
    policy = frame.get('arc_point_policy')
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        raise ValueError('Saved volcanic points require complete policy metadata.')
    for key, expected in POLICY.items():
        actual = policy[key]
        if (isinstance(expected, str) and (not isinstance(actual, str) or actual != expected)
                or isinstance(expected, int) and (not _integer(actual) or actual != expected)
                or isinstance(expected, float) and (not _scalar(actual) or actual != expected)):
            raise ValueError('Invalid saved volcanic point policy: '+key)
    pending = frame.get('arc_pending_source')
    if not isinstance(pending, dict):
        raise ValueError('Volcanic points need their canonical pending source inventory.')
    area = np.asarray(pending.get('area_km2'))
    owners = np.asarray(pending.get('owner'))
    points = np.asarray(pending.get('geometry_xyz'))
    if points.size == 0:
        points = points.reshape(0, 3)
    if owners.size == 0:
        owners = owners.astype(np.int64)
    if (area.ndim != 1 or area.dtype.kind not in 'fiu' or not np.isfinite(area).all() or np.any(area < 0.)
            or owners.shape != area.shape or owners.dtype.kind not in 'iu' or np.any(owners < 0)
            or points.shape != (len(area), 3) or points.dtype.kind not in 'fiu'
            or not np.isfinite(points).all()
            or not np.allclose(np.linalg.norm(points, axis=1), 1., rtol=0., atol=2e-10)):
        raise ValueError('Invalid pending inventory for volcanic points.')
    if (not _integer(pending.get('version')) or pending['version'] != 1
            or not _scalar(pending.get('source_column_km')) or pending['source_column_km'] != SOURCE_COLUMN_KM):
        raise ValueError('Volcanic points require the unchanged source volume convention.')
    rows = pending.get('source_provenance')
    arc_source_cohorts.validate(rows, len(area))
    epoch = frame.get('time_myr')
    if not _scalar(epoch, 0.):
        raise ValueError('Volcanic point snapshots require a finite epoch.')
    expected = {int(row['origin_id']): _feature(point, owner, amount, row)
                for point, owner, amount, row in zip(points, owners, area, rows) if amount > 0.}
    features = frame.get('volcanic_point_features')
    if not isinstance(features, list) or len(features) != len(expected):
        raise ValueError('Volcanic points must reference every positive pending origin exactly once.')
    seen = set()
    for feature in features:
        if not isinstance(feature, dict) or not _integer(feature.get('id'), 1):
            raise ValueError('Invalid volcanic point identity.')
        identity = int(feature['id'])
        if identity in seen or identity not in expected:
            raise ValueError('Volcanic point origin is duplicated or not pending.')
        seen.add(identity)
        reference = expected[identity]
        if set(feature) != set(reference):
            raise ValueError('Volcanic point fields cannot introduce a separate material inventory.')
        for key in ('owner', 'owner_uid'):
            if not _integer(feature[key], 0) or feature[key] != reference[key]:
                raise ValueError('Volcanic point changed its recorded owner or persistent host.')
        point = np.asarray(feature['geometry_xyz'])
        if (point.shape != (1, 3) or point.dtype.kind not in 'fiu' or not np.isfinite(point).all()
                or not np.allclose(point, reference['geometry_xyz'], rtol=0., atol=2e-12)):
            raise ValueError('Volcanic point moved away from its actual pending source.')
        for key in ('volume_km3', 'source_time_myr'):
            if not _scalar(feature[key], 0.) or feature[key] != reference[key]:
                raise ValueError('Volcanic point changed its pending volume or source time.')
        if feature['source_time_myr'] > epoch+1e-10:
            raise ValueError('Volcanic point source cannot be in the future.')
        if feature['status'] != 'pending' or feature['reason'] != reference['reason']:
            raise ValueError('Volcanic point changed its pending provenance decision.')
    _validate_promotions(frame, enabled=True)
    return int(tag)


def _validate_promotions(frame, *, enabled):
    diagnostics = frame.get('arc_material_diagnostics', {})
    if not isinstance(diagnostics, dict):
        raise ValueError('Volcanic point diagnostics must be a mapping.')
    geometry = diagnostics.get('emplacement_geometry', {})
    if not isinstance(geometry, dict):
        raise ValueError('Volcanic point emplacement diagnostics must be a mapping.')
    rows = geometry.get('sources', [])
    if not isinstance(rows, list):
        raise ValueError('Volcanic point source decisions must be a list.')
    promotions = [row for row in rows if isinstance(row, dict) and 'point_promotion' in row]
    if promotions and not enabled:
        raise ValueError('A volcanic point promotion requires its explicit policy version.')
    if not promotions:
        return
    provenance = {}
    originals = frame.get('arc_pending_source', {}).get('source_provenance', [])
    ledger = frame.get('arc_source_placements', [])
    if not isinstance(originals, list) or not isinstance(ledger, list):
        raise ValueError('Point promotion needs pending and enduring source provenance.')
    all_rows = list(originals)
    for entry in ledger:
        if not isinstance(entry, dict) or not isinstance(entry.get('source_provenance'), dict):
            raise ValueError('Point promotion cannot resolve its enduring source provenance.')
        all_rows.append(entry['source_provenance'])
    for origin in all_rows:
        arc_source_cohorts.validate([origin], 1)
        identity = int(origin['origin_id'])
        if identity in provenance:
            previous = provenance[identity]
            fields = (*arc_source_cohorts.LINEAGE, 'component_id', 'advection_host_plate_uid',
                      'source_time_myr', 'original_area_km2')
            if any(previous[field] != origin[field] for field in fields):
                raise ValueError('Point promotion source changed its original lineage or host.')
        provenance[identity] = origin

    def near(actual, expected):
        return abs(actual-expected) <= max(1e-7, abs(expected)*1e-11)

    fields = {'version', 'seed_key', 'source_origin_ids', 'source_volume_km3',
              'footprint_area_km2', 'volume_density_km'}
    for row in promotions:
        receipt = row['point_promotion']
        if (not isinstance(receipt, dict) or set(receipt) != fields
                or not _integer(receipt.get('version')) or receipt['version'] != VERSION
                or row.get('mode') != 'birth'):
            raise ValueError('Point promotion requires a versioned birth receipt.')
        key = receipt['seed_key']
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                or not 0 <= int(key) < 1 << 64 or str(int(key)) != key):
            raise ValueError('Point promotion requires its stateless 64-bit seed key.')
        origins = row.get('source_origin_ids')
        placed = np.asarray(row.get('source_placed_area_km2'))
        if (not isinstance(origins, list) or any(not _integer(value, 1) for value in origins)
                or len(set(origins)) != len(origins) or placed.shape != (len(origins),)
                or placed.dtype.kind not in 'fiu' or not np.isfinite(placed).all() or np.any(placed < 0.)):
            raise ValueError('Point promotion requires the actual per-origin placed partition.')
        consumed = [int(origin) for origin, area in zip(origins, placed) if area > 0.]
        declared = receipt['source_origin_ids']
        if (not isinstance(declared, list) or not consumed or declared != consumed
                or any(not _integer(value, 1) for value in declared)):
            raise ValueError('Point promotion source IDs must match exactly the consumed origins.')
        accepted = row.get('accepted_area_km2')
        profile = row.get('profile_capacity')
        if (not _scalar(accepted, 0.) or accepted <= 0. or not near(float(placed.sum()), accepted)
                or not isinstance(profile, dict) or profile.get('admissible') is not True):
            raise ValueError('Point promotion lacks accepted physical source capacity.')
        values = [receipt[name] for name in ('source_volume_km3', 'footprint_area_km2', 'volume_density_km')]
        if not all(_scalar(value, 0.) and value > 0. for value in values):
            raise ValueError('Point promotion needs positive volume, area and physical density.')
        volume, area, density = values
        if (not near(volume, SOURCE_COLUMN_KM*accepted) or not near(density, volume/area)
                or density < 8.-1e-9 or density > 75.+1e-9
                or not _scalar(profile.get('actual_area_km2'), 0.)
                or not near(area, profile['actual_area_km2'])
                or not _scalar(profile.get('actual_volume_km3'), 0.)
                or not near(volume, profile['actual_volume_km3'])):
            raise ValueError('Point promotion density must follow its conserved physical columns.')
        sources = []
        for identity in consumed:
            source = provenance.get(identity)
            if source is None or source['valid'] is not True or source['advection_host_plate_uid'] <= 0:
                raise ValueError('Point promotion requires qualified surviving source lineage.')
            sources.append(source)
        key = arc_source_cohorts.compatibility_key(sources[0])
        host = sources[0]['advection_host_plate_uid']
        if any(arc_source_cohorts.compatibility_key(source) != key
               or source['advection_host_plate_uid'] != host for source in sources):
            raise ValueError('Point density cannot join unrelated sources or owners.')
        from arc_cohort_footprint import connected
        links = [(source['origin_id'], other) for source in sources for other in source['links']]
        if len(connected(consumed, links, consumed[0])) != len(consumed):
            raise ValueError('Point promotion requires connected consumed source origins.')


def _patch(centre, direction, physical_area_km2, key):
    from native_arc_material import _solve_scale
    centre = np.asarray(centre, float)
    direction = np.asarray(direction, float)
    if (centre.shape != (3,) or direction.shape != (3,) or not np.isfinite(centre).all()
            or not np.isfinite(direction).all() or abs(np.linalg.norm(centre)-1.) > 2e-10):
        raise ValueError('A volcanic point proposal requires a finite unit source and finite strike.')
    centre = centre/np.linalg.norm(centre)
    along = direction-centre*np.dot(centre, direction)
    if np.linalg.norm(along) < 1e-12:
        along = np.cross(centre, [0., 0., 1.] if abs(centre[2]) < .9 else [0., 1., 0.])
    along /= np.linalg.norm(along)
    across = np.cross(centre, along)
    phase = (key/(1 << 64))*2.*np.pi
    phi = np.arange(RAYS)*2.*np.pi/RAYS
    # Low-amplitude broad harmonics preserve a compact convex foundation.
    # The seed changes shape/orientation, never the source point or supply.
    angle = phi+phase
    modulation = 1.+.02*np.cos(3.*phi+phase)+.01*np.cos(2.*phi-phase)
    xy = np.column_stack((np.cos(angle), np.sin(angle)))*modulation[:, None]
    coordinates = np.vstack((np.zeros((1, 2)), .48*xy, xy))
    radius = np.r_[0., np.full(RAYS, .48), np.ones(RAYS)]
    faces = []
    for i in range(RAYS):
        j = (i+1) % RAYS
        faces.extend(((0, 1+i, 1+j), (1+i, 1+RAYS+i, 1+RAYS+j), (1+i, 1+RAYS+j, 1+j)))
    faces = np.asarray(faces, np.int64)
    tangents = coordinates[:, 0, None]*along+coordinates[:, 1, None]*across
    distance = np.linalg.norm(tangents, axis=1)
    bearing = tangents/np.maximum(distance[:, None], 1e-30)

    def evaluate(scale):
        angles = distance*scale
        vertices = centre*np.cos(angles)[:, None]+bearing*np.sin(angles)[:, None]
        vertices[0] = centre
        return vertices, material_surface.spherical_face_areas(vertices, faces)

    high = 1.1/distance.max()
    upper_area = float(evaluate(high)[1].sum())
    if not _scalar(physical_area_km2, 0.) or not 0. < physical_area_km2 < upper_area:
        raise ValueError('Compact volcanic foundation must fit within a minor spherical cap.')
    scale = _solve_scale(evaluate, 0., high, physical_area_km2, low_area=0., high_area=upper_area,
                         seed=high*np.sqrt(physical_area_km2/upper_area))
    vertices, areas = evaluate(scale)
    if (not np.isfinite(vertices).all() or np.any(vertices@centre <= 0.)
            or not np.isfinite(areas).all() or np.any(areas <= 0.)):
        raise ValueError('Compact volcanic foundation has invalid minor-cap geometry.')
    return dict(vertices=vertices, faces=faces, area_km2=areas,
                reference_area_km2=areas*(physical_area_km2/float(areas.sum())),
                profile_radius=radius[faces].mean(axis=1))


def propose(centre, direction, source_area_km2, basement_m, seed_key):
    """A funded compact alternative; caller still checks lineage and exclusion."""
    if not _integer(seed_key, 0) or seed_key >= 1 << 64:
        raise ValueError('A volcanic footprint requires its stateless 64-bit seed key.')
    if not _scalar(source_area_km2, 0.) or source_area_km2 <= 0. or not _scalar(basement_m):
        raise ValueError('Volcanic point promotion requires positive finite supply and a finite datum.')
    factory = lambda area: _patch(centre, direction, area, int(seed_key))
    result = arc_birth_footprint.propose(factory, source_area_km2, basement_m)
    physical_area = (0. if result['plan'] is None else float(result['plan']['area_km2'].sum()))
    result['point_nucleation'] = dict(version=VERSION, seed_key=int(seed_key),
        source_volume_km3=float(source_area_km2)*SOURCE_COLUMN_KM,
        physical_footprint_area_km2=physical_area,
        mean_funded_column_km=(0. if not physical_area else float(source_area_km2)*SOURCE_COLUMN_KM/physical_area),
        admissible=result['plan'] is not None, relocation_km=0.,
        density_basis='conserved_volume_per_physical_footprint_not_point_count')
    return result
