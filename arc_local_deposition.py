"""Conserved juvenile magma on an existing source-containing material face.

This future-only fallback creates no footprint and transports no source to a
different face. Each face spends only its own offered magma. The complete arc
component retains its pre-transaction constructive slope limits, including
tectonic oversteepening; unplaceable volume remains pending.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
import numpy as np
import crustal_structure as columns
import crust_inventory

VERSION = 1
SOURCE_COLUMN_KM = 25.
MIN_SOURCE_AREA_KM2 = .001
POLICY = dict(version=VERSION, source_column_km=SOURCE_COLUMN_KM,
              maximum_column_km=75., minimum_source_area_km2=MIN_SOURCE_AREA_KM2,
              routing='source_containing_native_face', fallback='failed_lateral_growth')


def version(s):
    value = getattr(s, 'native_arc_deposition_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (0, VERSION):
        raise ValueError('Unsupported local juvenile deposition policy.')
    # Explicit old-profile experiments disable the associated future-only
    # emplacement extensions, as arc_birth_footprint.version already does.
    if value and getattr(s, 'native_arc_birth_profile_version', 0) != 1:
        return 0
    if value and getattr(s, 'native_arc_emplacement_version', 0) != 1:
        raise ValueError('Local juvenile deposition requires physical profiles and geographic sources.')
    return int(value)


def snapshot_fields(s):
    return dict(arc_deposition_version=VERSION, arc_deposition_policy=dict(POLICY)) if version(s) else {}


def _indices(value, count, name):
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in 'iu' or np.any(result < 0) or np.any(result >= count):
        raise ValueError('Invalid local deposition '+name+'.')
    return result.astype(np.int64, copy=False)


def _digest(*values):
    result = hashlib.sha256()
    for value in values:
        if value is None:
            result.update(b'None')
        elif isinstance(value, dict):
            for name in sorted(value):
                result.update(name.encode()); result.update(_digest(value[name]).encode())
        else:
            value = np.ascontiguousarray(value)
            result.update(str((value.dtype.str, value.shape)).encode()); result.update(value.tobytes())
    return result.hexdigest()


def _state_digest(s, selected, trace):
    surface = s.material_surface
    faces = surface['faces'][selected]
    vertices = np.unique(faces)
    return _digest(faces, surface['vertices'][vertices], surface['area_km2'][selected],
        surface['reference_area_km2'][selected], s.mass[selected], s.parcel_patch[selected],
        s.parcel_plate[selected], s.parcel_arc_id[selected], s.kind[selected], s.parcel_arc_basal_m[selected],
        {k: np.asarray(v)[selected] for k, v in s.structure.items()}, s.relief[selected],
        s.trace_patch[trace], s.trace_xyz[trace], s.trace_relief_m[trace], s.trace_adjustment_m[trace],
        {k: np.asarray(v)[trace] for k, v in s.trace_structure.items()})


def _plan_digest(plan):
    return _digest(plan['selected'], plan['trace_indices'], plan['fields'], plan['trace_fields'],
                   plan['spent_area_km2'], plan['relief_delta_m'], plan['trace_relief_delta_m'],
                   plan['accepted_area_km2'], plan['added_volume_km3'], plan['column_deposition'],
                   plan['profile_capacity'], plan['column_evaluations'], plan['accepted_footprint'])


def _added_column(fields, index, thickness):
    """Stage one column with its own inventory and heat; never reset history."""
    result = {name: np.asarray(value)[index:index+1].copy() for name, value in fields.items()}
    before = columns.elevation(result)
    result['thickness_km'] += thickness
    volume = thickness*result['area_factor']
    crust_inventory.add(result, volume)
    uplift = columns.elevation(result)-before
    result['added_volume_km_per_reference_km2'] += volume
    result['magmatic_uplift_m'] += uplift
    columns._state(result)
    return result


def propose(s, selected, source_faces, source_areas):
    """Return a nonmutating transaction for one existing connected arc.

    ``source_faces`` and ``source_areas`` align with offered magma origins.
    The caller has geographically classified each origin at its exact point.
    There is no transfer of offered volume between distinct receiving faces.
    """
    if not version(s):
        raise ValueError('Local juvenile deposition is not enabled.')
    import arc_birth_profile as profile
    import arc_emplacement_geometry as geometry
    from collision_contacts import _new_components
    selected = _indices(selected, len(s.mass), 'component indices')
    source_faces = _indices(source_faces, len(s.mass), 'source faces')
    source_areas = np.asarray(source_areas, float)
    if (not len(selected) or len(np.unique(selected)) != len(selected)
            or source_areas.shape != source_faces.shape or not np.isfinite(source_areas).all()
            or np.any(source_areas < 0) or not np.isin(source_faces, selected).all()):
        raise ValueError('Local deposition requires aligned source-containing component faces.')
    owner = int(s.parcel_plate[selected[0]])
    arc = int(s.parcel_arc_id[selected[0]])
    surface = s.material_surface
    if (arc <= 0 or np.any(s.kind[selected] != 3) or np.any(s.parcel_arc_id[selected] != arc)
            or np.any(s.parcel_plate[selected] != owner)
            or hasattr(s, 'active') and not s.active[owner]
            or len(np.unique(_new_components(surface['faces'][selected], s.parcel_plate[selected]))) != 1):
        raise ValueError('Local deposition requires one active same-owner connected juvenile component.')
    # A partial component would turn an actual interior edge into an artificial
    # ocean boundary during slope measurement. Require the complete component.
    entire = np.flatnonzero((s.parcel_arc_id == arc) & (s.parcel_plate == owner))
    labels = _new_components(surface['faces'][entire], s.parcel_plate[entire])
    component = labels[np.flatnonzero(entire == selected[0])[0]]
    if set(map(int, selected)) != set(map(int, entire[labels == component])):
        raise ValueError('Local deposition slope stencil omits connected arc material.')
    ids, local_faces = np.unique(surface['faces'][selected], return_inverse=True)
    faces = local_faces.reshape(-1, 3)
    refs = np.bincount(surface['faces'].ravel(), minlength=len(surface['vertices']))
    _, counts = np.unique(surface['faces'][selected], return_counts=True)
    if np.any(refs[ids] != counts):
        raise ValueError('Local deposition requires its complete shared-node slope stencil.')
    vertices = surface['vertices'][ids]
    area = np.asarray(surface['area_km2'])[selected]
    reference = np.asarray(s.mass)[selected]
    basal = s.parcel_arc_basal_m[selected]
    fields = {name: np.asarray(value)[selected].copy() for name, value in s.structure.items()}
    old_height = columns.elevation(fields)
    old_thickness = fields['thickness_km'].copy()
    before = profile.measure(vertices, faces, area, old_height, basal)
    limits = np.maximum(profile.MAX_GRADE, before['face_grade'])
    allowance = max(1e-10, float(limits.max(initial=0.))*1e-10)
    patches = s.parcel_patch[selected]
    trace = np.flatnonzero(np.isin(s.trace_patch, patches))
    at_patch = {int(patch): i for i, patch in enumerate(patches)}
    trace_local = np.array([at_patch[int(patch)] for patch in s.trace_patch[trace]], int)
    trace_fields = {name: np.asarray(value)[trace].copy() for name, value in s.trace_structure.items()}
    old_trace_height = columns.elevation(trace_fields)
    index = {int(face): i for i, face in enumerate(selected)}
    spent = np.zeros(len(source_faces)); evaluations = 0
    for face in sorted(set(map(int, source_faces)), key=lambda face: int(s.parcel_patch[face])):
        local = index[face]; origins = np.flatnonzero(source_faces == face)
        offered = float(source_areas[origins].sum())
        if offered < MIN_SOURCE_AREA_KM2:
            continue
        tracers = np.flatnonzero(trace_local == local)
        capacity = max(0., columns.MAX_THICKNESS_KM-fields['thickness_km'][local])
        if len(tracers):
            capacity = min(capacity, float(np.maximum(columns.MAX_THICKNESS_KM-
                trace_fields['thickness_km'][tracers], 0.).min()))
        high = min(offered*SOURCE_COLUMN_KM, capacity*area[local])
        if high < MIN_SOURCE_AREA_KM2*SOURCE_COLUMN_KM:
            continue

        def candidate(volume):
            nonlocal evaluations
            evaluations += 1
            increment = volume/area[local]
            changed = _added_column(fields, local, increment)
            trace_changed = [(int(i), _added_column(trace_fields, int(i), increment)) for i in tracers]
            heights = columns.elevation(fields)
            heights[local] = columns.elevation(changed)[0]
            measured = profile.measure(vertices, faces, area, heights, basal)
            accepted = bool(np.max(measured['face_grade']-limits, initial=0.) <= allowance
                            and changed['thickness_km'][0] <= columns.MAX_THICKNESS_KM
                            and all(row['thickness_km'][0] <= columns.MAX_THICKNESS_KM for _, row in trace_changed))
            return accepted, changed, trace_changed

        accepted, changed, trace_changed = candidate(high)
        low = high if accepted else 0.
        best = (changed, trace_changed) if accepted else None
        if not accepted:
            for _ in range(40):
                middle = (low+high)*.5
                accepted, changed, trace_changed = candidate(middle)
                if accepted:
                    low = middle; best = changed, trace_changed
                else:
                    high = middle
                if high-low <= max(1e-8, offered*SOURCE_COLUMN_KM*1e-10):
                    break
        if best is None or low < MIN_SOURCE_AREA_KM2*SOURCE_COLUMN_KM:
            continue
        changed, trace_changed = best
        # Account for the actually representable retained column increment.
        actual = float(area[local]*(changed['thickness_km'][0]-fields['thickness_km'][local]))
        if actual > offered*SOURCE_COLUMN_KM+max(1e-8, offered*SOURCE_COLUMN_KM*1e-12):
            raise ValueError('Local deposition exceeds its offered source volume.')
        for name, value in changed.items():
            fields[name][local] = value[0]
        for i, row in trace_changed:
            for name, value in row.items():
                trace_fields[name][i] = value[0]
        spent[origins] = source_areas[origins]*(min(actual/SOURCE_COLUMN_KM, offered)/offered)

    new_height = columns.elevation(fields)
    after = profile.measure(vertices, faces, area, new_height, basal)
    volume_by_face = area*(fields['thickness_km']-old_thickness)
    changed = np.flatnonzero(volume_by_face > 0.)
    added = float(volume_by_face.sum()); accepted_area = float(spent.sum())
    if not np.isclose(added, SOURCE_COLUMN_KM*accepted_area, rtol=2e-11, atol=1e-7):
        raise ValueError('Local juvenile deposition failed its physical/source volume ledger.')
    receipt = dict(version=VERSION, source_faces=source_faces.tolist(), source_face_ids=s.parcel_patch[source_faces].tolist(),
        receiving_faces=selected[changed].tolist(), receiving_face_ids=patches[changed].tolist(),
        face_area_km2=area[changed].tolist(), face_reference_area_km2=reference[changed].tolist(),
        face_volume_added_km3=volume_by_face[changed].tolist(), face_thickness_before_km=old_thickness[changed].tolist(),
        face_thickness_after_km=fields['thickness_km'][changed].tolist(), actual_added_volume_km3=added,
        source_volume_km3=SOURCE_COLUMN_KM*accepted_area, reference_area_change_km2=0.,
        footprint_area_change_km2=0., geometry_unchanged=True)
    capacity = dict(version=1, admissible=accepted_area > 0.,
        reason='accepted' if accepted_area > 0. else 'local_column_or_slope_capacity',
        maximum_grade=after['maximum_grade'], maximum_slope_deg=after['maximum_slope_deg'],
        slope_limit_deg=profile.MAX_CONSTRUCTIVE_SLOPE_DEG, maximum_allowed_grade=float(limits.max(initial=0.)),
        maximum_per_face_grade_excess=float(np.max(after['face_grade']-limits, initial=0.)),
        previous_maximum_grade=before['maximum_grade'])
    footprint_area = float(area.sum())
    unchanged = dict(admissible=True, candidate_area_km2=footprint_area, old_area_km2=footprint_area,
        new_footprint_area_km2=0., lost_old_footprint_area_km2=0., material_obstruction_km2=0.,
        foreign_owner_obstruction_km2=0., foreign_owner_vertices=0, eligible_new_footprint_area_km2=0.,
        area_tolerance_km2=geometry.tolerance(footprint_area), admission_tolerance_km2=1e-9,
        material_exclusion_version=1, exact_material_obstruction=None)
    plan = dict(selected=selected.copy(), trace_indices=trace.copy(), fields=fields, trace_fields=trace_fields,
        spent_area_km2=spent, accepted_area_km2=accepted_area, added_volume_km3=added,
        profile_capacity=capacity, column_deposition=receipt, column_evaluations=evaluations,
        accepted_footprint=unchanged if accepted_area > 0. else None,
        relief_delta_m=new_height-old_height, trace_relief_delta_m=columns.elevation(trace_fields)-old_trace_height,
        state_digest=_state_digest(s, selected, trace))
    plan['plan_digest'] = _plan_digest(plan)
    return plan


def commit(s, plan):
    """Apply a checked proposal atomically after all column/trace checks pass."""
    if not version(s):
        raise ValueError('Local juvenile deposition is not enabled.')
    selected, trace = plan['selected'], plan['trace_indices']
    if (_state_digest(s, selected, trace) != plan['state_digest']
            or _plan_digest(plan) != plan['plan_digest']):
        raise ValueError('Local deposition geometry or column proposal changed before commit.')
    if plan['accepted_area_km2'] <= 0.:
        return plan
    for name, value in plan['fields'].items():
        s.structure[name][selected] = value
    for name, value in plan['trace_fields'].items():
        s.trace_structure[name][trace] = value
    s.relief[selected] += plan['relief_delta_m']
    s.trace_relief_m[trace] += plan['trace_relief_delta_m']
    s.trace_adjustment_m[trace] += plan['trace_relief_delta_m']
    return plan
