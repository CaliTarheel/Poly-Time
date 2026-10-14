"""Material lifecycle bridge for reduced crustal columns.

This module changes neither plate motion nor buoyant reference-area parcels.
``structure`` and ``trace_structure`` are dictionaries of checkpoint-safe arrays.
The existing trace ledger remains height = birth + uplift - tectonic lowering
- NET erosion + explicit adjustment. Rebound is diagnostic, never a second
uplift term. Rift-memory extension is only the mechanical lowering subset.
"""
from __future__ import annotations

import numpy as np

import crustal_structure as columns
import crust_inventory
import dense_crust
import material_forcing
import normal_partition
from ridge_interaction import sample_ridge_effects


DIAGNOSTIC_FIELDS = (
    'denudation_m', 'rebound_m', 'thermal_uplift_m', 'thermal_subsidence_m',
    'foreland_subsidence_m', 'foreland_rebound_m', 'magmatic_uplift_m',
    'added_volume_km_per_reference_km2', 'erosion_rate_m_myr', 'foundered_m')
FIELDS = columns.STATE_FIELDS + DIAGNOSTIC_FIELDS


def material_height(kind, relief):
    """Full column elevation; class labels alone must never raise terrain."""
    return 220. + np.asarray(relief) + 180.*(np.asarray(kind) == 2) - 100.*(np.asarray(kind) == 3)


def _new(kind, relief):
    state = columns.initialize_structure(kind, material_height(kind, relief))
    state.update({name: np.zeros(len(kind)) for name in DIAGNOSTIC_FIELDS})
    return state


def initialize_parcels(s):
    """Call once after all original parcel arrays have been created."""
    s.structure = _new(s.kind, s.relief)


def initialize_traces(s, chosen):
    """Copy each marker's actual source column, without sampling or new RNG."""
    selected = np.asarray(chosen)
    if selected.ndim != 1 or len(selected) != len(s.trace_kind):
        raise ValueError('Initial traces must select one source column per marker.')
    s.trace_structure = {name: value[selected].copy() for name, value in s.structure.items()}


def _append(s, count, trace=False):
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('Appended column count must be a nonnegative integer.')
    name = 'trace_structure' if trace else 'structure'
    state = getattr(s, name)
    kinds = s.trace_kind if trace else s.kind
    relief = s.trace_relief_m if trace else s.relief
    if any(len(state[field]) + count != len(kinds) for field in FIELDS):
        raise ValueError('Append columns after material arrays grow, exactly once.')
    if not count:
        return
    new = _new(kinds[-count:], relief[-count:])
    if crust_inventory.present(state):
        crust_inventory.initialize(new)
    if dense_crust.present(state):
        dense_crust.initialize(new, 1200.)
    if 'lip_heat_m' in state:
        new['lip_heat_m'] = np.zeros(count)
    import numerical_accuracy
    if numerical_accuracy.present(state):
        if trace:
            ids=np.asarray(s.parcel_patch);order=np.argsort(ids)
            found=np.searchsorted(ids[order],s.trace_patch[-count:])
            if np.any(found>=len(ids)) or np.any(ids[order[found]]!=s.trace_patch[-count:]):
                raise ValueError('New trace accuracy needs its exact containing material identity.')
            for field in numerical_accuracy.FIELDS:new[field]=s.structure[field][order[found]].copy()
        else:
            numerical_accuracy.initialize_columns(new,s.material_surface['area_km2'][-count:])
    setattr(s, name, {field: np.concatenate((state[field], new[field])) for field in state})


def append_parcels(s, count):
    """Append neutral columns at the actual new juvenile parcel heights."""
    _append(s, count)


def append_traces(s, count):
    """Append independent juvenile marker columns at their birth heights."""
    _append(s, count, trace=True)


def coalesce(s, arc, keep, inverse):
    """Call BEFORE old arc mass/kind/relief arrays are modified.

    Same-owner arc groups are chosen by the existing engine. This preserves
    effective crust volume, mean surface height and mass-weighted diagnostics;
    it neither coalesces continental material nor changes independent traces.
    """
    selected = np.asarray(arc)
    groups = np.asarray(inverse)
    state = s.structure
    subset = {name: state[name][selected] for name in columns.STATE_FIELDS}
    if crust_inventory.present(state):
        subset.update({name: state[name][selected] for name in crust_inventory.FIELDS})
    if dense_crust.present(state):
        subset.update({name: state[name][selected] for name in dense_crust.FIELDS})
    if 'lip_heat_m' in state:
        subset['lip_heat_m'] = state['lip_heat_m'][selected]
    import numerical_accuracy
    if numerical_accuracy.present(state):
        subset.update({name:state[name][selected] for name in numerical_accuracy.FIELDS})
    weights = s.mass[selected]
    new = columns.coalesce_structure(subset, weights, groups)
    count = len(new['thickness_km'])
    totals = np.bincount(groups, weights=weights, minlength=count)
    for name in DIAGNOSTIC_FIELDS:
        new[name] = np.bincount(groups, weights=weights*state[name][selected], minlength=count)/totals
    s.structure = {name: np.concatenate((state[name][keep], new[name])) for name in state}


def _surface_change(state, requested_m, *, conserve_volume):
    """Realize a signed surface change by bounded column thickness alone.

    For shortening, reciprocal effective area preserves T*A. For magmatism,
    area is fixed and actual added volume is returned. Existing neutral
    reference offsets, heat and foreland loads remain unchanged.
    """
    before = columns.elevation(state)
    requested = np.broadcast_to(np.asarray(requested_m, dtype=float), before.shape)
    if not np.isfinite(requested).all():
        raise ValueError('Requested column elevation change must be finite.')
    old_t = state['thickness_km'].copy()
    loaded = before + state['foreland_m']
    desired = loaded + requested
    air_before = np.where(loaded >= 0, loaded, loaded/columns.WATER_FACTOR)
    air_after = np.where(desired >= 0, desired, desired/columns.WATER_FACTOR)
    if conserve_volume:
        minimum = columns.conserved_volume_floor(state)
    else:
        minimum = old_t-columns.fixed_area_removal_capacity(state)
    import numerical_accuracy
    if numerical_accuracy.present(state):
        # Source/erosion capacities still use nominal limits. A zero source
        # request must preserve an admitted V/A column rather than clip it.
        if conserve_volume and np.any(requested!=0):
            raise ValueError('Numerical native columns cannot realize scalar strain without matching material geometry.')
        delta=(air_after-air_before)/columns.AIRY_M_PER_KM
        delta=np.clip(delta,np.minimum(minimum-old_t,0.),np.maximum(columns.MAX_THICKNESS_KM-old_t,0.))
        new_t=old_t+delta
    else:
        new_t = np.clip(old_t + (air_after-air_before)/columns.AIRY_M_PER_KM,
                        minimum, columns.MAX_THICKNESS_KM)
    if not conserve_volume:
        crust_inventory.erode(state, np.maximum(old_t-new_t, 0.))
    state['thickness_km'] = new_t
    if conserve_volume:
        state['area_factor'] *= old_t/state['thickness_km']
        added = np.zeros(before.shape)
    else:
        added = (state['thickness_km']-old_t)*state['area_factor']
        crust_inventory.add(state, np.maximum(added, 0.))
    return columns.elevation(state)-before, added


def _foreland_inputs(s):
    version = getattr(s, 'foreland_loading_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, 1):
        raise ValueError('Unsupported foreland loading version.')
    enabled = getattr(s, 'foreland_loading_enabled', True)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('Foreland loading enabled must be Boolean.')
    if not enabled:
        # A fresh reviewed profile starts with zero foreland displacement.
        # Existing nonzero columns are never reset here: that would erase their
        # recorded history and create an unbooked jump in surface elevation.
        s.foreland_loading_diagnostics = dict(version=version, state='disabled',
            model='local Airy isostasy only; regional correction disabled',
            reason='Positive-only regional correction lacks balancing uplift; signed flexure remains unresolved.',
            changes_material_volume=False)
        return None
    if version == 1:
        import foreland_loading
        return foreland_loading.prepare_native(s)
    collision = normal_partition.collision(s) & (s.normal_speed < 0)
    edges = np.flatnonzero(collision)
    if not len(edges):
        return None
    surface = material_height(s.crust, s.land_relief)
    load = np.maximum(surface[s.ba[edges]], surface[s.bb[edges]])
    if not np.any(load > 500.):
        return None
    return (s.bmid[edges], s.bn[edges], s.bl[edges], s.bp[edges], s.bq[edges], load)


def _target(points, owners, kinds, foreland, *, face_ids=None):
    target = np.zeros(len(points))
    selected = (kinds == 1) | (kinds == 2)
    if foreland is not None and np.any(selected):
        if isinstance(foreland, dict) and foreland.get('version') == 1:
            import foreland_loading
            if face_ids is None:
                raise ValueError('Regional foreland loading requires exact source material identities.')
            target[selected] = foreland_loading.target(points[selected], owners[selected],
                np.asarray(face_ids)[selected], foreland, max_depth_m=columns.MAX_FORELAND_M)
        else:
            target[selected] = columns.foreland_target(points[selected], owners[selected], *foreland)
    return target


def _erosion_request(s, state, kinds, supported_height, dt, exposed):
    return _erosion_plan(s, state, kinds, supported_height, dt, exposed)['denudation_m']


def _erosion_plan(s, state, kinds, supported_height, dt, exposed):
    """Versioned calibration; absence of the flag preserves gross-removal law."""
    version = getattr(s, 'erosion_relief_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, 1, 2):
        raise ValueError('Unsupported erosion relief version.')
    if version == 2:
        import phase_evolution
        if not phase_evolution.enabled(s):
            raise ValueError('Erosion relief version 2 requires active retained phase state.')
    erosion = float(s.config['erosion'])
    timescale = np.where(kinds == 2, 350., 180.)
    height = np.maximum(supported_height, 0.)
    if version == 0:
        return dict(denudation_m=height*(-np.expm1(-dt*erosion/timescale))*exposed)
    # Exposure is a fractional rate, so a fixed exposure and supported height
    # obey the same exponential under timestep subdivision. 180/350 Myr now
    # refer explicitly to NET relief loss, before a thickness floor intervenes.
    target_loss = height*(-np.expm1(-dt*erosion*exposed/timescale))
    if version == 2:
        return columns.ordinary_denudation_for_net_loss(state, target_loss)
    return dict(denudation_m=columns.denudation_for_net_loss(state, target_loss))


def _seed_boundary_memory(s, prefix, points, kinds, belts, source, extension_loss):
    """Carry the actual sampled rift's identity beyond its display pixels."""
    ids = getattr(s, prefix+'rift_id')
    selected = (source >= 0) & (extension_loss > 0) & (ids < 0) & ((kinds == 1) | (kinds == 2))
    if not np.any(selected):
        return
    indices = np.flatnonzero(selected)
    indices = indices[belts['rift_id'][source[indices]] > 0]
    if not len(indices):
        return
    segments = source[indices]
    ids[indices] = belts['rift_id'][segments]
    births = {row['id']: row['started_myr'] for row in s.rift_records}
    getattr(s, prefix+'rift_birth_myr')[indices] = [births[int(rid)] for rid in ids[indices]]
    strike = np.cross(belts['mid'][segments], belts['normal'][segments])
    strike -= points[indices]*np.sum(strike*points[indices], axis=1)[:, None]
    strike /= np.maximum(np.linalg.norm(strike, axis=1)[:, None], 1e-20)
    getattr(s, prefix+'rift_tangent')[indices] = strike


def _evolve(s, shortening, extension, volcanic, dt, foreland, *, trace=False, belts=None,
            parcel_support_m=None, parcel_eligible_km=None, parcel_phase_prepared=None,
            parcel_surface_height_m=None):
    prefix = 'trace_' if trace else ''
    name = prefix+'structure'
    old = getattr(s, name)
    points = s.trace_xyz if trace else s.pos
    owners = s.trace_plate if trace else s.parcel_plate
    kinds = s.trace_kind if trace else s.kind
    relief = s.trace_relief_m if trace else s.relief
    cells = s._indices(points) if trace else s.parcel_cell
    if belts is None:
        local_shortening, local_extension, local_volcanic = (field[cells] for field in (shortening, extension, volcanic))
    else:
        values, sources = material_forcing.sample_belts(points, owners, belts, return_sources=True)
        local_shortening, local_extension, local_volcanic = values.T
    strength = np.where(kinds == 2, .35, 1.)
    target = _target(points, owners, kinds, foreland,
                     face_ids=getattr(s, 'trace_patch' if trace else 'parcel_patch', None))
    state, budget = columns.evolve_structure(
        old, dt, shortening_km_myr=local_shortening, extension_km_myr=local_extension,
        strength=strength, foreland_target_m=target,
        geometric_log_area=getattr(s, prefix+'geometric_log_area', None),
        geometric_accuracy=__import__('numerical_accuracy').column_endpoint(s,trace=trace))
    state.update({field: old[field].copy() for field in DIAGNOSTIC_FIELDS})
    setattr(s, name, state)
    relief += budget['total_delta_m']
    if belts is not None:
        _seed_boundary_memory(s, prefix, points, kinds, belts, sources[:, 1], budget['extension_loss_m'])
    s._remember_rift_extension(prefix, budget['extension_loss_m'])

    # The inversion helper only uses positive/zero stretch to veto active
    # opening. Its finite reservoir is consumed by realized compensated uplift.
    inversion_request = (s._rift_inversion_gain(prefix, extension, dt) if belts is None else
                         s._rift_inversion_gain(prefix, extension, dt, sampled_extension=local_extension))
    geometric = getattr(s, prefix+'geometric_log_area', None) is not None
    if geometric:
        # Rift inversion identifies a portion of realized compressional uplift.
        # It must not shrink the scalar column area a second time while the
        # corresponding geometric face remains unchanged.
        inversion = np.minimum(inversion_request, budget['shortening_gain_m'])
        extra_inversion = np.zeros_like(inversion)
    else:
        inversion, _ = _surface_change(state, inversion_request, conserve_volume=True)
        extra_inversion = inversion
    relief += extra_inversion
    getattr(s, prefix+'inversion_uplift_m')[:] += inversion
    if not trace:
        s._record_rift_inversion(inversion, dt)

    saturation = np.clip(1-relief/7000., 0, 1)
    ordinary = local_volcanic*dt*strength*saturation
    ridge = sample_ridge_effects(s.ridge_episodes, points, s.plate_uid[owners], s.t, dt)['volcanic_addition_m']
    ridge_request = ridge*strength*saturation
    request = ordinary+ridge_request
    magma, volume = _surface_change(state, request, conserve_volume=False)
    # Proportional allocation at the thickness cap does not prefer one source
    # or let the ridge-specific counter exceed the actual total construction.
    ridge_gain = np.divide(magma*ridge_request, request, out=np.zeros(len(points)), where=request > 0)
    relief += magma
    state['magmatic_uplift_m'] += magma
    state['added_volume_km_per_reference_km2'] += volume

    lip = None
    if getattr(s, 'lip_version', 0) == 1:
        import lip_events
        lip = lip_events.evolve_columns(s, dt, trace=trace)
        relief += lip['magmatic_m']+lip['thermal_m']

    # Only positive exposed surface height supplies a denudation request.
    before_erosion = columns.elevation(state)
    supported_height = before_erosion
    support_for_erosion = None
    if getattr(s, 'collision_surface_version', 0) == 1:
        from collision_surface import erosion_support
        support_for_erosion = erosion_support(s, trace=trace, parcel_support_m=parcel_support_m)
        supported_height = before_erosion+support_for_erosion
    exposed = np.ones_like(supported_height)
    if hasattr(s, 'parcel_exposed_fraction'):
        from collision_contacts import erosion_fraction
        exposed = erosion_fraction(s, trace=trace)
        if exposed.shape != supported_height.shape:
            raise ValueError('Physical erosion exposure must align with the evolving material.')
    import surface_erosion
    surface_height = None
    if surface_erosion.version(s):
        if trace:
            surface_height = surface_erosion.trace_values(s, parcel_surface_height_m)
        elif float(s.config['erosion']) == 0.:
            surface_height = np.zeros(len(s.mass))
            s.surface_erosion_diagnostics = dict(version=1, sampling_skipped=True,
                sample_epoch_myr=float(s.t), reason='erosion coefficient is zero')
        else:
            surface_height = surface_erosion.prepare(s)
        supported_height = surface_height
        exposed = np.ones_like(surface_height)  # Integral already includes exposure.
    erosion_plan = _erosion_plan(s, state, kinds, supported_height, dt, exposed)
    requested = erosion_plan['denudation_m']
    removed = np.minimum(requested, columns.fixed_area_removal_capacity(state)*1000.)
    crust_inventory.erode(state, removed/1000.)
    state['thickness_km'] -= removed/1000.
    net_loss = before_erosion-columns.elevation(state)
    relief -= net_loss
    state['denudation_m'] += removed
    state['rebound_m'] += removed-net_loss
    state['erosion_rate_m_myr'] = net_loss/dt

    # Eclogitization and foundering of crust below ECLOGITE_DEPTH_KM: the only
    # sink that reaches a buried sheet, since the erosion above is gated by
    # exposure and a covered column requests essentially nothing. It takes the
    # same slot and the same column semantics as erosion -- thickness leaves, the
    # surface follows by Airy -- but the mass is booked to an explicit
    # mantle-return reservoir instead of a denudation counter, because it does
    # not become sediment. The trace pass reuses the parcel pass's eligibility
    # exactly as it reuses the parcel pass's collision support.
    sink_loss = 0.
    foundered_km = eligible_km = None
    sink_scope = {}
    sink_attribution = None
    phase_prepared = None
    if getattr(s,'retained_dense_crust_version',0)==1:
        import phase_evolution
        before_phase=columns.elevation(state)
        staged,phase_report,sink_attribution=phase_evolution.evolve(s,state,dt,trace=trace,
                                                                  prepared=parcel_phase_prepared)
        state.update(staged)
        foundered_km=phase_report['returned_km']
        phase_prepared=phase_report['prepared']
        sink_scope=phase_report['diagnostics']
        sink_scope['step_contraction_km3']=(float(s.material_surface['area_km2']@phase_report['contraction_km']) if not trace else 0.)
        sink_loss=before_phase-columns.elevation(state)
        relief-=sink_loss
    elif getattr(s, 'foundering_version', 0) in (1, 2):
        import eclogite_sink
        if trace:
            eligible_km = eclogite_sink.trace_values(s, parcel_eligible_km)
        else:
            eligible_km, sink_scope, sink_attribution = eclogite_sink.eligible_thickness_km(
                s, state['thickness_km'])
            sink_scope['ordinary_loss_diagnostics'] = eclogite_sink.ordinary_loss_diagnostics(
                state, eligible_km, dt, area_km2=s.material_surface['area_km2'],
                attribution=sink_attribution)
        foundered_km, sink_loss = eclogite_sink.apply(state, eligible_km, dt)
        relief -= sink_loss

    thermal, foreland_delta = budget['thermal_delta_m'], budget['foreland_delta_m']
    state['thermal_uplift_m'] += np.maximum(thermal, 0.)
    state['thermal_subsidence_m'] += np.maximum(-thermal, 0.)
    state['foreland_rebound_m'] += np.maximum(foreland_delta, 0.)
    state['foreland_subsidence_m'] += np.maximum(-foreland_delta, 0.)
    gain = (budget['shortening_gain_m'] + np.maximum(thermal, 0.)
            + np.maximum(foreland_delta, 0.) + extra_inversion + magma + np.maximum(-sink_loss,0.))
    # Foundering subsidence is TECTONIC lowering, not denudation: it must never
    # reach trace_erosion_m, which gospl_export reads as the sediment supplied to
    # the surface-process model. No rock arrives at the surface here.
    lowering = (budget['extension_loss_m'] + np.maximum(-thermal, 0.)
                + np.maximum(-foreland_delta, 0.) + np.maximum(sink_loss,0.))
    if lip is not None:
        gain += lip['magmatic_m']+np.maximum(lip['thermal_m'], 0.)
        lowering += np.maximum(-lip['thermal_m'], 0.)
    if trace:
        s.trace_uplift_m += gain
        s.trace_extension_m += lowering
        s.trace_erosion_m += net_loss
        s.trace_ridge_uplift_m += ridge_gain
    # Preserve the existing long-lived deformation/suture proxy. It records
    # imposed compressive/volcanic activity even when column bounds are reached.
    suture_name = prefix+'suture'
    old_suture = getattr(s, suture_name)
    active_belt = (local_shortening*2.8 + local_volcanic) > 8.
    setattr(s, suture_name, np.clip(old_suture*np.exp(-dt/600.) + active_belt*dt*.008, 0, 1))
    result = dict(uplift_m=gain, tectonic_lowering_m=lowering, net_erosion_m=net_loss,
                extension_strain=np.maximum(-np.log1p(budget['mechanical_thickness_delta_km']/old['thickness_km']), 0.),
                inversion_uplift_m=inversion, ridge_uplift_m=ridge_gain,
                total_delta_m=gain-lowering-net_loss)
    if getattr(s, 'erosion_relief_version', 0) == 2:
        result['requested_net_erosion_m'] = erosion_plan['requested_net_loss_m']
        result['unmet_net_erosion_m'] = np.maximum(erosion_plan['requested_net_loss_m']-net_loss, 0.)
        result['erosion_limited'] = erosion_plan['limited']
    if not trace and surface_height is not None:
        result['_parcel_surface_height_m'] = surface_height
    if not trace and support_for_erosion is not None:
        result['_parcel_erosion_support_m'] = support_for_erosion
    if not trace and foundered_km is not None:
        result['_parcel_eligible_km'] = eligible_km
        result['_parcel_foundered_km'] = foundered_km
        result['_parcel_sink_loss_m'] = sink_loss
        result['_sink_scope'] = sink_scope
        result['_sink_attribution'] = sink_attribution
        result['_parcel_phase_prepared'] = phase_prepared
    return result


def deform(s, shortening_grid, extension_grid, volcanic_grid, dt, *, spherical_boundaries=False):
    """Evolve all material once per step; grids are km/Myr, km/Myr, m/Myr.

    Root must remove the former direct mechanical uplift/stretch/erosion calls.
    Production uses spherical_boundaries=True to sample continuous finite
    boundary belts directly at each material column. The explicit-grid adapter
    remains available for controlled callers and older engine fixtures.
    This function owns both parcel and trace budgets. The current oceanic
    thermal bathymetry and temporary ridge-window support stay outside these
    buoyant columns. The opt-in surface erosion policy includes them when
    determining the shared exposed elevation that drives a denudation request.
    """
    import phase_evolution, surface_erosion
    surface_erosion.version(s)  # Reject invalid policy before mutating either column set.
    if phase_evolution.enabled(s):
        phase_evolution.require_current_floor(s)
    grids = [np.asarray(value, dtype=float) for value in (shortening_grid, extension_grid, volcanic_grid)]
    if any(grid.shape != (s.n,) or not np.isfinite(grid).all() or np.any(grid < 0) for grid in grids):
        raise ValueError('Structure forcing needs aligned finite nonnegative flat grids.')
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Structure time step must be positive and finite.')
    s._prepare_rift_memory()
    # Unconditional: the foundering counter is one of FIELDS, so a world that
    # predates it needs the column padded before _evolve copies the diagnostic
    # set, whatever its foundering version. Only the clock and the reservoir are
    # version gated, inside these calls.
    import eclogite_sink
    eclogite_sink.ensure_fields(s)
    eclogite_sink.advance_clocks(s, float(dt))
    foreland = _foreland_inputs(s)
    belts = material_forcing.boundary_belts(s) if spherical_boundaries else None
    parcel = _evolve(s, *grids, float(dt), foreland, belts=belts)
    # Markers see the same physical-time support as their source columns.
    # Refreshing it after parcel erosion would prematurely apply a changed
    # lower-column load to traces while parcels used the earlier load.
    support = parcel.pop('_parcel_erosion_support_m', None)
    surface_height = parcel.pop('_parcel_surface_height_m', None)
    eligible = parcel.pop('_parcel_eligible_km', None)
    foundered = parcel.pop('_parcel_foundered_km', None)
    collapse = parcel.pop('_parcel_sink_loss_m', None)
    scope = parcel.pop('_sink_scope', {})
    attribution = parcel.pop('_sink_attribution', None)
    phase_prepared = parcel.pop('_parcel_phase_prepared', None)
    trace = _evolve(s, *grids, float(dt), foreland, trace=True, belts=belts,
                    parcel_support_m=support, parcel_eligible_km=eligible, parcel_phase_prepared=phase_prepared,
                    parcel_surface_height_m=surface_height)
    if getattr(s, 'erosion_relief_version', 0) == 2:
        area = np.asarray(s.material_surface['area_km2'], float)
        total_area = float(area.sum())
        mean = lambda value: float(area@value)/total_area if total_area > 0. else 0.
        s.erosion_relief_diagnostics = dict(version=2,
            model='Net relief decay limited to ordinary upper crust and the phase-free-equivalent 8 km mechanics reserve; dense basement is retained for phase evolution/drainage.',
            limited_columns=int(np.count_nonzero(parcel['erosion_limited'])),
            requested_mean_net_loss_m=mean(parcel['requested_net_erosion_m']),
            realized_mean_net_loss_m=mean(parcel['net_erosion_m']),
            unmet_mean_net_loss_m=mean(parcel['unmet_net_erosion_m']),
            maximum_unmet_net_loss_m=float(parcel['unmet_net_erosion_m'].max(initial=0.)),
            dense_basement_erosion=False)
    if foundered is not None:
        import eclogite_sink
        area = np.asarray(s.material_surface['area_km2'], float)
        returned = eclogite_sink.record(s, foundered, attribution)
        touched = foundered > 0
        if getattr(s,'retained_dense_crust_version',0)==1:
            s.foundering_diagnostics=dict(foundering_version=s.foundering_version,
                model='retained dense phase with local hydrostatic pressure, conductive bath and strength-limited drainage',
                step_returned_km3=returned,cumulative_returned_km3=s.mantle_return_km3['total'],
                faces_touched=int(np.count_nonzero(touched)),
                maximum_surface_subsidence_m=float(np.maximum(collapse,0.).max(initial=0.)),
                maximum_surface_rebound_m=float(np.maximum(-collapse,0.).max(initial=0.)),
                density_or_temperature_resolved='reduced column state',lateral_escape_resolved=False,**scope)
        else:
            s.foundering_diagnostics = dict(version=s.foundering_version,
                model='depth-gated first-order eclogitization and foundering of retained columns',
                eclogite_depth_km=eclogite_sink.ECLOGITE_DEPTH_KM,
                tau_myr=eclogite_sink.FOUNDERING_TAU_MYR,
                heating_delay_myr=eclogite_sink.HEATING_DELAY_MYR,
                returned_fraction=eclogite_sink.RETURNED_FRACTION,
                residual_floor_km=eclogite_sink.RESIDUAL_FLOOR_KM,
                step_returned_km3=returned, faces_touched=int(np.count_nonzero(touched)),
                maximum_removed_km=float(foundered.max(initial=0.)),
                cumulative_returned_km3=s.mantle_return_km3['total'],
                # England & Houseman (1989) put post-delamination collapse at 1-2 km
                # in 5-10 Myr; a catch-up transient can exceed that and must be
                # visible in every frame rather than inferred afterwards.
                maximum_surface_collapse_m_per_myr=float(np.max(collapse, initial=0.)/dt),
                mean_surface_collapse_m_per_myr=float(
                    (area[touched]@collapse[touched])/max(float(area[touched].sum()), 1e-300)/dt)
                    if np.any(touched) else 0.,
                contacts=eclogite_sink.contact_diagnostics(s, attribution, area*foundered),
                continental_slab_pull_coefficient=eclogite_sink.CONTINENTAL_SLAB_PULL,
                changes_reference_area=False, density_or_temperature_resolved=False,
                lateral_escape_resolved=False, **scope)
    if support is not None:
        from collision_surface import refresh
        # Thickness is inside the support digest (collision_surface.py:72), so an
        # upper sheet loses its share of every root the sink has just thinned in
        # this same step rather than a step later.
        refresh(s, getattr(s, '_collision_overlap', None))
    return dict(parcel=parcel, trace=trace)


def extend_interior(s, parcel_strain, trace_strain):
    """Apply one resolved interior-extension increment to material columns.

    Inputs are aligned, finite, nonnegative *logarithmic horizontal strain*
    increments, already integrated over the caller's time step. The mechanical
    solver/caller has already handled relative strength and strain partition;
    this function does not apply those factors a second time. It conserves
    effective column volume with T_new = T_old*exp(-strain) and reciprocal area,
    bounded by the phase-free-equivalent 8 km mechanics reserve. Parcel positions, actual
    footprints, reference area and mass are unchanged.

    Realized extension supplies the existing bounded strain-heating law, with
    no second cooling, erosion, foreland evolution, inversion or time advance.
    Cooling age resets only where thickness really decreases. Trace lowering
    records mechanical subsidence; trace uplift records the actual thermal
    surface gain. Rift-extension memory receives the mechanical subset through
    the normal helper, so callers should seed a progressive scar identity first
    when that history is available. Exclude boundary extension from these
    increments if ``deform`` already accounted for it in the same step.

    The returned parcel/trace budgets include actual_strain, extension_loss_m,
    thermal_uplift_m and total_delta_m. Both inputs and column states are checked
    before either set of material arrays is mutated.
    """
    prepared = []
    for trace, supplied in ((False, parcel_strain), (True, trace_strain)):
        state = s.trace_structure if trace else s.structure
        relief = s.trace_relief_m if trace else s.relief
        strain = np.asarray(supplied, dtype=float)
        if (strain.shape != relief.shape or not np.isfinite(strain).all()
                or np.any(strain < 0)):
            raise ValueError('Interior extension needs one finite nonnegative strain increment per material column.')
        before = columns.elevation(state)
        old_t = state['thickness_km']
        restored = columns.restored_thickness(state)
        limited = np.minimum(strain, np.maximum(np.log(restored/columns.MIN_THICKNESS_KM), 0.))
        new = dict(state)
        new['thickness_km'] = np.maximum(old_t*np.exp(-limited), columns.conserved_volume_floor(state))
        new['area_factor'] = state['area_factor']*old_t/new['thickness_km']
        actual = np.maximum(np.log(old_t/new['thickness_km']), 0.)
        mechanical = columns.elevation(new)
        loss = np.maximum(before-mechanical, 0.)
        # Exact heating-only integration in accumulated strain. The ordinary
        # once-per-step evolution owns the elapsed-time cooling term.
        heating_fraction = -np.expm1(-columns.HEATING_COUPLING*actual)
        new['rift_heat_m'] = np.minimum(
            state['rift_heat_m']+(columns.MAX_RIFT_HEAT_M-state['rift_heat_m'])*heating_fraction,
            columns.MAX_RIFT_HEAT_M)
        new['rift_age_myr'] = np.where(actual > 1e-14, 0., state['rift_age_myr'])
        final = columns.elevation(new)
        thermal = np.maximum(final-mechanical, 0.)
        new['thermal_uplift_m'] = state['thermal_uplift_m']+thermal
        budget = dict(actual_strain=actual, extension_loss_m=loss,
                      thermal_uplift_m=thermal, total_delta_m=final-before)
        prepared.append((trace, new, budget))

    result = {}
    changed = ('thickness_km', 'area_factor', 'rift_heat_m', 'rift_age_myr', 'thermal_uplift_m')
    for trace, new, budget in prepared:
        prefix = 'trace_' if trace else ''
        state = s.trace_structure if trace else s.structure
        relief = s.trace_relief_m if trace else s.relief
        for field in changed:
            state[field] = new[field]
        relief += budget['total_delta_m']
        s._remember_rift_extension(prefix, budget['extension_loss_m'])
        if trace:
            s.trace_extension_m += budget['extension_loss_m']
            s.trace_uplift_m += budget['thermal_uplift_m']
        result['trace' if trace else 'parcel'] = budget
    return result


def _selected_change(s, mask, requested, *, trace, conserve_volume, ledger):
    state_name = 'trace_structure' if trace else 'structure'
    state = getattr(s, state_name)
    kinds = s.trace_kind if trace else s.kind
    relief = s.trace_relief_m if trace else s.relief
    selected = np.asarray(mask)
    if selected.dtype == bool:
        if selected.shape != (len(kinds),):
            raise ValueError('Material selection must align with its columns.')
        selected = np.flatnonzero(selected)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError('Material selection must be a Boolean mask or integer indices.')
    if np.any(selected < 0) or np.any(selected >= len(kinds)):
        raise ValueError('Material selection references a nonexistent column.')
    if len(np.unique(selected)) != len(selected):
        raise ValueError('A column cannot occur twice in a direct material change.')
    subset = {field: value[selected].copy() for field, value in state.items()}
    change, volume = _surface_change(subset, requested, conserve_volume=conserve_volume)
    for field in subset:
        state[field][selected] = subset[field]
    relief[selected] += change
    if trace:
        getattr(s, 'trace_'+ledger+'_m')[selected] += change if ledger == 'adjustment' else -change
    return selected, change, volume


def cut_rift(s, parcel_mask, trace_mask, requested_m=250.):
    """Thin a newly accepted structural rift, recording realized loss once.

    Call after the engine seeds the accepted scar ID and tangent, replacing all
    old direct relief and extension-counter edits. There is no instantaneous
    fictitious thermal pulse; subsequent actual opening supplies rift heating.
    """
    if not np.isscalar(requested_m) or not np.isfinite(requested_m) or requested_m < 0:
        raise ValueError('Requested rift lowering must be finite and nonnegative.')
    result = {}
    for trace, mask in ((False, parcel_mask), (True, trace_mask)):
        prefix = 'trace_' if trace else ''
        if getattr(s, prefix+'geometric_log_area', None) is not None:
            # A new fault changes connectivity, not the thickness of a face
            # whose area has not changed. Subsequent resolved opening thins it.
            selected = np.asarray(mask)
            count = np.count_nonzero(selected) if selected.dtype == bool else len(selected)
            result['trace' if trace else 'parcel'] = np.zeros(count)
            continue
        selected, change, _ = _selected_change(s, mask, -float(requested_m), trace=trace,
                                               conserve_volume=True, ledger='extension')
        state = getattr(s, prefix+'structure')
        state['rift_age_myr'][selected[change < 0]] = 0.
        kinds = s.trace_kind if trace else s.kind
        eligible = (getattr(s, prefix+'rift_id')[selected] > 0) & np.isin(kinds[selected], (1, 2))
        getattr(s, prefix+'rift_extension_m')[selected[eligible]] -= change[eligible]
        result['trace' if trace else 'parcel'] = -change
    return result


def replenish(s, parcel_indices, trace_indices, floor_relief_m):
    """Replace old arc relief floors with actual bounded column addition.

    Arc area flux is handled by the caller. This only raises existing columns
    toward the requested relief floor. Independent marker corrections retain
    the historical explicit-adjustment convention, not a duplicate uplift.
    """
    if not np.isscalar(floor_relief_m) or not np.isfinite(floor_relief_m):
        raise ValueError('Arc replenishment floor must be finite.')
    result = {}
    for trace, indices in ((False, parcel_indices), (True, trace_indices)):
        relief = s.trace_relief_m if trace else s.relief
        indices = np.asarray(indices)
        requested = np.maximum(floor_relief_m-relief[indices], 0.)
        selected, change, volume = _selected_change(s, indices, requested, trace=trace,
                                                    conserve_volume=False, ledger='adjustment')
        state = s.trace_structure if trace else s.structure
        state['magmatic_uplift_m'][selected] += change
        state['added_volume_km_per_reference_km2'][selected] += volume
        result['trace' if trace else 'parcel'] = change
    return result
