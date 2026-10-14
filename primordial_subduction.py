"""Optional inherited subduction at every initial primordial-ocean margin.

This is an explicit geological initial condition, not elapsed model history or
a persistent kinematic constraint. Slab weight drives the ordinary force solve;
actual convergence still gates consumption, and normal shutdown remains active.
"""
from copy import deepcopy
import math
import numpy as np

import slab_memory
import trench_history

VERSION = 1
DEFAULT_DEPTH_KM = 100.


def normalize(value=None):
    defaults = dict(enabled=False, initial_slab_depth_km=DEFAULT_DEPTH_KM,
                    target_margin_fraction=1., selection_seed=0)
    if value is None:
        return defaults
    if not isinstance(value, dict) or set(value)-set(defaults):
        raise ValueError('Primordial subduction settings contain unknown fields.')
    result = dict(defaults, **value)
    if type(result['enabled']) is not bool:
        raise ValueError('Primordial subduction enabled must be boolean.')
    depth = result['initial_slab_depth_km']
    if (isinstance(depth, (bool, np.bool_)) or not np.isscalar(depth)
            or not isinstance(depth, (int, float, np.number)) or not np.isfinite(depth)
            or not 1. <= depth <= slab_memory.MAX_SLAB_DEPTH_KM):
        raise ValueError('Initial slab vertical depth must be finite and between 1 and 660 km.')
    fraction = result['target_margin_fraction']
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float, np.number))
            or not np.isfinite(fraction) or not .05 <= float(fraction) <= 1.):
        raise ValueError('Primordial subduction target_margin_fraction must be between 0.05 and 1.')
    seed = result['selection_seed']
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= int(seed) < 2**32):
        raise ValueError('Primordial subduction selection_seed must be an unsigned 32-bit integer.')
    result.update(initial_slab_depth_km=float(depth),
                  target_margin_fraction=float(fraction), selection_seed=int(seed))
    return result


def target_edges(s):
    """Genuine coast interfaces; scalar incoming slot for legacy single oceans.

    Split initial oceans return a per-contact incoming array, which the finite
    capture and trench grouping operators already accept. Ocean-ocean ridges
    are never included in inherited coastal subduction.
    """
    import supercontinent_ring
    if supercontinent_ring.enabled(s):
        source = getattr(s, 'native_initial_owner_interfaces', None)
        if not isinstance(source, dict) or getattr(s, 'native_initial_ownership_consumed', True):
            raise ValueError('Ring subduction requires the unconsumed initial owner interfaces.')
        selected, source_down = supercontinent_ring.source_targets(s, source)
        return _mapped_targets(s, source, selected, source_down)
    uid = getattr(s, 'initial_ocean_plate_uid', None)
    uids = np.asarray(getattr(s, 'initial_ocean_plate_uids', [] if uid is None else [uid]), np.int64)
    oceans = np.flatnonzero(np.asarray(s.active) & np.isin(np.asarray(s.plate_uid), uids))
    if not len(oceans) or len(oceans) != len(uids) or len(np.unique(uids)) != len(uids):
        raise ValueError('Primordial subduction requires identifiable initial primordial ocean plates.')
    continental = np.unique(np.asarray(s.parcel_plate)[np.asarray(s.kind) > 0])
    incoming_p, incoming_q = np.isin(s.bp, oceans), np.isin(s.bq, oceans)
    down = np.where(incoming_p & ~incoming_q, s.bp, np.where(incoming_q & ~incoming_p, s.bq, -1))
    other = np.where(incoming_p, s.bq, s.bp)
    mask = (down >= 0) & np.isin(other, continental) & (s.bl > 1e-10)
    source = getattr(s, 'native_initial_owner_interfaces', None)
    if isinstance(source, dict) and not getattr(s, 'native_initial_ownership_consumed', True):
        if 'kind_a' in source and 'kind_b' in source:
            # A mixed plate can be incoming water on one coast and overriding
            # continent on another. Material on each finite source side, not
            # a plate-wide ocean/continent label, determines initial polarity.
            return _material_targets(s, source, oceans)
        if getattr(s, 'primordial_connectivity_version', 0):
            raise ValueError('Mixed plate subduction requires source-side material metadata.')
        source_pairs = np.sort(np.column_stack((source['owner_a'], source['owner_b'])), axis=1)
        graph_pairs = set(map(tuple, np.sort(np.column_stack((s.bp, s.bq)), axis=1)))
        mapped = np.array([tuple(pair) in graph_pairs for pair in source_pairs])
        coastal = (np.isin(source['owner_a'], oceans) ^ np.isin(source['owner_b'], oceans)) & ~source['ocean']
        parents = np.asarray(s.native_boundary_geometry['contact_index'])
        if len(parents) != int(mapped.sum()):
            raise ValueError('Initial margin source geometry does not align with its boundary mapping.')
        target_length = np.bincount(parents, weights=source['length_km'][mapped]*coastal[mapped], minlength=len(mask))
        mask &= target_length > 1e-10
        expected = float(source['length_km'][coastal].sum())
        if (not math.isclose(float(s.bl[mask].sum()), expected, rel_tol=1e-10, abs_tol=1e-7)
                or not np.allclose(target_length[mask], s.bl[mask], rtol=1e-10, atol=1e-7)):
            raise ValueError('Initial primordial margins are unresolved or mixed with ocean-ocean fronts; refine the control mesh.')
    return (int(oceans[0]) if len(oceans) == 1 else down), mask


def _material_targets(s, source, oceans):
    owner_a, owner_b = np.asarray(source['owner_a']), np.asarray(source['owner_b'])
    water_a, water_b = np.asarray(source['kind_a']) == 0, np.asarray(source['kind_b']) == 0
    coastal = water_a != water_b
    source_down = np.where(water_a, owner_a, owner_b)
    carriers = np.asarray(getattr(s, 'primordial_ocean_carrier_uids', s.plate_uid[oceans]))
    if np.any(~np.isin(s.plate_uid[source_down[coastal]], carriers)):
        raise ValueError('Initial coastal incoming water does not belong to a declared water-region carrier.')
    down, mask = _mapped_targets(s, source, coastal, source_down)
    if not getattr(s, 'primordial_connectivity_version', 0) and len(oceans) == 1:
        return int(oceans[0]), mask
    return down, mask


def _mapped_targets(s, source, coastal, source_down):
    """Map selected source interfaces with their incoming owner onto control contacts."""
    owner_a, owner_b = np.asarray(source['owner_a']), np.asarray(source['owner_b'])
    source_pairs = np.sort(np.column_stack((owner_a, owner_b)), axis=1)
    graph_pairs = set(map(tuple, np.sort(np.column_stack((s.bp, s.bq)), axis=1)))
    mapped = np.array([tuple(pair) in graph_pairs for pair in source_pairs])
    parents = np.asarray(s.native_boundary_geometry['contact_index'])
    if len(parents) != int(mapped.sum()):
        raise ValueError('Initial margin source geometry does not align with its boundary mapping.')
    down = np.full(len(s.bl), -1, int)
    source_ids = np.flatnonzero(mapped)
    for index, parent in zip(source_ids, parents):
        if not coastal[index]:
            continue
        incoming = int(source_down[index])
        if down[parent] >= 0 and down[parent] != incoming:
            raise ValueError('Initial primordial margins have unresolved opposite polarities in one contact; refine the control mesh.')
        down[parent] = incoming
    target_length = np.bincount(parents, weights=source['length_km'][mapped]*coastal[mapped], minlength=len(down))
    mask = (down >= 0) & (target_length > 1e-10)
    expected = float(source['length_km'][coastal].sum())
    if (not math.isclose(float(s.bl[mask].sum()), expected, rel_tol=1e-10, abs_tol=1e-7)
            or not np.allclose(target_length[mask], s.bl[mask], rtol=1e-10, atol=1e-7)):
        raise ValueError('Initial primordial margins are unresolved or mixed with ocean-ocean fronts; refine the control mesh.')
    return down, mask


def selected_target_edges(s, settings=None):
    """Select complete connected inherited arcs from the eligible coast network.

    Defaults retain the historical all-margin behavior. A reduced target uses
    whole local trench components so initialization never creates alternating
    edge-scale on/off polarity along one connected arc. Longer arcs are
    preferred, with a seeded tie/bias term so equally plausible worlds remain
    reproducible without always selecting the same geographic ordering.
    """
    settings = normalize(s.config.get('primordial_subduction') if settings is None else settings)
    down, candidate = target_edges(s)
    if not np.any(candidate) or settings['target_margin_fraction'] >= 1.-1e-12:
        return down, candidate, dict(candidate_length_km=float(np.asarray(s.bl)[candidate].sum()),
                                     selected_length_km=float(np.asarray(s.bl)[candidate].sum()),
                                     candidate_components=len(trench_history._components(
                                         s, candidate, np.broadcast_to(down, (len(s.ba),))
                                         if np.ndim(down)==0 else np.asarray(down))),
                                     selected_components=None,
                                     target_margin_fraction=settings['target_margin_fraction'])
    down_array = np.broadcast_to(down, (len(s.ba),)).astype(int).copy() if np.ndim(down)==0 else np.asarray(down, int).copy()
    groups = trench_history._components(s, candidate, down_array)
    if not groups:
        return down, candidate, dict(candidate_length_km=0., selected_length_km=0.,
                                     candidate_components=0, selected_components=0,
                                     target_margin_fraction=settings['target_margin_fraction'])
    rng = np.random.default_rng(np.random.SeedSequence([settings['selection_seed'], 0x534C4142]))
    ranked = []
    for group in groups:
        length = float(np.asarray(s.bl)[group].sum())
        center = np.sum(np.asarray(s.bmid)[group]*np.asarray(s.bl)[group,None], axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-30)
        # Length dominates; the bounded seeded term breaks near-equal choices
        # and changes geographic selection across seeds without fragmenting arcs.
        bias = .85 + .30*float(rng.random())
        ranked.append((length*bias, length, tuple(np.round(center, 12)), group))
    ranked.sort(key=lambda row:(-row[0], -row[1], row[2]))
    target = settings['target_margin_fraction']*sum(row[1] for row in ranked)
    selected = np.zeros(len(s.ba), bool)
    chosen = 0
    length = 0.
    for _, physical, _, group in ranked:
        if chosen and length >= target:
            break
        selected[group] = True
        length += physical
        chosen += 1
    return down, selected, dict(candidate_length_km=float(sum(row[1] for row in ranked)),
                                selected_length_km=float(length),
                                candidate_components=len(groups),
                                selected_components=int(chosen),
                                target_margin_fraction=settings['target_margin_fraction'],
                                selection_seed=settings['selection_seed'])


def initialize(s):
    import supercontinent_ring
    settings = normalize(s.config.get('primordial_subduction'))
    if not settings['enabled']:
        return False
    if getattr(s, 'primordial_subduction_version', 0) == VERSION:
        return True
    if getattr(s, 'physics_profile_version', 0) != 1 or not slab_memory.conservative(s):
        raise ValueError('Primordial subduction requires reviewed_v1 and the conservative slab inventory.')
    if s.t != 0. or s.steps != 0 or s.trench_systems:
        raise ValueError('Inherited primordial subduction is only initialized in a fresh world.')
    ocean, mask, selection = selected_target_edges(s, settings)
    if not np.any(mask):
        raise ValueError('No resolved primordial-ocean continental interfaces exist in this starting world.')
    depth = settings['initial_slab_depth_km']
    down_dip = depth/math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
    s.primordial_subduction_version = VERSION
    down = np.broadcast_to(ocean, (len(s.ba),)).astype(int).copy()
    groups = trench_history._components(s, mask, down)
    initial_area = initial_mass = 0.
    for edges in groups:
        row = trench_history._birth(s, edges, down, inherited=True)
        incoming = down[edges]
        below = np.where(s.bp[edges] == incoming, s.ba[edges], s.bb[edges])
        age = np.asarray(s.age)[below]
        length = np.asarray(s.bl)[edges]
        area = float(length.sum()*down_dip)
        mass = float(length@slab_memory.excess_mass_per_area_kg_m2(age)*down_dip*1e6)
        buoyancy = float(length@trench_history.ocean_buoyancy(age)*down_dip)
        row.update(slab_initial_area_km2=area, slab_retained_area_km2=area,
                   slab_retained_buoyancy_area_km2=buoyancy,
                   slab_initial_excess_mass_kg=mass, slab_retained_excess_mass_kg=mass,
                   initial_subduction=dict(version=VERSION, vertical_depth_km=depth,
                       down_dip_length_km=down_dip, dip_deg=slab_memory.SUBDUCTION_DIP_DEG,
                       ocean_age_myr=float(np.average(age, weights=length)),
                       mass_source='Existing ocean cooling-age law integrated over the initial trench trace.',
                       provenance='Inherited geological initial condition; no model time or runtime ocean consumption.'))
        slab_memory.refresh_line_load(row)
        slab_memory.validate_row(row, require_mass=True)
        initial_area += area
        initial_mass += mass
        others = np.where(s.bp[edges] == incoming, s.bq[edges], s.bp[edges])
        s.polarity[incoming, others] = incoming
        s.polarity[others, incoming] = incoming
    ocean_uids = np.asarray(s.plate_uid)[np.unique(down[mask])]
    s.primordial_subduction_diagnostics = dict(version=VERSION, enabled=True,
        initial_slab_depth_km=depth, dip_deg=slab_memory.SUBDUCTION_DIP_DEG,
        down_dip_length_km=down_dip, initial_maturity=1.,
        initial_ocean_plate_uid=int(s.initial_ocean_plate_uid),
        initial_ocean_plate_uids=np.asarray(getattr(s, 'initial_ocean_plate_uids', ocean_uids)).tolist(),
        initial_subducting_ocean_plate_uids=ocean_uids.tolist(), initial_trenches=len(groups),
        initial_subducting_water_carrier_uids=ocean_uids.tolist(),
        initial_boundary_edges=int(mask.sum()), initial_trench_length_km=float(s.bl[mask].sum()),
        candidate_trench_length_km=float(selection['candidate_length_km']),
        selected_trench_length_km=float(selection['selected_length_km']),
        target_margin_fraction=float(selection['target_margin_fraction']),
        candidate_margin_components=int(selection['candidate_components']),
        selected_margin_components=(None if selection['selected_components'] is None else int(selection['selected_components'])),
        selection_seed=int(settings['selection_seed']),
        initial_slab_area_km2=initial_area, initial_slab_excess_mass_kg=initial_mass,
        inherited=True, runtime_consumption_km2_at_start=0.,
        scope=('Every resolved interface of the declared supercontinent ring; the carrier plate\'s own skirt is incoming.'
               if supercontinent_ring.enabled(s) else
               'Selected complete connected initial water/continental margin arcs between distinct plates; '
               'the local water carrier is incoming. Shared coast within a mixed plate is internal.'),
        assumption='Specified uniform vertical slab extent, fixed dip and fully developed initial mechanism.',
        evolution='Ordinary force balance, actual-convergence capture and trench shutdown; no permanent forced velocities.')
    trench_history.prepare(s)
    s._record('primordial_subduction', 'Inherited oceanic slabs occupy the selected initial primordial-ocean margins.',
              details=deepcopy(s.primordial_subduction_diagnostics))
    return True


def record_initial_motion(s):
    if getattr(s, 'primordial_subduction_version', 0) != VERSION:
        return
    ocean, mask, _ = selected_target_edges(s)
    import native_subduction
    water = native_subduction.edge_ocean_fraction(s, ocean)[mask]
    speed, length = np.asarray(s.normal_speed)[mask], np.asarray(s.bl)[mask]
    closing, opening = speed < -1e-10, speed > 1e-10
    s.primordial_subduction_diagnostics.update(
        initial_closing_length_km=float(length[closing].sum()),
        initial_opening_length_km=float(length[opening].sum()),
        initial_stationary_length_km=float(length[~(closing | opening)].sum()),
        initial_normal_speed_min_km_myr=float(speed.min()),
        initial_normal_speed_max_km_myr=float(speed.max()),
        initial_incoming_ocean_fraction_min=float(water.min()),
        initial_incoming_ocean_length_km=float(length@water),
        actual_motion='Inherited subduction is an initial mechanism; solved rigid plate motions need not converge at every margin simultaneously.')


def annotate_snapshot(s, frame, control):
    """Show established mechanisms without changing any live kinematic code.

    At t=0 the declared inherited interfaces are shown. Subsequently only
    actually closing, oceanic attached pieces are annotated as subduction.
    Exact spreading pieces keep their divergent classification.
    """
    if getattr(s, 'primordial_subduction_version', 0) != VERSION:
        return
    import boundary_labels
    import native_subduction
    labels = boundary_labels.enabled(s)
    ids = [r['id'] for r in s.trench_systems if r.get('initial_subduction')
           and r['phase'] not in ('shutdown', 'joined')]
    valid = np.isin(s.trench_id, ids) & ((s.down == s.bp) | (s.down == s.bq))
    valid &= native_subduction.edge_ocean_fraction(s, s.down) > 1e-10
    if s.t != 0.:
        valid &= (s.normal_speed < 0.) & (s.trench_maturity > 0.)
    # Mechanism labels already call every closing water-fed trench subduction,
    # inherited or not, so the running override is theirs. What they cannot
    # express is the declared initial condition: at t=0 the inherited interfaces
    # are a statement about the world, not yet a measured motion.
    if labels:
        frame['initial_declared_interface'] = sorted(int(i) for i in ids)
        # Provenance only: which displayed pieces belong to a declared inherited
        # trench. The label itself already comes from the running mechanism.
        for row in frame.get('boundary_segments', []):
            if valid[int(row['contact_index'])]:
                row.setdefault('mechanism', 'inherited_subduction')
    if not labels or s.t == 0.:
        codes = np.asarray(frame['native_boundary_code']).copy()
        codes[valid] = 2
        frame['native_boundary_kinematic_code'] = np.asarray(s.bcode).copy()
        frame['native_boundary_code'] = codes
        for row in frame.get('boundary_segments', []):
            parent = int(row['contact_index'])
            row.setdefault('kinematic_code', int(row['code']))
            if valid[parent] and row['code'] not in (1, 5):
                row.update(code=2, mechanism='inherited_subduction',
                           normal_speed_km_myr=float(s.normal_speed[parent]))
        frame['boundary'] = boundary_labels.cells(s, codes, s.n)[control]
    frame['primordial_subduction_review'] = dict(
        declared_initial_mechanism=s.t == 0., active_mechanism_edges=int(valid.sum()),
        classification=('Inherited initial interfaces at t=0; mechanism labels thereafter.' if labels else
                        'Inherited initial interfaces at t=0; closing oceanic attached subduction thereafter.'),
        boundary_label_version=int(getattr(s, 'boundary_label_version', 0)),
        live_kinematic_codes_unchanged=True)


def snapshot(s):
    if not getattr(s, 'primordial_subduction_version', 0):
        return {}
    return dict(primordial_subduction_version=VERSION,
                primordial_subduction_diagnostics=deepcopy(s.primordial_subduction_diagnostics))
