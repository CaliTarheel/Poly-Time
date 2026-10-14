"""Opt-in force-limit rifting: tear plates where their own loads can drive a cut.

Every integration step, each plate that carries slab pull is checked by the
rigid-plastic limit analysis in ``plate_limit_analysis``. The check is
warm-started from the previous step's mechanism axis, with a full axis scan at
``rescan_interval_myr``. When the worst mechanism's loading ratio is at least
one, the attached path reevaluates the active slab and boundary operators for
a prospective daughter motion and admits its paid opening work. Historical
nonattached worlds retain their earlier basal-drag estimate. Opening
accumulates along that mechanism. While it stretches, the
cut's edges keep strength S0/beta, with beta = 1 + opening/rift_width_km, so
the rift weakens and localizes (necking) instead of hopping between cuts. The
plate splits at breakup: beta = breakup_stretch for the continental share of
the cut and ``ocean_breakup_opening_km`` for its oceanic share. Below failure
the accumulated opening heals over ``heal_myr``.

Accumulation follows one mechanism. A new worst cut whose smaller side shares
less than ``SAME_MECHANISM_OVERLAP`` with the previous smaller side restarts
the count, because opening on one cut does not weaken a different one. The
larger sides are not compared: two different corner cuts leave nearly the
same remainder.

The commit uses ``native_topology._commit``, the same transaction as every
other native split: daughter viability, intact cratons, trench and slab
transfer, and a fresh force solve on the new geometry. Material moves in whole
source patches: each patch goes to the side holding its centroid, and each
material-bearing cell to the side holding most of its material. The tear is
therefore ragged at patch resolution, and never splits a craton group. With
``forbid_cratons`` the mechanism search itself cannot cut craton cells.

Until the split, the plate stays rigid; the accumulated opening is a
bookkeeping delay for rift development, not represented stretching. Work
receipts refer to independent virtual candidate paths, not simultaneous world
energy or material heating. See SLAB-FORCE-LEDGER.md.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import numpy as np

import plate_balance
import plate_limit_analysis

VERSION = 1
BOUNDED_POLICY_VERSION = 1
BOUNDED_SEARCH = dict(version=1, mode='bounded_axes', max_axes=10)
POLICY_STATE_FIELDS = frozenset({'force_rifting_version', 'force_rifting_state',
                                'force_rifting_diagnostics', 'force_rifting_policy'})
# Breakup is a stretching process, not a fixed opening distance. A continental
# rift of width rift_width_km breaks when stretched by breakup_stretch
# (beta ~ 3-4 at continental breakup: crust thinned from ~35 km to ~10 km);
# oceanic lithosphere localizes its rupture over tens of km. A mixed cut needs
# the continental share of the first plus the oceanic share of the second.
DEFAULTS = dict(enabled=False, rift_width_km=150., breakup_stretch=3., ocean_breakup_opening_km=30.,
                rescan_interval_myr=10., heal_myr=50., forbid_cratons=True, inherited_weakness=False)
_RANGES = dict(rift_width_km=(1., 1000.), breakup_stretch=(1.01, 10.), ocean_breakup_opening_km=(1., 1000.),
               rescan_interval_myr=(0., 1000.), heal_myr=(1., 1000.))
SAME_MECHANISM_OVERLAP = .6


def normalize(value=None):
    if value is None:
        return deepcopy(DEFAULTS)
    if not isinstance(value, dict) or set(value)-set(DEFAULTS)-{'search_policy'}:
        raise ValueError('Invalid force_limit_rifting settings; allowed keys are '+', '.join(sorted(set(DEFAULTS)|{'search_policy'}))+'.')
    result = dict(deepcopy(DEFAULTS), **value)
    for name in ('enabled', 'forbid_cratons', 'inherited_weakness'):
        if type(result[name]) is not bool:
            raise ValueError(f'Force-limit rifting {name} must be boolean.')
    for name, (low, high) in _RANGES.items():
        number = result[name]
        if (isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, float, np.number))
                or not np.isfinite(number) or not low <= number <= high):
            raise ValueError(f'Force-limit rifting {name} must be finite and between {low} and {high}.')
        result[name] = float(number)
    if 'search_policy' in result:
        search = result['search_policy']
        if (not isinstance(search, dict) or set(search)-set(BOUNDED_SEARCH)
                or type(search.get('version', 1)) is not int or search.get('version', 1) != 1
                or search.get('mode', 'bounded_axes') != 'bounded_axes'):
            raise ValueError('Force-rift search policy requires version 1 and bounded_axes mode.')
        search = dict(BOUNDED_SEARCH, **search)
        count = search['max_axes']
        if type(count) is not int or not 6 <= count <= 16:
            raise ValueError('Force-rift bounded search max_axes must be an integer from 6 to 16.')
        result['search_policy'] = search
    return result


def enabled(s):
    return getattr(s, 'force_rifting_version', 0) == VERSION


def initialize(s):
    settings = normalize(s.config.get('force_limit_rifting'))
    if not settings['enabled'] or enabled(s):
        return
    if s.t != 0. or s.steps != 0:
        raise ValueError('Force-limit rifting is a fresh-world policy, not a resume migration.')
    policy = _bounded_record(s, settings) if 'search_policy' in settings else None
    s.force_rifting_version = VERSION
    s.force_rifting_state = {}
    s.force_rifting_diagnostics = dict(version=VERSION, checks=0, commits=0, refusals=[])
    if policy is not None:
        s.force_rifting_policy = policy


def _bounded_record(s, settings):
    import effective_subduction
    if not effective_subduction.enabled(s) or not plate_balance.resistance_version(s) == 1:
        raise ValueError('Bounded force breakup requires Lite and its passive complete force ledger.')
    return dict(version=BOUNDED_POLICY_VERSION, mode='bounded_axes',
        parameters=deepcopy(settings), activation_myr=float(s.t), epoch_myr=float(s.t),
        history='Future matched cut observations only; virtual development, not realized material extension.')


def upgrade(s, value):
    """Explicit future-only policy activation; not a generic resume migration.

    A reviewed source-policy producer must verify exact inherited state and
    publish the new checkpoint. No force solve, constructor, history rebuild,
    retrospective opening, or existing material/rift/initiation change occurs.
    """
    settings = normalize(value)
    if not settings['enabled'] or 'search_policy' not in settings:
        raise ValueError('Explicit force breakup activation requires enabled bounded_axes policy.')
    if any(hasattr(s, name) for name in POLICY_STATE_FIELDS):
        raise ValueError('Force breakup policy/history already exists; it cannot be reactivated or credited twice.')
    if normalize(s.config.get('force_limit_rifting'))['enabled']:
        raise ValueError('Parent already requests force breakup; an explicit disabled parent is required.')
    if s.config.get('physics_profile') != 'reviewed_v1':
        raise ValueError('Force breakup activation requires reviewed_v1.')
    if not np.isfinite(s.t) or s.t < 0.:
        raise ValueError('Force breakup activation requires an accepted nonnegative epoch.')
    # Validate all prerequisites before publishing any field.
    policy = _bounded_record(s, settings)
    s.config = dict(s.config, force_limit_rifting=deepcopy(settings))
    s.force_rifting_version = VERSION
    s.force_rifting_state = {}
    s.force_rifting_diagnostics = dict(version=VERSION, checks=0, commits=0, refusals=[])
    s.force_rifting_policy = policy
    return deepcopy(s.force_rifting_policy)


def bounded_policy(s):
    settings = normalize(s.config.get('force_limit_rifting'))
    if 'search_policy' not in settings:
        if hasattr(s, 'force_rifting_policy'):
            raise ValueError('Recorded bounded force policy lost its selected search configuration.')
        return None
    policy = getattr(s, 'force_rifting_policy', None)
    if (not isinstance(policy, dict) or type(policy.get('version')) is not int
            or policy.get('version') != BOUNDED_POLICY_VERSION
            or policy.get('mode') != 'bounded_axes' or policy.get('parameters') != settings
            or not np.isfinite(policy.get('epoch_myr', np.nan))
            or not np.isfinite(policy.get('activation_myr', np.nan))
            or not 0. <= policy['activation_myr'] <= policy['epoch_myr'] <= float(s.t)):
        raise ValueError('Bounded force breakup lacks exact recorded policy/activation state.')
    return policy


def _loaded_plates(s, balance, *, ledger=None):
    if ledger is not None:
        # Retain local loads even when their rigid sum is zero. Ridge/contact
        # and future upper-plate sources are just as eligible as incoming pull.
        return sorted(int(p) for p, row in ledger['plates'].items()
            if s.active[int(p)] and np.any(s.plate == int(p))
            and np.any(np.asarray(row['cell_generalized_force'], float) != 0.))
    trench = balance.trench_edges
    owners = set(int(p) for p in balance.slab_owner[trench])
    if plate_balance.subduction_response_version(s) == 1:
        owners |= {int(balance.bq[e]) if balance.slab_owner[e] == balance.bp[e] else int(balance.bp[e]) for e in trench}
    return sorted(p for p in owners if s.active[p] and np.any(s.plate == p))


def _stable_cut(s, plate, found, *, continuity=None):
    """Exact owner/control/material support identity; no overlap-based credit."""
    cells = np.asarray(found['cells'], dtype='<i8')
    piece = np.asarray(found['piece'], bool)
    side = np.sort(cells[piece])
    other = np.sort(cells[~piece])
    digest = hashlib.sha256()
    # The oriented side matters: the same cut with exchanged moving sides
    # cannot borrow a path paid for the opposite relative opening direction.
    for value in (np.array([int(s.plate_uid[plate])], dtype='<i8'), np.sort(cells), side, other,
                  np.sort(np.asarray(found['cut_edges'], dtype='<i8'))):
        digest.update(np.array([len(value)], dtype='<i8').tobytes()); digest.update(value.tobytes())
    material_fields = ('parcel_plate', 'parcel_patch', 'pos', 'mass',
                       'parcel_craton', 'parcel_cell', '_indices', 'n')
    represented = [hasattr(s, name) for name in material_fields]
    if any(represented) and not all(represented):
        raise ValueError('Force-rift support identity requires complete represented material support.')
    if all(represented):
        mask = np.zeros(len(s.xyz), bool); mask[side] = True
        patch_side, _ = _material_sides(s, plate, mask.copy())
        selected = np.asarray(s.parcel_plate) == plate
        pairs = np.column_stack((np.asarray(s.parcel_patch)[selected], patch_side[selected])).astype('<i8')
        pairs = np.unique(pairs, axis=0)
        digest.update(pairs.tobytes())
        if continuity is not None:
            continuity.update(_fixed_interface_proof(s, plate, found, side, pairs))
    return digest.hexdigest()


def _same_supported_cut(previous, identity, axis):
    if previous is None or previous.get('support_identity') != identity:
        return False
    old = np.asarray(previous['axis'], float); now = np.asarray(axis, float)
    return old.shape == now.shape == (3,) and np.linalg.norm(old-now) <= 1e-10


def _fixed_interface_proof(s, plate, found, side, material_pairs):
    """Future-only proof for one fixed native virtual interface, not transport.

    Remote complementary control labels do not enter the global piece indicator.
    The finite-mode field is U + (I_piece-alpha)*q*axis: a changed support-area
    fraction only shifts the freely solved common U. Exact oriented cut geometry
    keeps its signed relative velocity and length-weighted opening coordinate.
    Current forces, strengths and capacity still price every new increment.
    """
    if found.get('opening_mode_supported') is not True or found.get('common_motion_relaxed') is not True:
        return {}
    fields = ('edge_a', 'edge_b', 'edge_mid', 'edge_length')
    if not all(hasattr(s, name) for name in fields):
        return {}
    cut = np.sort(np.asarray(found['cut_edges'], dtype='<i8'))
    a, b = np.asarray(s.edge_a), np.asarray(s.edge_b)
    mid, length = np.asarray(s.edge_mid), np.asarray(s.edge_length)
    xyz = np.asarray(s.xyz)
    if (not len(cut) or cut.ndim != 1 or len(np.unique(cut)) != len(cut)
            or a.ndim != 1 or b.shape != a.shape or mid.shape != (len(a), 3)
            or length.shape != a.shape or xyz.ndim != 2 or xyz.shape[1] != 3
            or np.any(cut < 0) or np.any(cut >= len(a))):
        return {}
    endpoints = np.column_stack((a[cut], b[cut])).astype('<i8')
    if np.any(endpoints < 0) or np.any(endpoints >= len(xyz)):
        return {}
    selected = np.zeros(len(xyz), bool)
    selected[side] = True
    signs = selected[endpoints[:, 1]].astype('<i8')-selected[endpoints[:, 0]].astype('<i8')
    geometry = (np.asarray(xyz[endpoints], dtype='<f8'),
                np.asarray(mid[cut], dtype='<f8'), np.asarray(length[cut], dtype='<f8'))
    # Every edge must cross this oriented partition. A patch must have exactly
    # one assigned side; ambiguous represented material never gains continuity.
    if (np.any(signs == 0) or not all(np.isfinite(v).all() for v in geometry)
            or np.any(geometry[-1] <= 0.)
            or len(np.unique(material_pairs[:, 0])) != len(material_pairs)):
        return {}
    # Apply the existing _cut geometric validity thresholds; hashing a
    # degenerate contour must never qualify continuity on its own.
    axis = np.asarray(found['axis'], float)
    radial_norm = np.linalg.norm(geometry[1], axis=1)
    if (axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0.
            or np.any(radial_norm <= 0.)):
        return {}
    radial = geometry[1]/radial_norm[:, None]
    normal = geometry[0][:, 1]-geometry[0][:, 0]
    normal -= radial*np.sum(normal*radial, axis=1)[:, None]
    if (np.any(np.linalg.norm(normal, axis=1) <= 1e-14)
            or float(np.average(np.linalg.norm(np.cross(axis/np.linalg.norm(axis), radial), axis=1),
                                weights=geometry[2])) <= 1e-14):
        return {}
    digest = hashlib.sha256()
    for value in (np.array([int(s.plate_uid[plate])], dtype='<i8'), side, cut,
                  endpoints, signs, *geometry, material_pairs):
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape, dtype='<i8').tobytes())
        digest.update(value.tobytes())
    return dict(version=1, scope='fixed_native_virtual_interface', identity=digest.hexdigest(),
                owner_uid=int(s.plate_uid[plate]), piece_count=len(side),
                cut_edge_count=len(cut), material_patch_count=len(material_pairs))


def _valid_fixed_interface_proof(proof):
    keys = {'version', 'scope', 'identity', 'owner_uid', 'piece_count', 'cut_edge_count', 'material_patch_count'}
    if (not isinstance(proof, dict) or set(proof) != keys or type(proof['version']) is not int
            or proof['version'] != 1 or proof['scope'] != 'fixed_native_virtual_interface'
            or not isinstance(proof['identity'], str) or len(proof['identity']) != 64
            or any(c not in '0123456789abcdef' for c in proof['identity'])):
        return False
    return all(type(proof[name]) is int and proof[name] >= minimum for name, minimum in
               (('owner_uid', 0), ('piece_count', 1), ('cut_edge_count', 1), ('material_patch_count', 0)))


def _same_fixed_interface(previous, continuity, axis):
    """No legacy reconstruction, overlap match, reversal or retired-path lookup."""
    if previous is None:
        return False
    before = previous.get('fixed_interface_continuity')
    if (not _valid_fixed_interface_proof(before) or not _valid_fixed_interface_proof(continuity)
            or before != continuity):
        return False
    old, now = np.asarray(previous['axis'], float), np.asarray(axis, float)
    return old.shape == now.shape == (3,) and np.linalg.norm(old-now) <= 1e-10


def _smaller_side(piece, cells):
    """The smaller of a cut's two sides within the plate's current cells."""
    cells = set(int(c) for c in cells)
    side = set(int(c) for c in piece) & cells
    other = cells-side
    return side if len(side) <= len(other) else other


def _overlap(first, second):
    """Jaccard overlap of two cell sets."""
    first, second = set(first), set(second)
    if not first or not second:
        return 0.
    return len(first & second)/len(first | second)


def _retire_mode(diagnostics, uid, row, time_myr, reason):
    """Retain paid virtual-path provenance after its active row disappears.

    These independently evaluated paths are not simultaneous physical world
    energy or material heating. Healing and mode replacement never refund
    already recorded resistance work.
    """
    if row is None or row.get('paid_cut_work_j', 0.) <= 0.:
        return
    edges = np.sort(np.asarray(row['cut_edges'], dtype='<i8'))
    diagnostics.setdefault('retired_candidate_paths', []).append(dict(
        plate_uid=int(uid), since_myr=float(row['since_myr']),
        retired_myr=float(time_myr), reason=reason,
        cut_edges_sha256=hashlib.sha256(edges.tobytes()).hexdigest(),
        cut_edge_count=len(edges), axis=deepcopy(row['axis']),
        opened_km=float(row['opened_km']), paid_cut_work_j=float(row['paid_cut_work_j']),
        last_opening_work=deepcopy(row.get('last_opening_work')),
        interpretation='Retired independent virtual candidate path; not world energy or material heat.'))


def _paid_opening(s, plate, found, opened_km, settings, dt):
    """Accept held-mode opening only within its explicit cut-work budget.

    This is the existing rigid-until-breakup bookkeeping path. Other forces
    and the selected mode are held over this accepted source interval; this
    receipt is not a resolved deforming strip or a finite-time energy solve.
    """
    rate = float(found['predicted_opening_cm_yr_mean'])*10.
    proposed_m = max(rate*dt*1000., 0.)
    if proposed_m == 0.:
        return 0., dict(cut_work_j=0., held_mode_heat_j=0., proposed_increment_m=0., accepted_increment_m=0.)
    if not found.get('opening_mode_supported', False):
        raise ValueError('Loaded force-rift mode has no complete opening law: '+str(found.get('opening_mode_reason')))
    import cohesive_failure
    mode = found['mode_work']
    seconds = dt*plate_balance.SECONDS_PER_MYR
    signed_remaining = float(mode['driving_power_w']-mode['resisting_power_w'])
    cut_power = float(mode['cut_power_w'])
    closure_w = signed_remaining-cut_power
    if not np.all(np.isfinite([signed_remaining, cut_power, closure_w])) or cut_power < 0.:
        raise ValueError('Force-rift cut work is not finite and nonnegative; no history accepted.')
    available = max(min(cut_power, signed_remaining), 0.)*seconds
    width_m, start_m = settings['rift_width_km']*1000., opened_km*1000.
    # An equivalent frozen-mode intact capacity preserves the currently
    # selected/scaled cut force at this starting coordinate. A changed cut
    # is not credited with a different mechanism's past paid work.
    capacity = float(found['current_capacity_n'])*(1.+start_m/width_m)
    admitted = cohesive_failure.increment_for_work_m(capacity, start_m, width_m, available, proposed_m)
    if admitted < proposed_m*(1.-1e-7):
        raise ValueError('Force-rift opening has unresolved or insufficient cut work; no history accepted. '
                         f'Cut power={cut_power!r} W, signed remaining={signed_remaining!r} W, '
                         f'absolute closure={abs(closure_w)!r} W.')
    work = cohesive_failure.necking_work_j(capacity, start_m, admitted, width_m)
    fraction = admitted/proposed_m
    if fraction < 1.:
        # Preserve the common rotation while rounding the differential
        # increment down to its affordable value. A substantial discrepancy
        # is rejected above, rather than replacing the mechanical solution.
        support = np.asarray(s.support[plate] if hasattr(s, 'support') else s.plate == plate, float)
        selected = np.zeros(len(s.xyz), bool)
        selected[found['cells'][found['piece']]] = True
        area = np.asarray(s.cell_area)*support
        alpha = float(area[selected].sum()/area.sum())
        first, rest = np.asarray(found['piece_rotation_rad_myr']), np.asarray(found['rest_rotation_rad_myr'])
        common, differential = alpha*first+(1.-alpha)*rest, first-rest
        found['piece_rotation_rad_myr'] = (common+(1.-alpha)*fraction*differential).tolist()
        found['rest_rotation_rad_myr'] = (common-alpha*fraction*differential).tolist()
        found['relative_rotation_rad_myr'] *= fraction
    return admitted/(1000.*dt), dict(cut_work_j=work, held_mode_heat_j=available-work,
        proposed_increment_m=proposed_m, accepted_increment_m=admitted,
        held_mode_available_cut_work_j=available,
        signed_remaining_power_w=signed_remaining, cut_power_w=cut_power,
        power_closure_absolute_w=abs(closure_w),
        power_closure_relative_to_cut=abs(closure_w)/cut_power if cut_power > 0. else None,
        interpretation='Resistance work and unused virtual-path budget; current geometry and mode '
                       'held over the source interval. Independent candidate path, not simultaneous '
                       'world energy, material heating or full finite-time qualification.')


def update(s, dt):
    """Check every loaded plate and accumulate opening; never changes topology."""
    if not enabled(s) or dt <= 0.:
        return
    settings = normalize(s.config.get('force_limit_rifting'))
    policy = bounded_policy(s)
    if policy is not None:
        elapsed = float(s.t)-policy['epoch_myr']
        if elapsed <= 0. or abs(elapsed-dt) > 1e-9*max(abs(dt), 1.):
            raise ValueError('Bounded force breakup needs one new matching accepted source interval.')
    balance = plate_limit_analysis.active_balance(s, dt)
    ledger = None
    if balance.uses_force_ledger:
        import balance_force_ledger
        # Topology checks run after source/geometry changes; s.omega can be
        # the accepted interval average. Obtain the CURRENT intact equilibrium
        # for onset without changing that accepted transport or its cadence.
        balance.solve()
        ledger = balance_force_ledger.export(balance, balance.x)
    # A rejected force export or opening solve must not leave half the plates
    # with advanced opening history. Publish the complete check atomically.
    state, seen = deepcopy(s.force_rifting_state), set()
    diagnostics = deepcopy(s.force_rifting_diagnostics)
    if balance.uses_force_ledger:
        diagnostics.setdefault('accepted_cut_work_j', 0.)
        diagnostics.setdefault('held_mode_heat_j', 0.)
        diagnostics['work_interpretation'] = ('Sum of independently evaluated virtual candidate paths; '
            'not simultaneous world opening, material heat or a complete physical energy budget.')
    loaded = (_loaded_plates(s, balance, ledger=ledger) if policy is not None
              else _loaded_plates(s, balance))
    for p in loaded:
        uid = str(int(s.plate_uid[p]))
        seen.add(uid)
        previous = state.get(uid)
        coarse = previous is None or float(s.t)-previous['scanned_myr'] >= settings['rescan_interval_myr']
        seeds = [] if previous is None else [np.asarray(previous['axis'])]
        scale = None
        if previous is not None and previous['opened_km'] > 0.:
            beta = 1.+previous['opened_km']/settings['rift_width_km']
            scale = {int(e): 1./beta for e in previous['cut_edges']}
        search_options = (dict(bounded_axes=settings['search_policy']['max_axes'])
                          if policy is not None else {})
        found = plate_limit_analysis.worst_mechanism(s, p, balance, seed_axes=seeds, coarse=coarse,
            forbid_cratons=settings['forbid_cratons'], edge_strength_scale=scale,
            inherited=settings['inherited_weakness'], ledger=ledger, **search_options)
        diagnostics['checks'] += 1
        if found['axis'] is None:
            _retire_mode(diagnostics, uid, previous, s.t, 'no resolved mechanism')
            state.pop(uid, None)
            continue
        piece = found['cells'][found['piece']].tolist()
        cells = found['cells'].tolist()
        continuity = {}
        identity = _stable_cut(s, p, found, continuity=continuity) if policy is not None else None
        same = (previous is not None and previous.get('last_observation_myr') == policy['epoch_myr']
                and (_same_supported_cut(previous, identity, found['axis'])
                     or balance.uses_force_ledger and _same_fixed_interface(previous, continuity, found['axis'])) if policy is not None else
                previous is not None and _overlap(
                    _smaller_side(piece, cells), _smaller_side(previous['piece_cells'], cells)) >= SAME_MECHANISM_OVERLAP)
        if balance.uses_force_ledger and scale is not None and not same:
            # Weakening belongs to the tracked cut. An unrelated candidate
            # cannot borrow weakened edges and then forget their paid history.
            # Restart selection with intact strengths and no carried opening;
            # this conservative reset avoids transferring another path's work.
            found = plate_limit_analysis.worst_mechanism(s, p, balance,
                seed_axes=seeds+[found['axis']], coarse=coarse,
                forbid_cratons=settings['forbid_cratons'], edge_strength_scale=None,
                inherited=settings['inherited_weakness'], ledger=ledger, **search_options)
            diagnostics['checks'] += 1
            if found['axis'] is None:
                _retire_mode(diagnostics, uid, previous, s.t, 'changed cut has no intact resolved mechanism')
                state.pop(uid, None)
                continue
            piece = found['cells'][found['piece']].tolist()
            cells = found['cells'].tolist()
            continuity = {}
            identity = _stable_cut(s, p, found, continuity=continuity) if policy is not None else None
            same = False
        if previous is not None and not same:
            _retire_mode(diagnostics, uid, previous, s.t, 'mechanism changed; conservative intact reset')
        opened = previous['opened_km'] if same else 0.
        # A first or changed support observation starts at zero. Its endpoint
        # solve cannot price an interval that has no matching starting cut.
        supported_dt = dt if policy is None or same else 0.
        rate_km_myr = found['predicted_opening_cm_yr_mean']*10. if found['ratio'] >= 1. else 0.
        work_receipt = None
        if balance.uses_force_ledger:
            if found['ratio'] >= 1. and not found.get('opening_mode_supported', False):
                raise ValueError('Force-rift opening operator is unsupported: '+str(found.get('opening_mode_reason')))
            if rate_km_myr > 0. and supported_dt > 0.:
                rate_km_myr, work_receipt = _paid_opening(s, p, found, opened, settings, supported_dt)
                diagnostics['accepted_cut_work_j'] += work_receipt['cut_work_j']
                diagnostics['held_mode_heat_j'] += work_receipt['held_mode_heat_j']
        opened = (opened+rate_km_myr*supported_dt if rate_km_myr > 0.
                  else opened*math.exp(-supported_dt/settings['heal_myr']))
        continental = found['cut_continental_fraction']
        state[uid] = dict(axis=found['axis'].tolist(), piece_cells=piece, ratio=found['ratio'],
            opened_km=opened, opening_km_myr=rate_km_myr, cut_length_km=found['cut_length_km'],
            breakup_opening_km=continental*settings['rift_width_km']*(settings['breakup_stretch']-1.)
                               + (1.-continental)*settings['ocean_breakup_opening_km'],
            stretch=1.+opened/settings['rift_width_km'],
            cut_continental_fraction=found['cut_continental_fraction'],
            cut_edges=found['cut_edges'].tolist(), piece_rotation_rad_myr=found['piece_rotation_rad_myr'],
            rest_rotation_rad_myr=found['rest_rotation_rad_myr'],
            scanned_myr=float(s.t) if coarse else previous['scanned_myr'],
            since_myr=previous['since_myr'] if same else float(s.t))
        if policy is not None:
            state[uid].update(support_identity=identity, fixed_interface_continuity=continuity,
                support_match_kind=('exact' if same and _same_supported_cut(previous, identity, found['axis'])
                    else 'fixed_interface' if same else 'new_observation'),
                last_observation_myr=float(s.t),
                supported_elapsed_myr=(previous.get('supported_elapsed_myr', 0.) if same else 0.)+supported_dt,
                search_policy=found.get('search_policy'), searched_axis_count=found.get('searched_axis_count'),
                search_scope=found.get('search_scope'), candidate_axis_sources=deepcopy(found.get('candidate_axis_sources')),
                search_interpretation=found.get('search_interpretation'))
        if balance.uses_force_ledger:
            state[uid].update(force_accounting=('active effective-subduction ledger'
                if balance.effective_subduction else 'active attached-slab ledger'),
                paid_cut_work_j=(previous.get('paid_cut_work_j', 0.) if same else 0.)
                    +(work_receipt['cut_work_j'] if work_receipt else 0.),
                last_opening_work=work_receipt if work_receipt is not None else
                    deepcopy(previous.get('last_opening_work')) if same else None,
                opening_mode={key: value for key, value in found.items() if key.startswith(('mode_', 'opening_mode_',
                    'common_restriction_', 'differential_virtual_work_', 'common_motion_'))})
    for uid in set(state)-seen:
        _retire_mode(diagnostics, uid, state[uid], s.t, 'owner no longer loaded or active')
        del state[uid]
    s.force_rifting_state = state
    s.force_rifting_diagnostics = diagnostics
    if policy is not None:
        s.force_rifting_policy = dict(deepcopy(policy), epoch_myr=float(s.t))


def _material_sides(s, p, region):
    """Whole-patch parcel sides and a region agreeing with them on material cells."""
    from material_geometry import patch_centres
    selected = s.parcel_plate == p
    parcel_side = np.zeros(len(s.mass), bool)
    owned = np.flatnonzero(s.plate == p)

    def side_of(points):
        # Moving material can overhang its plate's cells; such a point takes
        # the side of the nearest cell that the plate actually owns.
        cell = s._indices(points)
        off = s.plate[cell] != p
        if np.any(off):
            cell[off] = owned[np.argmax(points[off]@s.xyz[owned].T, axis=1)]
        return region[cell]
    if np.any(selected):
        ids, centres, inverse = patch_centres(s.pos[selected], s.mass[selected], s.parcel_patch[selected])
        parcel_side[selected] = side_of(centres)[inverse]
        # A craton group moves whole, to the side holding most of its mass.
        # The mechanism search already keeps each group's cells together, so
        # this only reassigns sub-resolution overhangs.
        for group in np.unique(s.parcel_craton[selected]):
            if group < 0:
                continue
            mine = selected & (s.parcel_craton == group)
            parcel_side[mine] = s.mass[mine & parcel_side].sum() > .5*s.mass[mine].sum()
    weight = np.bincount(s.parcel_cell[selected], weights=s.mass[selected], minlength=s.n)
    positive = np.bincount(s.parcel_cell[selected & parcel_side], weights=s.mass[selected & parcel_side], minlength=s.n)
    material = (s.plate == p) & (weight > 0.)
    region = region.copy()
    region[material] = positive[material] > .5*weight[material]
    return parcel_side, region


def commit(s):
    """Split the plate whose accumulated opening is largest past the threshold."""
    if not enabled(s):
        return False
    import native_topology
    settings = normalize(s.config.get('force_limit_rifting'))
    ready = sorted(((row['opened_km']/row['breakup_opening_km'], uid) for uid, row in s.force_rifting_state.items()
                    if row['opened_km'] >= row['breakup_opening_km']), reverse=True)
    for _, uid in ready:
        row = s.force_rifting_state[uid]
        slots = np.flatnonzero(s.active & (s.plate_uid == int(uid)))
        if len(slots) != 1:
            continue
        p = int(slots[0])
        region = np.zeros(s.n, bool)
        region[np.asarray(row['piece_cells'], int)] = True
        region &= s.plate == p
        parcel_side, region = _material_sides(s, p, region)
        if not region.any() or not ((s.plate == p) & ~region).any():
            continue
        traces = native_topology._trace_side(s, parcel_side, region)
        rotations = np.array([row['rest_rotation_rad_myr'], row['piece_rotation_rad_myr']], float)
        center = native_topology._unit(np.sum(s.xyz[region]*s.cell_area[region, None], axis=0))
        setting = 'continental' if row['cut_continental_fraction'] > 0. else 'oceanic'
        loading = dict(model='force-limit rifting (rigid-plastic limit analysis)', cause='force_limit',
                       loading_ratio=row['ratio'], accumulated_opening_km=row['opened_km'],
                       opening_km_myr=row['opening_km_myr'], mechanism_since_myr=row['since_myr'],
                       breakup_opening_km=row['breakup_opening_km'], stretch_at_breakup=row['stretch'],
                       cut_length_km=row['cut_length_km'], cut_continental_fraction=row['cut_continental_fraction'])
        if 'paid_cut_work_j' in row:
            loading.update(paid_cut_work_j=row['paid_cut_work_j'], opening_mode=deepcopy(row.get('opening_mode')),
                           last_opening_work=deepcopy(row.get('last_opening_work')))
        before = s.next_plate_uid
        if native_topology._commit(s, p, region, parcel_side, traces, rotations, setting, loading, center):
            q = int(np.flatnonzero(s.active & (s.plate_uid == before))[0])
            _mark_rift(s, p, q, np.asarray(row['cut_edges'], int), center, np.asarray(row['axis'], float))
            s.force_rifting_diagnostics['commits'] += 1
            _retire_mode(s.force_rifting_diagnostics, uid, row, s.t, 'committed breakup')
            del s.force_rifting_state[uid]
            return True
        refusals = s.force_rifting_diagnostics['refusals']
        refusals.append(dict(time_myr=float(s.t), plate_uid=int(uid), loading_ratio=row['ratio'],
                             accumulated_opening_km=row['opened_km'],
                             reason='commit refused: daughter viability or intact material'))
        del refusals[:-20]
    return False


def _mark_rift(s, p, q, cut_edges, center, axis):
    """Record inherited rift structure on material in the cells along the cut."""
    if not len(cut_edges):
        return
    cells = np.unique(s.native_mesh['edge_faces'][cut_edges])
    rid = s._new_rift_record('force-limit breakup', (p, q), center)
    for prefix, points, owners, kinds in (('', s.pos, s.parcel_plate, s.kind),
                                          ('trace_', s.trace_xyz, s.trace_plate, s.trace_kind)):
        belt = np.isin(s._indices(points), cells) & np.isin(owners, (p, q))
        ids = getattr(s, prefix+'rift_id')
        new = belt & (ids < 0) & (kinds != 3)
        ids[new] = rid
        getattr(s, prefix+'rift_birth_myr')[new] = s.t
        # Opening about the mechanism axis is along axis x r, so the rift
        # runs along the axis projected into each tangent plane.
        along = axis[None, :]-points[new]*(points[new]@axis)[:, None]
        getattr(s, prefix+'rift_tangent')[new] = along/np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-30)
        sutures = getattr(s, prefix+'suture')
        sutures[belt] = np.maximum(sutures[belt], .8)


def snapshot(s):
    if not enabled(s):
        return {}
    plates = {uid: {k: v for k, v in row.items() if k not in ('piece_cells', 'cut_edges')}
              for uid, row in s.force_rifting_state.items()}
    result = dict(force_rifting_version=VERSION, force_rifting_plates=deepcopy(plates),
                  force_rifting_diagnostics=deepcopy(s.force_rifting_diagnostics))
    policy = bounded_policy(s)
    if policy is not None:
        result['force_rifting_policy'] = deepcopy(policy)
        result['force_rifting_policy_version'] = BOUNDED_POLICY_VERSION
    return result


def validate_frame(frame):
    """Validate new bounded-policy receipts; leave legacy frame contracts alone."""
    if 'force_rifting_policy' not in frame:
        rows = frame.get('force_rifting_plates', {})
        if ('force_rifting_policy_version' in frame or isinstance(rows, dict)
                and any(isinstance(row, dict) and ('support_identity' in row or 'search_policy' in row
                    or 'fixed_interface_continuity' in row or 'support_match_kind' in row)
                        for row in rows.values())):
            raise ValueError('Bounded force-breakup frame lost its declared policy.')
        return

    def reject():
        raise ValueError('Invalid bounded force-breakup frame policy or future-only work history.')

    def number(value, *, nonnegative=True):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or nonnegative and value < 0.):
            reject()
        return float(value)

    policy = frame['force_rifting_policy']
    if (type(frame.get('force_rifting_policy_version')) is not int
            or frame['force_rifting_policy_version'] != BOUNDED_POLICY_VERSION
            or not isinstance(policy, dict) or set(policy) != {'version', 'mode', 'parameters',
            'activation_myr', 'epoch_myr', 'history'} or type(policy['version']) is not int
            or policy['version'] != BOUNDED_POLICY_VERSION or policy['mode'] != 'bounded_axes'
            or policy['history'] != 'Future matched cut observations only; virtual development, not realized material extension.'):
        reject()
    settings = normalize(policy['parameters'])
    if settings != policy['parameters'] or not settings['enabled'] or 'search_policy' not in settings:
        reject()
    activation, epoch, time = (number(policy['activation_myr']), number(policy['epoch_myr']),
                                number(frame.get('time_myr')))
    # Native saved frames round display time to six decimals; the policy
    # retains the exact accepted epoch for interval accounting.
    if not activation <= epoch or time not in (epoch, round(epoch, 6)):
        reject()
    if type(frame.get('force_rifting_version')) is not int or frame['force_rifting_version'] != VERSION:
        reject()
    plates, diagnostics = frame.get('force_rifting_plates'), frame.get('force_rifting_diagnostics')
    if not isinstance(plates, dict) or not isinstance(diagnostics, dict):
        reject()
    if type(diagnostics.get('version')) is not int or diagnostics['version'] != VERSION:
        reject()
    for name in ('checks', 'commits'):
        if type(diagnostics.get(name)) is not int or diagnostics[name] < 0:
            reject()
    if not isinstance(diagnostics.get('refusals'), list):
        reject()
    paid = 0.
    for uid, row in plates.items():
        if not isinstance(uid, str) or not uid.isdecimal() or not isinstance(row, dict):
            reject()
        if (row.get('search_policy') != 'bounded_axes_v1'
                or type(row.get('searched_axis_count')) is not int
                or not 1 <= row['searched_axis_count'] <= settings['search_policy']['max_axes']):
            reject()
        identity = row.get('support_identity')
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            reject()
        if 'fixed_interface_continuity' in row or 'support_match_kind' in row:
            proof = row.get('fixed_interface_continuity')
            kind = row.get('support_match_kind')
            if (not isinstance(proof, dict) or kind not in ('exact', 'fixed_interface', 'new_observation')
                    or proof and (not _valid_fixed_interface_proof(proof) or proof['owner_uid'] != int(uid))
                    or kind == 'fixed_interface' and not proof):
                reject()
        axis = np.asarray(row.get('axis'), float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1.) > 1e-8:
            reject()
        since, observation = number(row.get('since_myr')), number(row.get('last_observation_myr'))
        elapsed = number(row.get('supported_elapsed_myr'))
        if row.get('support_match_kind') == 'fixed_interface' and elapsed <= 0.:
            reject()
        if not activation <= since <= observation == epoch or elapsed > observation-since+1e-9:
            reject()
        for name in ('ratio', 'opened_km', 'opening_km_myr', 'cut_length_km',
                     'breakup_opening_km', 'stretch', 'cut_continental_fraction'):
            number(row.get(name))
        if row['cut_continental_fraction'] > 1. or row['breakup_opening_km'] <= 0. or row['stretch'] < 1.:
            reject()
        work = number(row.get('paid_cut_work_j'))
        paid += work
        if elapsed == 0. and (row['opened_km'] != 0. or work != 0.):
            reject()
        receipt = row.get('last_opening_work')
        if receipt is not None:
            if not isinstance(receipt, dict):
                reject()
            for name in ('cut_work_j', 'held_mode_heat_j', 'proposed_increment_m', 'accepted_increment_m',
                         'held_mode_available_cut_work_j', 'cut_power_w', 'power_closure_absolute_w'):
                number(receipt.get(name))
            number(receipt.get('signed_remaining_power_w'), nonnegative=False)
            if receipt.get('power_closure_relative_to_cut') is not None:
                number(receipt['power_closure_relative_to_cut'])
            if (receipt['accepted_increment_m'] > receipt['proposed_increment_m']*(1.+1e-8)
                    or receipt['cut_work_j'] > receipt['held_mode_available_cut_work_j']*(1.+1e-8)):
                reject()
        elif work > 0.:
            reject()
    retired = diagnostics.get('retired_candidate_paths', [])
    if not isinstance(retired, list):
        reject()
    for row in retired:
        if not isinstance(row, dict) or not activation <= number(row.get('since_myr')) <= number(row.get('retired_myr')) <= epoch:
            reject()
        paid += number(row.get('paid_cut_work_j'))
    total = number(diagnostics.get('accepted_cut_work_j', 0.))
    number(diagnostics.get('held_mode_heat_j', 0.))
    if abs(total-paid) > 1e-8*max(total, paid, 1.):
        reject()
    if epoch == activation and (plates or retired or total != 0. or diagnostics['checks'] != 0
                                or diagnostics['commits'] != 0 or diagnostics['refusals']):
        reject()
