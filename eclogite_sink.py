"""Legacy instantaneous sink and shared foundering inventory/frame utilities.

The description below applies to worlds without retained_dense_crust_version=1.
That explicit policy replaces this sink with phase_evolution.py's retained
density, temperature and mechanically admitted drainage. The ledger and frame
utilities here are shared by both policies.

Crust deeper than ECLOGITE_DEPTH_KM below the top of its stack (own thickness plus
the overburden of every sheet the contact graph records above it) converts to
eclogite and founders into the mantle on FOUNDERING_TAU_MYR once it has been
buried for HEATING_DELAY_MYR, up to RETURNED_FRACTION of any column. Mass leaves
thickness_km into an explicit per-plate/per-sheet/per-contact mantle-return
ledger; nothing is deleted. A reduced kinetic closure on state the model already
carries (overlap ledger, contact order, thickness, burial clock): no temperature,
density or slab geometry is resolved, and retained eclogite's negative buoyancy is
not represented.

Version 2 enforces the compositional cap with conservative removable-crust
volume per reference area (crust_inventory.py). Version 1 retains its historical
thickness-counter law until explicit migration. Neither resolves a dense phase.

WHY THIS EXISTS. The only other sink on buoyant columns is exposure-gated erosion
(structure_engine.py:246-267, gated by collision_contacts.erosion_fraction), which
by construction cannot reach a buried sheet: it removes 0.003-0.02 m/Myr from the
covered crust that supplies most of a stacked plateau's height. Continental crust
is stratified and its mafic lower part passes garnet granulite at ~1.0-1.2 GPa and
plagioclase-out eclogite at ~1.5-2.0 GPa for 500-800 C (Ito & Kennedy 1971; Hacker
1996; Schorn & Diener 2017), becoming 0.1-0.3 Mg/m3 denser than peridotite (Hacker,
Abers & Peacock 2003) and therefore no longer part of the buoyant column. Roughly
half to two thirds of the pre-collisional Indian crust is absent from the plateau,
sediment and extrusion budgets today (Ingalls et al. 2016; Copley, Avouac & Royer
2010; Yakovlev & Clark 2014).

WHAT IT CANNOT DO, stated because the closure is deliberately reduced:
  * There is no temperature field. ECLOGITE_DEPTH_KM, HEATING_DELAY_MYR and
    FOUNDERING_TAU_MYR stand in for a geotherm, argued from residence time. The
    falsifier is large and measured: at 70 km (cold, fast-underthrust India,
    Hetenyi et al. 2007) this law returns only about a quarter of a stacked load.
  * There is no density state, so eclogite is never retained as a dense layer; the
    extra ~45 m/km of pull-down of retained eclogite is not represented, which
    makes the surface response conservative.
  * There is no lithospheric-mantle thickness, so the England & Houseman (1989)
    uplift-then-collapse of a removed thermal boundary layer has no representation.
  * crustal_root_km (geology_snapshot.py) will NOT show the loss, because the
    neutral reference thins with the column; material_foundered_m is the record.

INTERACTIONS. The sink reads the overlap ledger and contact order that
collision_contacts.refresh has just rebuilt for the post-advect geometry, plus
thickness and the burial clock. It never reads bcode, row['state'],
suture_strength or parcel_collision_suture, and it never writes plate, sheet,
exposure, burial or suture, so a welded and consolidated stack keeps sinking and
the sink cannot be switched off by a label. It runs as the last mass writer of a
step, inside structure_engine.deform and after erosion, never between advect and
the geometric commit (crustal_structure.py:180-192), so the geometric strain
residual, the relaxation volume assertion and the adaptivity budget are all
outside its window. It adds no torque: the per-contact eclogite load is exported
with coefficient CONTINENTAL_SLAB_PULL of zero.
"""
from __future__ import annotations

from copy import copy, deepcopy
import math
import numpy as np

import crustal_structure as columns
import collision_surface
import crust_inventory

VERSION = 2
ARRAY_FIELDS = {'material_foundered_m': np.float64}
INVENTORY_ARRAY_FIELDS = {'material_'+name: name for name in crust_inventory.FIELDS}
ARRAY_FIELDS.update({name: np.float64 for name in INVENTORY_ARRAY_FIELDS})
import dense_crust
ARRAY_FIELDS.update(dense_crust.ARRAY_FIELDS)
PARCEL_FIELDS = ('parcel_root_age_myr',)
COLUMN_FIELD = 'foundered_m'

# Plagioclase-out eclogite at an orogenic geotherm. 45-60 km is the physical band
# (Ito & Kennedy 1971; Hacker 1996; Schorn & Diener 2017 JMG); 70-80 km is the
# cold-slab value under southern Tibet (Hetenyi et al. 2007 EPSL 264:226;
# Wittlinger et al. 2009 GJI 177:1037). 50 km is chosen because this model's
# buried sheets have residence times of several tens of Myr against a conductive
# relaxation time L^2/kappa of ~50 Myr for a 40 km overburden, i.e. they are at
# least half-way back to an ordinary continental geotherm. That argument is an
# ASSUMPTION, not a solved temperature; see the falsifier note in the docstring.
ECLOGITE_DEPTH_KM = 50.
# Sequential first-order chain: eclogitization 5-20 Myr once T exceeds ~600 C
# (Hetenyi et al. 2007; Shi et al. 2018 Nat Comms 9:3483) followed by foundering
# in 1-25 Myr at 1e18-1e20 Pa s (Jull & Kelemen 2001; Ye et al. 2022 Solid Earth
# 13:745). Mean residence tau_e + tau_f ~ 20 Myr; the band is 10-40 Myr, and dry
# granulite metastability (Austrheim 1987; Jackson et al. 2004) argues the long end.
FOUNDERING_TAU_MYR = 20.
# Conductive heating of a 20-40 km underthrust sheet, L^2/kappa = 13-50 Myr
# (Turcotte & Schubert half-space). Band 5-20 Myr.
HEATING_DELAY_MYR = 10.
# Fraction of a column that can ever leave: ~50% of India's crust is unaccounted
# for (Ingalls et al. 2016), 60-70% on the Copley, Avouac & Royer (2010) budget,
# 25-50% on Yakovlev & Clark (2014); only the felsic part relaminates (Hacker,
# Kelemen & Behn 2011). Band 0.4-0.7. This compositional cap, not the floor
# below, is what bounds the loss.
RETURNED_FRACTION = crust_inventory.FRACTION
# Engineering bound, not physics: stay two kilometres clear of the hard minimum
# at crustal_structure.py:31 so the sink never pins a column on a bound that the
# geometric deformation solver and crustal_structure._state both enforce.
RESIDUAL_FLOOR_KM = columns.MIN_THICKNESS_KM+2.
# Eclogite minus peridotite, 3450 +/- 100 against 3300 kg/m3 (Hacker, Abers &
# Peacock 2003). Used ONLY to export a candidate continental slab-pull load per
# metre of suture trace (Capitanio et al. 2010 would give ~1e13 N/m over a 500 km
# underthrust width). No torque is applied in this version; the coefficient is
# zero in source so that a later decision has the number already logged beside
# the slab attachment it would have to be partitioned against.
ECLOGITE_DENSITY_CONTRAST = 150.
GRAVITY_M_S2 = 9.81
CONTINENTAL_SLAB_PULL = 0.


def enabled(s):
    return getattr(s, 'foundering_version', 0) in (1, VERSION)


def upgrade_depth_integration(s):
    """Explicitly select local stack integration; the old heating law remains."""
    import burial_depth
    if not enabled(s):
        raise ValueError('Local burial integration requires an enabled foundering model.')
    version = getattr(s, 'foundering_depth_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, burial_depth.VERSION):
        raise ValueError('Unsupported burial-depth version.')
    if version == burial_depth.VERSION:
        return deepcopy(getattr(s, 'foundering_depth_migration', {}))
    # Compute once before committing the flag. This checks layer order, geometry
    # and area closure without changing any column, clock or mantle inventory.
    _, scope, _ = burial_depth.integrate(s, s.structure['thickness_km'], _depth_pairs(s),
        depth_km=ECLOGITE_DEPTH_KM, heating_delay_myr=HEATING_DELAY_MYR)
    report = dict(from_version=0, to_version=burial_depth.VERSION,
                  time_myr=float(getattr(s, 't', 0.)), eligible_km3=scope['eligible_km3'],
                  heating_model_changed=False)
    s.foundering_depth_version = burial_depth.VERSION
    s.foundering_depth_migration = report
    return deepcopy(report)


def _depth_pairs(s):
    """Use the refreshed contact cache, or compute it for a detached migration."""
    if getattr(s, '_collision_overlap', None) is not None:
        return _pair_order(s)
    import mesh_coverage
    staged = copy(s)
    surface = s.material_surface
    staged._collision_overlap = mesh_coverage.material_overlaps(
        surface['vertices'], surface['faces'], s.parcel_collision_sheet,
        radius_km=surface.get('radius_km', columns.RADIUS_KM))
    return _pair_order(staged)


def upgrade_inventory(s):
    """Explicit, atomic migration to the conservative composition ledger.

    Preserve the version-1 allowance at this boundary, not an invented fresh
    60% of the remaining crust. Old volume losses cannot be reconstructed.
    No physical columns, root clocks, existing mantle returns or RNG change.
    """
    version = getattr(s, 'foundering_version', 0)
    if version == VERSION:
        validate_alignment(s)
        return deepcopy(getattr(s, 'foundering_inventory_migration', {}))
    if version not in (0, 1) or isinstance(version, (bool, np.bool_)):
        raise ValueError('Unsupported foundering inventory migration.')
    staged = copy(s)
    staged.structure = deepcopy(s.structure)
    staged.trace_structure = deepcopy(s.trace_structure)
    for name in ('mantle_return_km3', 'process_totals', 'parcel_root_age_myr'):
        if hasattr(s, name):
            setattr(staged, name, deepcopy(getattr(s, name)))
    staged.foundering_version = 1
    ensure_fields(staged)
    validate_alignment(staged)
    for state in (staged.structure, staged.trace_structure):
        crust_inventory.initialize(state, legacy=True)
    staged.foundering_version = VERSION
    validate_alignment(staged)
    report = dict(from_version=int(version), to_version=VERSION,
                  time_myr=float(getattr(s, 't', 0.)),
                  baseline_removable_km3=float(np.asarray(s.mass)@staged.structure[crust_inventory.BASELINE]),
                  historical_volume_reconstructed=False,
                  basis='remaining legacy allowance at migration boundary')
    for name in ('structure', 'trace_structure', 'parcel_root_age_myr', 'mantle_return_km3', 'process_totals'):
        if hasattr(staged, name):
            setattr(s, name, getattr(staged, name))
    s.foundering_inventory_migration = report
    s.foundering_version = VERSION
    return deepcopy(report)


def ensure_fields(s):
    """Create or pad the per-column ledger, the root clock and the reservoir.

    The column counter is padded WHETHER OR NOT the sink is enabled: it belongs
    to structure_engine.FIELDS, so a version-0 world resumed on this source would
    otherwise lose the field alignment that _evolve and _append require. Only the
    physics, the mantle reservoir and the saved frame are version gated.
    """
    for state, n in ((getattr(s, 'structure', None), len(s.mass)),
                     (getattr(s, 'trace_structure', None), len(s.trace_patch))):
        if state is None:
            continue
        old = state.get(COLUMN_FIELD)
        if old is None:
            state[COLUMN_FIELD] = np.zeros(n)
        elif len(old) < n:
            state[COLUMN_FIELD] = np.r_[old, np.zeros(n-len(old))]
        elif len(old) > n:
            raise ValueError('Foundering ledger lost material alignment.')
    if not enabled(s):
        return
    if getattr(s, 'foundering_version', 0) == VERSION:
        for state in (s.structure, s.trace_structure):
            if not crust_inventory.present(state):
                raise ValueError('Version 2 requires explicit removable-crust inventory migration.')
            crust_inventory.validate(state)
    n = len(s.mass)
    clock = getattr(s, 'parcel_root_age_myr', None)
    if clock is None:
        # An existing world carries no record of when a column first exceeded the
        # eclogite depth. A root that is already deeper than z_e has been at depth
        # since it thickened -- roughly a Myr per kilometre -- so it is treated as
        # heated, and everything shallower starts its clock now. New faces padded
        # below are juvenile columns and correctly start at zero.
        thickness = np.asarray(s.structure['thickness_km'], float)
        s.parcel_root_age_myr = np.where(thickness > ECLOGITE_DEPTH_KM, HEATING_DELAY_MYR, 0.)
    elif len(clock) < n:
        s.parcel_root_age_myr = np.r_[np.asarray(clock, float), np.zeros(n-len(clock))]
    elif len(clock) > n:
        raise ValueError('Foundering root clock lost material alignment.')
    if not hasattr(s, 'mantle_return_km3'):
        s.mantle_return_km3 = dict(total=0., by_plate_uid={}, by_sheet={}, by_contact={})
    s.mantle_return_km3.setdefault('by_contact', {})
    if hasattr(s, 'process_totals'):
        s.process_totals.setdefault('crust_returned_to_mantle_km3', 0.)


def validate_alignment(s):
    """Reject a stale ledger before a geometry transaction can omit it."""
    if not enabled(s):
        return
    n = len(s.mass)
    for label, state, count in (('material columns', s.structure, n),
                                ('trace columns', s.trace_structure, len(s.trace_patch))):
        value = state.get(COLUMN_FIELD)
        if (not isinstance(value, np.ndarray) or value.shape != (count,)
                or not np.isfinite(value).all() or np.any(value < 0)):
            raise ValueError('The foundering ledger must align with every column: '+label)
        if getattr(s, 'foundering_version', 0) == VERSION:
            if not crust_inventory.present(state):
                raise ValueError('Missing removable-crust inventory: '+label)
            crust_inventory.validate(state)
    clock = getattr(s, 'parcel_root_age_myr', None)
    if (not isinstance(clock, np.ndarray) or clock.shape != (n,)
            or not np.isfinite(clock).all() or np.any(clock < 0)):
        raise ValueError('The foundering root clock must align with every material face.')


def advance_clocks(s, dt):
    """Advance the own-root heating clock at most once per step.

    The covered branch is gated by collision_contacts' burial clock; an exposed
    column carries no burial, so its deep crust needs its own residence record
    (garnet-granulite metastability: Austrheim 1987, Jackson et al. 2004 Geology
    32:625). The clock advances while the column itself is deeper than z_e and is
    never reset, so a root that thins and rethickens is not re-cooled -- the
    model has no temperature to cool.
    """
    if not enabled(s) or dt <= 0. or getattr(s,'retained_dense_crust_version',0)==1:
        return
    if getattr(s, '_root_age_updated_myr', None) == float(s.t):
        return
    ensure_fields(s)
    deep = np.asarray(s.structure['thickness_km'], float) > ECLOGITE_DEPTH_KM
    s.parcel_root_age_myr = s.parcel_root_age_myr + deep*float(dt)
    s._root_age_updated_myr = float(s.t)


def _pair_order(s):
    """Order the cached overlap pairs into (upper, lower) exactly as support does.

    Same persistent layer graph and same failure mode as collision_surface.refresh
    (:82-87): a pair with no recorded vertical order is a corrupt contact graph,
    not something to guess at.
    """
    overlap = getattr(s, '_collision_overlap', None)
    sheets = np.asarray(getattr(s, 'parcel_collision_sheet', np.empty(0, np.int64)))
    if overlap is None or not len(sheets) or not len(overlap['first']):
        empty = np.empty(0, int)
        return empty, empty, np.empty(0, float), empty
    first, second, weight = (np.asarray(overlap[name]) for name in ('first', 'second', 'area_km2'))
    graph = collision_surface.descendants(s.collision_contacts)
    rows = {tuple(sorted((int(row['top_sheet']), int(row['under_sheet'])))): int(row['id'])
            for row in s.collision_contacts}
    upper = np.empty(len(first), int); lower = np.empty(len(first), int)
    contact = np.zeros(len(first), int)
    for k, (a, b) in enumerate(zip(first, second)):
        one, two = int(sheets[a]), int(sheets[b])
        if two in graph.get(one, ()):
            upper[k] = int(a); lower[k] = int(b)
        elif one in graph.get(two, ()):
            upper[k] = int(b); lower[k] = int(a)
        else:
            raise ValueError('A foundering pair has no persistent vertical order.')
        contact[k] = rows.get(tuple(sorted((one, two))), 0)
    return upper, lower, np.asarray(weight, float), contact


def eligible_thickness_km(s, thickness):
    """Kilometres of each column lying deeper than ECLOGITE_DEPTH_KM.

    With foundering_depth_version=1, disjoint local stacks are thresholded
    before integration by burial_depth.py. The following describes the retained
    version-0 law only; its mean-depth threshold is not refinement invariant.

    Depth is measured against the WHOLE overburden of a face, not one pair at a
    time: a lower face whose covering pairs sum to more than its own area would
    otherwise have its deep part counted once per cover. The covered area
    fraction fc and the area-weighted mean overburden give one depth per face,

        fc = min(sum_j w_ji / A_i, 1)
        depth_covered_i = (sum_j w_ji T_j)/(A_i fc_i) + T_i

    so the covered branch contributes fc*clip(depth_covered - z_e, 0, T) and the
    uncovered remainder -- exposed columns and single columns -- contributes
    (1 - fc)*clip(T - z_e, 0, T), its own root. Returns the eligible thickness,
    a diagnostic scope and the attribution the ledger needs to split the removed
    volume between the sutures it came from.
    """
    version = getattr(s, 'foundering_depth_version', 0)
    if version == 1:
        import burial_depth
        return burial_depth.integrate(s, thickness, _depth_pairs(s),
            depth_km=ECLOGITE_DEPTH_KM, heating_delay_myr=HEATING_DELAY_MYR)
    if version != 0:
        raise ValueError('Unsupported burial-depth version.')
    thickness = np.asarray(thickness, float)
    area = np.asarray(s.material_surface['area_km2'], float)
    n = len(area)
    upper, lower, weight, contact = _pair_order(s)
    covered_area = np.bincount(lower, weights=weight, minlength=n)
    overburden = np.bincount(lower, weights=weight*thickness[upper], minlength=n)
    fraction = np.minimum(np.divide(covered_area, area, out=np.zeros(n), where=area > 0), 1.)
    depth = np.divide(overburden, area*fraction, out=np.zeros(n), where=fraction > 0)+thickness
    # The burial clock is advanced by collision_contacts.refresh; the root clock
    # by advance_clocks above. Both gate their own branch with the same delay.
    heated = np.asarray(s.parcel_burial_myr, float) >= HEATING_DELAY_MYR
    root_heated = np.asarray(getattr(s, 'parcel_root_age_myr', np.zeros(n)), float) >= HEATING_DELAY_MYR
    covered = fraction*np.clip(depth-ECLOGITE_DEPTH_KM, 0., thickness)
    own = (1.-fraction)*np.clip(thickness-ECLOGITE_DEPTH_KM, 0., thickness)
    gated = covered*heated+own*root_heated
    eligible = np.minimum(gated, thickness)
    share = np.divide(covered*heated, gated, out=np.zeros(n), where=gated > 0)
    scope = dict(covered_km3=float(area@covered), own_root_km3=float(area@own),
                 root_inventory_km3=float(area@(covered+own)),
                 eligible_km3=float(area@eligible),
                 heated_covered_fraction=float(area@(covered*heated)/max(float(area@covered), 1e-300)),
                 heated_root_fraction=float(area@(own*root_heated)/max(float(area@own), 1e-300)),
                 overlapping_face_pairs=int(len(weight)),
                 faces_with_cover_beyond_own_area=int(np.count_nonzero(covered_area > area)))
    attribution = dict(lower=lower, weight=weight, contact=contact,
                       covered_area=covered_area, covered_share=share)
    return eligible, scope, attribution


def trace_values(s, parcel_values):
    """Map a parcel-pass field to markers; pattern of collision_surface.erosion_support."""
    if parcel_values is None:
        raise ValueError('The foundering trace pass needs the parcel pass eligibility.')
    values = np.asarray(parcel_values)
    ids = np.asarray(s.parcel_patch); order = np.argsort(ids)
    query = np.asarray(s.trace_patch); at = np.searchsorted(ids[order], query)
    valid = at < len(ids)
    valid[valid] &= ids[order[at[valid]]] == query[valid]
    if not valid.all():
        raise ValueError('A foundering trace has no material face.')
    return values[order[at]].copy()


def ordinary_loss_diagnostics(state, eligible_km, dt, *, area_km2, attribution=None):
    """Observe the ordinary sink's next transaction without changing its inputs.

    This reports the existing face-mean law, not a resolved spatial depletion
    model. Volumes are physical km3. Limiter losses are disjoint: inventory is
    applied first, then the residual floor to the inventory-admitted request.
    Thus the two blocked volumes are not independent counterfactual losses.
    Exact partial-cover classification needs the union footprint already found
    by burial integration; summed pair areas cannot supply it for triple stacks.
    """
    thickness = np.asarray(state['thickness_km'], float)
    eligible = np.asarray(eligible_km, float)
    area = np.asarray(area_km2, float)
    if (thickness.ndim != 1 or eligible.shape != thickness.shape or area.shape != thickness.shape
            or not np.isfinite(thickness).all() or np.any(thickness <= 0)
            or not np.isfinite(eligible).all() or np.any(eligible < 0)
            or not np.isfinite(area).all() or np.any(area <= 0)
            or not np.isfinite(dt) or dt < 0):
        raise ValueError('Ordinary foundering diagnostics need finite aligned nonnegative inputs.')
    foundered = np.asarray(state[COLUMN_FIELD], float)/1000.
    cap = np.maximum(RETURNED_FRACTION*thickness-(1.-RETURNED_FRACTION)*foundered, 0.)
    inventory_present = crust_inventory.present(state)
    if inventory_present:
        cap = np.asarray(state[crust_inventory.REMAINING], float)/state['area_factor']
    requested = eligible*(-np.expm1(-dt/FOUNDERING_TAU_MYR))
    inventory_admitted = np.minimum(requested, cap)
    floor_capacity = np.maximum(thickness-RESIDUAL_FLOOR_KM, 0.)
    admitted = np.minimum(inventory_admitted, floor_capacity)
    inventory_blocked = requested-inventory_admitted
    floor_blocked = inventory_admitted-admitted
    total = lambda values: float(area@values)
    report = dict(version=1, column_assignment='uniform whole-face thickness decrement',
        spatial_depletion_resolved=False, limiter_attribution_order='inventory_then_residual_floor',
        removable_volume_inventory=inventory_present, eligible_km3=total(eligible),
        requested_loss_km3=total(requested), admitted_loss_km3=total(admitted),
        inventory_blocked_requested_loss_km3=total(inventory_blocked),
        residual_floor_blocked_requested_loss_km3=total(floor_blocked),
        inventory_limited_faces=int(np.count_nonzero(inventory_blocked > 0)),
        residual_floor_limited_faces=int(np.count_nonzero(floor_blocked > 0)),
        eligible_faces_at_residual_floor=int(np.count_nonzero((eligible > 0)&(floor_capacity == 0))),
        exact_partial_cover_available=False)
    union = attribution.get('covered_union_area_km2') if isinstance(attribution, dict) else None
    if union is not None:
        union = np.asarray(union, float)
        tolerance = 2e-10
        if (union.shape != area.shape or not np.isfinite(union).all()
                or np.any(union < -tolerance*area) or np.any(union > (1.+tolerance)*area)):
            raise ValueError('Invalid exact covered-union footprint for foundering diagnostics.')
        fraction = np.clip(union/area, 0., 1.)
        partial = (fraction > tolerance)&(fraction < 1.-tolerance)
        report.update(exact_partial_cover_available=True,
            partial_cover_fraction_tolerance=tolerance,
            partially_covered_faces=int(np.count_nonzero(partial)),
            eligible_on_partially_covered_faces_km3=total(np.where(partial, eligible, 0.)),
            removed_on_partially_covered_faces_km3=total(np.where(partial, admitted, 0.)),
            uniform_removal_assigned_to_uncovered_area_km3=total(
                np.where(partial, (1.-fraction)*admitted, 0.)))
    return report


def apply(state, eligible_km, dt):
    """Founder eligible crust from one column set. Returns (removed_km, lowering_m).

    Version 2 limits loss to the remaining removable volume divided by current
    area. Version 1 retains F + d <= phi(T + F), which is not volume-conservative
    after strain. The residual floor keeps either law off the hard minimum.
    """
    thickness = state['thickness_km']
    foundered = state[COLUMN_FIELD]/1000.
    conversion = -np.expm1(-dt/FOUNDERING_TAU_MYR)
    cap = np.maximum(RETURNED_FRACTION*thickness-(1.-RETURNED_FRACTION)*foundered, 0.)
    if crust_inventory.present(state):
        crust_inventory.validate(state)
        cap = state[crust_inventory.REMAINING]/state['area_factor']
    removed = np.minimum(np.minimum(np.asarray(eligible_km, float)*conversion, cap),
                         np.maximum(thickness-RESIDUAL_FLOOR_KM, 0.))
    before = columns.elevation(state)
    crust_inventory.founder(state, removed)
    # Airy subsidence of exactly AIRY_M_PER_KM per kilometre removed, water
    # loaded below the datum: the same surface response as net denudation.
    target = columns.elevation(dict(state, thickness_km=thickness-removed))
    state['thickness_km'] = thickness-removed      # area_factor untouched: mass leaves, no strain
    # A column that shed its lower crust to the mantle has a THINNER NEUTRAL
    # REFERENCE. Without this, gravitational_relaxation's self term 0.5 A (h-h0)^2
    # would read a thinned buried column as compressed below its neutral state and
    # drive it to re-thicken, a restoring force of the same order as the collision
    # drive. Decrement the reference with the column and re-anchor the reference
    # elevation exactly as crustal_structure.coalesce_structure does at :256-261,
    # so the surface drops by the Airy amount while h - h0 is unchanged. Erosion
    # does not do this because its per-step loss is two orders smaller.
    state['reference_thickness_km'] = np.maximum(state['reference_thickness_km']-removed,
                                                 columns.MIN_THICKNESS_KM)
    reference_air = (columns._air_height(target+state['foreland_m'])-state['rift_heat_m']
                     - columns.AIRY_M_PER_KM*(state['thickness_km']-state['reference_thickness_km']))
    if 'lip_heat_m' in state:
        reference_air = reference_air-state['lip_heat_m']
    state['reference_elevation_m'] = columns._water_load(reference_air)
    state[COLUMN_FIELD] = state[COLUMN_FIELD]+removed*1000.
    return removed, before-columns.elevation(state)


def record(s, removed_km, attribution=None):
    """Book the removed physical volume to the explicit mantle-return reservoir.

    Keyed three ways because each survives a different event: by plate UID the
    owner at removal time, by sheet an identity that survives consolidation onto
    the top plate, and by contact an identity that survives the daughter-sheet
    split at collision_contacts.py:93-110. Contact key '0' is crust that lay under
    no suture -- an exposed column's own root. All three sum to the total.
    """
    area = np.asarray(s.material_surface['area_km2'], float)
    removed = np.asarray(removed_km, float)
    if (removed.shape != area.shape or not np.isfinite(area).all()
            or np.any(area <= 0) or not np.isfinite(removed).all() or np.any(removed < 0)):
        raise ValueError('Mantle return needs aligned finite nonnegative removal and positive areas.')
    volume = area*removed
    if not np.isfinite(volume).all():
        raise ValueError('Mantle return volume must be finite.')
    total = float(volume.sum())
    ledger = s.mantle_return_km3
    validate_ledger(ledger)
    # Validate every attribution before committing any reservoir or process
    # total. A failed contact split must not leave a partly booked transaction.
    additions = {}
    for key, index in (('by_plate_uid', np.asarray(s.plate_uid)[np.asarray(s.parcel_plate)]),
                       ('by_sheet', np.asarray(s.parcel_collision_sheet))):
        additions[key] = _grouped(index, volume)
    additions['by_contact'] = _contact_shares(s, volume, attribution)
    validate_ledger(dict(total=total, **additions), require_closure=True)
    staged = deepcopy(ledger)
    staged['total'] += total
    for key, shares in additions.items():
        for value, share in shares.items():
            staged[key][str(value)] = staged[key].get(str(value), 0.)+float(share)
    # The key's zero default is declared beside the other process totals in
    # raster_engine; read it defensively so an older state that predates the
    # declaration still books the volume rather than raising mid-step.
    totals = getattr(s, 'process_totals', None)
    process_returned = None
    if totals is not None:
        previous = totals.get('crust_returned_to_mantle_km3', 0.)
        if not isinstance(previous, (float, int, np.floating)) or not np.isfinite(previous) or previous < 0:
            raise ValueError('Invalid process mantle-return total.')
        process_returned = previous+total
        if not np.isfinite(process_returned):
            raise ValueError('Invalid process mantle-return total.')
    validate_ledger(staged)
    ledger.clear()
    ledger.update(staged)
    if totals is not None:
        totals['crust_returned_to_mantle_km3'] = process_returned
    return total


def validate_ledger(ledger, *, require_closure=False):
    """Validate finite returns; optionally require complete attribution.

    Older ledgers can have an inherited gap, including a contact bucket added
    after some mantle return was already recorded. Never reconstruct that past.
    Every new increment closes; a frame explicitly marked ledger version 1 also
    promises full cumulative closure in all three attribution buckets.
    """
    if (not isinstance(ledger, dict)
            or set(ledger) != {'total', 'by_plate_uid', 'by_sheet', 'by_contact'}
            or not isinstance(ledger['total'], float)
            or not math.isfinite(ledger['total']) or ledger['total'] < 0):
        raise ValueError('Invalid mantle return ledger.')
    for key in ('by_plate_uid', 'by_sheet', 'by_contact'):
        bucket = ledger[key]
        if not isinstance(bucket, dict) or any(
                not isinstance(value, float) or not math.isfinite(value) or value < 0
                for value in bucket.values()):
            raise ValueError('Invalid mantle return ledger: '+key)
        try:
            subtotal = math.fsum(bucket.values())
        except OverflowError as error:
            raise ValueError('Invalid mantle return ledger: '+key) from error
        if require_closure and not math.isclose(subtotal, ledger['total'], rel_tol=2e-10, abs_tol=1e-9):
            raise ValueError('Mantle return attribution does not close: '+key)


def _grouped(index, values):
    """Nonzero group sums of values keyed by the integer labels in index."""
    selected = np.flatnonzero(values != 0.)
    if not len(selected):
        return {}
    keys, inverse = np.unique(np.asarray(index)[selected], return_inverse=True)
    sums = np.bincount(inverse, weights=values[selected], minlength=len(keys))
    return {int(key): float(value) for key, value in zip(keys, sums) if value}


def _contact_shares(s, volume, attribution):
    """Split each face's returned volume between the sutures that buried it."""
    if attribution is None or not len(attribution['lower']):
        return {0: float(volume.sum())} if volume.sum() else {}
    lower, weight = attribution['lower'], attribution['weight']
    covered = volume*attribution['covered_share']
    # covered_share is a per-face fraction, so the uncovered remainder cannot be
    # negative in exact arithmetic; only the summation order of the same terms
    # can make it one. The ledger this feeds refuses negative volumes with no
    # tolerance, and a first step returning 1.4e-11 km3 in total produced a
    # -1.6e-27 km3 remainder that stopped a run before its second frame. Absorb
    # cancellation roundoff without hiding a real imbalance.
    uncovered = float(volume.sum()-covered.sum())
    if uncovered < -1e-9*max(float(np.abs(volume).sum()), 1.):
        raise ValueError('Suture-covered volume exceeds the volume returned to the mantle.')
    result = {0: max(0., uncovered)}
    denominator = attribution['covered_area']
    if 'eligible_pair_volume_km3' in attribution:
        weight = attribution['eligible_pair_volume_km3']
        denominator = np.bincount(lower, weights=weight, minlength=len(volume))
    per_pair = weight*np.divide(covered[lower], denominator[lower],
                                out=np.zeros(len(lower)), where=denominator[lower] > 0)
    for key, share in _grouped(attribution['contact'], per_pair).items():
        result[key] = result.get(key, 0.)+share
    return result


def underthrust_width_km(s, lower_faces):
    """Distance from each underthrust face to the nearest exposed face of its sheet.

    The honest width diagnostic the sink does NOT change: eclogitization thins a
    buried sheet, it does not narrow it, so a model orogen can stay thousands of
    kilometres wide where Earth's are 300-700 km (no lateral escape is resolved).
    Logged, never read by any law.
    """
    surface = s.material_surface
    centres = np.asarray(surface['vertices'], float)[np.asarray(surface['faces'])].mean(axis=1)
    centres /= np.maximum(np.linalg.norm(centres, axis=1)[:, None], 1e-30)
    radius = float(surface.get('radius_km', columns.RADIUS_KM))
    sheets = np.asarray(s.parcel_collision_sheet)
    exposed = np.asarray(s.parcel_exposed_fraction, float) >= .5
    result = np.full(len(lower_faces), np.nan)
    for sheet in np.unique(sheets[lower_faces]):
        targets = np.flatnonzero((sheets == sheet) & exposed)
        selected = np.flatnonzero(sheets[lower_faces] == sheet)
        if not len(targets) or not len(selected):
            continue
        rows = centres[lower_faces[selected]]
        best = np.full(len(selected), -1.)
        for begin in range(0, len(targets), 4096):
            cosine = rows@centres[targets[begin:begin+4096]].T
            best = np.maximum(best, cosine.max(axis=1))
        result[selected] = np.arccos(np.clip(best, -1., 1.))*radius
    return result


def _pull_state(s):
    """One retained-pull sample per step; it loops over every trench system."""
    import slab_memory
    if not slab_memory.enabled(s) or not hasattr(s, 'bcode') or not hasattr(s, 'trench_systems'):
        return None
    return slab_memory.pull_state(s)[1]


def _slab_attachment(s, row, weight):
    """Length-weighted retained slab strength on this contact's collision edges."""
    if weight is None:
        return None
    top, under = int(row.get('top_owner', -1)), int(row.get('under_owner', -1))
    pair = (((s.bp == top) & (s.bq == under)) | ((s.bp == under) & (s.bq == top)))
    import normal_partition
    selected = np.flatnonzero(normal_partition.collision(s) & pair)
    if not len(selected):
        return 0.
    length = np.asarray(s.bl, float)[selected]
    if not length.sum():
        return 0.
    return float(weight[selected]@length/length.sum())


def contact_diagnostics(s, attribution, volume):
    """Per-contact width, attachment and the candidate continental slab-pull load.

    The load is Capitanio et al. (2010) read on this model's own numbers,
    drho*g*(returned volume per metre of front). It is EXPORTED WITH COEFFICIENT
    ZERO: a continental slab-pull torque would have to be partitioned against the
    collision GPE the relaxation already spends and against the megathrust
    resistance, and none of that is decided here.
    """
    rows = {int(row['id']): row for row in s.collision_contacts}
    shares = _contact_shares(s, volume, attribution)
    booked = s.mantle_return_km3.get('by_contact', {})
    # Only contacts that have actually returned crust are logged: a run carries
    # hundreds of retained rows, and a frame does not need a zero for each.
    active = sorted(identity for identity in rows
                    if shares.get(identity, 0.) > 0. or booked.get(str(identity), 0.) > 0.)
    if not active:
        return []
    lower = attribution['lower'] if attribution is not None else np.empty(0, int)
    faces = np.unique(lower) if len(lower) else np.empty(0, int)
    width = underthrust_width_km(s, faces) if len(faces) else np.empty(0)
    at = {int(face): k for k, face in enumerate(faces)}
    pull = _pull_state(s)
    contact = attribution['contact'] if attribution is not None else np.empty(0, int)
    order = np.argsort(contact, kind='stable')
    bounds = np.searchsorted(contact[order], [active, np.asarray(active)+1])
    result = []
    for k, identity in enumerate(active):
        row = rows[identity]
        pairs = order[bounds[0][k]:bounds[1][k]]
        values = width[[at[int(f)] for f in np.unique(lower[pairs])]] if len(pairs) else np.empty(0)
        values = values[np.isfinite(values)]
        length_km = sum(float(front.get('length_km', 0.)) for front in row.get('local_fronts', ()))
        overlap = float(row.get('overlap_area_km2', 0.))
        # The ledger already carries this step, because record() ran first.
        cumulative = float(booked.get(str(identity), 0.))
        load = (CONTINENTAL_SLAB_PULL*ECLOGITE_DENSITY_CONTRAST*GRAVITY_M_S2
                * cumulative*1e9/(length_km*1e3)) if length_km > 0 else 0.
        result.append(dict(contact_id=identity,
            step_returned_km3=float(shares.get(identity, 0.)), cumulative_returned_km3=cumulative,
            front_length_km=length_km, overlap_area_km2=overlap,
            area_over_length_km=overlap/length_km if length_km > 0 else 0.,
            geometric_underthrust_p90_km=float(np.percentile(values, 90)) if len(values) else 0.,
            slab_attachment=_slab_attachment(s, row, pull),
            continental_slab_pull_n_per_m=load, continental_slab_pull_coefficient=CONTINENTAL_SLAB_PULL))
    return result


def snapshot_fields(s):
    if not enabled(s):
        return {}
    ensure_fields(s)
    result = dict(foundering_version=s.foundering_version,
                material_foundered_m=np.asarray(s.structure[COLUMN_FIELD], float).copy(),
                mantle_return_km3=deepcopy(s.mantle_return_km3),
                foundering_diagnostics=deepcopy(getattr(s, 'foundering_diagnostics', {})))
    validate_ledger(s.mantle_return_km3)
    # An inherited attribution gap stays legacy; emitting a new frame must not
    # silently assert that missing historical contact accounting is complete.
    result['mantle_return_ledger_version'] = int(all(math.isclose(
        math.fsum(s.mantle_return_km3[key].values()), s.mantle_return_km3['total'],
        rel_tol=2e-10, abs_tol=1e-9)
        for key in ('by_plate_uid', 'by_sheet', 'by_contact')))
    if s.foundering_version == VERSION:
        result.update({name: s.structure[field].copy() for name, field in INVENTORY_ARRAY_FIELDS.items()})
        result['foundering_inventory_migration'] = deepcopy(getattr(s, 'foundering_inventory_migration', {}))
    if getattr(s, 'foundering_depth_version', 0):
        result['foundering_depth_version'] = s.foundering_depth_version
        result['foundering_depth_migration'] = deepcopy(getattr(s, 'foundering_depth_migration', {}))
    import phase_evolution
    result.update(phase_evolution.snapshot_fields(s))
    return result


def validate_frame(frame):
    import phase_evolution
    phase_evolution.validate_frame(frame)
    version = frame.get('foundering_version', 0)
    depth_version = frame.get('foundering_depth_version', 0)
    ledger_version = frame.get('mantle_return_ledger_version', 0)
    if (isinstance(ledger_version, (bool, np.bool_)) or ledger_version not in (0, 1)
            or (ledger_version and not version)):
        raise ValueError('Unsupported or unversioned mantle-return ledger closure.')
    if (isinstance(depth_version, (bool, np.bool_)) or depth_version not in (0, 1)
            or (depth_version and not version)):
        raise ValueError('Unsupported or unversioned local burial integration.')
    if isinstance(version, (bool, np.bool_)) or version not in (0, 1, VERSION):
        raise ValueError('Unsupported foundering version.')
    for name, selected, required in (
            ('foundering_inventory_migration', version, VERSION),
            ('foundering_depth_migration', depth_version, 1)):
        if name in frame and (selected != required or not isinstance(frame[name], dict)):
            raise ValueError('Foundering migration record requires its matching version: '+name)
    if not version:
        if set(ARRAY_FIELDS).intersection(frame) or 'mantle_return_km3' in frame:
            raise ValueError('Foundering fields need their version.')
        return
    values = np.asarray(frame.get('material_foundered_m'))
    if (values.shape != (len(frame['material_faces']),) or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Invalid saved foundering ledger.')
    fields = set(INVENTORY_ARRAY_FIELDS).intersection(frame)
    if version == 1 and fields:
        raise ValueError('Removable-crust inventory requires foundering version 2.')
    if version == VERSION:
        if fields != set(INVENTORY_ARRAY_FIELDS):
            raise ValueError('Missing saved removable-crust inventory.')
        inventory = {field: np.asarray(frame[name]) for name, field in INVENTORY_ARRAY_FIELDS.items()}
        for value in inventory.values():
            if value.shape != values.shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Invalid saved removable-crust inventory.')
        if not np.allclose(inventory[crust_inventory.BASELINE]+inventory[crust_inventory.ADDED],
                           inventory[crust_inventory.ERODED]+inventory[crust_inventory.RETURNED]+inventory[crust_inventory.REMAINING],
                           rtol=2e-12, atol=1e-12):
            raise ValueError('Saved removable-crust inventory does not close.')
    validate_ledger(frame.get('mantle_return_km3'), require_closure=ledger_version == 1)
