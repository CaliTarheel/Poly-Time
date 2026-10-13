"""Declared finite trench traction, without a slab inventory or neck solve.

The horizontal line force is a prescribed constitutive source in N/m. The
default acts on the incoming plate only; the optional carrier policy adds a
declared smaller upper-plate source. Reaction and power exchange belong to an
omitted slab/mantle reservoir. The ordinary balance retains passive resistance;
this source does not infer slab mass or an independent rollback velocity.
"""
from copy import deepcopy
import math
import numpy as np

VERSION = 1
DEFAULT_FORCE_N_PER_M = 5e12


def normalize(value=None):
    defaults = dict(enabled=False, force_n_per_m=DEFAULT_FORCE_N_PER_M)
    if value is None:
        return defaults
    if not isinstance(value, dict) or set(value)-set(defaults)-{'initiation', 'carrier_traction'}:
        raise ValueError('Effective subduction settings contain unknown fields.')
    result = dict(defaults, **value)
    if type(result['enabled']) is not bool:
        raise ValueError('Effective subduction enabled must be boolean.')
    force = result['force_n_per_m']
    if (isinstance(force, (bool, np.bool_)) or not isinstance(force, (int, float, np.number))
            or not np.isfinite(force) or force <= 0.):
        raise ValueError('Effective subduction force_n_per_m must be finite and positive.')
    result['force_n_per_m'] = float(force)
    if 'initiation' in result:
        import effective_subduction_initiation
        result['initiation'] = effective_subduction_initiation.normalize(result['initiation'])
        if result['initiation']['enabled'] and not result['enabled']:
            raise ValueError('Lite initiation requires the existing effective subduction law.')
    if 'carrier_traction' in result:
        import effective_subduction_carrier_traction as carrier
        result['carrier_traction'] = carrier.normalize(result['carrier_traction'])
        if result['carrier_traction']['enabled'] and not result['enabled']:
            raise ValueError('Carrier traction requires the existing effective subduction law.')
    return result


def enabled(s):
    version = getattr(s, 'effective_subduction_version', 0)
    if (isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer))
            or version not in (0, VERSION)):
        raise ValueError('Unsupported effective subduction version.')
    return version == VERSION


def validate_compatibility(s):
    """Reject runtime attachment/entry experiments rather than dropping loads."""
    if not enabled(s):
        return
    if getattr(s, 'slab_memory_version', 0) != 0:
        raise ValueError('Effective subduction cannot carry a detailed slab inventory.')
    if getattr(s, 'subduction_response_version', 0) != 0:
        raise ValueError('Effective subduction requires its one-sided fixed_trench closure.')
    if (getattr(s, 'continental_entry_regions', None) is not None
            or hasattr(s, 'channel_region_store') or getattr(s, 'trench_shutdown_version', 0)):
        raise ValueError('Effective subduction is incompatible with detailed continental entry and rupture histories.')
    if not normalize(s.config.get('effective_subduction'))['enabled']:
        raise ValueError('Effective subduction state and saved configuration disagree.')
    import effective_subduction_carrier_traction as carrier
    carrier.validate(s)


def initialize(s):
    """Seed declared arcs using the existing finite primordial selection law."""
    settings = normalize(s.config.get('effective_subduction'))
    if not settings['enabled']:
        return False
    if enabled(s):
        validate_compatibility(s)
        return True
    if getattr(s, 'physics_profile_version', 0) != 1:
        raise ValueError('Effective subduction requires the reviewed_v1 physics profile.')
    if float(s.t) != 0. or int(s.steps) != 0 or s.trench_systems:
        raise ValueError('Effective subduction declarations require a fresh world without trench history.')
    import primordial_subduction
    import trench_history
    declaration = primordial_subduction.normalize(s.config.get('primordial_subduction'))
    if not declaration['enabled']:
        raise ValueError('Effective subduction requires explicit primordial_subduction declarations.')
    incoming, selected, selection = primordial_subduction.selected_target_edges(s, declaration)
    if not np.any(selected):
        raise ValueError('No resolved initial interfaces exist for effective subduction.')
    s.slab_memory_version = 0
    s.effective_subduction_version = VERSION
    s.effective_subduction_settings = deepcopy(settings)
    import effective_subduction_carrier_traction as carrier
    carrier.initialize(s)
    validate_compatibility(s)
    down = np.broadcast_to(incoming, (len(s.ba),)).astype(int).copy()
    groups = trench_history._components(s, selected, down)
    for edges in groups:
        row = trench_history._birth(s, edges, down, inherited=True)
        row['effective_subduction'] = dict(version=VERSION,
            force_n_per_m=settings['force_n_per_m'],
            initial_trace_length_km=float(np.asarray(s.bl)[edges].sum()),
            provenance='Declared finite initial trench; no inherited slab mass or model elapsed time.')
        owners = down[edges]
        others = np.where(s.bp[edges] == owners, s.bq[edges], s.bp[edges])
        s.polarity[owners, others] = owners
        s.polarity[others, owners] = owners
    s.effective_subduction_diagnostics = dict(version=VERSION, enabled=True,
        force_n_per_m=settings['force_n_per_m'], initial_trenches=len(groups),
        initial_trench_length_km=float(np.asarray(s.bl)[selected].sum()),
        selection=deepcopy(selection), initial_slab_inventory_created=False,
        closure='One-sided horizontal traction on the incoming plate; unresolved slab/mantle reservoir supplies the reaction and power.',
        evolution='Declared traces persist through stall and opening; only actual oceanic convergence permits intake.',
        limitations='No resolved slab energetics, spontaneous initiation, breakoff, or independent trench rollback.')
    if settings.get('initiation', {}).get('enabled', False):
        import effective_subduction_initiation
        effective_subduction_initiation.initialize(s)
    prepare(s)
    s._record('effective_subduction', 'Declared finite trenches supply effective incoming-plate traction.',
              ('effective_subduction', VERSION), details=deepcopy(s.effective_subduction_diagnostics))
    return True


def _water_fraction(s, owners):
    import native_subduction
    if native_subduction.enabled(s):
        water = np.asarray(native_subduction.edge_ocean_fraction(s, owners), float)
    else:
        below = np.where(owners == np.asarray(s.bp), s.ba, s.bb)
        water = (np.asarray(s.crust)[below] == 0).astype(float)
    if water.shape != owners.shape or not np.isfinite(water).all() or np.any((water < 0.) | (water > 1.)):
        raise ValueError('Effective subduction needs aligned incoming ocean fractions in [0,1].')
    return np.where((owners >= 0) & (water > 1e-10), water, 0.)


def line_state(s):
    """Return incoming owners, horizontal N/m forces, and matched live traces.

    The last array is a boolean geometric declaration, including stalled dry
    pieces whose force is zero. It is never a slab length or a mass proxy.
    Existing trace identity, polarity and owners select the source; solved
    convergence and boundary color do not select its force.
    """
    count = len(s.ba)
    owners = np.full(count, -1, int)
    forces = np.zeros(count)
    live = np.zeros(count, bool)
    if not enabled(s):
        return owners, forces, live
    validate_compatibility(s)
    ids = np.asarray(s.trench_id)
    if ids.shape != (count,):
        raise ValueError('Effective subduction trace IDs must align with current contacts.')
    slots = {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(s.active)}
    for row in s.trench_systems:
        law = row.get('effective_subduction')
        if not law or law.get('activation_pending', False) or row['phase'] in ('shutdown', 'joined'):
            continue
        if law.get('version') != VERSION:
            raise ValueError('Unsupported declared effective trench law.')
        force = normalize(dict(enabled=True, force_n_per_m=law['force_n_per_m']))['force_n_per_m']
        down, over = slots.get(row['downgoing_plate_uid']), slots.get(row['overriding_plate_uid'])
        if down is None or over is None or down == over:
            continue
        select = ((ids == row['id']) & (np.asarray(s.bl) > 1e-10)
                  & (((s.bp == down) & (s.bq == over)) | ((s.bp == over) & (s.bq == down))))
        share = None
        if law.get('initiation_finite_support',False):
            import effective_subduction_initiation
            # One where the whole contact carries the trench. The local initiation policy
            # gives a contact the share of its length that lies on the declared trace.
            share=effective_subduction_initiation.support_share(s,row,select)
            select=share>0.
        owners[select], forces[select], live[select] = down, force, True
        if share is not None:forces[select]*=share[select]
    forces *= _water_fraction(s, owners)
    return owners, forces, live


def prepare(s):
    """Rematch the established mechanism; finite capture gates real intake."""
    if not enabled(s):
        return
    import trench_history
    import physics_profile
    s.trench_id = trench_history._match(s, include_shutdown=False)
    owners, force, live = line_state(s)
    s.down[:] = -1
    s.down[live] = owners[live]
    s.trench_maturity = np.zeros(len(s.ba))
    # A parent's mean speed can hide a locally closing portion of its finite
    # trace. Maturity describes the declared mechanism; native capture clips
    # the actual Euler convergence footprint and excludes incoming continent.
    s.trench_maturity[live & (force > 0.)] = 1.
    physics_profile.stall_buoyant_incoming(s)


def update(s, dt):
    """Persist declared mechanisms without speed-dependent aging or shutdown."""
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0.:
        raise ValueError('Effective trench history needs finite nonnegative elapsed time.')
    previous = getattr(s, '_trench_last_update_myr', None)
    if previous is not None and float(s.t) <= previous:
        prepare(s)
        return
    prepare(s)
    import trench_history
    slots = trench_history._slots(s)
    for row in s.trench_systems:
        if (not row.get('effective_subduction') or row['effective_subduction'].get('activation_pending',False)
                or row['phase'] in ('shutdown', 'joined')):
            continue
        if row['downgoing_plate_uid'] not in slots or row['overriding_plate_uid'] not in slots:
            trench_history._shutdown(s, row, 'plate_owner_lost')
            continue
        select=s.trench_id == row['id']
        if row['effective_subduction'].get('initiation_finite_support',False):
            import effective_subduction_initiation
            select=effective_subduction_initiation.supported_parents(s,row,select)
        edges = np.flatnonzero(select)
        if not len(edges):
            row['blocking_reason'] = 'declared_trace_unmatched'
            continue
        # Keep the finite declaration's advected support. Replacing it by all
        # nearby matches would grow the arc by a matching-radius halo each step.
        _, center, length, _ = trench_history._geometry(s, edges)
        row.update(observed_center=center, observed_length_km=length,
                   last_seen_myr=float(s.t), phase='mature', maturity=1.,
                   quiet_myr=0., blocked_myr=0., blocking_reason='')
        speed = float(np.average(np.maximum(-np.asarray(s.normal_speed)[edges], 0.), weights=np.asarray(s.bl)[edges]))
        row['active_myr'] += dt
        row['total_active_myr'] += dt
        row['shortening_km'] += speed*dt
        row['total_shortening_km'] += speed*dt
    import effective_subduction_initiation
    effective_subduction_initiation.update(s, dt)
    s._trench_last_update_myr = float(s.t)
    prepare(s)
    _, forces, live = line_state(s)
    s.trench_history_diagnostics = dict(systems=len(s.trench_systems),
        mature=sum(r['phase'] == 'mature' for r in s.trench_systems),
        shutdown=sum(r['phase'] == 'shutdown' for r in s.trench_systems),
        active_length_km=float(np.asarray(s.bl)[live].sum()),
        force_bearing_length_km=float(np.asarray(s.bl)[forces > 0.].sum()),
        policy=('Persistent effective traction plus explicit local induced-fault initiation; no quiet shutdown timer.'
            if effective_subduction_initiation.enabled(s) else
            'Persistent declared effective traction; no kinematic initiation or quiet shutdown timer.'))


def snapshot(s):
    if not enabled(s):
        return {}
    _, forces, live = line_state(s)
    diagnostics = deepcopy(s.effective_subduction_diagnostics)
    import effective_subduction_carrier_traction as carrier
    if carrier.enabled(s):
        diagnostics['closure'] = carrier.POLICY
    diagnostics.update(current_matched_length_km=float(np.asarray(s.bl)[live].sum()),
        current_force_bearing_length_km=float(np.asarray(s.bl)[forces > 0.].sum()),
        integrated_incoming_force_n=float(forces @ (np.asarray(s.bl)*1e3)))
    import effective_subduction_initiation
    if effective_subduction_initiation.enabled(s):
        diagnostics['evolution']='Established traces persist; accepted local induced-fault gates create pending trenches, activated before their next solved source interval.'
    return dict(effective_subduction_version=VERSION, effective_subduction_diagnostics=diagnostics,
        **effective_subduction_initiation.snapshot(s), **carrier.snapshot(s))


def validate_frame(frame):
    """Validate optional carrier metadata without changing legacy frame contracts."""
    import effective_subduction_carrier_traction as carrier
    return carrier.validate_frame(frame)
