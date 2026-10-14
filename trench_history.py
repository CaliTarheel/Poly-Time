"""Persistent, local subduction histories for the reduced-complexity engine.

This module observes real convergent boundaries. It never creates a boundary,
changes angular velocity, or supplies a mantle force. Geometry follows the
overriding plate between observations, then is reconciled with the actual
contact. Distinct connected trenches between the same two plates have distinct
identities. A short interruption preserves memory; sustained collision,
extension, disappearance, or owner loss ends an episode.

The 10 Myr maturation and shutdown delays below are worldbuilding closures,
not fitted predictions. The distinction between initiation, episodic restart,
polarity reversal and simple separation follows the SZI database compilation:
https://www.nature.com/articles/s41467-020-17522-9 .

Integration: initialize after initial boundaries; advect once before moving
time; prepare after each boundary refresh; update once after trench migration
and before arc/back-arc production. prepare is strictly match-only. Missing
history retains legacy behavior for old fixtures; an initialized history has
zero production at an as-yet uninitiated trench.
"""
from __future__ import annotations

import math
from copy import deepcopy
import numpy as np
import native_subduction
import slab_memory
import physics_profile
import normal_partition
import effective_subduction

from ridge_geometry import rotate, candidate_cells

RADIUS_KM = 6371.
MATURATION_MYR = 10.
MATURATION_SHORTENING_KM = 100.
COLLISION_SHUTDOWN_MYR = 6.
EXTENSION_SHUTDOWN_MYR = 10.
QUIET_SHUTDOWN_MYR = 20.
REACTIVATION_MEMORY_MYR = 100.
# Attached-slab persistence (opt-in, version 1): below this normal speed a
# loaded trench is stationary, not opening or quiet. A plate held still by
# balanced pulls still carries its hanging slab (Scotese Rule IV); solver
# noise of ~1e-3 km/Myr must not read as extension and shut the pull off.
# 0.5 km/Myr is 0.05 cm/yr, two orders below plate-driving speeds.
PERSISTENCE_FLOOR_KM_MYR = .5
PERSISTENCE_POLICIES = ('kinematic', 'attached_slab_v1')
SEPARATION_PERSISTENCE_MYR = 16.
MAX_GEOMETRY_POINTS = 192


def _slots(s):
    return {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(s.active)}


def _unit(v):
    return np.asarray(v, float)/max(float(np.linalg.norm(v)), 1e-30)


def _spacing(s):
    if isinstance(getattr(s, 'native_mesh', None), dict):
        return math.sqrt(float(np.mean(s.cell_area)))
    return np.pi*RADIUS_KM/int(s.h)


def _nearest(points, anchors):
    """Chord distance in km, bounded temporary memory and pole/seam invariant."""
    points, anchors = np.asarray(points, float), np.asarray(anchors, float)
    if not len(anchors):
        return np.full(len(points), np.inf)
    answer = np.empty(len(points))
    for start in range(0, len(points), 1024):
        dot = points[start:start+1024] @ anchors.T
        answer[start:start+1024] = RADIUS_KM*np.sqrt(np.maximum(0., 2.-2.*dot.max(axis=1)))
    return answer


def _match(s, *, include_shutdown=True):
    """Match each current face to the nearest local historical trench."""
    count = len(s.ba)
    ids, best = np.zeros(count, np.int64), np.full(count, np.inf)
    if not getattr(s, 'trench_systems', None) or not count:
        return ids
    uid_p, uid_q = s.plate_uid[s.bp], s.plate_uid[s.bq]
    current = _slots(s)
    initial_selection = None
    if ((getattr(s, 'primordial_subduction_version', 0) == 1 or effective_subduction.enabled(s)) and
            float(s.t) == 0. and int(s.steps) == 0):
        # The initial condition specifies complete selected arcs. The normal
        # rematching radius accommodates moving traces, but at the unchanged
        # initial geometry it must not seed neighboring unselected margins.
        import primordial_subduction
        _, initial_selection, _ = primordial_subduction.selected_target_edges(s)
    for row in s.trench_systems:
        if row['phase'] == 'joined':
            continue
        if row['downgoing_plate_uid'] not in current or row['overriding_plate_uid'] not in current:
            continue
        if row['phase'] == 'shutdown' and (not include_shutdown or
                float(s.t)-row['last_seen_myr'] > REACTIVATION_MEMORY_MYR):
            continue
        p, q = row['plate_uids']
        edges = np.flatnonzero(((uid_p == p) & (uid_q == q)) | ((uid_p == q) & (uid_q == p)))
        if initial_selection is not None and (row.get('initial_subduction') or row.get('effective_subduction')):
            edges = edges[initial_selection[edges]]
        if not len(edges):
            continue
        distance = _nearest(s.bmid[edges], row['geometry_xyz'])
        radius = max(120., 1.8*_spacing(s), row.get('anchor_gap_km', 0.)*.65)
        incoming_owner = current[row['downgoing_plate_uid']]
        incoming = np.where(s.bp[edges] == incoming_owner, s.ba[edges], s.bb[edges])
        # A newly reversed polarity may occupy the same trace as a blocked old
        # trench. Prefer the physically admissible current episode there, while
        # retaining the old trace for its separate shutdown history.
        buoyant = (1.-native_subduction.edge_ocean_fraction(s,incoming_owner)[edges]
                   if native_subduction.enabled(s) else (s.crust[incoming] > 0))
        rank = distance + radius*(2.*buoyant
                                  + .5*(row['phase'] == 'shutdown')
                                  + .05*(row['phase'] == 'quiet'))
        take = (distance <= radius) & (rank < best[edges])
        chosen = edges[take]
        ids[chosen], best[chosen] = row['id'], rank[take]
    return ids


def prepare(s):
    """Refresh edge IDs, local polarity, and mechanism maturity without aging.

    The returned arrays refer to the current boundary cache. A quiet historical
    trench can be identified on a transform/collision face for review, but its
    mechanism weight is zero unless that face is genuinely subducting now.
    """
    if not hasattr(s, 'trench_systems'):
        return
    if effective_subduction.enabled(s):
        return effective_subduction.prepare(s)
    s.trench_id = _match(s)
    s.trench_maturity = np.zeros(len(s.ba), float)
    current = _slots(s)
    rupture_governed = _rupture_governed(s)
    for row in s.trench_systems:
        if rupture_governed and not _local_neck_state(row)[1]:
            # Source capture runs before the lifecycle update in a native
            # step; detached or unrepresented slabs must be gated here too.
            continue
        candidates = normal_partition.convergence(s) if normal_partition.enabled(s) else (s.bcode == 2)
        if row.get('initial_subduction') and row['phase'] not in ('shutdown', 'joined'):
            # An inherited attached slab remains a mechanism and force source
            # before its solved relative motion becomes convergent. Capture
            # still receives maturity only on actually closing oceanic pieces.
            candidates = np.asarray(s.bl) > 1e-10
        edges = np.flatnonzero((s.trench_id == row['id']) & candidates)
        p = current.get(row['downgoing_plate_uid'])
        if not len(edges) or p is None:
            continue
        below = np.where(s.bp[edges] == p, s.ba[edges], s.bb[edges])
        water = (native_subduction.edge_ocean_fraction(s,p)[edges] > 0.
                 if native_subduction.enabled(s) else s.crust[below] == 0)
        valid = water & (s.normal_speed[edges] < 0)
        # Age changes on the incoming ocean cannot flip an established trench.
        # An incoming buoyant block, however, is never forced beneath the plate.
        if physics_profile.polarity_enabled(s):
            # Preserve the historical incoming side even where its continent
            # blocks the old trench. The stall operator disables consumption;
            # it must never hand maturity or polarity to the opposite plate.
            s.down[edges] = p
        else:
            s.down[edges[valid]] = p
        if row['phase'] not in ('shutdown', 'joined'):
            s.trench_maturity[edges[valid]] = row['maturity']
    physics_profile.stall_buoyant_incoming(s)


def weights(s):
    """Mechanism scaling, with a backwards-compatible uninitialized fallback."""
    if not hasattr(s, 'trench_systems'):
        return np.ones(len(s.ba), float)
    value = getattr(s, 'trench_maturity', None)
    if value is None or len(value) != len(s.ba):
        prepare(s)
    return np.asarray(s.trench_maturity)


def ocean_buoyancy(age_myr):
    """Bounded cooling-age proxy for slab negative buoyancy, not a density solve.

    Fresh ridge material has no developed cooling contrast. Older oceanic
    lithosphere strengthens this proxy up to the retained traction ceiling.
    Actual slab depth, mantle flow and interface friction remain unresolved.
    """
    age = np.asarray(age_myr, float)
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError('Slab buoyancy requires finite nonnegative ocean age.')
    return np.minimum(np.sqrt(age / 80.), 1.8)


def slab_pull_weights(s, *, legacy=False):
    """Use the SAME local developed-trench maturity as physical consumption.

    No initialized history means no established slab pull. Old age-only
    behavior requires an explicit legacy request; missing fields never opt a
    new world into it. Basal driving can still establish initial convergence.
    """
    if type(legacy) is not bool:
        raise ValueError('Legacy slab-force mode must be explicit boolean.')
    if slab_memory.enabled(s) and not legacy:
        return slab_memory.pull_state(s)[1]
    count = len(s.ba)
    result = np.zeros(count)
    down = np.asarray(s.down)
    valid = normal_partition.subduction(s) & (np.asarray(s.normal_speed) < 0) & (down >= 0)
    if not np.any(valid):
        return result
    below = np.where(down == s.bp, s.ba, s.bb)
    age = np.asarray(s.age)[below[valid]]
    buoyancy = ocean_buoyancy(age)
    if legacy:
        result[valid] = np.maximum(.2, buoyancy)
        return result
    if not hasattr(s, 'trench_systems') or not s.trench_systems:
        return result
    maturity = weights(s)
    if (maturity.shape != (count,) or not np.isfinite(maturity).all()
            or np.any((maturity < 0) | (maturity > 1))):
        raise ValueError('Slab pull requires aligned finite trench maturity in [0,1].')
    water = (native_subduction.edge_ocean_fraction(s, down) if native_subduction.enabled(s)
             else (np.asarray(s.crust)[below] == 0).astype(float))
    result[valid] = maturity[valid] * water[valid] * buoyancy
    return result


def slab_pull_state(s):
    """Force owners may retain a developed slab during brief collision shutdown."""
    if slab_memory.enabled(s):
        return slab_memory.pull_state(s)
    return np.asarray(s.down), slab_pull_weights(s)


def _corners(a, b, width, height):
    """Dual-grid corner IDs shared by successive true contact faces.

    The longitude seam is a single seam; all longitude corners at a geographic
    pole are the same physical vertex. No distance through a projected map is
    used for topology.
    """
    ra, ca, rb, cb = a//width, a % width, b//width, b % width
    if ra == rb:
        col = cb if (cb-ca) % width == 1 else ca
        vertices = ((ra, col), (ra+1, col))
    else:
        row = max(ra, rb)
        vertices = ((row, ca), (row, (ca+1) % width))
    return tuple(-1 if r == 0 else -2 if r == height else r*width+c for r, c in vertices)


def _components(s, select, down):
    """Connected chains, split by pair and polarity; no pair-wide grouping."""
    edges = np.flatnonzero(select)
    parents = np.arange(len(edges))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i
    corners = {}
    native = getattr(s, 'native_mesh', None)
    native_edges = None
    if isinstance(native, dict):
        indices = np.asarray(s.boundary_edge_indices, dtype=np.int64)
        if indices.shape != (len(s.ba),):
            raise ValueError('Native trench edges must align with the current boundary cache.')
        native_edges = native['edge_vertices'][indices]
    for j, edge in enumerate(edges):
        pair = tuple(sorted((int(s.plate_uid[s.bp[edge]]), int(s.plate_uid[s.bq[edge]]))))
        key = (*pair, int(s.plate_uid[down[edge]]))
        vertices = (native_edges[edge] if native_edges is not None else
                    _corners(int(s.ba[edge]), int(s.bb[edge]), s.w, s.h))
        for corner in vertices:
            k = (*key, corner)
            if k in corners:
                first, second = find(j), find(corners[k])
                if first != second:
                    parents[max(first, second)] = min(first, second)
            else:
                corners[k] = j
    groups = {}
    for j, edge in enumerate(edges):
        groups.setdefault(find(j), []).append(int(edge))
    return [np.asarray(group, int) for group in groups.values()]


def _geometry(s, edges):
    # Deterministic farthest-point anchors cover bent chains without privileging
    # longitude ordering or losing a branch at the map seam.
    points = s.bmid[edges]
    if len(points) > MAX_GEOMETRY_POINTS:
        selected = [0]
        distance = np.full(len(points), np.inf)
        for _ in range(MAX_GEOMETRY_POINTS-1):
            distance = np.minimum(distance, 2.-2.*(points @ points[selected[-1]]))
            selected.append(int(np.argmax(distance)))
        points = points[selected]
    else:
        points = points.copy()
    length = float(s.bl[edges].sum())
    center = _unit(np.sum(s.bmid[edges]*s.bl[edges, None], axis=0))
    if np.linalg.norm(center) < .5:
        center = points[0]
    # Covering error is an explicit local sampling allowance, not a global
    # pair radius or a centroid-based trench identity.
    gap = 2.*float(_nearest(s.bmid[edges], points).max(initial=0.))
    return points.tolist(), center.tolist(), length, gap


def _event(s, row, phase, reason):
    row['phase_serial'] += 1
    row['history'].append(dict(time_myr=float(s.t), phase=phase, reason=reason,
                               episode=row['episode']))
    current = _slots(s)
    owner_slots = tuple(current[u] for u in row['plate_uids'] if u in current)
    descriptions = {
        'initiating': 'Sustained local convergence is establishing a new subduction system.',
        'inherited': 'An established subduction system is present in the specified geological initial condition.',
        'mature': 'The local subduction system has developed a mature modeled slab and arc supply.',
        'quiet': 'The local trench has become inactive; its geometry and polarity are retained briefly.',
        'shutdown': 'The local subduction episode ended after ' + reason.replace('_', ' ') + '.',
        'reactivated': 'Convergence restarted a new episode at a preserved former trench.',
        'resumed': 'The trench resumed convergence before its subduction episode shut down.',
        'separated': 'A continuous subduction system separated into independently tracked local trenches.',
        'transferred': 'The existing slab and trench history followed a local change of overriding or subducting plate.'}
    effective = effective_subduction.enabled(s)
    if effective:
        descriptions.update(
            inherited='A declared finite trench supplies effective pull in the specified geological initial condition.',
            mature='The declared local trench remains an established effective force source.',
            transferred='The declared trench traction and polarity followed a local change of incoming or overriding plate.')
    if hasattr(s, '_record'):
        s._record('trench_'+phase, descriptions[phase], ('trench', row['id'], row['phase_serial']),
                  plates=owner_slots, xyz=row['center'], details={
                      'trench_id': row['id'], 'episode': row['episode'], 'phase': row['phase'],
                      'reason': reason, 'downgoing_plate_uid': row['downgoing_plate_uid'],
                      'overriding_plate_uid': row['overriding_plate_uid'],
                      'maturity': row['maturity'], 'length_km': row['length_km'],
                      'parent_trench_id': row.get('parent_trench_id'),
                      'mechanism': ('persistent declared incoming-plate traction; no convergence initiation or quiet timer'
                          if effective else 'local convergence history with parameterized initiation and shutdown hysteresis')})


def _birth(s, edges, down, *, parent=None, predecessor=None, inherited=False):
    first = int(edges[0])
    p = int(down[first])
    q = int(s.bq[first] if s.bp[first] == p else s.bp[first])
    geometry, center, length, gap = _geometry(s, edges)
    row = dict(id=int(s.next_trench_id), plate_uids=sorted([int(s.plate_uid[p]), int(s.plate_uid[q])]),
               downgoing_plate_uid=int(s.plate_uid[p]), overriding_plate_uid=int(s.plate_uid[q]),
               born_myr=float(s.t), last_seen_myr=float(s.t), phase='initiating', phase_serial=0,
               maturity=0., active_myr=0., shortening_km=0., total_active_myr=0.,
               total_shortening_km=0., quiet_myr=0., blocked_myr=0., blocking_reason='',
               episode=1, geometry_xyz=geometry, center=center, length_km=length,
               anchor_gap_km=gap, history=[], episodes=[dict(start_myr=float(s.t), end_myr=None)],
               parent_trench_id=None if parent is None else parent['id'],
               predecessor_trench_id=None if predecessor is None else predecessor['id'])
    s.next_trench_id += 1
    s.trench_systems.append(row)
    if slab_memory.enabled(s):
        slab_memory.ensure(row, conservative=slab_memory.conservative(s))
    if parent is not None:
        for key in ('maturity', 'active_myr', 'shortening_km'):
            row[key] = parent[key]
        for key in ('initial_maturity', 'initial_subduction', 'effective_subduction'):
            if key in parent:
                row[key] = deepcopy(parent[key])
        row['phase'] = 'mature' if row['maturity'] >= 1.-1e-12 else 'initiating'
        _event(s, row, 'separated', 'continuous_subduction_separation')
    elif inherited:
        row.update(phase='mature', maturity=1., initial_maturity=1.)
        _event(s, row, 'inherited', 'declared_geological_initial_condition')
    else:
        _event(s, row, 'initiating', 'polarity_reversal' if predecessor else 'new_convergent_contact')
    return row


def _shutdown(s, row, reason):
    if row['phase'] == 'shutdown':
        return
    row['phase'], row['maturity'] = 'shutdown', 0.
    if 'initial_maturity' in row:
        row['initial_maturity'] = 0.
    row['episodes'][-1].update(end_myr=float(s.t), shutdown_reason=reason)
    _event(s, row, 'shutdown', reason)


def _join(s, row, successor):
    """Resolve bookkeeping identities without claiming a slab shut down."""
    if slab_memory.enabled(s):
        target=next(r for r in s.trench_systems if r['id']==successor)
        slab_memory.join(row,target)
    row['phase'] = 'joined'
    row['successor_trench_id'] = int(successor)
    row['pending_branches'] = []
    row['episodes'][-1].update(continued_as_trench_id=int(successor), continued_myr=float(s.t))
    row['history'].append(dict(time_myr=float(s.t), phase='joined',
                               reason='continuous_trench_geometry_rejoined',
                               episode=row['episode'], successor_trench_id=int(successor)))
    # Existing back-arc load belongs to the continuous slab, not a raster branch
    # label. Avoid both resetting it and creating a replacement loading record.
    for basin in getattr(s, 'backarc_basins', []):
        if basin.get('trench_id') == row['id'] and basin.get('phase') != 'closed':
            basin['trench_id'] = int(successor)


def _pending_branch(s, row, edges, dt):
    """A spatially persistent split must outlive brief threshold flicker."""
    geometry, center, _, gap = _geometry(s, edges)
    pending = row.setdefault('pending_branches', [])
    candidates = []
    for branch in pending:
        if branch['last_seen_myr'] == float(s.t):
            continue
        distance = float(np.mean(_nearest(np.asarray(geometry), branch['geometry_xyz'])))
        if distance <= max(120., 1.8*_spacing(s), gap*.65):
            candidates.append((distance, branch))
    if candidates:
        branch = min(candidates, key=lambda value: value[0])[1]
        consecutive = abs(branch['last_seen_myr']+dt-float(s.t)) < 1e-6
        branch['separated_myr'] = branch['separated_myr']+dt if consecutive else dt
    else:
        branch = dict(started_myr=float(s.t), separated_myr=dt)
        pending.append(branch)
    branch.update(last_seen_myr=float(s.t), geometry_xyz=geometry, center=center)
    return branch


def initialize(s):
    """Seed initial local systems once; zero elapsed history means zero maturity."""
    if hasattr(s, 'trench_systems'):
        prepare(s)
        return
    s.trench_systems, s.next_trench_id = [], 1
    s._trench_last_update_myr = None
    s._trench_last_advection_myr = None
    update(s, 0.)


def advect(s, dt):
    """Rotate live and recently shut-down trench geometry with its upper plate."""
    if not hasattr(s, 'trench_systems'):
        return
    key = (float(s.t), float(dt))
    if getattr(s, '_trench_last_advection_myr', None) == key:
        return
    current = _slots(s)
    for row in s.trench_systems:
        owner = current.get(row['overriding_plate_uid'])
        if row['phase'] == 'joined' or owner is None or (row['phase'] == 'shutdown' and
                float(s.t)-row['last_seen_myr'] > REACTIVATION_MEMORY_MYR):
            continue
        rotation = np.asarray(s.omega[owner])*float(dt)
        import slab_tether_local
        slab_tether_local.advect(row,rotation)
        row['geometry_xyz'] = rotate(np.asarray(row['geometry_xyz']), rotation).tolist()
        import slab_anchors
        slab_anchors.advect(row, rotation)
        row['center'] = rotate(np.asarray(row['center']), rotation).tolist()
        for branch in row.get('pending_branches', []):
            branch['geometry_xyz'] = rotate(np.asarray(branch['geometry_xyz']), rotation).tolist()
            branch['center'] = rotate(np.asarray(branch['center']), rotation).tolist()
    s._trench_last_advection_myr = key


def _rupture_governed(s):
    """Select explicit local-neck termination only for its native experiment."""
    version=getattr(s,'trench_shutdown_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,1):
        raise ValueError('Unsupported trench shutdown version.')
    return int(version)==1


def normalize_persistence(value='kinematic'):
    if not isinstance(value, str) or value not in PERSISTENCE_POLICIES:
        raise ValueError('Trench persistence must be kinematic or attached_slab_v1.')
    return value


def persistence_floor(s):
    """Stationary-trench speed floor in km/Myr; zero keeps the kinematic law."""
    version = getattr(s, 'trench_persistence_version', 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version not in (0, 1):
        raise ValueError('Unsupported trench persistence version.')
    return PERSISTENCE_FLOOR_KM_MYR if version == 1 else 0.


def _local_neck_state(row):
    """Report complete rupture and surviving attached mass on a local trace."""
    import slab_tether_history as necks
    import slab_tether_local as local
    if not local.enabled(row):return False,False
    necks.validate(row)
    channels=row[necks.FIELD]
    ruptured=bool(channels) and all(c['neck']['damage']==1. for c in channels)
    attached=any(c['neck']['damage']<1. and c['retained_area_km2']>0.
                 and c['retained_excess_mass_kg']>0. for c in channels)
    return ruptured,attached


def update(s, dt):
    """Commit local lifecycle transitions at most once for each simulation time."""
    dt = float(dt)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError('Trench history needs a finite, nonnegative elapsed time.')
    if effective_subduction.enabled(s):
        return effective_subduction.update(s, dt)
    rupture_governed=_rupture_governed(s)
    if not hasattr(s, 'trench_systems'):
        initialize(s)
        if dt == 0:
            return
    previous = getattr(s, '_trench_last_update_myr', None)
    if previous is not None and float(s.t) <= previous:
        prepare(s)
        return
    prepare(s)
    matched = s.trench_id.copy()
    rows = {row['id']: row for row in s.trench_systems}
    current = _slots(s)
    down = np.asarray(s.down).copy()
    active = normal_partition.subduction(s) & (s.normal_speed < 0) & (down >= 0)
    attached_trace = np.ones(len(s.ba), bool)
    if rupture_governed:
        # A boundary color is not by itself a slab continuation. Existing
        # channels retain only their inherited incoming owner. The production
        # lifecycle may additionally admit an independent fresh contact that
        # passes its post-breakoff distance/time guard.
        attached_trace[:] = False
        inherited_down = np.full(len(s.ba),-1,int)
        for row in s.trench_systems:
            if row['phase'] in ('shutdown','joined') or not _local_neck_state(row)[1]:
                continue
            owner=current.get(row['downgoing_plate_uid'])
            if owner is None:continue
            trace=matched==row['id']
            attached_trace[trace]=True
            inherited_down[trace]=owner
        continuation=attached_trace & (down==inherited_down)
        replacement=np.zeros(len(s.ba),bool)
        candidates=np.flatnonzero(active & (matched==0))
        if len(candidates) and getattr(s,'replacement_subduction_version',0)==1:
            import continental_lifecycle
            replacement[candidates]=continental_lifecycle.replacement_allowed(s,candidates)
        active &= continuation | replacement
    floor = persistence_floor(s)
    if floor > 0. and not rupture_governed:
        # A loaded trench whose plates are merely stationary keeps its
        # established polarity and pull. Genuine opening above the floor, a
        # buoyant arrival or a lost owner still quiet and shut it down.
        for row in s.trench_systems:
            if (row['phase'] in ('shutdown', 'joined')
                    or float(row.get(slab_memory.RETAINED_MASS_FIELD, 0.)) <= 0.):
                continue
            own, over = current.get(row['downgoing_plate_uid']), current.get(row['overriding_plate_uid'])
            if own is None or over is None:
                continue
            trace = ((matched == row['id']) & (np.abs(s.normal_speed) <= floor) & (s.bl > 1e-10)
                     & (((s.bp == own) & (s.bq == over)) | ((s.bp == over) & (s.bq == own))))
            down[trace] = own
            active |= trace
    # At new contacts choose polarity locally, never from a pair-wide ocean age.
    fresh = np.flatnonzero(active & (matched == 0))
    if len(fresh):
        canonical = np.where(s.plate_uid[s.bp] < s.plate_uid[s.bq], s.bp, s.bq)
        fresh_mask = np.zeros(len(s.ba), bool)
        fresh_mask[fresh] = True
        for group in _components(s, fresh_mask, canonical):
            first = int(group[0])
            p, q = int(s.bp[first]), int(s.bq[first])
            aa = np.where(s.bp[group] == p, s.ba[group], s.bb[group])
            bb = np.where(s.bp[group] == p, s.bb[group], s.ba[group])
            age_p = float(np.average(s.age[aa], weights=s.bl[group]))
            age_q = float(np.average(s.age[bb], weights=s.bl[group]))
            chosen = np.full(len(group), p if age_p >= age_q else q, int)
            if native_subduction.enabled(s):
                wet_p=native_subduction.edge_ocean_fraction(s,p)[group] > 0.
                wet_q=native_subduction.edge_ocean_fraction(s,q)[group] > 0.
                chosen[~wet_p & wet_q]=q
                chosen[~wet_q & wet_p]=p
            else:
                chosen[s.crust[aa] > 0] = q
                chosen[s.crust[bb] > 0] = p
            down[group] = chosen
    below = np.where(down == s.bp, s.ba, s.bb)
    active &= (native_subduction.edge_ocean_fraction(s,down) > 0.
               if native_subduction.enabled(s) else s.crust[below] == 0)
    # Quiet pieces can bridge brief speed-threshold flicker within one existing
    # local chain. Divergence and collision never provide such a bridge.
    quiet_bridge = (matched > 0) & attached_trace & (s.bcode == 3) & (s.normal_speed <= 2.)
    for row in s.trench_systems:
        p = current.get(row['downgoing_plate_uid'])
        if p is not None:
            down[quiet_bridge & (matched == row['id'])] = p
    groups = _components(s, active | quiet_bridge, down)
    groups = [group for group in groups if np.any(active[group])]
    # Largest continuation keeps its ID when a trench separates. Further local
    # pieces inherit maturity and are explicitly separation, not new initiation.
    # Symmetric native chains can have equal physical length. Floating-point
    # roundoff under a globe rotation must not reverse their identity order.
    native = isinstance(getattr(s, 'native_mesh', None), dict)
    groups.sort(key=lambda e: (-(round(float(s.bl[e][active[e]].sum()), 7) if native
                                else float(s.bl[e][active[e]].sum())), int(e[0])))
    used, observed, joined = set(), {}, {}
    slab_branches = {}
    for group in groups:
        edges = group[active[group]]
        uid = int(s.plate_uid[down[edges[0]]])
        candidates = [int(i) for i in np.unique(matched[group]) if int(i) in rows and
                      rows[int(i)]['downgoing_plate_uid'] == uid]
        candidates.sort(key=lambda i: (-float(s.bl[group][matched[group] == i].sum()), i))
        available = [i for i in candidates if i not in used]
        row = rows[available[0]] if available else None
        if row is None:
            parent_id = candidates[0] if candidates else None
            while parent_id in joined:
                parent_id = joined[parent_id]
            parent = rows[parent_id] if parent_id is not None and rows[parent_id]['phase'] not in ('shutdown', 'joined') else None
            if parent is not None:
                branch = _pending_branch(s, parent, edges, dt)
                if branch['separated_myr'] < SEPARATION_PERSISTENCE_MYR:
                    # Both current pieces retain their old identity during the
                    # observation interval. Only their actual convergent faces
                    # produce arcs/retreat; no mechanism is painted across the gap.
                    observed[parent['id']] = np.concatenate((observed[parent['id']], edges))
                    continue
                parent['pending_branches'].remove(branch)
            reversed_ids = [int(i) for i in np.unique(matched[group]) if int(i) in rows and
                            rows[int(i)]['downgoing_plate_uid'] != uid]
            predecessor = rows[reversed_ids[0]] if reversed_ids else None
            row = _birth(s, edges, down, parent=parent, predecessor=predecessor)
            if parent is not None and slab_memory.enabled(s):
                slab_branches.setdefault(parent['id'],[]).append(row['id'])
            rows[row['id']] = row
        used.add(row['id'])
        observed[row['id']] = edges
        for prior in available[1:]:
            joined[prior] = row['id']
            used.add(prior)
    # Allocate one prior inventory over all continuing pieces together. Using
    # the original parent length repeatedly would bias later-born branches.
    for parent_id, children in slab_branches.items():
        lengths={ident:float(s.bl[observed[ident]].sum()) for ident in [parent_id,*children]}
        remaining=math.fsum(lengths.values())
        pending=[parent_id,*children]
        for ident in children:
            import slab_tether_local as local
            shares=None
            if local.enabled(rows[parent_id]):
                all_edges=np.concatenate([observed[i] for i in pending])
                shares=local.transfer_fractions(rows[parent_id],s,all_edges,np.isin(all_edges,observed[ident]))
            others=np.concatenate([observed[i] for i in pending if i!=ident])
            slab_memory.partition(rows[parent_id],rows[ident],lengths[ident]/max(remaining,1e-30),channel_fractions=shares,
                                  split_points=(s.bmid[observed[ident]],s.bmid[others]))
            pending.remove(ident)
            remaining-=lengths[ident]
    for row in s.trench_systems:
        if rupture_governed and _local_neck_state(row)[0]:
            if row['phase'] not in ('shutdown','joined'):
                _shutdown(s,row,'slab_necks_detached')
            continue
        edges = observed.get(row['id'])
        if edges is not None:
            if row['phase'] == 'shutdown':
                row.update(episode=row['episode']+1, active_myr=0., shortening_km=0.,
                           phase='initiating', maturity=0.)
                row['episodes'].append(dict(start_myr=float(s.t), end_myr=None))
                _event(s, row, 'reactivated', 'episodic_subduction')
            was_quiet = row['phase'] == 'quiet'
            geometry, center, length, gap = _geometry(s, edges)
            row.update(geometry_xyz=geometry, center=center, length_km=length,
                       anchor_gap_km=gap, last_seen_myr=float(s.t), quiet_myr=0.,
                       blocked_myr=0., blocking_reason='')
            slab_memory.refresh_line_load(row)
            speed = float(np.average(np.maximum(-s.normal_speed[edges], 0.), weights=s.bl[edges]))
            row['active_myr'] += dt
            row['shortening_km'] += speed*dt
            row['total_active_myr'] += dt
            row['total_shortening_km'] += speed*dt
            inherited_maturity = float(row.get('initial_maturity', 0.))
            row['maturity'] = float(min(1., inherited_maturity+row['active_myr']/MATURATION_MYR,
                                       inherited_maturity+row['shortening_km']/MATURATION_SHORTENING_KM))
            phase = 'mature' if row['maturity'] >= 1.-1e-12 else 'initiating'
            previous_phase = row['phase']
            row['phase'] = phase
            if was_quiet:
                _event(s, row, 'resumed', 'convergence_resumed_before_shutdown')
            elif phase == 'mature' and previous_phase != 'mature':
                _event(s, row, 'mature', 'persistent_convergence_and_slab_development')
            continue
        if row['phase'] in ('shutdown', 'joined'):
            continue
        if row['id'] in joined:
            _join(s, row, joined[row['id']])
            continue
        if row['downgoing_plate_uid'] not in current or row['overriding_plate_uid'] not in current:
            _shutdown(s, row, 'plate_owner_lost')
            continue
        near = np.flatnonzero(matched == row['id'])
        reason, delay = 'convergence_ceased', QUIET_SHUTDOWN_MYR
        if len(near):
            total = max(float(s.bl[near].sum()), 1e-10)
            collision = float(s.bl[near][normal_partition.collision(s)[near]].sum())/total
            extending = (s.normal_speed > floor if normal_partition.enabled(s)
                         else np.isin(s.bcode, [1, 5]))
            extension = float(s.bl[near][extending[near]].sum())/total
            p = current[row['downgoing_plate_uid']]
            incoming = np.where(s.bp[near] == p, s.ba[near], s.bb[near])
            buoyant = (float(s.bl[near]@(1.-native_subduction.edge_ocean_fraction(s,p)[near]))/total
                       if native_subduction.enabled(s) else float(s.bl[near][s.crust[incoming] > 0].sum())/total)
            if collision >= .5 or buoyant >= .5:
                reason, delay = 'buoyant_collision', COLLISION_SHUTDOWN_MYR
            elif extension >= .5:
                reason, delay = 'sustained_extension', EXTENSION_SHUTDOWN_MYR
        else:
            reason = 'contact_disappeared'
        row['quiet_myr'] += dt
        row['blocked_myr'] = row['blocked_myr']+dt if row['blocking_reason'] == reason else dt
        row['blocking_reason'] = reason
        if row['phase'] != 'quiet':
            row['phase'] = 'quiet'
            _event(s, row, 'quiet', reason)
        if rupture_governed and len(near) and _local_neck_state(row)[1]:
            # A real attached slab still exerts its coupled force on this
            # trace. Kinematic quiet/collision labels do not sever its neck.
            continue
        if row['blocked_myr'] >= delay or row['quiet_myr'] >= QUIET_SHUTDOWN_MYR:
            _shutdown(s, row, reason)
    s._trench_last_update_myr = float(s.t)
    for row in s.trench_systems:
        if 'pending_branches' in row:
            row['pending_branches'] = [b for b in row['pending_branches'] if b['last_seen_myr'] == float(s.t)]
    prepare(s)
    s.trench_history_diagnostics = dict(
        systems=len(s.trench_systems),
        initiating=sum(r['phase'] == 'initiating' for r in s.trench_systems),
        mature=sum(r['phase'] == 'mature' for r in s.trench_systems),
        quiet=sum(r['phase'] == 'quiet' for r in s.trench_systems),
        shutdown=sum(r['phase'] == 'shutdown' for r in s.trench_systems),
        joined=sum(r['phase'] == 'joined' for r in s.trench_systems),
        active_length_km=float(s.bl[normal_partition.subduction(s) & (s.trench_id > 0)].sum()),
        maturation_myr=MATURATION_MYR)


def transfer_overriding(s, old, new):
    """Rehost only trench faces whose actual overriding cell changed owners.

    Call after local membership transfer, while the preceding boundary cache
    still exists. A complete transfer preserves the identity; a partial transfer
    creates a child with inherited maturity. Returns old-ID -> transferred-ID.
    Incoming terranes are not silently converted into overriding-plate history.
    """
    return _transfer(s, old, new, 'overriding_plate_uid', 'local_overriding_plate_transfer')


def transfer_downgoing(s, old, new):
    """Rehost trench faces whose actual downgoing cell moved to another plate.

    A rift through a subducting plate does not detach its slab: the slab hangs
    from the leading edge, and that edge now belongs to the daughter. Trenches
    are matched by plate-ID pair, so without this the mature record loses its
    contact ("contact_disappeared") and the same margin restarts as a new
    initiating trench with no slab pull. In a 2026-09-21 run a 1.3e6 km2
    retained slab (~25 GW of slab power) did this when a rift split its
    subducting plate, and the three plates involved slowed from ~16 to ~5.5
    km/Myr. Same contract as transfer_overriding.
    """
    return _transfer(s, old, new, 'downgoing_plate_uid', 'local_downgoing_plate_transfer')


def transfer_split(s, old, new):
    """Carry both trench roles across a committed ownership change old -> new.

    Used for a plate split (new is the daughter) and for whole-plate absorption
    (new is the receiving neighbour). Call after membership changes, while the
    preceding boundary cache still exists.
    """
    mapping = transfer_downgoing(s, old, new)
    mapping.update(transfer_overriding(s, old, new))
    return mapping


def _relocate(s, row, edges):
    """Record where a rehosted trace now lies.

    An induced trench carries the finite fault it formed on as endpoint pairs and may load only
    boundary those arcs cover. A change of plate ownership moves neither that fault nor its ends,
    so parent and daughter both keep the whole carried support and their plate pairs decide which
    of its edges each one loads. Replacing it by the edge anchors used for every other trench
    would discard the endpoints and let a trace claim boundary it never weakened.
    """
    law = row.get('effective_subduction') or {}
    if law.get('initiation_finite_support', False):
        # The declared owner of the support follows the record that now carries it.
        law['support_origin_trench_id'] = row['id']
        return
    geometry, center, length, gap = _geometry(s, edges)
    row.update(geometry_xyz=geometry, center=center, length_km=length, anchor_gap_km=gap)


def _transfer(s, old, new, role, reason):
    if not hasattr(s, 'trench_systems') or old == new:
        return {}
    old_uid, new_uid = int(s.plate_uid[old]), int(s.plate_uid[new])
    other_role = 'downgoing_plate_uid' if role == 'overriding_plate_uid' else 'overriding_plate_uid'
    mapping = {}
    for row in list(s.trench_systems):
        if row[role] != old_uid or row['phase'] in ('shutdown', 'joined'):
            continue
        edges = np.flatnonzero(s.trench_id == row['id'])
        if not len(edges):
            continue
        side = np.where(s.bp[edges] == old, s.ba[edges], s.bb[edges])
        moved = s.plate[side] == new
        if not np.any(moved):
            continue
        if row[other_role] == new_uid:
            # The receiving plate already holds the other side, so the moved
            # faces are no longer a plate boundary and a plate cannot subduct
            # beneath itself. A wholly absorbed trench ends; a partial one keeps
            # its identity on the faces that still separate the two plates.
            if np.all(moved):
                _shutdown(s, row, 'boundary_absorbed')
            continue
        target = row
        if not np.all(moved):
            target = deepcopy(row)
            if slab_memory.enabled(s):
                import slab_tether_history
                if slab_tether_history.FIELD in target:
                    # The daughter is not additional material. Partition the
                    # original channels instead of accepting their cloned load.
                    target[slab_tether_history.FIELD]=[]
                import slab_tether_local as local
                shares=local.transfer_fractions(row,s,edges,moved) if local.enabled(row) else None
                slab_memory.partition(row,target,float(s.bl[edges[moved]].sum()/s.bl[edges].sum()),channel_fractions=shares,
                                      split_points=(s.bmid[edges[moved]],s.bmid[edges[~moved]]))
            target.update(id=int(s.next_trench_id), born_myr=float(s.t), history=[],
                          phase_serial=0, parent_trench_id=row['id'],
                          episodes=[dict(start_myr=float(s.t), end_myr=None,
                                         inherited_trench_id=row['id'])])
            s.next_trench_id += 1
            s.trench_systems.append(target)
            _relocate(s, row, edges[~moved])
            slab_memory.refresh_line_load(row)
        target.update({role: new_uid, 'plate_uids': sorted([target[other_role], new_uid])})
        _relocate(s, target, edges[moved])
        slab_memory.refresh_line_load(target)
        _event(s, target, 'transferred', reason)
        mapping[row['id']] = target['id']
    return mapping


def downgoing_weight(s, p, cells=None, *, over=None):
    """Legacy local mature-trench suppression for overlapping supports.

    New versioned native transport uses finite polygon capture in
    native_subduction.removal instead; this historical radius operator remains
    unchanged for raster and restored unversioned native states.

    Use ``transported[p] *= 1-compression*downgoing_weight(s,p)`` instead of a
    plate-pair polarity lookup. Only real current convergent corridors and
    their two present owners qualify; remote parts of the same pair do not.
    """
    cells = np.arange(s.n, dtype=np.int64) if cells is None else np.asarray(cells, np.int64)
    if not hasattr(s, 'trench_systems'):
        return (s.polarity[p, s.plate[cells]] == p).astype(float)
    result = np.zeros(s.n, float)
    maturity = weights(s)
    edges = np.flatnonzero(normal_partition.subduction(s) & (s.down == p) & (maturity > 0))
    if over is not None:
        edges = edges[(s.bp[edges] == over) | (s.bq[edges] == over)]
    if not len(edges):
        return result[cells]
    radius = max(90., 1.1*_spacing(s))
    if isinstance(getattr(s, 'native_mesh', None), dict):
        # The computational surface owns these face centres. A spherical cap
        # lookup never samples the display raster or assumes four neighbours.
        # Both axes are chunked so refined meshes and long trenches do not
        # create an all-face by all-boundary temporary matrix.
        cosine = math.cos(min(radius/RADIUS_KM, math.pi))
        candidates = np.unique(cells)
        for start in range(0, len(edges), 128):
            local = edges[start:start+128]
            other = np.where(s.bp[local] == p, s.bq[local], s.bp[local])
            for begin in range(0, len(candidates), 4096):
                chosen = candidates[begin:begin+4096]
                near = s.xyz[chosen] @ s.bmid[local].T >= cosine-2e-14
                near &= ((s.plate[chosen, None] == p) |
                         (s.plate[chosen, None] == other[None, :]))
                near &= (s.crust[chosen, None] == 0)
                result[chosen] = np.maximum(result[chosen],
                    np.max(np.where(near, maturity[local][None, :], 0.), axis=1))
        return result[cells]
    for start in range(0, len(edges), 256):
        local = edges[start:start+256]
        segment, candidates = candidate_cells(s.bmid[local], radius, s.w, s.h)
        e = local[segment]
        other = np.where(s.bp[e] == p, s.bq[e], s.bp[e])
        valid = ((s.plate[candidates] == p) | (s.plate[candidates] == other)) & (s.crust[candidates] == 0)
        np.maximum.at(result, candidates[valid], maturity[e[valid]])
    return result[cells]


def downgoing_mask(s, p, cells=None):
    return downgoing_weight(s, p, cells) > 0


def retreat_weight(s, down, over, cells):
    """Associate trailing fractional forearc support with its local trench.

    Bilinear advection leaves a low-amplitude strip behind a retreating front.
    Cutting that strip to one cell silently reduces the prescribed retreat.
    Here each candidate belongs to its nearest *current* pair boundary; a
    transform, ridge, uninitiated or differently dipping trench blocks use of a
    mature system farther away. The 3,000-km cap bounds old diffuse residuals;
    actual support gradients and local convergence still determine all motion.
    """
    cells = np.asarray(cells, np.int64)
    if not hasattr(s, 'trench_systems'):
        return np.ones(len(cells))
    edges = np.flatnonzero(((s.bp == down) & (s.bq == over)) |
                          ((s.bp == over) & (s.bq == down)))
    result = np.zeros(len(cells))
    if not len(edges) or not len(cells):
        return result
    eligible = normal_partition.subduction(s)[edges] & (s.down[edges] == down)
    edge_weight = (slab_pull_weights(s)[edges] if slab_memory.enabled(s) else weights(s)[edges])*eligible
    # Matrix blocks stay bounded even on a detailed or fragmented boundary.
    for start in range(0, len(cells), 512):
        points = s.xyz[cells[start:start+512]]
        best = np.full(len(points), -np.inf)
        assigned = np.zeros(len(points))
        for section in range(0, len(edges), 1024):
            local = edges[section:section+1024]
            dots = points @ s.bmid[local].T
            index = np.argmax(dots, axis=1)
            value = dots[np.arange(len(points)), index]
            take = value > best
            assigned[take] = edge_weight[section+index[take]]
            best[take] = value[take]
        assigned[best < math.cos(max(3000., 4.*_spacing(s))/RADIUS_KM)] = 0.
        result[start:start+512] = assigned
    return result
