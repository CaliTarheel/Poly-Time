"""Arc magma as a capped submarine ridge with emergent centres (the "ribbon").

Under this policy waiting and newly supplied arc magma is not offered to one
compact footprint at its source point. Each origin feeds the arc crust of its
own plate lying along the stretch of arc front it was erupted for: its recorded
trench trace carried to the source point, with a finite half-width. That crust
thickens in place, lowest surface first, by a bounded amount per Myr.

How high a face may be built depends on how far inside its patch it lies.
Within the rim width of an open edge it may rise to the ridge-top depth only.
Further in, the limit climbs at the constructive slope to a summit cap. Every
patch therefore keeps a submarine rim, neighbouring patches stay separated by
submarine ridge, and land appears as separate islands, not one strip.

Magma the existing crust can never hold founds new ground: a flat
minimum-column foundation in free water of the same plate inside the origin's
band, admitted by the same exact footprint test as every juvenile birth. What
still does not fit stays pending at its origin, exactly as before.

Thickening is intrusive growth of established crust, so it is limited by these
caps and not by the constructive slope rule that governs a new volcanic pile.
All of the numbers are explicit, uncalibrated model choices. The policy is
future-only: a state without the version marker keeps the previous path.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real
import numpy as np

import crust_inventory
import crustal_structure as columns

VERSION = 1
SOURCE_COLUMN_KM = 25.
RADIUS_KM = 6371.
# Share of the largest footprint the minimum column can fund. The remainder keeps every stored
# column above the mechanical floor after the surface normalizes the newborn vertices.
FOOTPRINT_FILL = .999
# Footprint area of the compact eight-ray foundation per squared radius of its farthest vertex.
FOOTPRINT_AREA_PER_RADIUS2 = 2.6
LATTICE_KM = 10.
POLICY = dict(version=VERSION, source_column_km=SOURCE_COLUMN_KM, ridge_top_m=-1000., summit_m=2500.,
              constructive_slope_deg=20., rim_km=5., band_km=50., maximum_column_km=75.,
              maximum_thickening_km_per_myr=1., foundation_column_km=8.,
              foundation_minimum_radius_km=11., foundation_maximum_radius_km=25.,
              foundation_clearance_km=2., foundation_births_per_myr=8, foundation_attempts_per_myr=16,
              routing='same_owner_arc_crust_within_band_of_origin_arc_front_span',
              order='oldest_origin_first_lowest_surface_first',
              widening='flat_foundation_in_free_owner_water_within_band',
              calibration='Uncalibrated model parameters')
STATE_COUNTERS = ('transactions', 'foundations')
STATE_TOTALS = ('thickened_source_area_km2', 'foundation_source_area_km2')


def _integer(value, minimum=None):
    return (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and (minimum is None or value >= minimum))


def _scalar(value, minimum=None):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and (minimum is None or value >= minimum))


def _unit(value):
    value = np.asarray(value, float)
    return value/np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-300)


def version(s):
    value = getattr(s, 'native_arc_ribbon_version', 0)
    if not _integer(value) or value not in (0, VERSION):
        raise ValueError('Unsupported arc ribbon policy.')
    if value and any(getattr(s, name, 0) != 1 for name in (
            'native_arc_emplacement_version', 'native_arc_birth_profile_version',
            'native_arc_deposition_version', 'native_arc_footprint_version', 'native_arc_point_version')):
        raise ValueError('The arc ribbon requires geographic sources, physical profiles and source provenance.')
    return int(value)


def activate(s):
    """Mark a state for the ribbon from its current epoch onward; nothing else changes."""
    if getattr(s, 'native_arc_ribbon_version', 0):
        raise ValueError('The arc ribbon is already active.')
    s.native_arc_ribbon_version = VERSION
    version(s)
    s.arc_ribbon_state = dict(version=VERSION, activated_myr=float(s.t),
                              **{name: 0 for name in STATE_COUNTERS}, **{name: 0. for name in STATE_TOTALS})
    return s.arc_ribbon_state


def _state(s):
    state = getattr(s, 'arc_ribbon_state', None)
    if (not isinstance(state, dict) or not _integer(state.get('version')) or state['version'] != VERSION
            or not _scalar(state.get('activated_myr'), 0.)
            or any(not _integer(state.get(name), 0) for name in STATE_COUNTERS)
            or any(not _scalar(state.get(name), 0.) for name in STATE_TOTALS)):
        raise ValueError('An active arc ribbon requires its recorded activation state.')
    return state


def snapshot_fields(s):
    if not version(s):
        return {}
    return dict(arc_ribbon_version=VERSION, arc_ribbon_policy=dict(POLICY), arc_ribbon_state=deepcopy(_state(s)))


def validate_frame(frame):
    """Check the saved policy, activation record and the last transaction's source closure."""
    tag = frame.get('arc_ribbon_version', 0)
    if not _integer(tag) or tag not in (0, VERSION):
        raise ValueError('Unsupported saved arc ribbon policy.')
    diagnostics = frame.get('arc_material_diagnostics', {})
    report = diagnostics.get('ribbon') if isinstance(diagnostics, dict) else None
    if not tag:
        if report is not None or any(key in frame for key in ('arc_ribbon_policy', 'arc_ribbon_state')):
            raise ValueError('Saved arc ribbon records require their explicit policy version.')
        return 0
    if any(not _integer(frame.get(key)) or frame[key] != 1 for key in (
            'arc_emplacement_version', 'arc_birth_profile_version', 'arc_deposition_version',
            'arc_footprint_version', 'arc_point_version')):
        raise ValueError('A saved arc ribbon requires geographic sources, physical profiles and provenance.')
    policy = frame.get('arc_ribbon_policy')
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        raise ValueError('A saved arc ribbon requires complete policy metadata.')
    for key, expected in POLICY.items():
        actual = policy[key]
        if (type(actual) is not type(expected) and not (isinstance(expected, float) and _scalar(actual))
                or actual != expected):
            raise ValueError('Invalid saved arc ribbon policy: '+key)
    state = frame.get('arc_ribbon_state')
    epoch = frame.get('time_myr')
    if (not isinstance(state, dict) or not _integer(state.get('version')) or state['version'] != VERSION
            or not _scalar(state.get('activated_myr'), 0.) or not _scalar(epoch, 0.)
            or state['activated_myr'] > epoch+1e-10
            or any(not _integer(state.get(name), 0) for name in STATE_COUNTERS)
            or any(not _scalar(state.get(name), 0.) for name in STATE_TOTALS)):
        raise ValueError('A saved arc ribbon requires its activation record.')
    if report is None:
        if state['transactions']:
            raise ValueError('An arc ribbon that has run requires its last transaction ledger.')
        return int(tag)
    if not isinstance(report, dict) or not _integer(report.get('version')) or report['version'] != VERSION:
        raise ValueError('Invalid saved arc ribbon transaction.')

    def near(actual, expected):
        return abs(actual-expected) <= max(1e-7, abs(expected)*1e-11)

    areas = {}
    for key in ('incoming_area_km2', 'previous_pending_area_km2', 'thickened_source_area_km2',
                'foundation_source_area_km2', 'placed_area_km2', 'pending_area_km2',
                'thickened_volume_km3', 'foundation_volume_km3', 'placed_volume_km3',
                'maximum_thickening_km', 'interval_myr'):
        if not _scalar(report.get(key), 0.):
            raise ValueError('Invalid arc ribbon scalar: '+key)
        areas[key] = float(report[key])
    for key in ('sources', 'thickened_faces', 'foundations_born', 'foundation_attempts'):
        if not _integer(report.get(key), 0):
            raise ValueError('Invalid arc ribbon count: '+key)
    placed = areas['thickened_source_area_km2']+areas['foundation_source_area_km2']
    if (not near(areas['placed_area_km2'], placed)
            or not near(areas['incoming_area_km2']+areas['previous_pending_area_km2'],
                        areas['placed_area_km2']+areas['pending_area_km2'])
            or not near(areas['thickened_volume_km3'], SOURCE_COLUMN_KM*areas['thickened_source_area_km2'])
            or not near(areas['foundation_volume_km3'], SOURCE_COLUMN_KM*areas['foundation_source_area_km2'])
            or not near(areas['placed_volume_km3'], SOURCE_COLUMN_KM*areas['placed_area_km2'])):
        raise ValueError('The arc ribbon source ledger does not close.')
    limit = POLICY['maximum_thickening_km_per_myr']*areas['interval_myr']
    if areas['maximum_thickening_km'] > limit*(1.+1e-9)+1e-9:
        raise ValueError('Arc ribbon thickening exceeded its rate limit.')
    if report['foundations_born'] > report['foundation_attempts']:
        raise ValueError('An arc ribbon foundation requires an admission attempt.')
    inventory = frame.get('arc_pending_source')
    if (not isinstance(inventory, dict) or not _scalar(inventory.get('total_area_km2'), 0.)
            or not near(float(inventory['total_area_km2']), areas['pending_area_km2'])):
        raise ValueError('The arc ribbon pending area disagrees with the saved source inventory.')
    return int(tag)


class Arcs:
    """Minor arcs prepared for great-circle distance queries.

    `nearest` is the distance from each point to the closest arc. With a cutoff the arcs are
    binned on a Cartesian grid and only nearby ones are examined: a point farther than the cutoff
    from every arc reports infinity, and every finite answer is the exact nearest distance.
    """

    def __init__(self, start, end, cutoff_km=None):
        self.start = np.asarray(start, float).reshape(-1, 3)
        self.end = np.asarray(end, float).reshape(-1, 3)
        normal = np.cross(self.start, self.end)
        length = np.linalg.norm(normal, axis=1)
        self.proper = length > 1e-14
        self.normal = normal/np.where(self.proper, length, 1.)[:, None]
        self.forward = np.cross(self.normal, self.start)
        self.backward = np.cross(self.normal, self.end)
        self.cutoff = cutoff_km
        self.bins = None
        if cutoff_km is not None and len(self.start):
            self.cell = max(float(cutoff_km), 20.)/RADIUS_KM
            # A minor arc leaves the box of its endpoints by no more than its sagitta.
            sagitta = 1.-np.sqrt(np.clip(.5*(1.+np.sum(self.start*self.end, axis=1)), 0., 1.))
            reach = (float(cutoff_km)/RADIUS_KM+sagitta)[:, None]
            low = np.floor((np.minimum(self.start, self.end)-reach)/self.cell).astype(np.int64)
            high = np.floor((np.maximum(self.start, self.end)+reach)/self.cell).astype(np.int64)
            bins = {}
            for index, (a, b) in enumerate(zip(low.tolist(), high.tolist())):
                for x in range(a[0], b[0]+1):
                    for y in range(a[1], b[1]+1):
                        for z in range(a[2], b[2]+1):
                            bins.setdefault((x, y, z), []).append(index)
            self.bins = {key: np.asarray(value, np.int64) for key, value in bins.items()}

    def _distance(self, points, take=None):
        """Angles (points, arcs) in radians for all arcs or the selected ones."""
        pick = (lambda value: value) if take is None else (lambda value: value[take])
        ends = np.arccos(np.clip(np.maximum(np.einsum('pi,ai->pa', points, pick(self.start)),
                                            np.einsum('pi,ai->pa', points, pick(self.end))), -1., 1.))
        off = np.abs(np.arcsin(np.clip(np.einsum('pi,ai->pa', points, pick(self.normal)), -1., 1.)))
        between = (pick(self.proper)[None, :] & (np.einsum('pi,ai->pa', points, pick(self.forward)) >= 0.)
                   & (np.einsum('pi,ai->pa', points, pick(self.backward)) <= 0.))
        return np.where(between, off, ends)

    def each(self, point):
        """Distance in km from one point to every arc."""
        return self._distance(np.asarray(point, float)[None])[0]*RADIUS_KM

    def nearest(self, points, *, chunk=2048):
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.full(len(points), np.inf)
        if not len(points) or not len(self.start):
            return result
        if self.bins is None:
            for first in range(0, len(points), chunk):
                result[first:first+chunk] = self._distance(points[first:first+chunk]).min(axis=1)*RADIUS_KM
            return result
        cells = np.floor(points/self.cell).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(inverse.ravel(), kind='stable')
        bounds = np.searchsorted(inverse.ravel()[order], np.arange(len(unique)+1))
        for index, key in enumerate(map(tuple, unique.tolist())):
            arcs = self.bins.get(key)
            if arcs is None:
                continue
            members = order[bounds[index]:bounds[index+1]]
            distance = self._distance(points[members], arcs).min(axis=1)*RADIUS_KM
            result[members] = np.where(distance <= self.cutoff, distance, np.inf)
        return result


def nearest_arc_km(points, start, end):
    """Great-circle distance from each unit point to the nearest of the minor arcs start->end."""
    return Arcs(start, end).nearest(points)


def spans(points, provenance):
    """Each origin's stretch of arc front: its trench trace carried rigidly to its source point.

    Both the trace and the point already move with the origin's host plate. An origin without a
    recorded trace keeps the degenerate span of its own point.
    """
    result = []
    for point, row in zip(np.asarray(points, float), provenance):
        trace = np.asarray(row.get('trace_xyz', []), float).reshape(-1, 2, 3)
        if not len(trace):
            result.append((point[None].copy(), point[None].copy()))
            continue
        middle = _unit(trace.reshape(-1, 3).sum(axis=0))
        axis = np.cross(middle, point)
        sine = float(np.linalg.norm(axis))
        cosine = float(middle@point)
        if sine > 1e-12:
            axis = axis/sine
            flat = trace.reshape(-1, 3)
            trace = (flat*cosine+np.cross(axis, flat)*sine
                     + axis*(flat@axis)[:, None]*(1.-cosine)).reshape(-1, 2, 3)
        result.append((_unit(trace[:, 0]), _unit(trace[:, 1])))
    return result


def cap_m(edge_km):
    """Highest surface a point this far inside its patch may be built to."""
    grade = 1000.*np.tan(np.deg2rad(POLICY['constructive_slope_deg']))
    return np.minimum(POLICY['summit_m'],
                      POLICY['ridge_top_m']+grade*np.maximum(0., np.asarray(edge_km, float)-POLICY['rim_km']))


def level_fill(base, limit, weight, amount):
    """Raise one common level over columns standing at `base`, none above its `limit`.

    Returns the height added to each column. `weight` converts a column's added height to the
    conserved quantity `amount`. The lowest columns are raised first; the result never exceeds
    `amount`, and equals it whenever the columns have that much room.
    """
    base = np.asarray(base, float)
    room = np.maximum(np.asarray(limit, float)-base, 0.)
    weight = np.asarray(weight, float)
    capacity = float(room@weight)
    if amount <= 0. or capacity <= 0.:
        return np.zeros(len(base))
    if amount >= capacity:
        return room
    low, high = float(base.min()), float((base+room).max())
    for _ in range(64):
        middle = .5*(low+high)
        if float(np.clip(middle-base, 0., room)@weight) < amount:
            low = middle
        else:
            high = middle
    added = np.clip(low-base, 0., room)
    # Finish exactly on the linear piece above the bracket's lower end.
    rising = (base < high) & (added < room)
    slope = float(weight[rising].sum())
    if slope > 0.:
        added = np.clip(low+(amount-float(added@weight))/slope-base, 0., room)
    total = float(added@weight)
    if total > amount:
        added *= amount/total
    return added


def _open_edges(surface):
    """Mask (face, edge) of the edges used by exactly one material face."""
    faces = np.asarray(surface['faces'])
    first, second = faces.astype(np.int64), faces[:, [1, 2, 0]].astype(np.int64)
    key = (np.minimum(first, second) << 32 | np.maximum(first, second)).ravel()
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return (counts[inverse] == 1).reshape(-1, 3), key.reshape(-1, 3)


def eligible_faces(s):
    """Exposed juvenile arc faces of active plates that take part in no collision stack."""
    take = (np.asarray(s.kind) == 3) & (np.asarray(s.parcel_arc_id) > 0)
    owners = np.asarray(s.parcel_plate)
    if hasattr(s, 'active'):
        take &= np.asarray(s.active, bool)[owners]
    if hasattr(s, 'parcel_exposed_fraction'):
        take &= np.asarray(s.parcel_exposed_fraction, float) >= 1.-1e-12
    contacts = getattr(s, 'collision_contacts', [])
    if contacts and hasattr(s, 'parcel_collision_sheet'):
        stacked = {int(row[name]) for row in contacts for name in ('top_sheet', 'under_sheet')}
        take &= ~np.isin(np.asarray(s.parcel_collision_sheet), np.fromiter(stacked, np.int64, len(stacked)))
    return np.flatnonzero(take)


def belt(s, interval_myr):
    """Geometry and room of every face the ribbon may thicken; nothing is changed."""
    surface = s.material_surface
    faces = eligible_faces(s)
    vertices = np.asarray(surface['vertices'], float)
    corners = np.asarray(surface['faces'])
    open_edge, key = _open_edges(surface)
    following = corners[:, [1, 2, 0]]
    start, end = vertices[corners[open_edge]], vertices[following[open_edge]]
    # The cap is averaged over the three edge midpoints, so a face that spans the rise from
    # rim to summit takes its mean and not the value at one point.
    unique, first, inverse = np.unique(key[faces].ravel(), return_index=True, return_inverse=True)
    flat_face, flat_edge = np.divmod(first, 3)
    a = vertices[corners[faces][flat_face, flat_edge]]
    b = vertices[following[faces][flat_face, flat_edge]]
    # Beyond the rise from rim to summit the cap no longer depends on the distance, and no
    # foundation is wider than its largest radius, so one cutoff serves both questions.
    grade = 1000.*np.tan(np.deg2rad(POLICY['constructive_slope_deg']))
    cutoff = max(POLICY['rim_km']+(POLICY['summit_m']-POLICY['ridge_top_m'])/grade,
                 POLICY['foundation_maximum_radius_km']+POLICY['foundation_clearance_km'])+1.
    coast = Arcs(start, end, cutoff)
    depth = coast.nearest(_unit(a+b))
    cap = cap_m(depth)[inverse].reshape(-1, 3).mean(axis=1)
    fields = {name: np.asarray(value)[faces] for name, value in s.structure.items()}
    elevation = columns.elevation(fields)
    foreland = np.asarray(fields['foreland_m'], float)
    air = columns._air_height(elevation+foreland)
    room = np.maximum(columns._air_height(cap+foreland)-air, 0.)/columns.AIRY_M_PER_KM
    thickness = np.asarray(fields['thickness_km'], float)
    ceiling = np.maximum(POLICY['maximum_column_km']-thickness, 0.)
    patches = np.asarray(s.parcel_patch)[faces]
    order = np.argsort(patches, kind='stable')
    trace = np.flatnonzero(np.isin(s.trace_patch, patches))
    trace_face = order[np.searchsorted(patches[order], np.asarray(s.trace_patch)[trace])]
    if len(trace):
        trace_room = np.maximum(POLICY['maximum_column_km']
                                - np.asarray(s.trace_structure['thickness_km'], float)[trace], 0.)
        np.minimum.at(ceiling, trace_face, trace_room)
    room = np.minimum(room, ceiling)
    step = np.minimum(room, POLICY['maximum_thickening_km_per_myr']*max(float(interval_myr), 0.))
    return dict(faces=faces, owner=np.asarray(s.parcel_plate)[faces], area=np.asarray(surface['area_km2'], float)[faces],
                centre=_unit(vertices[corners[faces]].sum(axis=1)), air=air, room_km=room, step_km=step,
                elevation=elevation, cap_m=cap, trace=trace, trace_face=trace_face, coast=coast)


def _reach(ground, order, points, owners, arcs):
    """Belt faces of each origin's own plate within the band of its arc-front span."""
    band = POLICY['band_km']
    by_owner = {}
    for owner in np.unique(owners[order]) if len(order) else ():
        by_owner[int(owner)] = np.flatnonzero(ground['owner'] == owner)
    result = {}
    for i in order:
        local = by_owner[int(owners[i])]
        start, end = arcs[i]
        if not len(local):
            result[int(i)] = local
            continue
        ends = np.vstack((start, end))
        middle = _unit(ends.sum(axis=0))
        radius = float(np.arccos(np.clip(ends@middle, -1., 1.)).max())+(band+1.)/RADIUS_KM
        near = local[ground['centre'][local]@middle >= np.cos(min(radius, np.pi))]
        result[int(i)] = near[nearest_arc_km(ground['centre'][near], start, end) <= band]
    return result


def _fill(ground, order, volumes, reach, room_km):
    """Sequential lowest-first filling. Returns volume per face and per origin."""
    air = ground['air'].copy()
    room = np.asarray(room_km, float).copy()
    weight = ground['area']/columns.AIRY_M_PER_KM
    added = np.zeros(len(air))
    placed = np.zeros(len(volumes))
    leading = np.full(len(volumes), -1, np.int64)
    for i in order:
        take = reach[int(i)]
        if not len(take) or volumes[i] <= 0.:
            continue
        take = take[room[take] > 0.]
        if not len(take):
            continue
        rise = level_fill(air[take], air[take]+room[take]*columns.AIRY_M_PER_KM, weight[take], volumes[i])
        volume = rise*weight[take]
        total = float(volume.sum())
        if total <= 0.:
            continue
        air[take] += rise
        room[take] = np.maximum(room[take]-rise/columns.AIRY_M_PER_KM, 0.)
        added[take] += volume
        placed[i] = total
        leading[i] = take[int(np.argmax(volume))]
    return added, placed, leading


def _thicken(s, faces, thickness_km, trace, trace_face):
    """Add juvenile column to existing faces and their enduring points; geometry is untouched."""
    fields = {name: np.asarray(value)[faces].copy() for name, value in s.structure.items()}
    before = columns.elevation(fields)
    fields['thickness_km'] = fields['thickness_km']+thickness_km
    volume = thickness_km*fields['area_factor']
    crust_inventory.add(fields, volume)
    uplift = columns.elevation(fields)-before
    fields['added_volume_km_per_reference_km2'] = fields['added_volume_km_per_reference_km2']+volume
    fields['magmatic_uplift_m'] = fields['magmatic_uplift_m']+uplift
    columns._state(fields)
    points = {name: np.asarray(value)[trace].copy() for name, value in s.trace_structure.items()}
    trace_uplift = np.zeros(len(trace))
    if len(trace):
        increment = thickness_km[trace_face]
        trace_before = columns.elevation(points)
        points['thickness_km'] = points['thickness_km']+increment
        trace_volume = increment*points['area_factor']
        crust_inventory.add(points, trace_volume)
        trace_uplift = columns.elevation(points)-trace_before
        points['added_volume_km_per_reference_km2'] = points['added_volume_km_per_reference_km2']+trace_volume
        points['magmatic_uplift_m'] = points['magmatic_uplift_m']+trace_uplift
        columns._state(points)
    for name, value in fields.items():
        s.structure[name][faces] = value
    s.relief[faces] += uplift
    for name, value in points.items():
        s.trace_structure[name][trace] = value
    s.trace_relief_m[trace] += trace_uplift
    s.trace_adjustment_m[trace] += trace_uplift
    return uplift


def _candidates(order, owners, arcs):
    """One trial site per lattice cell of every overflowing origin's band, in origin order.

    Each site keeps how far it lies from the nearest arc-front span that proposed it.
    """
    band, step = POLICY['band_km'], LATTICE_KM
    offsets = np.arange(-int(band//step), int(band//step)+1)*step/RADIUS_KM
    sites, site_owner, site_offset = [], [], []
    for i in order:
        start, end = arcs[i]
        for a, b in zip(start, end):
            angle = float(np.arccos(np.clip(a@b, -1., 1.)))
            count = max(int(np.ceil(angle*RADIUS_KM/step)), 1)
            if angle > 1e-12:
                t = np.arange(count+1)/count
                along = _unit((np.sin((1.-t)*angle)[:, None]*a+np.sin(t*angle)[:, None]*b)/np.sin(angle))
                tangent = _unit(b-a*(a@b))
            else:
                along = a[None]
                tangent = _unit(np.cross(a, [0., 0., 1.] if abs(a[2]) < .9 else [0., 1., 0.]))
            # Across the arc: a fixed great-circle normal serves the whole short segment.
            across = _unit(np.cross(a, tangent))
            site = _unit(along[:, None, :]*np.cos(offsets)[None, :, None]
                         + across[None, None, :]*np.sin(offsets)[None, :, None]).reshape(-1, 3)
            sites.append(site)
            site_owner.append(np.full(len(site), owners[i], np.int64))
            site_offset.append(np.tile(np.abs(offsets)*RADIUS_KM, len(along)))
    if not sites:
        return np.empty((0, 3)), np.empty(0, np.int64), np.empty(0)
    sites, site_owner, site_offset = np.vstack(sites), np.concatenate(site_owner), np.concatenate(site_offset)
    cell = np.floor(sites*(RADIUS_KM/step)).astype(np.int64)
    _, inverse = np.unique(np.column_stack((site_owner, cell)), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    nearest = np.full(int(inverse.max())+1, np.inf)
    np.minimum.at(nearest, inverse, site_offset)
    # Of the proposals for one cell keep the first that lies closest to its arc front.
    closest = np.flatnonzero(site_offset == nearest[inverse])
    first = np.full(len(nearest), len(sites), np.int64)
    np.minimum.at(first, inverse[closest], closest)
    first.sort()
    return sites[first], site_owner[first], site_offset[first]


def _found(s, ground, order, points, owners, provenance, arcs, overflow, interval_myr):
    """Found flat foundations in free owner water for magma the belt can never hold."""
    import arc_birth_profile
    import arc_emplacement_geometry as emplacement
    import arc_point_nucleation
    import arc_source_geometry
    import mesh_geometry
    import native_arc_material
    import native_boundary_geometry
    minimum_radius, maximum_radius = POLICY['foundation_minimum_radius_km'], POLICY['foundation_maximum_radius_km']
    column = POLICY['foundation_column_km']

    def source_area(radius):
        return FOOTPRINT_AREA_PER_RADIUS2*radius*radius*column/SOURCE_COLUMN_KM

    spent = np.zeros(len(points))
    born = []
    result = dict(spent=spent, born=born, attempts=0, candidates=0, refused={})
    births = int(np.floor(POLICY['foundation_births_per_myr']*interval_myr+1e-9))
    attempts = int(np.floor(POLICY['foundation_attempts_per_myr']*interval_myr+1e-9))
    funders = np.array([i for i in order if overflow[i] > 0.], np.int64)
    if births < 1 or not len(funders) or float(overflow[funders].sum()) < source_area(minimum_radius):
        return result
    sites, site_owner, site_offset = _candidates(funders, owners, arcs)
    if not len(sites):
        return result
    surface = s.material_surface
    locator = mesh_geometry.build_locator(surface['vertices'], surface['faces'])
    covered, _ = mesh_geometry.locate_points(sites, locator)
    placement = emplacement.prepare(s)
    keep = (covered < 0) & (native_boundary_geometry.sample_owners(sites, placement['owner']) == site_owner)
    sites, site_owner, site_offset = sites[keep], site_owner[keep], site_offset[keep]
    clearance = ground['coast'].nearest(sites)
    radius = np.minimum(maximum_radius, clearance-POLICY['foundation_clearance_km'])
    keep = radius >= minimum_radius
    sites, site_owner, site_offset, radius = sites[keep], site_owner[keep], site_offset[keep], radius[keep]
    result['candidates'] = int(len(sites))
    if not len(sites):
        return result
    left = overflow.copy()
    # Every funder's span as one flat list, so a site finds its funders in one pass.
    funder_spans = Arcs(np.vstack([arcs[i][0] for i in funders]), np.vstack([arcs[i][1] for i in funders]))
    segment_funder = np.concatenate([np.full(len(arcs[i][0]), k, np.int64) for k, i in enumerate(funders)])

    def funding(site, owner):
        nearest = np.full(len(funders), np.inf)
        np.minimum.at(nearest, segment_funder, funder_spans.each(site))
        return funders[(nearest <= POLICY['band_km']) & (owners[funders] == owner) & (left[funders] > 0.)]

    bearing = np.arange(8)*np.pi/4.

    def owned_radius(site, owner, allowed):
        """Largest of a few radii whose rim lies wholly in the owner's own water; zero if none."""
        east = _unit(np.cross([0., 0., 1.] if abs(site[2]) < .9 else [0., 1., 0.], site))
        north = np.cross(site, east)
        around = np.cos(bearing)[:, None]*east+np.sin(bearing)[:, None]*north
        for trial in sorted({allowed, .5*(allowed+minimum_radius), minimum_radius}, reverse=True):
            angle = (trial+POLICY['foundation_clearance_km'])/RADIUS_KM
            rim = _unit(site*np.cos(angle)+around*np.sin(angle))
            if np.all(native_boundary_geometry.sample_owners(rim, placement['owner']) == owner):
                return trial
        return 0.

    def refuse(reason):
        result['refused'][reason] = result['refused'].get(reason, 0)+1

    centres = np.empty((0, 3))
    reserved = np.empty(0)
    # Widest first, because a wide foundation holds far more than a narrow one; among equals the
    # one nearest its arc front, so that new ground widens the belt before it scatters.
    for index in np.lexsort((np.arange(len(sites)), site_offset, -np.floor(radius))):
        if len(born) >= births or result['attempts'] >= attempts:
            break
        site, owner, allowed = sites[index], int(site_owner[index]), float(radius[index])
        if len(centres):
            gap = RADIUS_KM*np.arccos(np.clip(centres@site, -1., 1.))-reserved-POLICY['foundation_clearance_km']
            allowed = min(allowed, float(gap.min()))
            if allowed < minimum_radius:
                continue
        allowed = owned_radius(site, owner, allowed)
        if allowed < minimum_radius:
            continue
        paying = funding(site, owner)
        request = min(float(left[paying].sum()), source_area(allowed))
        if request < source_area(minimum_radius):
            continue
        result['attempts'] += 1
        cell = int(s._indices(site[None])[0])
        strike = native_arc_material._strike(s, cell, owner, site)
        key = arc_point_nucleation.rank(s.config['seed'], [provenance[i]['origin_id'] for i in paying])
        basal = float(arc_source_geometry.basement(s, site[None])[0])

        def factory(amount):
            return arc_point_nucleation._patch(site, strike, amount*SOURCE_COLUMN_KM/column*FOOTPRINT_FILL, key)

        admission = emplacement.admit(placement, factory, request, owner,
                                      minimum_area_km2=source_area(minimum_radius))
        accepted = float(admission['accepted_area_km2'])
        if accepted <= 0.:
            last = admission['diagnostics']['last_examined_footprint'] or {}
            refuse(last.get('reason') or ('foreign_owner' if last.get('foreign_owner_vertices')
                   or last.get('foreign_owner_obstruction_km2', 0.) > last.get('admission_tolerance_km2', 0.)
                   else 'material' if last.get('material_obstruction_km2', 0.) > last.get('admission_tolerance_km2', 0.)
                   else 'exact_material' if last.get('exact_material_obstruction') is not None else 'other'))
            continue
        plan = admission['plan']
        if not arc_birth_profile.birth(plan, accepted, basal)['diagnostics']['admissible']:
            refuse('constructive_profile')
            continue
        first = len(s.mass)
        birth = native_arc_material._birth(s, cell, owner, accepted, position=site, basal_m=basal,
                                           fixed_source_volume=True, prepared_patch=plan)
        emplacement.commit(placement, surface, np.arange(first, len(s.mass)))
        share = np.zeros(len(points))
        remaining = accepted
        for i in paying:
            used = min(float(left[i]), remaining)
            share[i] = used
            remaining -= used
            if remaining <= 0.:
                break
        if remaining > accepted*1e-12+1e-12:
            raise ValueError('An arc ribbon foundation spent more magma than its funders hold.')
        left -= share
        np.maximum(left, 0., out=left)
        spent += share
        extent = float(RADIUS_KM*np.arccos(np.clip(_unit(plan['vertices'])@site, -1., 1.)).max())
        centres = np.vstack((centres, site[None]))
        reserved = np.r_[reserved, extent]
        born.append(dict(arc_id=int(birth['arc_id']), owner=owner, source_area_km2=accepted,
                         volume_km3=float(birth['added_volume_km3']), footprint_area_km2=float(plan['area_km2'].sum()),
                         radius_km=extent, share=share, new_faces=int(birth['new_faces'])))
    return result


def _book(s, provenance, points, thickened, leading, ground, foundations):
    """Record every origin's placed magma in the enduring source ledger."""
    ledger = getattr(s, 'native_arc_source_placements', [])
    earlier = {int(entry['source_provenance']['origin_id']): entry for entry in ledger
               if entry.get('ribbon') == VERSION and entry['mode'] == 'deposition'}
    for i in np.flatnonzero(thickened > 0.):
        origin = int(provenance[i]['origin_id'])
        entry = earlier.get(origin)
        if entry is not None:
            # One enduring record per origin: thickening usually spans many intervals.
            entry['area_km2'] = float(entry['area_km2']+thickened[i])
            entry['volume_km3'] = float(SOURCE_COLUMN_KM*entry['area_km2'])
            entry['placement_time_myr'] = float(s.t)
            continue
        face = int(ground['faces'][leading[i]])
        ledger.append(dict(placement_id=len(ledger)+1, arc_id=int(s.parcel_arc_id[face]),
            placement_time_myr=float(s.t), source_provenance=deepcopy(provenance[i]),
            geometry_xyz=[np.asarray(points[i], float).tolist()], area_km2=float(thickened[i]),
            volume_km3=float(SOURCE_COLUMN_KM*thickened[i]), mode='deposition',
            receiving_face_id=int(s.parcel_patch[face]), ribbon=VERSION))
    for row in foundations:
        for i in np.flatnonzero(row['share'] > 0.):
            ledger.append(dict(placement_id=len(ledger)+1, arc_id=row['arc_id'],
                placement_time_myr=float(s.t), source_provenance=deepcopy(provenance[i]),
                geometry_xyz=[np.asarray(points[i], float).tolist()], area_km2=float(row['share'][i]),
                volume_km3=float(SOURCE_COLUMN_KM*row['share'][i]), mode='birth', ribbon=VERSION))
    s.native_arc_source_placements = ledger


def transaction(s, cells, additions, *, positions=None, owners=None, source_provenance=None, dt=None):
    """The whole arc source transaction of one interval under the ribbon policy."""
    import arc_emplacement_geometry as emplacement
    import arc_source_cohorts as cohorts
    import arc_source_geometry as sources
    import native_arc_material
    state = _state(s)
    cells, additions = np.asarray(cells), np.asarray(additions, float)
    if (cells.ndim != 1 or (len(cells) and cells.dtype.kind not in 'iu') or additions.shape != cells.shape
            or np.any(cells < 0) or np.any(cells >= s.n)
            or not np.isfinite(additions).all() or np.any(additions < 0)):
        raise ValueError('Arc additions need valid native cells and finite nonnegative areas.')
    if positions is None:
        if owners is not None:
            raise ValueError('Explicit arc owners require actual source positions.')
        positions, owners = s.xyz[cells].copy(), s.plate[cells].copy()
    if owners is None:
        raise ValueError('Actual arc source positions require intended owners.')
    if dt is None or not _scalar(dt, 0.):
        raise ValueError('The arc ribbon requires the length of its interval.')
    interval = float(dt)
    new_points, new_owners = sources.validate_positions(positions, owners, len(cells), len(s.support))
    native_arc_material.ensure_fields(s)
    empty = dict(xyz=np.empty((0, 3)), owner=np.empty(0, np.int16), area=np.empty(0),
                 emplacement_version=1, source_column_km=SOURCE_COLUMN_KM, source_provenance=[])
    pending = getattr(s, 'native_arc_pending', empty)
    emplacement.validate_pending(pending)
    old_points, old_owners = sources.validate_positions(pending['xyz'], pending['owner'], len(pending['area']),
                                                        len(s.support))
    old_area = np.asarray(pending['area'], float)
    if len(old_area) and 'source_provenance' not in pending:
        raise ValueError('Pending magma without proven source provenance cannot enter the arc ribbon.')
    current = cohorts.normalize(s, source_provenance, len(new_points), owners=new_owners, areas=additions)
    provenance = cohorts.concat(current, pending.get('source_provenance', []))
    points = np.vstack((new_points, old_points))
    owner = np.r_[new_owners, old_owners].astype(np.int64)
    area = np.r_[additions, old_area]
    incoming, prior = float(additions.sum()), float(old_area.sum())
    usable = cohorts.host_mask(s, owner, provenance) & (area > 0.)
    order = sorted(map(int, np.flatnonzero(usable)), key=lambda i: int(provenance[i]['origin_id']))
    volume = area*SOURCE_COLUMN_KM
    arcs = spans(points, provenance)
    ground = belt(s, interval)
    reach = _reach(ground, order, points, owner, arcs)
    # What the belt can ever hold decides who founds new ground; this interval's bounded
    # thickening decides what is placed now.
    _, fits, _ = _fill(ground, order, volume, reach, ground['room_km'])
    added, placed, leading = _fill(ground, order, volume, reach, ground['step_km'])
    before_volume = float(s.material_surface['area_km2']@s.structure['thickness_km'])
    changed = np.flatnonzero(added > 0.)
    thickness = np.zeros(len(added))
    thickness[changed] = added[changed]/ground['area'][changed]
    if len(changed):
        take = np.isin(ground['trace_face'], changed)
        local = np.searchsorted(changed, ground['trace_face'][take])
        _thicken(s, ground['faces'][changed], thickness[changed], ground['trace'][take], local)
    thickened_volume = float(s.material_surface['area_km2']@s.structure['thickness_km'])-before_volume
    planned = float(added.sum())
    if not np.isclose(thickened_volume, planned, rtol=2e-11, atol=max(1e-7, abs(before_volume)*2e-15)):
        raise ValueError('Arc ribbon thickening failed its physical column-volume ledger.')
    thickened = placed/SOURCE_COLUMN_KM
    overflow = np.maximum(area-np.maximum(fits, placed)/SOURCE_COLUMN_KM, 0.)
    overflow[~usable] = 0.
    foundations = _found(s, ground, order, points, owner, provenance, arcs, overflow, interval)
    founded = foundations['spent']
    founded_volume = float(sum(row['volume_km3'] for row in foundations['born']))
    if not np.isclose(founded_volume, SOURCE_COLUMN_KM*float(founded.sum()), rtol=2e-11, atol=1e-7):
        raise ValueError('Arc ribbon foundations must use exactly their funded magma volume.')
    _book(s, provenance, points, thickened, leading, ground, foundations['born'])
    spent = thickened+founded
    left = area-spent
    left[(left < 0.) & (left > -np.maximum(area, 1.)*1e-12)] = 0.
    if np.any(left < 0.):
        raise ValueError('The arc ribbon spent more magma than an origin holds.')
    keep = np.flatnonzero(left > 0.)
    totals = s.process_totals
    totals['arc_added_km2'] += float(spent.sum())
    totals['arc_added_volume_km3'] = totals.get('arc_added_volume_km3', 0.)+SOURCE_COLUMN_KM*float(spent.sum())
    totals['arc_deposited_source_area_km2'] = totals.get('arc_deposited_source_area_km2', 0.)+float(thickened.sum())
    totals['arc_deposited_volume_km3'] = (totals.get('arc_deposited_volume_km3', 0.)
                                          + SOURCE_COLUMN_KM*float(thickened.sum()))
    state['transactions'] += 1
    state['foundations'] += len(foundations['born'])
    state['thickened_source_area_km2'] += float(thickened.sum())
    state['foundation_source_area_km2'] += float(founded.sum())
    # The earlier emplacement code closes the interval with nothing left to examine: it keeps its
    # ledgers, events and cache invalidation, and the ribbon's own account is recorded beside it.
    s.native_arc_pending = deepcopy(empty)
    report = native_arc_material.add_arc_crust(s, np.empty(0, np.int64), np.empty(0), positions=np.empty((0, 3)),
                                               owners=np.empty(0, np.int64), source_provenance=[], _ribbon=False)
    s.native_arc_pending = dict(xyz=points[keep].copy(), owner=owner[keep].astype(np.int16), area=left[keep].copy(),
                                emplacement_version=1, source_column_km=SOURCE_COLUMN_KM,
                                source_provenance=cohorts.subset(provenance, keep))
    waiting = float(left[keep].sum())
    report.update(pending_area_km2=waiting, pending_magma_volume_km3=waiting*SOURCE_COLUMN_KM,
                  patch_count=len(np.unique(s.parcel_arc_id[s.parcel_arc_id > 0])))
    report['emplacement_geometry']['unexamined_pending_area_km2'] = waiting
    placed_area = float(spent.sum())
    final = columns.elevation({name: np.asarray(value)[ground['faces']] for name, value in s.structure.items()})
    report['ribbon'] = dict(version=VERSION, interval_myr=interval, sources=len(order),
        sources_without_host=int(np.count_nonzero((area > 0.) & ~usable)),
        sources_reaching_arc_crust=int(sum(len(reach[i]) > 0 for i in order)),
        incoming_area_km2=incoming, previous_pending_area_km2=prior,
        belt_faces=int(len(ground['faces'])), belt_area_km2=float(ground['area'].sum()),
        belt_room_km3=float(ground['room_km']@ground['area']),
        fits_eventually_area_km2=float(fits.sum())/SOURCE_COLUMN_KM,
        overflow_area_km2=float(overflow.sum()),
        thickened_source_area_km2=float(thickened.sum()), thickened_volume_km3=SOURCE_COLUMN_KM*float(thickened.sum()),
        thickened_faces=int(len(changed)), maximum_thickening_km=float(thickness.max(initial=0.)),
        physical_volume_residual_km3=thickened_volume-planned,
        foundation_source_area_km2=float(founded.sum()), foundation_volume_km3=SOURCE_COLUMN_KM*float(founded.sum()),
        foundations_born=len(foundations['born']), foundation_attempts=int(foundations['attempts']),
        foundation_candidates=int(foundations['candidates']),
        foundation_refusals=dict(sorted(foundations['refused'].items())),
        foundations=[{name: row[name] for name in ('arc_id', 'owner', 'source_area_km2', 'volume_km3',
                                                   'footprint_area_km2', 'radius_km', 'new_faces')}
                     for row in foundations['born']],
        placed_area_km2=placed_area, placed_volume_km3=SOURCE_COLUMN_KM*placed_area, pending_area_km2=waiting,
        source_area_residual_km2=incoming+prior-placed_area-waiting,
        origins_used_up=int(np.count_nonzero((area > 0.) & (left <= 0.))), origins_waiting=int(len(keep)),
        belt_above_sea_level_km2=float(ground['area'][final > 0.].sum()),
        belt_mean_surface_m=float(np.average(final, weights=ground['area'])) if len(final) else 0.)
    if abs(report['ribbon']['source_area_residual_km2']) > max(1e-9, (incoming+prior)*1e-12):
        raise ValueError('The arc ribbon failed its source area ledger.')
    s.arc_material_diagnostics = report
    return report
