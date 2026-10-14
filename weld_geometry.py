"""Physical free-edge integration for the reduced, opening-only suture weld.

Internal edges of either material sheet carry no weld traction. Adjacent
collinear clipped pieces are merged before integration, making its resolution
independent of the material triangulation. This remains a plan-view cohesive
opening law; it is not an area-based basal shear or 3-D interface model.
"""
import math
import numpy as np

import collision_fronts as fronts

VERSION = 1


def upgrade(s):
    """Explicit coordinate-law boundary; do not change material or weld history."""
    from copy import deepcopy
    import collision_contacts
    if not collision_contacts.weld_enabled(s):
        raise ValueError('Local weld coordinates require the persistent suture model.')
    version = getattr(s, 'suture_weld_coordinate_version', 0)
    if isinstance(version, (bool, np.bool_)) or version not in (0, VERSION):
        raise ValueError('Unsupported local weld coordinate version.')
    if version == VERSION:
        return deepcopy(getattr(s, 'suture_weld_coordinate_migration', {}))
    report = dict(from_version=0, to_version=VERSION, time_myr=float(s.t),
                  coordinate='local free-edge normal opening',
                  integration='analytic great-circle unilateral Huber law',
                  capacity_changed=False, material_changed=False)
    s.suture_weld_coordinate_version = VERSION
    s.suture_weld_coordinate_migration = report
    return deepcopy(report)


def _intersection_arc(a, b, c, d, tolerance):
    # A point touch has no directed arc normal. Without this guard the zero
    # normal passes every plane/tangent test and can admit a whole free edge.
    # Use the same positive-length threshold as collision_fronts.shared_arc.
    # Translation avoids cancellation between order-one products when the
    # clipped edge is short; a x (b-a) is algebraically the same normal.
    cross = np.cross(a, b-a)
    if np.linalg.norm(cross) <= tolerance or np.linalg.norm(np.cross(c, d-c)) <= tolerance:
        return None
    normal = fronts._unit(cross)
    if max(abs(float(c@normal)), abs(float(d@normal))) > tolerance:
        return None
    points = [p for p in (a, b, c, d)
              if fronts._on_arc(p, a, b, tolerance) and fronts._on_arc(p, c, d, tolerance)]
    best = max(((fronts._angle(p, q), p, q) for i, p in enumerate(points)
                for q in points[i+1:]), key=lambda entry: entry[0], default=None)
    return (best[1], best[2]) if best is not None and best[0] > tolerance else None


def _intervals(segments, normal, tolerance):
    """Canonical periodic union on a great circle; no internal mesh breakpoints."""
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    basis = fronts._unit(np.cross(normal, axis))
    tangent = np.cross(normal, basis)
    intervals = []
    period = 2.*math.pi
    for a, b in segments:
        start, end = sorted((math.atan2(float(a@tangent), float(a@basis)) % period,
                             math.atan2(float(b@tangent), float(b@basis)) % period))
        if end-start > math.pi:
            intervals.extend(((0., start), (end, period)))
        else:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]+tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[0][0] <= tolerance and merged[-1][1] >= period-tolerance:
        merged = [[merged[-1][0], merged[0][1]+period]]+merged[1:-1]
    return basis, tangent, merged


def arcs(surface, sheets, overlap, front, top_sheet, geometry):
    """Return canonical arcs, their outward normals and material sides.

    Each side sees the relative velocity. Half weights retain the existing
    force per metre for two opposing straight fronts: two edges represent one
    contact. This symmetric convention also covers a contained sheet, whose
    closed perimeter has zero net area rate but nonzero local peeling.
    """
    radius = float(surface.get('radius_km', 6371.))
    tolerance = fronts.CONNECT_TOLERANCE_KM/radius
    groups = []
    for index in front['overlap_indices']:
        first, second = int(overlap['first'][index]), int(overlap['second'][index])
        pair = (first, second) if sheets[first] == top_sheet else (second, first)
        polygon = fronts._polygon(geometry, index)
        for side, face in enumerate(pair):
            start, end = geometry['free_offsets'][face:face+2]
            for c, d, inward in geometry['free_edges'][start:end]:
                outward = -inward
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    arc = _intersection_arc(a, b, c, d, tolerance)
                    if arc is None:
                        continue
                    group = next((g for g in groups if g['side'] == side
                                  and np.linalg.norm(g['normal']-outward) < 2e-11), None)
                    if group is None:
                        group = dict(side=side, normal=outward, segments=[])
                        groups.append(group)
                    group['segments'].append(arc)
    result = []
    for group in groups:
        normal = group['normal']
        basis, tangent, intervals = _intervals(group['segments'], normal, tolerance)
        for start, end in intervals:
            result.append(dict(basis=basis, tangent=tangent, normal=normal, side=group['side'],
                               start=start, end=end, length_km=.5*(end-start)*radius))
    return result


def moments(start, end):
    """Exact first and second trigonometric moments on an arc."""
    span = end-start
    middle = .5*(start+end)
    first = 2.*math.sin(.5*span)*np.array([math.cos(middle), math.sin(middle)])
    direction = np.array([math.cos(middle), math.sin(middle)])
    transverse = np.array([-direction[1], direction[0]])
    parallel = .5*(span+math.sin(span))
    perpendicular = (.5*(span-math.sin(span)) if span >= 1e-3
                     else span**3*(1./12.-span**2/240.+span**4/10080.))
    second = parallel*np.outer(direction, direction)+perpendicular*np.outer(transverse, transverse)
    return first, second


def integrate(operator, x, start, end, delta, *, velocity=None):
    """Exact arc integral of unilateral Huber opening resistance and derivatives.

    Closing speed is a*cos(theta)+b*sin(theta). The dissipation density is zero
    for closing, v*v/(2*delta) for -delta<v<0, and -v-delta/2 below -delta.
    Splitting at the analytic zero/yield crossings avoids quadrature error.
    Its C1 potential is convex and its resisting work is nonnegative, including
    zero motion. No smooth-tail attraction is applied to a closing contact.
    """
    if not np.isfinite([start, end, delta]).all() or end <= start or delta <= 0.:
        raise ValueError('Weld integration requires a finite positive arc and smoothing width.')
    # Reduced modes may preserve the native-width common product separately
    # from their differential columns; derivatives still use the full operator.
    velocity = operator@x if velocity is None else np.asarray(velocity, float)
    if velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError('Weld integration requires two finite velocity modes.')
    amplitude = float(np.linalg.norm(velocity))
    cuts = [start, end]
    if amplitude:
        phase = math.atan2(float(velocity[1]), float(velocity[0]))
        for threshold in (0., -delta):
            if abs(threshold) > amplitude:
                continue
            offset = math.acos(float(np.clip(threshold/amplitude, -1., 1.)))
            for root in (phase-offset, phase+offset):
                for k in range(math.ceil((start-root)/(2.*math.pi)), math.floor((end-root)/(2.*math.pi))+1):
                    angle = root+k*2.*math.pi
                    if start < angle < end:
                        cuts.append(angle)
    gradient = np.zeros(2); hessian = np.zeros((2, 2)); value = 0.; opening_flux = 0.
    cuts = sorted(set(cuts))
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = .5*(a+b)
        speed = float(velocity@np.array([math.cos(mid), math.sin(mid)]))
        if speed >= 0.:
            continue
        first, second = moments(a, b)
        opening_flux -= float(velocity@first)
        if speed <= -delta:
            value += -float(velocity@first)-.5*delta*(b-a)
            gradient -= first
        else:
            value += float(velocity@second@velocity)/(2.*delta)
            gradient += second@velocity/delta
            hessian += second/delta
    return value, gradient@operator, operator.T@hessian@operator, opening_flux
