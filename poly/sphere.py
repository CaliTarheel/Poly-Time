"""Points, rotations and polygons on the sphere.

Points are unit vectors (x, y, z). Polygons are S2 geographies from `spherely`, built from rings whose
interior lies to the left (counter-clockwise seen from outside), so a polygon larger than a hemisphere
is the region its ring encloses, not the complement S2 would pick for an unoriented ring. Areas are in
km2 on a sphere of radius RADIUS_KM.
"""
from __future__ import annotations

import math
import re

import numpy as np
import spherely

RADIUS_KM = 6371.
SPHERE_KM2 = 4. * math.pi * RADIUS_KM ** 2
_S2_RADIUS_M = spherely.EARTH_RADIUS_METERS


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def from_lonlat(lon, lat):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def to_lonlat(xyz):
    xyz = unit(xyz)
    return np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0])), np.degrees(np.arcsin(np.clip(xyz[..., 2], -1., 1.)))


def rotation(vector):
    """Rotation matrix for a rotation vector (axis times angle in radians)."""
    vector = np.asarray(vector, float)
    angle = float(np.linalg.norm(vector))
    if angle == 0.:
        return np.eye(3)
    k = vector / angle
    cross = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3) + math.sin(angle) * cross + (1. - math.cos(angle)) * cross @ cross


def rotate(points, matrix):
    return unit(np.asarray(points, float) @ np.asarray(matrix, float).T)


def arc_km(a, b):
    """Great-circle distance between unit vectors, km."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return RADIUS_KM * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def polygon(ring, holes=()):
    """Polygon whose interior lies to the left of `ring` (unit vectors, not closed)."""
    def coords(points):
        lon, lat = to_lonlat(np.asarray(points, float))
        return list(zip(lon.tolist(), lat.tolist()))
    return spherely.create_polygon(coords(ring), [coords(h) for h in holes] or None, oriented=True)


EMPTY = spherely.from_wkt('POLYGON EMPTY')
DEBRIS_LIMIT_KM2 = 1.
SLIT_KM2 = 1e-3  # a hole enclosing less than this is a slit left where edges nearly coincide
# Rebuilding a shape read back from S2 can cut a slit or fold of tens of km2 (an absolute amount,
# whatever the plate's size). Allow 1e-4 of the shape and at least 100 km2: far inside the 0.1%
# standard, and still orders of magnitude below a lost lobe or an inside-out shape.
ROTATION_TOLERANCE = 1e-4
ROTATION_FLOOR_KM2 = 100.


def allowed_change(area, tolerance=None):
    """How much rebuilding a shape of `area` km2 read back from S2 may change it (see ROTATION_TOLERANCE)."""
    return max(ROTATION_FLOOR_KM2, (ROTATION_TOLERANCE if tolerance is None else tolerance) * area)


def _full():
    ring = from_lonlat([0., 90., 180., 270.], [0., 0., 0., 0.])
    return spherely.union(polygon(ring), polygon(ring[::-1]))


def strip(points, half_width_km):
    """A band `half_width_km` either side of a polyline, as one polygon or the union of per-segment pieces."""
    points = np.asarray(points, float)
    angle = half_width_km / RADIUS_KM
    if len(points) >= 2:
        length = np.r_[0., np.cumsum(arc_km(points[:-1], points[1:]))]
        for spacing in (2. * half_width_km, 4. * half_width_km):
            marks = np.unique(np.r_[np.searchsorted(length, np.arange(0., length[-1], spacing)), len(points) - 1])
            coarse = points[marks]
            # Coincident points have no direction between them (and would give NaN normals).
            coarse = coarse[np.r_[True, arc_km(coarse[:-1], coarse[1:]) > 1e-6]]
            if len(coarse) < 2:
                return EMPTY
            segment = unit(np.cross(coarse[:-1], coarse[1:]))
            normal = unit(np.vstack((segment[:1], segment[:-1] + segment[1:], segment[-1:])))
            left = unit(coarse * np.cos(angle) + normal * np.sin(angle))
            right = unit(coarse * np.cos(angle) - normal * np.sin(angle))
            ring = np.vstack((right, left[::-1]))
            try:
                band = polygon(ring)
            except (ValueError, RuntimeError):
                continue
            if area_km2(band) > .5 * SPHERE_KM2:
                band = polygon(ring[::-1])
            return band
    pieces = []
    for a, b in zip(points[:-1], points[1:]):
        if float(arc_km(a, b)) < 1e-6:
            continue
        normal = unit(np.cross(a, b))
        corners = [unit(q * np.cos(angle) + side * normal * np.sin(angle)) for q in (a, b) for side in (-1., 1.)]
        pieces.append(polygon(np.array([corners[0], corners[2], corners[3], corners[1]])))
    return union(*pieces) if pieces else EMPTY


def area_km2(geography) -> float:
    return float(spherely.area(geography)) / _S2_RADIUS_M ** 2 * RADIUS_KM ** 2


def union(*parts):
    """The union of any number of geographies, joined in pairs so no operation is much larger than its inputs."""
    parts = list(parts)
    if not parts:
        return EMPTY
    if len(parts) == 1:
        return spherely.union(EMPTY, parts[0])
    while len(parts) > 1:
        parts = [spherely.union(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i]
                 for i in range(0, len(parts), 2)]
    return parts[0]


def intersection(a, b):
    return spherely.intersection(a, b)


def difference(a, b):
    return spherely.difference(a, b)


def is_empty(geography) -> bool:
    return bool(spherely.is_empty(geography))


def _ring(text):
    values = np.array([[float(v) for v in pair.split()] for pair in text.split(',')])
    points = from_lonlat(values[:, 0], values[:, 1])
    if len(points) > 1 and np.allclose(points[0], points[-1], atol=1e-15):
        points = points[:-1]
    return points


def polygons(geography):
    """The polygons of a geography as lists of rings (unit-vector arrays): [shell, hole, ...].

    Rings keep the orientation S2 reports (shells counter-clockwise, holes clockwise), so
    `polygon(shell, holes)` rebuilds the same region.
    """
    return _polygons_from_wkb(spherely.to_wkb(geography))


def _polygons_from_wkb(data):
    """The polygons in a WKB geometry (points and lines left out), read with numpy: exact and fast."""
    import struct

    result = []

    def read(pos):
        endian = '<' if data[pos] == 1 else '>'
        (kind,) = struct.unpack_from(endian + 'I', data, pos + 1)
        pos += 5
        dims = 2 + (kind // 1000 in (1, 2)) + 2 * (kind // 1000 == 3)
        kind %= 1000
        if kind == 1:
            return pos + 8 * dims
        (count,) = struct.unpack_from(endian + 'I', data, pos)
        pos += 4
        if kind == 2:
            return pos + 8 * dims * count
        if kind == 3:
            rings = []
            for _ in range(count):
                (n,) = struct.unpack_from(endian + 'I', data, pos)
                pos += 4
                values = np.frombuffer(data, endian + 'f8', n * dims, pos).reshape(n, dims)
                pos += 8 * dims * n
                points = from_lonlat(values[:, 0], values[:, 1])
                if len(points) > 1 and np.allclose(points[0], points[-1], atol=1e-15):
                    points = points[:-1]
                rings.append(points)
            if rings and len(rings[0]):
                result.append(rings)
            return pos
        for _ in range(count):  # multi-geometries and collections
            pos = read(pos)
        return pos

    if data:
        read(0)
    return result


def _members(text):
    """Top-level members of a GEOMETRYCOLLECTION's WKT."""
    body = text[text.index('(') + 1:text.rindex(')')]
    members, depth, start = [], 0, 0
    for i, char in enumerate(body):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            members.append(body[start:i].strip())
            start = i + 1
    members.append(body[start:].strip())
    return [m for m in members if m]


def _polygons_from_wkt(text):
    kind = text.split('(')[0].strip().split()[0] if '(' in text else text.split()[0]
    if text.endswith('EMPTY'):
        return []
    if kind == 'GEOMETRYCOLLECTION':
        # Boolean operations can return zero-area points and lines alongside the polygons.
        return [rings for member in _members(text) if member.startswith(('POLYGON', 'MULTIPOLYGON'))
                for rings in _polygons_from_wkt(member)]
    if kind in ('POINT', 'MULTIPOINT', 'LINESTRING', 'MULTILINESTRING'):
        return []
    if kind not in ('POLYGON', 'MULTIPOLYGON'):
        raise ValueError('Expected polygons, got ' + kind)
    ring_depth = 2 if kind == 'POLYGON' else 3
    result, current, depth, start = [], [], 0, None
    for i, char in enumerate(text):
        if char == '(':
            depth += 1
            if depth == ring_depth:
                start = i + 1
        elif char == ')':
            if depth == ring_depth:
                current.append(_ring(text[start:i]))
            elif depth == ring_depth - 1:
                result.append(current)
                current = []
            depth -= 1
    return [rings for rings in result if rings]


def loop_area_km2(loop) -> float:
    """Signed area of a loop smaller than a hemisphere (positive when counter-clockwise), km2."""
    loop = np.asarray(loop, float)
    if len(loop) < 3:
        return 0.
    a, b, c = loop[0], loop[1:-1], loop[2:]
    numerator = np.einsum('j,ij->i', a, np.cross(b, c))
    denominator = 1. + b @ a + np.einsum('ij,ij->i', b, c) + c @ a
    return float(2. * np.arctan2(numerator, denominator).sum()) * RADIUS_KM ** 2


def tidy_ring(ring, tolerance=1e-10):
    """A ring without near-duplicate vertices or zero-width spikes (A -> B -> A).

    Reading S2's shapes through longitude and latitude moves points by ~1e-16, leaving vertices
    that are almost repeated and spikes that turn straight back; S2 rejects both when reading the
    ring back. `tolerance` is in radians (1e-10 is about 0.6 mm).
    """
    points = np.asarray(ring, float)
    while len(points) > 3:
        # A vertex (almost) the same as the one before it goes; a run of them keeps its first.
        repeat = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) <= tolerance
        if repeat.any():
            points = points[~repeat] if not repeat.all() else points[:1]
            continue
        u = points - np.roll(points, 1, axis=0)
        w = np.roll(points, -1, axis=0) - points
        cosine = np.einsum('ij,ij->i', u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
        spike = cosine < -1. + 1e-9  # the ring turns straight back here: a spike of no width
        if not spike.any():
            break
        spike &= ~np.roll(spike, 1)  # one of two neighbouring spikes at a time
        if len(points) - spike.sum() < 3:
            break
        points = points[~spike]
    return points


def simple_loops(ring, tolerance=1e-10):
    """Split a ring that touches itself at repeated vertices into simple loops (order kept)."""
    ring = np.asarray(ring, float)
    if len(ring) == 0:
        return []
    step = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    keep = ring[np.r_[True, step > tolerance]]
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tolerance:
        keep = keep[:-1]
    keys = [tuple(k) for k in np.round(keep / tolerance).astype(np.int64)]
    loops, path, where = [], [], {}
    for point, key in zip(keep, keys):
        if key in where:
            start = where[key]
            loops.append(np.array(path[start:]))
            for dropped in path[start + 1:]:
                where.pop(tuple(np.round(dropped / tolerance).astype(np.int64)), None)
            path = path[:start + 1]
        else:
            where[key] = len(path)
            path.append(point)
    if path:
        loops.append(np.array(path))
    return [loop for loop in loops if len(loop) >= 3]


_EDGE = re.compile(r'(?:loop (\d+)\D*?)?edge (\d+)', re.IGNORECASE)


def polygon_robust(ring, holes=(), attempts=24):
    """`polygon`, dropping the vertex S2 names when it rejects a ring (crossing or repeated edges).

    S2's own outputs are valid, but writing them as longitude and latitude moves points by ~1e-16,
    which can make edges that nearly touch cross. Dropping the vertex at the offending edge moves
    the outline by about that much; callers check the area against S2's own.
    """
    loops = [tidy_ring(ring)] + [tidy_ring(h) for h in holes]
    loops = [loops[0]] + [h for h in loops[1:] if len(h) >= 3]
    for _ in range(attempts):
        try:
            return polygon(loops[0], loops[1:])
        except (ValueError, RuntimeError) as error:
            found = [(int(m.group(1) or 0), int(m.group(2))) for m in _EDGE.finditer(str(error))]
            # S2 numbers loops in its own nesting order, which is ours only while that is simple;
            # an edge that is not there means it is not, and the caller rebuilds loop by loop.
            if not found or any(loop >= len(loops) or edge >= len(loops[loop]) for loop, edge in found):
                raise
            if 'cross' in str(error) and len(found) == 2 and found[0][0] == found[1][0] and found[0][0] < len(loops):
                # Two edges of one loop cross: a slit or fold. Cut out the shorter stretch of the
                # loop between them and put the crossing point in its place.
                cut = _cut_crossing(loops[found[0][0]], found[0][1], found[1][1])
                if cut is not None:
                    loops[found[0][0]] = cut
                    continue
            # When two loops are named (a crossing), drop the vertex from the smaller one.
            loop, edge = min(found, key=lambda le: len(loops[le[0]]) if le[0] < len(loops) else 1 << 30)
            if loop >= len(loops):
                raise
            if len(loops[loop]) <= 3:
                if loop == 0:
                    raise
                del loops[loop]  # a hole down to three vertices that S2 still rejects: no width left
                continue
            loops[loop] = np.delete(loops[loop], (edge + 1) % len(loops[loop]), axis=0)
    return polygon(loops[0], loops[1:])


def _cut_crossing(loop, i, j):
    """`loop` with the shorter stretch between crossing edges i and j replaced by their crossing."""
    n = len(loop)
    i, j = sorted((i, j))
    a, b, c, d = loop[i], loop[(i + 1) % n], loop[j], loop[(j + 1) % n]
    meet = np.cross(np.cross(a, b), np.cross(c, d))
    norm = float(np.linalg.norm(meet))
    if norm < 1e-300:
        return None
    meet /= norm
    if float(meet @ (a + b + c + d)) < 0.:
        meet = -meet
    # Two loops meet at the crossing: vertices i+1..j, and the rest round the start. A slit or fold
    # encloses (almost) nothing, so keep the stretch that encloses more and cut the other.
    inner = np.vstack((meet[None], loop[i + 1:j + 1]))
    outer = np.vstack((loop[:i + 1], meet[None], loop[j + 1:]))
    if abs(loop_area_km2(inner)) <= abs(loop_area_km2(outer)) and len(inner) < len(loop):
        result = outer
    else:
        result = inner
    return result if len(result) >= 3 else None


def part(rings, minimum_km2=1e-6, whole_km2=None):
    """A valid geography for one polygon's rings (shell, holes), and the debris area dropped from it.

    Boolean operations between edges that coincide exactly (a plate's crust and its boundary, both
    moved by the same rotation) leave zero-width spikes: repeated vertices, zero-length edges and
    loops that enclose nothing. A ring S2 will not take is split at its repeated vertices. A loop that
    encloses almost nothing on either side is debris. Otherwise the ring's order defines its region:
    a shell's loops are joined and a hole's loops cut out, except that a shell loop whose region is
    larger than the whole polygon (`whole_km2`) is an inner loop and is cut out too.
    """
    loops = simple_loops(rings[0])
    if not loops:
        return EMPTY, 0.
    # Holes that enclose (almost) nothing are slits left where crust pieces meet without exactly
    # shared edges; S2 keeps them, but read back they can turn into loops S2 rejects or reads as the
    # whole sphere. They are debris.
    holes, slits = [], 0.
    for hole in rings[1:]:
        area = sum(abs(loop_area_km2(loop)) for loop in simple_loops(hole))
        if area < max(minimum_km2, SLIT_KM2):
            slits += area
        else:
            holes.append(hole)
    rings = [rings[0]] + holes
    if sum(abs(loop_area_km2(loop)) for loop in loops) < minimum_km2:
        return EMPTY, sum(abs(loop_area_km2(loop)) for loop in loops)
    try:
        whole = polygon_robust(rings[0], [h for h in rings[1:] if simple_loops(h)])
    except (ValueError, RuntimeError):
        whole = None
    if whole is not None and whole_km2 is not None and area_km2(whole) > .5 * SPHERE_KM2 > 1.5 * whole_km2:
        # Read back inside out (cutting its folds left the ring running the other way): turn it
        # the right way out if that is what S2 had, else rebuild it loop by loop below.
        flipped = spherely.difference(_full(), whole)
        whole = flipped if area_km2(flipped) <= whole_km2 + allowed_change(whole_km2) else None
    if whole is not None:
        # Repairs (vertices dropped, crossings cut) must not change the area the loops enclose; a
        # ring that touches itself can be repaired into something else, and is then built loop by
        # loop instead. (Loop areas are reliable below half the sphere.)
        shell = [loop_area_km2(loop) for loop in loops]
        cut = [abs(loop_area_km2(loop)) for hole in rings[1:] for loop in simple_loops(hole)]
        if all(abs(a) < .45 * SPHERE_KM2 for a in shell) and sum(shell) > 0.:
            expected = sum(shell) - sum(cut)
            if abs(area_km2(whole) - expected) > max(1., 1e-4 * expected):
                whole = None
    if whole is not None:
        area = area_km2(whole)
        return (EMPTY, area + slits) if area < minimum_km2 else (whole, slits)
    region, dropped = EMPTY, 0.
    # A clockwise (inner) loop read as a shell encloses nearly the whole sphere; a true shell is larger
    # than the polygon only by its holes. So a shell loop is inner only beyond half the sphere and
    # beyond the polygon's area by more than 1%.
    limit = SPHERE_KM2 if whole_km2 is None else max(.5 * SPHERE_KM2, 1.01 * whole_km2 + 1.)
    cuts = []
    for k, ring in enumerate(rings):
        for loop in simple_loops(ring):
            if abs(loop_area_km2(loop)) < minimum_km2:
                dropped += abs(loop_area_km2(loop))
                continue
            try:
                piece = polygon_robust(loop)
            except (ValueError, RuntimeError):
                # Real area that S2 will not take: give up on rebuilding rather than lose it.
                raise ValueError('A loop of %.3g km2 cannot be rebuilt.' % abs(loop_area_km2(loop)))
            if k == 0 and area_km2(piece) <= limit:
                region = spherely.union(region, piece)
            else:
                # A hole, or an inner loop of the shell: what it encloses is cut out, whichever way
                # round it runs (a hole folded over itself can run both ways), after every shell
                # loop is in.
                cuts.append(piece if area_km2(piece) <= .5 * SPHERE_KM2 else spherely.difference(_full(), piece))
    if cuts:
        region = spherely.difference(region, union(*cuts))
    return region, dropped + slits


def parts(geography, minimum_km2=1e-6):
    """The polygons of a geography as valid geographies, and the debris area dropped; checked against S2."""
    result, dropped = [], 0.
    whole = area_km2(geography)
    if whole > SPHERE_KM2 - allowed_change(SPHERE_KM2):
        return [_full()], 0.  # the whole sphere is one piece
    for rings in polygons(geography):
        piece, lost = part(rings, minimum_km2, whole)
        dropped += lost
        if not is_empty(piece):
            result.append(piece)
    error = sum(area_km2(piece) for piece in result) + dropped - whole
    if abs(error) > allowed_change(whole):
        raise ValueError('Splitting into polygons changed the area by %.3g km2.' % error)
    if dropped > allowed_change(whole):
        # Debris has no width; dropping this much would be losing real crust.
        raise ValueError('Splitting into polygons would drop %.3g km2.' % dropped)
    return result, dropped


def clean(geography, minimum_km2=1e-6):
    """The same region rebuilt from valid polygons only (no zero-width lines or debris).

    Returns (shape, debris area dropped, whether it was rebuilt). If the rebuilt area differs from
    S2's own area of the input, the input is returned unchanged, so cleaning can never lose crust.
    """
    if spherely.get_type_id(geography) != spherely.GeographyType.GEOMETRYCOLLECTION.value:
        # Polygons only already; rebuild only if some hole is a slit (encloses nothing), which
        # otherwise rides along and multiplies with every later operation.
        slit = any(abs(loop_area_km2(hole)) < SLIT_KM2 for rings in polygons(geography) for hole in rings[1:])
        if not slit:
            return geography, 0., True
    try:
        pieces, dropped = parts(geography, minimum_km2)
    except (ValueError, RuntimeError):
        return geography, 0., False
    return (union(*pieces) if pieces else EMPTY), dropped, True


def rotate_geography(geography, matrix, tolerance=None):
    """The same polygons rotated by `matrix` (rings keep their orientation)."""
    pieces, dropped = [], 0.
    whole = area_km2(geography)
    if whole > SPHERE_KM2 - allowed_change(SPHERE_KM2, tolerance):
        return _full()  # the whole sphere (a plate that is all there is), which S2 writes as one point
    for rings in polygons(geography):
        piece, lost = part([rotate(ring, matrix) for ring in rings], whole_km2=whole)
        dropped += lost
        if not is_empty(piece):
            pieces.append(piece)
    result = union(*pieces) if pieces else EMPTY
    # Rebuilding moves the outline where S2's own output does not survive longitude and latitude
    # (tiny holes, edges that nearly touch); allow 1e-5 of the shape (1,000 km2 on a 100 Mkm2
    # plate), a hundred times inside the 0.1% standard, and refuse anything more.
    allowed = allowed_change(area_km2(geography), tolerance)
    if abs(area_km2(result) - (SPHERE_KM2 - area_km2(geography))) <= allowed and area_km2(geography) < .5 * SPHERE_KM2:
        # Rebuilt inside out (a ring read the other way round): turn it the right way out.
        result = spherely.difference(_full(), result)
    if dropped > allowed or abs(area_km2(result) - area_km2(geography)) > allowed:
        raise ValueError('Rotating changed the area by %.3g km2.' % (area_km2(result) - area_km2(geography)))
    return result


SHARED_KM = 1e-3  # vertices of different rings closer than this (1 m) are the same vertex


NODE_KM = 1e-2  # a point of one ring this close to another ring's edge is put into that edge too


def _shared_clusters(points):
    """Which of `points` are the same point (within SHARED_KM): (number of clusters, cluster of each)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    count = len(points)
    pairs = cKDTree(points).query_pairs(SHARED_KM / RADIUS_KM, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)


def _to_line_km(points, line):
    """Great-circle distance (km) from each of `points` to the polyline `line`."""
    a, b = line[:-1], line[1:]
    pole = np.cross(a, b)
    norm = np.linalg.norm(pole, axis=1)
    pole = pole / np.maximum(norm, 1e-300)[:, None]
    off = points @ pole.T  # sine of the distance to each edge's great circle
    foot = points[:, None, :] - off[..., None] * pole[None]
    # The foot of the perpendicular is on the edge when it lies between a and b.
    within = (np.einsum('pij,ij->pi', foot, np.cross(pole, a)) >= 0.) & (np.einsum('pij,ij->pi', foot, np.cross(b, pole)) >= 0.)
    across = np.where(within & (norm[None] > 0.), np.arcsin(np.clip(np.abs(off), 0., 1.)), np.inf)
    ends = np.minimum(np.arccos(np.clip(points @ a.T, -1., 1.)), np.arccos(np.clip(points @ b.T, -1., 1.)))
    return np.min(np.minimum(across, ends), axis=1) * RADIUS_KM


def _merge_runs(first, second, where, node_km):
    """The points of two runs between the same shared vertices, in order, if they are one line.

    Runs are (ring, start, u, w, inner points); the two must be of different rings and run
    opposite ways, every point of each within `node_km` of the other's line, and both move
    steadily from u to w. Returns the merged inner points from the first run's u to its w.
    """
    (ra, _, ua, wa, xa), (rb, _, ub, wb, xb) = first, second
    if ra == rb or ua != wb or wa != ub or (not len(xa) and not len(xb)):
        return None
    xb = xb[::-1]  # both from ua to wa now
    direction = where[wa] - where[ua]
    if any(np.any(np.diff(where[x] @ direction) < 0.) for x in (xa, xb) if len(x) > 1):
        return None
    line_a = where[np.concatenate(([ua], xa, [wa]))]
    line_b = where[np.concatenate(([ua], xb, [wa]))]
    if len(xa) and _to_line_km(where[xa], line_b).max() > node_km:
        return None
    if len(xb) and _to_line_km(where[xb], line_a).max() > node_km:
        return None
    merged = np.concatenate((xa, xb))
    return merged[np.argsort((where[merged] - where[ua]) @ direction, kind='stable')]


def node_shared(rings, node_km=NODE_KM, groups=None):
    """Rings that share boundaries made to share their vertices too.

    Points within SHARED_KM become one point (and a ring's repeats of it go). Where two rings run
    between the same two shared vertices in opposite directions, and every point either has there
    lies within `node_km` of the other's line, both get all the points of both, in order: a point
    on another ring's edge (where an operation cut one ring and not its neighbour) is then a vertex
    of both, so the shared edge can be simplified on both sides at once.

    `groups` names the polygon each ring belongs to (default: each its own). Two points of one
    polygon are never made one: S2 will not take a polygon that touches itself or its holes.
    """
    rings = [np.asarray(r, float) for r in rings]
    sizes = np.array([len(r) for r in rings])
    if not len(sizes) or sizes.sum() == 0:
        return rings
    points = np.vstack(rings)
    clusters, cluster = _shared_clusters(points)
    ring_of = np.repeat(np.arange(len(rings)), sizes)
    group_of = (np.arange(len(rings)) if groups is None else np.asarray(groups))[ring_of]
    # A point is the same as the one before it in its ring: a repeat, which goes. Otherwise two
    # points of one cluster in one polygon would make it touch itself: those points stay apart.
    previous = np.roll(np.arange(len(points)), 1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    previous[starts] = starts + sizes - 1
    repeat = cluster == cluster[previous]
    keys = cluster[~repeat] * (group_of.max() + 1) + group_of[~repeat]
    _, counts = np.unique(keys, return_counts=True)
    pinched = np.zeros(clusters, bool)
    pinched[np.unique(keys[np.isin(keys, np.unique(keys)[counts > 1])] // (group_of.max() + 1))] = True
    if pinched.any():
        # Give each point of a pinched cluster a cluster of its own again.
        alone = np.flatnonzero(pinched[cluster])
        cluster = cluster.copy()
        cluster[alone] = clusters + np.arange(len(alone))
        clusters += len(alone)
    where = np.zeros((clusters, 3))
    where[cluster[::-1]] = points[::-1]  # each cluster at its first point
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    sequences = []
    for o, n in zip(offsets, sizes):
        c = cluster[o:o + n]
        c = c[c != np.roll(c, 1)] if n > 1 else c
        sequences.append(c if len(c) >= 3 else None)
    in_rings = np.zeros(clusters, int)
    for c in sequences:
        if c is not None:
            in_rings[np.unique(c)] += 1
    shared = in_rings >= 2
    runs = {}
    for r, c in enumerate(sequences):
        if c is None:
            continue
        at = np.flatnonzero(shared[c])
        for k in range(len(at) if len(at) >= 2 else 0):
            i, j = at[k], at[(k + 1) % len(at)]
            u, w = c[i], c[j]
            if u == w:
                continue
            inner = c[i + 1:j] if j > i else np.concatenate((c[i + 1:], c[:j]))
            runs.setdefault((min(u, w), max(u, w)), []).append((r, i, u, w, inner))
    replace = {}
    for items in runs.values():
        used = set()
        for p in range(len(items)):
            for q in range(p + 1, len(items)):
                if p in used or q in used:
                    continue
                merged = _merge_runs(items[p], items[q], where, node_km)
                if merged is not None:
                    (ra, ia, *_), (rb, ib, *_) = items[p], items[q]
                    replace[(ra, ia)], replace[(rb, ib)] = merged, merged[::-1]
                    used |= {p, q}
    result = []
    for r, c in enumerate(sequences):
        if c is None:
            result.append(rings[r])
            continue
        at = np.flatnonzero(shared[c])
        if len(at) < 2 or not any((r, i) in replace for i in at):
            result.append(where[c])
            continue
        pieces = []
        for k in range(len(at)):
            i, j = at[k], at[(k + 1) % len(at)]
            inner = replace.get((r, i))
            if inner is None:
                inner = c[i + 1:j] if j > i else np.concatenate((c[i + 1:], c[:j]))
            pieces.append(np.concatenate(([c[i]], inner)))
        result.append(where[np.concatenate(pieces)])
    return result


def simplify_shared(rings, area_km2=100., max_edge_km=150., alone_km2=1e-3, rounds=8, groups=None):
    """Rings with small zig-zags and points on straight edges taken out, keeping shared edges shared.

    A vertex goes only if, in every ring that has it, it cuts off a triangle smaller than
    `area_km2` with its two neighbours, and every such ring has the same two neighbours: an edge
    two rings share stays shared, so rings that tiled the sphere still tile it; the triangle just
    changes sides. A vertex only one ring has lies on another ring's edge, or nowhere shared, and
    goes only below `alone_km2`. Junctions (neighbours differ), and vertices whose removal would
    leave an edge longer than `max_edge_km`, stay. Returns the new rings, how many vertices went,
    and how many went from each ring.
    """
    from scipy.spatial import cKDTree

    rings = node_shared(rings, groups=groups)
    sizes = np.array([len(r) for r in rings])
    if not len(sizes) or sizes.sum() == 0:
        return rings, 0, np.zeros(len(rings), int)
    start = sizes.copy()
    points = np.vstack(rings)
    ring_of = np.repeat(np.arange(len(rings)), sizes)
    # Which vertices are the same point, once: taking vertices out never moves the others.
    clusters, cluster = _shared_clusters(points)
    removed = 0
    for _ in range(rounds):
        sizes = np.bincount(ring_of, minlength=len(rings))
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        count = len(points)
        at = np.arange(count) - offsets[ring_of]
        before = offsets[ring_of] + (at - 1) % sizes[ring_of]
        after = offsets[ring_of] + (at + 1) % sizes[ring_of]
        a, v, b = points[before], points, points[after]
        triangle = .5 * np.linalg.norm(np.cross(v - a, b - a), axis=1) * RADIUS_KM ** 2
        long = arc_km(a, b) > max_edge_km
        low, high = np.minimum(cluster[before], cluster[after]), np.maximum(cluster[before], cluster[after])
        occurrences = np.bincount(cluster, minlength=clusters)
        first = np.full(clusters, 0)
        first[cluster[::-1]] = np.arange(count)[::-1]  # an occurrence of each (gone ones unused)
        worst = np.zeros(clusters)
        np.maximum.at(worst, cluster, triangle)
        bad = np.zeros(clusters, bool)
        # Every ring that has the vertex must have the same neighbours. Rings under six vertices are
        # left alone, so the at most every other vertex a round takes always leaves a triangle.
        np.logical_or.at(bad, cluster, long | (sizes[ring_of] < 6) | (low == high) | (low == cluster)
                         | (high == cluster) | (low != low[first][cluster]) | (high != high[first][cluster]))
        limit = np.where(occurrences > 1, area_km2, alone_km2)
        candidate = ~bad & (worst < limit) & (occurrences > 0)
        if not candidate.any():
            break
        # Only where the triangle a vertex cuts off is empty: no vertex of any ring inside it or
        # on its sides. In a valid tiling an edge that crossed the new one would have to cross the
        # two old ones or end inside, so the new edge crosses nothing.
        cand = np.flatnonzero(candidate)
        rep = first[cand]
        A, V, B = points[before[rep]], points[rep], points[after[rep]]
        centre = unit(A + V + B)
        radius = np.max([np.linalg.norm(centre - X, axis=1) for X in (A, V, B)], axis=0) + 2. * SHARED_KM / RADIUS_KM
        found = cKDTree(points).query_ball_point(centre, radius)
        lengths = np.array([len(f) for f in found])
        if lengths.sum():
            ks = np.repeat(np.arange(len(cand)), lengths)
            js = np.concatenate([np.asarray(f, int) for f in found])
            other = cluster[js]
            keep = (other != cand[ks]) & (other != low[rep][ks]) & (other != high[rep][ks])
            ks, js = ks[keep], js[keep]
            turn = np.sign(np.einsum('ij,ij->i', np.cross(A, V), B))[ks]
            inside = np.ones(len(ks), bool)
            for X, Y in ((A, V), (V, B), (B, A)):
                pole = np.cross(X[ks], Y[ks])
                side = np.einsum('ij,ij->i', pole, points[js]) / np.maximum(np.linalg.norm(pole, axis=1), 1e-300)
                inside &= side * turn >= -SHARED_KM / RADIUS_KM
            blocked = np.zeros(len(cand), bool)
            blocked[ks[inside]] = True
            candidate[cand[blocked]] = False
        # A vertex goes when its triangle is smaller than its neighbours' (ties by position), so no
        # two neighbours go in one round and the smallest triangles go first.
        key = worst + np.arange(clusters) * 1e-12
        lo, hi = low[first], high[first]
        go = candidate & ((key < key[lo]) | ~candidate[lo]) & ((key < key[hi]) | ~candidate[hi])
        if not go.any():
            break
        keep = ~go[cluster]
        removed += int((~keep).sum())
        points, ring_of, cluster = points[keep], ring_of[keep], cluster[keep]
    sizes = np.bincount(ring_of, minlength=len(rings))
    return np.split(points, np.cumsum(sizes)[:-1]), removed, start - sizes
