"""Points, rotations and polygons on the sphere.

Points are unit vectors (x, y, z). Polygons are S2 geographies from `spherely`, built from rings whose
interior lies to the left (counter-clockwise seen from outside), so a polygon larger than a hemisphere
is the region its ring encloses, not the complement S2 would pick for an unoriented ring. Areas are in
km2 on a sphere of radius RADIUS_KM.
"""
from __future__ import annotations

import math

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


def area_km2(geography) -> float:
    return float(spherely.area(geography)) / _S2_RADIUS_M ** 2 * RADIUS_KM ** 2


def union(*parts):
    result = EMPTY
    for part in parts:
        result = spherely.union(result, part)
    return result


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
    return _polygons_from_wkt(spherely.to_wkt(geography, precision=17).strip())


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


def simple_loops(ring, tolerance=1e-12):
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
    if sum(abs(loop_area_km2(loop)) for loop in loops) < minimum_km2:
        return EMPTY, sum(abs(loop_area_km2(loop)) for loop in loops)
    try:
        whole = polygon(rings[0], [h for h in rings[1:] if simple_loops(h)])
    except (ValueError, RuntimeError):
        whole = None
    if whole is not None:
        area = area_km2(whole)
        return (EMPTY, area) if area < minimum_km2 else (whole, 0.)
    region, dropped = EMPTY, 0.
    limit = SPHERE_KM2 if whole_km2 is None else whole_km2 + max(1., 1e-9 * SPHERE_KM2)
    for k, ring in enumerate(rings):
        for loop in simple_loops(ring):
            if abs(loop_area_km2(loop)) < minimum_km2:
                dropped += abs(loop_area_km2(loop))
                continue
            try:
                piece = polygon(loop)
            except (ValueError, RuntimeError):
                # Real area that S2 will not take: give up on rebuilding rather than lose it.
                raise ValueError('A loop of %.3g km2 cannot be rebuilt.' % abs(loop_area_km2(loop)))
            if k == 0 and area_km2(piece) <= limit:
                region = spherely.union(region, piece)
            else:
                region = spherely.intersection(region, piece)
    return region, dropped


def parts(geography, minimum_km2=1e-6):
    """The polygons of a geography as valid geographies, and the debris area dropped; checked against S2."""
    result, dropped = [], 0.
    whole = area_km2(geography)
    for rings in polygons(geography):
        piece, lost = part(rings, minimum_km2, whole)
        dropped += lost
        if not is_empty(piece):
            result.append(piece)
    error = sum(area_km2(piece) for piece in result) + dropped - whole
    if abs(error) > max(1e-3, 1e-9 * SPHERE_KM2):
        raise ValueError('Splitting into polygons changed the area by %.3g km2.' % error)
    if dropped > DEBRIS_LIMIT_KM2:
        # Debris has no width; dropping this much would be losing real crust.
        raise ValueError('Splitting into polygons would drop %.3g km2.' % dropped)
    return result, dropped


def clean(geography, minimum_km2=1e-6):
    """The same region rebuilt from valid polygons only (no zero-width lines or debris).

    Returns (shape, debris area dropped, whether it was rebuilt). If the rebuilt area differs from
    S2's own area of the input, the input is returned unchanged, so cleaning can never lose crust.
    """
    if not spherely.to_wkt(geography, precision=1).startswith('GEOMETRYCOLLECTION'):
        return geography, 0., True  # polygons only already
    try:
        pieces, dropped = parts(geography, minimum_km2)
    except (ValueError, RuntimeError):
        return geography, 0., False
    return (union(*pieces) if pieces else EMPTY), dropped, True


def rotate_geography(geography, matrix):
    """The same polygons rotated by `matrix` (rings keep their orientation)."""
    parts = [polygon(rotate(rings[0], matrix), [rotate(h, matrix) for h in rings[1:]]) for rings in polygons(geography)]
    return union(*parts) if parts else EMPTY
