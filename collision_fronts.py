"""Current geometric fronts of persistent sheet contacts.

Polarity and geological contact identity belong to collision_contacts. These
front indices describe this geometry epoch only. A shared triangle is merely
a candidate: actual clipped polygons must share a positive-length arc before
they belong to one patch. Normals come from free material edges, never motion.
"""
from __future__ import annotations

import math
import numpy as np
import mesh_coverage
import vector_norm

# 0.1 millimetre on Earth; well above clipping roundoff, below material scale.
CONNECT_TOLERANCE_KM = 1e-7


def _unit(x):
    x = np.asarray(x, float)
    return x / max(vector_norm.length(x), 1e-30)


def _angle(a, b):
    return float(np.arctan2(vector_norm.length(np.cross(a, b)), np.dot(a, b)))


class _Arc:
    """One directed edge a->b, with everything that depends only on it computed once.

    Pair tests (`shared_arc`, `_on_arc`) used to recompute these from scratch for every
    partner edge. Each value below is the same NumPy call on the same operands, so
    it is bitwise what the per-pair code obtained.
    """
    __slots__ = ('a', 'b', '_cross', '_length', '_normal', '_angle')

    def __init__(self, a, b):
        self.a, self.b = a, b
        self._cross = self._length = self._normal = self._angle = None

    def cross(self):
        if self._cross is None:
            self._cross = np.cross(self.a, self.b)
        return self._cross

    def length(self):
        if self._length is None:
            self._length = vector_norm.length(self.cross())
        return self._length

    def normal(self):
        if self._normal is None:
            self._normal = _unit(self.cross())
        return self._normal

    def angle(self):
        if self._angle is None:
            self._angle = float(np.arctan2(self.length(), np.dot(self.a, self.b)))
        return self._angle


def _on_arc(point, a, b, tol):
    return _point_on_arc(point, _Arc(a, b), tol)


def _point_on_arc(point, arc, tol):
    normal = arc.normal()
    if abs(float(np.dot(point, normal))) > tol:
        return False
    # Directed tangential tests distinguish a minor arc from its antipode.
    return (np.dot(np.cross(arc.a, point), normal) >= -tol
            and np.dot(np.cross(point, arc.b), normal) >= -tol)


def _shared_length(first, second, tol):
    if first.length() <= tol or second.length() <= tol:
        return 0.
    normal = first.normal()
    if max(abs(float(second.a @ normal)), abs(float(second.b @ normal))) > tol:
        return 0.
    points = [p for p in (first.a, first.b, second.a, second.b)
              if _point_on_arc(p, first, tol) and _point_on_arc(p, second, tol)]
    # An edge's own angle is already known; every other pair is computed as before.
    longest = max((first.angle() if p is first.a and q is first.b
                   else second.angle() if p is second.a and q is second.b
                   else _angle(p, q)
                   for i, p in enumerate(points) for q in points[i+1:]), default=0.)
    return longest if longest > tol else 0.


def shared_arc(a, b, c, d, tol):
    """Positive common minor-arc length in radians, excluding point touches."""
    return _shared_length(_Arc(a, b), _Arc(c, d), tol)


def _arcs(polygon):
    return [_Arc(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]


def _touches(one, two, tol):
    second = _arcs(two)
    return any(_shared_length(first, other, tol) > 0.
               for first in _arcs(one) for other in second)


def prepare(surface, sheets, overlap):
    """Cache clipping/topology once for a fixed vertex/face/sheet signature."""
    vertices, faces = surface['vertices'], surface['faces']
    neighbors = [set((i,)) for i in range(len(faces))]
    edge_faces = {}
    for face, tri in enumerate(faces):
        for a, b in zip(tri, np.roll(tri, -1)):
            key = (int(sheets[face]), min(int(a), int(b)), max(int(a), int(b)))
            edge_faces.setdefault(key, []).append((face, int(a), int(b)))
    free = [[] for _ in faces]
    for entries in edge_faces.values():
        if len(entries) == 1:
            face, a, b = entries[0]
            free[face].append((vertices[a], vertices[b], _unit(np.cross(vertices[a], vertices[b]))))
        else:
            for face, _, _ in entries:
                neighbors[face].update(other[0] for other in entries)
    polygons = [mesh_coverage.clip_triangle(vertices[faces[a]], vertices[faces[b]])
                for a, b in zip(overlap['first'], overlap['second'])]
    # Packed numeric buffers avoid one tiny NPZ member per edge/polygon in
    # exact-resume checkpoints; this cache remains entirely reconstructible.
    width = max(map(len, neighbors), default=0)
    adjacency = np.full((len(faces), width), -1, np.int64)
    for i, row in enumerate(neighbors):
        adjacency[i, :len(row)] = sorted(row)
    free_offsets = np.r_[0, np.cumsum([len(row) for row in free])]
    free_edges = np.array([edge for row in free for edge in row], float).reshape(-1, 3, 3)
    polygon_offsets = np.r_[0, np.cumsum([len(row) for row in polygons])]
    polygon_vertices = np.concatenate(polygons) if polygons else np.empty((0, 3))
    return dict(neighbors=adjacency, free_offsets=free_offsets, free_edges=free_edges,
                polygon_offsets=polygon_offsets, polygon_vertices=polygon_vertices)


def _polygon(geometry, index):
    a, b = geometry['polygon_offsets'][index:index+2]
    return geometry['polygon_vertices'][a:b]


def force_stencil_groups(surface, sheets, overlap, fronts, force_overlap,
                         force_take, geometry=None):
    """Extend existing fronts through actual finite clipped arcs for work.

    Contact inventory can omit a tiny area whose derivative remains finite.
    This transient graph assigns those additional force terms to an existing
    front without changing its geometry, capacity, scale or saved history.
    A shared material face is only a candidate: clipped polygons must share
    a positive-length arc. Point touches and empty intersections do not join.

    A component without an existing front has no inherited weld. A component
    joining two existing fronts is ambiguous under their separate constitutive
    coordinates; reject it rather than merge capacities or count work twice.
    Returned arrays index force_overlap and align exactly with fronts.
    """
    take = np.asarray(force_take, dtype=np.int64)
    if (take.ndim != 1 or len(np.unique(take)) != len(take)
            or np.any(take < 0) or np.any(take >= len(force_overlap['first']))):
        raise ValueError('Force stencil indices must be distinct valid pair indices.')
    if geometry is None:
        geometry = prepare(surface, sheets, force_overlap)
    first, second = force_overlap['first'], force_overlap['second']
    tol = CONNECT_TOLERANCE_KM / float(surface['radius_km'])
    # Two distinct vertices can represent a real shared arc with zero area.
    # A point intersection, even if duplicated in a polygon buffer, cannot.
    nodes = {int(i) for i in take if any(_angle(a, b) > tol
        for a, b in zip(_polygon(geometry, int(i)),
                        np.roll(_polygon(geometry, int(i)), -1, axis=0)))}
    pair_index = {(int(first[i]), int(second[i])): int(i) for i in take}
    if len(pair_index) != len(take):
        raise ValueError('Force stencil contains duplicate material pairs.')
    parents = {i: i for i in nodes}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in sorted(nodes):
        a, b = int(first[i]), int(second[i])
        for aa in geometry['neighbors'][a]:
            if aa < 0:
                continue
            for bb in geometry['neighbors'][b]:
                if bb < 0:
                    continue
                # Ledger pairs are ordered by face index, not by sheet. The
                # indexed faces of two sheets can be interleaved after edits.
                other = pair_index.get(tuple(sorted((int(aa), int(bb)))))
                if other not in nodes or other <= i or root(i) == root(other):
                    continue
                if _touches(_polygon(geometry, i), _polygon(geometry, other), tol):
                    ra, rb = root(i), root(other)
                    parents[max(ra, rb)] = min(ra, rb)

    seeds = {}
    for front_index, front in enumerate(fronts):
        for old in front['overlap_indices']:
            key = (int(overlap['first'][old]), int(overlap['second'][old]))
            node = pair_index.get(key)
            if node not in nodes:
                raise ValueError('An existing weld front is missing from physical force geometry.')
            seeds.setdefault(root(node), set()).add(front_index)
    if any(len(identity) > 1 for identity in seeds.values()):
        raise ValueError('A finite clipped-arc bridge ambiguously joins existing weld fronts.')
    result = [[] for _ in fronts]
    for node in sorted(nodes):
        identity = seeds.get(root(node), ())
        if identity:
            result[next(iter(identity))].append(node)
    return [np.asarray(indices, dtype=np.int64) for indices in result]


def construct(surface, sheets, overlap, take, top_sheet, geometry):
    """Return area-additive connected fronts for one ordered sheet pair.

    The vector-area integral gives a subdivision-invariant spherical centroid.
    A contact normal combines upper outward and lower inward free-edge normals,
    weighted by the exact clipped edge lengths. Half its tangential magnitude
    is the effective interface length. Containment or cancelling geometry has
    no resolved direction; it receives no invented convergent front.
    """
    radius = float(surface['radius_km'])
    tol = CONNECT_TOLERANCE_KM / radius
    first, second = overlap['first'], overlap['second']
    top = np.where(sheets[first[take]] == top_sheet, first[take], second[take])
    under = np.where(sheets[first[take]] == top_sheet, second[take], first[take])
    pair_index = {(int(a), int(b)): int(i) for a, b, i in zip(top, under, take)}
    parents = {int(i): int(i) for i in take}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b, index in zip(top, under, take):
        index = int(index)
        for aa in geometry['neighbors'][a]:
            if aa < 0: continue
            for bb in geometry['neighbors'][b]:
                if bb < 0: continue
                other = pair_index.get((aa, bb))
                if other is None or other <= index or root(index) == root(other):
                    continue
                if _touches(_polygon(geometry, index), _polygon(geometry, other), tol):
                    ra, rb = root(index), root(other)
                    parents[max(ra, rb)] = min(ra, rb)
    components = {}
    for index in take:
        components.setdefault(root(int(index)), []).append(int(index))
    face_pair = {int(i): (int(a), int(b)) for a, b, i in zip(top, under, take)}
    free_arcs = {}  # a free edge meets many polygon edges; its own cross/normal/length are fixed
    result = []
    for indices in components.values():
        vector_terms, normal_terms = [], []
        points = []
        boundary_length = [0., 0.]
        for index in indices:
            polygon = _polygon(geometry, index)
            points.extend(polygon)
            for arc in _arcs(polygon):
                vector_terms.append(.5 * arc.angle() * arc.normal())
                for side, face in enumerate(face_pair[index]):
                    start, end = geometry['free_offsets'][face:face+2]
                    for k in range(start, end):
                        c, d, inward = geometry['free_edges'][k]
                        free = free_arcs.get(k)
                        if free is None:
                            free = free_arcs[k] = _Arc(c, d)
                        length = _shared_length(arc, free, tol) * radius
                        if length:
                            boundary_length[side] += length
                            normal_terms.append(inward * length * (-1. if side == 0 else 1.))
        vector = np.array([math.fsum(float(term[k]) for term in vector_terms) for k in range(3)])
        finite_centroid = np.linalg.norm(vector) > 1e-14*math.fsum(float(overlap['area_km2'][i])/radius**2 for i in indices)
        center = _unit(vector) if finite_centroid else np.asarray(points[0])
        raw_normal = np.array([math.fsum(float(term[k]) for term in normal_terms) for k in range(3)])
        raw_normal -= center * float(raw_normal @ center)
        length = .5 * float(np.linalg.norm(raw_normal))
        resolved = finite_centroid and min(boundary_length) > CONNECT_TOLERANCE_KM and length > CONNECT_TOLERANCE_KM
        normal = _unit(raw_normal) if resolved else np.zeros(3)
        # A cap smaller than a hemisphere is geodesically convex: enclosing
        # every vertex also encloses every clipped edge and polygon interior.
        # Large/nonlocal components deliberately get a whole-sphere bound.
        bound = max((_angle(center, p) for p in points), default=math.pi)
        bound = bound if finite_centroid and bound < math.pi*.5 else math.pi
        result.append(dict(front_index=len(result), overlap_indices=indices,
            overlap_area_km2=math.fsum(float(overlap['area_km2'][i]) for i in indices),
            center=center.tolist(), normal=normal.tolist(), length_km=length if resolved else 0.,
            footprint_radius_km=float(bound*radius), direction_resolved=bool(resolved),
            upper_free_edge_length_km=float(boundary_length[0]),
            lower_free_edge_length_km=float(boundary_length[1])))
    return result
