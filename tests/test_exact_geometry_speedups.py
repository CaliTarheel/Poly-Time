"""Faster spherical-geometry helpers return bitwise what the originals did.

Each `original_*` below is the implementation exactly as it stood before the speedup.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arc_surface
import collision_fronts
import convex_partition
import vector_norm


def same(a, b):
    """Bitwise equality of scalars/arrays, with NaN equal to NaN."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---- collision_fronts, before -------------------------------------------------------------
def original_unit(x):
    x = np.asarray(x, float)
    return x / max(float(np.linalg.norm(x)), 1e-30)


def original_angle(a, b):
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def original_on_arc(point, a, b, tol):
    normal = original_unit(np.cross(a, b))
    if abs(float(np.dot(point, normal))) > tol:
        return False
    return (np.dot(np.cross(a, point), normal) >= -tol
            and np.dot(np.cross(point, b), normal) >= -tol)


def original_shared_arc(a, b, c, d, tol):
    if np.linalg.norm(np.cross(a, b)) <= tol or np.linalg.norm(np.cross(c, d)) <= tol:
        return 0.
    normal = original_unit(np.cross(a, b))
    if max(abs(float(c @ normal)), abs(float(d @ normal))) > tol:
        return 0.
    points = [p for p in (a, b, c, d)
              if original_on_arc(p, a, b, tol) and original_on_arc(p, c, d, tol)]
    longest = max((original_angle(p, q) for i, p in enumerate(points) for q in points[i+1:]), default=0.)
    return longest if longest > tol else 0.


def original_touches(one, two, tol):
    return any(original_shared_arc(a, b, c, d, tol) > 0.
               for a, b in zip(one, np.roll(one, -1, axis=0))
               for c, d in zip(two, np.roll(two, -1, axis=0)))


# ---- convex_partition.split, before -------------------------------------------------------
def original_split(polygon, edge):
    polygon = np.asarray(polygon, float)
    if not len(polygon):
        return polygon, polygon
    d = edge.distances(polygon)
    if np.all(d >= 0.):
        return polygon, np.empty((0, 3))
    if np.all(d <= 0.):
        return np.empty((0, 3)), polygon
    positive = []; negative = []
    for i, point in enumerate(polygon):
        previous = (i-1) % len(polygon)
        if (d[i] > 0. and d[previous] < 0.) or (d[i] < 0. and d[previous] > 0.):
            fraction = abs(d[previous])/(abs(d[previous])+abs(d[i]))
            crossing = (1.-fraction)*polygon[previous]+fraction*point
            crossing /= np.linalg.norm(crossing)
            positive.append(crossing); negative.append(crossing)
        if d[i] >= 0.:
            positive.append(point)
        if d[i] <= 0.:
            negative.append(point)
    return convex_partition._clean(positive), convex_partition._clean(negative)


# ---- arc_surface.prepare, before ----------------------------------------------------------
def original_prepare(stencil, arc_ids):
    faces = np.asarray(stencil['face_vertices'])
    ids = np.asarray(arc_ids)
    original = np.asarray(stencil['vertex_index'])[faces]
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    edges = np.sort(original[:, pairs], axis=2).reshape(-1, 2)
    records = np.ascontiguousarray(edges).view(np.dtype((np.void, 2*edges.dtype.itemsize))).ravel()
    _, inverse, counts = np.unique(records, return_inverse=True, return_counts=True)
    open_arc = (counts[inverse] == 1) & np.repeat(ids > 0, 3)
    nodes = faces[:, pairs].reshape(-1, 2)[open_arc]
    boundary = np.zeros(stencil['vertex_count'], bool)
    boundary[nodes.ravel()] = True
    return dict(stencil=stencil, boundary_nodes=boundary, face_boundary_mask=boundary[faces],
                open_edge_count=int(open_arc.sum()), affected_faces=np.any(boundary[faces], axis=1))


def unit(points):
    points = np.asarray(points, float)
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


class VectorNormTests(unittest.TestCase):
    def test_bitwise_equal_to_numpy_norm_for_every_float64_vector_shape_we_see(self):
        rng = np.random.default_rng(3)
        specials = [0., -0., 1., np.inf, -np.inf, np.nan, 5e-324, 1e-310, 1e-160, 1e160, 1e300]
        cases = [rng.normal(size=3)*10.**rng.uniform(-40, 40) for _ in range(4000)]
        cases += [rng.uniform(-1., 1., 3) for _ in range(4000)]
        cases += [np.array([rng.choice(specials) for _ in range(3)]) for _ in range(500)]
        cases += [rng.normal(size=9)[::3] for _ in range(500)]      # strided view
        cases += [rng.normal(size=(2, 3))[1] for _ in range(500)]   # row of a matrix
        cases += [rng.normal(size=2), rng.normal(size=1), np.empty(0)]
        with np.errstate(all='ignore'):
            for x in cases:
                self.assertTrue(same(np.float64(vector_norm.length(x)), np.linalg.norm(x)), x)

    def test_other_dtypes_and_ranks_use_numpy_itself(self):
        for x in (np.array([3, 4]), np.array([3., 4.], np.float32), np.ones((2, 3)), [3., 4.]):
            self.assertEqual(vector_norm.length(x), float(np.linalg.norm(x)))
            self.assertIs(type(vector_norm.length(x)), float)


class ArcPairTests(unittest.TestCase):
    TOL = 1e-7/6371.

    def arcs(self, rng):
        """Random pairs of short arcs: apart, crossing, collinear overlap, shared ends, degenerate."""
        for number in range(2500):
            centre = unit(rng.normal(size=3))
            tangent = np.cross(centre, unit(rng.normal(size=3))); tangent /= np.linalg.norm(tangent)
            normal = np.cross(centre, tangent)
            scale = 10.**rng.uniform(-6., -1.)

            def at(t, lift=0.):
                return unit(centre + t*scale*tangent + lift*normal)
            kind = number % 6
            t = np.sort(rng.uniform(-1., 1., 4))
            lift = rng.choice([0., 1e-12, 1e-10, 1e-8, 1e-6, 1e-3], 4)*scale
            if kind == 0:    # independent
                a, b, c, d = (unit(centre + scale*rng.normal(size=3)) for _ in range(4))
            elif kind == 1:  # collinear, overlapping or not, sometimes nudged off the plane
                a, b, c, d = at(t[0]), at(t[2]), at(t[1], lift[1]), at(t[3], lift[3])
            elif kind == 2:  # shared endpoint
                a, b, c = at(t[0]), at(t[2]), at(t[1], lift[1])
                d = b
            elif kind == 3:  # identical, possibly reversed
                a, b = at(t[0]), at(t[3])
                c, d = (a, b) if number % 2 else (b, a)
            elif kind == 4:  # a zero-length or tiny edge
                a, b, c, d = at(t[0]), at(t[0]), at(t[1]), at(t[3])
            else:            # perpendicular crossing
                a, b = at(t[0]), at(t[3])
                c, d = at(0., -.5), at(0., .5)
            yield a, b, c, d

    def test_shared_arc_and_its_pieces_match_the_originals_bitwise(self):
        rng = np.random.default_rng(14)
        positive = 0
        for a, b, c, d in self.arcs(rng):
            expected = original_shared_arc(a, b, c, d, self.TOL)
            self.assertTrue(same(np.float64(collision_fronts.shared_arc(a, b, c, d, self.TOL)),
                                 np.float64(expected)), (a, b, c, d))
            positive += expected > 0.
            for p in (a, b, c, d):
                self.assertEqual(collision_fronts._on_arc(p, a, b, self.TOL),
                                 original_on_arc(p, a, b, self.TOL))
            self.assertTrue(same(np.float64(collision_fronts._angle(a, b)), np.float64(original_angle(a, b))))
            self.assertTrue(same(collision_fronts._unit(np.cross(a, b)), original_unit(np.cross(a, b))))
        self.assertGreater(positive, 200)   # the corpus really does include shared arcs

    def test_touching_polygons_match_the_originals(self):
        rng = np.random.default_rng(15)
        touching = 0
        for number in range(400):
            centre = unit(rng.normal(size=3))
            frame = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            scale = 10.**rng.uniform(-4., -1.5)

            def polygon(offset, count):
                angles = np.sort(rng.uniform(0., 2.*np.pi, count))
                local = np.column_stack((np.cos(angles), np.sin(angles)))*scale+offset
                return unit(centre + local[:, 0, None]*frame[:, 0] + local[:, 1, None]*frame[:, 1])
            one = polygon(0., int(rng.integers(3, 7)))
            if number % 2:   # a neighbour glued along one of one's edges
                a, b = one[0], one[1]
                outward = unit(a+b) - centre
                two = np.vstack((a, b, unit(b + outward*scale*.5 + rng.normal(size=3)*scale*.1)))
            else:
                two = polygon(rng.uniform(-2., 2., 2)*scale, int(rng.integers(3, 7)))
            expected = original_touches(one, two, self.TOL)
            self.assertEqual(collision_fronts._touches(one, two, self.TOL), expected)
            touching += expected
        self.assertGreater(touching, 50)


class SplitAndSurfaceTests(unittest.TestCase):
    def test_split_matches_original_bitwise(self):
        rng = np.random.default_rng(16)
        for _ in range(1500):
            polygon = unit(np.array([1., 0., 0.]) + rng.normal(size=(int(rng.integers(3, 8)), 3))*.05)
            a, b = unit(np.array([1., 0., 0.]) + rng.normal(size=(2, 3))*.05)
            edge = convex_partition.Edge(a, b, 1 if rng.random() < .5 else -1)
            for got, want in zip(convex_partition.split(polygon, edge), original_split(polygon, edge)):
                self.assertTrue(same(got, want))

    def test_arc_surface_prepare_matches_original(self):
        rng = np.random.default_rng(17)
        for number in range(60):
            count = int(rng.integers(1, 60))
            vertices = int(rng.integers(4, 40))
            faces = np.array([rng.choice(vertices, 3, replace=False) for _ in range(count)])
            if number % 3 == 0:                    # force shared edges
                faces[1:] = faces[0][np.argsort(rng.random((count-1, 3)), axis=1)]
            index = rng.permutation(vertices*2)[:vertices]
            if number % 4 == 1:
                index = index.astype(np.int64) + (1 << 40)    # beyond the packed-key range
            if number % 4 == 2:
                index = index.astype(np.uint32)
            stencil = dict(face_vertices=faces, vertex_index=index, vertex_count=vertices)
            ids = rng.integers(0, 3, count)
            got, want = arc_surface.prepare(stencil, ids), original_prepare(stencil, ids)
            for key in ('boundary_nodes', 'face_boundary_mask', 'affected_faces'):
                self.assertTrue(same(got[key], want[key]), key)
            self.assertEqual(got['open_edge_count'], want['open_edge_count'])

    def test_arc_surface_prepare_on_no_faces(self):
        stencil = dict(face_vertices=np.empty((0, 3), int), vertex_index=np.arange(4), vertex_count=4)
        got, want = arc_surface.prepare(stencil, np.empty(0, int)), original_prepare(stencil, np.empty(0, int))
        self.assertTrue(same(got['boundary_nodes'], want['boundary_nodes']))
        self.assertEqual(got['open_edge_count'], want['open_edge_count'])


if __name__ == '__main__':
    unittest.main()
