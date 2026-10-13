"""Binary64 stack regions are clipped only where a cover crosses them, and come out unchanged."""
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import mesh_coverage
from tests.test_burial_partition_precision import RUN_52_TRIANGLES, TRIANGLES as SHARED_VERTEX
from tests.test_burial_regional_precision import TRIANGLES as NARROW
from tests.test_small_spherical_faces import unit

RADIUS = 6371.
WIDE = ((1., -.05, -.04), (1., .05, -.04), (1., 0., .06))


def stack(seed, count, lower=WIDE, reach=.02, size=.006):
    """One lower face under small positively wound faces, every second one sharing an edge.

    Two covers that share an edge cut along nearly the same line: that is what leaves region
    vertices a few ulps apart and puts the winding of sound regions in doubt.
    """
    rng = np.random.default_rng(seed)
    covers = []
    while len(covers) < count:
        centre = rng.uniform(-reach, reach, 2)
        angle = rng.uniform(0., 2*np.pi)+np.array([0., 2.1, 4.2])
        corners = centre+size*rng.uniform(.5, 1.)*np.c_[np.cos(angle), np.sin(angle)]
        covers.append(unit(np.c_[np.ones(3), corners]))
        far = 2*corners[:2].mean(axis=0)-corners[2]
        covers.append(unit(np.c_[np.ones(3), np.array([corners[1], corners[0], far])]))
    return np.array([unit(lower)]+covers[:count])


def every_region_against_every_cover(triangles, selected, upper):
    """The loop the kernel used before it shared one sign test per cover."""
    regions = [(triangles[0], ())]
    visits = 0
    for pair in selected:
        triangle = triangles[upper[pair]]
        planes = mesh_coverage._triangle_planes(triangle)
        following = []
        for polygon, cover in regions:
            visits += 1
            inside, outside = burial_depth._split(polygon, planes, RADIUS, triangle)
            following.extend((piece, cover) for piece in outside)
            if mesh_coverage._polygon_area(inside, RADIUS) > 0.:
                following.append((inside, cover+(int(pair),)))
        regions = following
    return regions, visits


def clipped(triangles):
    count = len(triangles)-1
    regions = burial_depth._clipped_regions(triangles, 0, np.arange(count), np.arange(1, count+1), RADIUS)
    return regions, np.array([mesh_coverage._polygon_area(polygon, RADIUS) for polygon, _ in regions])


class ClippedRegionsTests(unittest.TestCase):
    def test_regions_are_those_of_clipping_every_region_against_every_cover(self):
        cases = [RUN_52_TRIANGLES, SHARED_VERTEX, NARROW]+[stack(seed, 30) for seed in range(3)]
        for triangles in cases:
            count = len(triangles)-1
            expected, _ = every_region_against_every_cover(triangles, np.arange(count), np.arange(1, count+1))
            actual = burial_depth._clipped_regions(triangles, 0, np.arange(count), np.arange(1, count+1), RADIUS)
            self.assertEqual([cover for _, cover in actual], [cover for _, cover in expected])
            for (polygon, _), (reference, _) in zip(actual, expected):
                self.assertEqual(polygon.tobytes(), reference.tobytes())

    def test_only_regions_a_cover_crosses_are_clipped(self):
        triangles = stack(0, 40)
        _, visits = every_region_against_every_cover(triangles, np.arange(40), np.arange(1, 41))
        with patch.object(burial_depth, '_split', wraps=burial_depth._split) as split:
            burial_depth._clipped_regions(triangles, 0, np.arange(40), np.arange(1, 41), RADIUS)
        self.assertLess(split.call_count, visits/4)

    def test_shared_signs_are_the_early_exits_of_split(self):
        triangles = stack(1, 30)
        polygons = [polygon for polygon, _ in clipped(triangles)[0]]
        for triangle in stack(2, 6)[1:]:
            excluded, held = burial_depth._cover_signs(polygons, triangle)
            planes = mesh_coverage._triangle_planes(triangle)
            for polygon, out, within in zip(polygons, excluded, held):
                inside, outside = burial_depth._split(polygon, planes, RADIUS, triangle)
                self.assertEqual(bool(out), len(inside) == 0 and len(outside) == 1 and outside[0] is polygon)
                self.assertEqual(bool(within), inside is polygon)

    def test_lower_face_without_area_is_not_admitted(self):
        # The lower face is the one region that never passed an area test of its own;
        # a single region is always clipped the plain way.
        flat = unit([[1., -.01, 0.], [1., 0., 0.], [1., .01, 0.]])
        triangles = np.array([flat, unit(WIDE)])
        expected, _ = every_region_against_every_cover(triangles, np.arange(1), np.arange(1, 2))
        actual = burial_depth._clipped_regions(triangles, 0, np.arange(1), np.arange(1, 2), RADIUS)
        self.assertEqual(len(actual), len(expected))


if __name__ == '__main__':
    unittest.main()
