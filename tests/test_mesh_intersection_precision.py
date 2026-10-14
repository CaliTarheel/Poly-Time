"""Rare rounded-vertex area loss is measured on the original source rays."""
import unittest
from unittest.mock import patch

import numpy as np

import mesh_coverage as coverage
from tests.test_burial_area_precision import TRIANGLES, RADIUS
from tests.test_small_spherical_faces import decimal_overlap, unit


class MeshIntersectionPrecisionTests(unittest.TestCase):
    def test_recorded_thin_source_matches_independent_oracle_in_both_orders(self):
        before = TRIANGLES.copy()
        expected = decimal_overlap(*TRIANGLES)
        with patch.object(coverage, '_precise_intersection_area',
                          wraps=coverage._precise_intersection_area) as refined:
            actual = coverage._intersection_areas(TRIANGLES, TRIANGLES[::-1], RADIUS)
        self.assertEqual(refined.call_count, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.)
        np.testing.assert_array_equal(TRIANGLES, before)
        vertices = TRIANGLES.reshape(-1, 3)
        overlap = coverage.material_overlaps(vertices, np.arange(6).reshape(2, 3), np.array([1, 2]))
        np.testing.assert_array_equal(overlap['first'], [0])
        np.testing.assert_array_equal(overlap['second'], [1])
        np.testing.assert_allclose(overlap['area_km2'], expected, rtol=2e-13, atol=0.)

    def test_thin_intersection_refines_even_with_well_conditioned_source_faces(self):
        first = unit([[1., -.03, 0.], [1., .03, 0.], [1., 0., .03]])
        second = unit([[1., .03, 1e-10], [1., -.03, 1e-10], [1., 0., -.03]])
        triangles = np.array([first, second])
        self.assertFalse(coverage._triangle_area_condition(triangles).any())
        expected = decimal_overlap(first, second)
        self.assertGreater(expected, 0.)
        with patch.object(coverage, '_precise_intersection_area',
                          wraps=coverage._precise_intersection_area) as refined:
            actual = coverage._intersection_areas(triangles, triangles[::-1], RADIUS)
        self.assertEqual(refined.call_count, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.)
        np.testing.assert_allclose(coverage._intersection_areas(triangles, triangles[::-1], 1.),
                                   actual/RADIUS**2, rtol=2e-15, atol=0.)

    def test_resolved_intersection_keeps_vectorized_fast_path(self):
        subject = unit([[1., -.03, -.02], [1., .03, -.02], [1., .02, .03]])
        clipper = unit([[1., -.02, -.04], [1., .05, 0.], [1., -.02, .04]])
        expected = decimal_overlap(subject, clipper)
        with patch.object(coverage, '_precise_intersection_area',
                          side_effect=AssertionError('resolved pair should stay vectorized')):
            actual = coverage._intersection_areas(subject[None], clipper[None], RADIUS)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.)

    def test_exact_shared_edge_and_degenerate_fans_have_zero_area(self):
        first = unit([[1., -.03, 0.], [1., .03, 0.], [1., 0., .03]])
        touching = first[[1, 0, 2]].copy()
        touching[2, 2] *= -1.
        for subject, clipper in ((first, touching), (touching, first)):
            self.assertEqual(coverage._precise_intersection_area(subject, clipper, RADIUS), 0.)
            np.testing.assert_array_equal(
                coverage._intersection_areas(subject[None], clipper[None], RADIUS), [0.])
        degenerate = first[[0, 0, 1]]
        self.assertEqual(coverage._precise_intersection_area(degenerate, first, RADIUS), 0.)


if __name__ == '__main__':
    unittest.main()
