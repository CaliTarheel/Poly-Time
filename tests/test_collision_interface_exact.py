"""Exact clipped-ray interface work, independently checked by surface quadrature."""
from decimal import Decimal, localcontext
from fractions import Fraction
from math import lcm
import pickle
import unittest
from unittest.mock import patch

import numpy as np

import collision_interface as interface
from exact_polygon import ExactPolygon


def rational_polygon(rows):
    points = []
    for row in rows:
        row = [Fraction(value) for value in row]
        denominator = lcm(*(value.denominator for value in row))
        points.append(tuple(int(value*denominator) for value in row)+(denominator,))
    return ExactPolygon(tuple(points))


def independent_ray_quadrature(polygon, omega):
    """Positive projected-triangle quadrature, without a boundary moment identity."""
    with localcontext() as context:
        context.prec = 90
        points = []
        for row in polygon.homogeneous:
            point = [Decimal(value)/Decimal(row[3]) for value in row[:3]]
            norm = sum(value*value for value in point).sqrt()
            points.append([value/norm for value in point])
        triangles = []
        a = points[0]
        for b, c in zip(points[1:-1], points[2:]):
            u, v = [[right-left for left, right in zip(a, other)] for other in (b, c)]
            cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            jacobian = float(sum(x*y for x, y in zip(a, cross)))
            if jacobian < 0:
                raise ValueError('Oracle requires positive winding.')
            triangles.append((np.array([a, b, c], float), jacobian))
    nodes, weights = np.polynomial.legendre.leggauss(24)
    nodes, weights = (nodes+1)/2, weights/2
    area = work = 0.
    for (a, b, c), jacobian in triangles:
        for u, weight in zip(nodes, weights):
            q = a+u*(b-a)+(1-u)*nodes[:, None]*(c-a)
            norm = np.linalg.norm(q, axis=1)
            da = jacobian*(1-u)/norm**3
            speed_squared = np.sum(np.cross(omega, q/norm[:, None])**2, axis=1)
            area += weight*float(weights@da)
            work += weight*float(weights@(da*speed_squared))
    return area, work


def subulp_strip():
    low, high = Fraction(1, 4), Fraction(1, 4)+Fraction(1, 10**25)
    return rational_polygon([(1, Fraction(-3, 100), low), (1, Fraction(3, 100), low),
                             (1, Fraction(3, 100), high), (1, Fraction(-3, 100), high)])


class ExactInterfaceGeometryTests(unittest.TestCase):
    def test_captured_106_region_work_matches_independent_surface_quadrature(self):
        from tests.test_exact_burial_geometry import captured_partition, RADIUS
        regions, areas, _ = captured_partition()
        selected = [(polygon, area) for (polygon, cover), area in zip(regions, areas)
                    if cover == (1,)]
        self.assertEqual(len(selected), 1)
        polygon, area = selected[0]
        self.assertIsInstance(polygon, ExactPolygon)
        metric = interface.rotation_metric(polygon, RADIUS)
        for omega in (*np.eye(3), np.array([.2, -.7, .3])):
            expected_area, expected_work = independent_ray_quadrature(polygon, omega)
            self.assertAlmostEqual(area/(expected_area*RADIUS**2), 1., delta=2e-14)
            actual = float(omega@metric@omega)
            self.assertGreater(actual, 0.)
            self.assertAlmostEqual(actual/(expected_work*(RADIUS*1000)**2), 1., delta=2e-14)

    def test_subulp_footprint_retains_area_and_positive_work(self):
        polygon = subulp_strip()
        represented = polygon.represented()
        np.testing.assert_array_equal(represented[0], represented[3])
        np.testing.assert_array_equal(represented[1], represented[2])
        metric = interface.rotation_metric(polygon, .001)
        for omega in (*np.eye(3), np.array([.2, -.7, .3]), np.array([1., 0., .25])):
            area, expected = independent_ray_quadrature(polygon, omega)
            actual = float(omega@metric@omega)
            self.assertGreater(actual, 0.)
            self.assertAlmostEqual(actual/expected, 1., delta=2e-12)
            self.assertAlmostEqual(float(np.trace(metric))/(2*area), 1., delta=2e-14)

    def test_exact_metric_is_additive_with_collinear_edge_and_pickle(self):
        polygon = subulp_strip()
        a, b, c, d = polygon.homogeneous
        refined = ExactPolygon((a, b, c, d,
                                tuple(a[i]*d[3]+d[i]*a[3] for i in range(3))+(2*a[3]*d[3],)))
        actual = interface.rotation_metric(pickle.loads(pickle.dumps(refined)), 6371.)
        expected = sum(interface.rotation_metric(ExactPolygon((a, one, two)), 6371.)
                       for one, two in ((b, c), (c, d)))
        np.testing.assert_allclose(actual, expected, rtol=3e-15,
                                   atol=np.finfo(float).eps*np.linalg.norm(expected))

    def test_duplicate_edges_and_integer_ray_rescaling_preserve_metric(self):
        polygon = subulp_strip()
        points = list(polygon.homogeneous)
        points.insert(2, points[1])
        scaled = ExactPolygon(tuple(tuple(value*(index+1) for value in row)
                                    for index, row in enumerate(points)))
        expected = interface.rotation_metric(polygon, .001)
        np.testing.assert_allclose(interface.rotation_metric(scaled, .001), expected,
                                   rtol=3e-15, atol=np.finfo(float).eps*np.linalg.norm(expected))

    def test_exact_rays_cannot_mix_in_cache_when_projection_matches(self):
        polygon = subulp_strip()
        wide = rational_polygon([(1, Fraction(-3, 100), Fraction(1, 4)),
                                 (1, Fraction(3, 100), Fraction(1, 4)),
                                 (1, Fraction(3, 100), Fraction(1, 4)+Fraction(2, 10**25)),
                                 (1, Fraction(-3, 100), Fraction(1, 4)+Fraction(2, 10**25))])
        np.testing.assert_array_equal(polygon.represented(), wide.represented())
        first = interface.rotation_metric(polygon, .001)
        first[:] = -1.  # Returned matrices must be detached from retained cache data.
        restored = interface.rotation_metric(polygon, .001)
        doubled = interface.rotation_metric(wide, .001)
        np.testing.assert_allclose(doubled, 2*restored, rtol=3e-15, atol=0.)
        self.assertGreater(np.trace(restored), 0.)

    def test_invalid_exact_winding_and_radius_fail_without_repair(self):
        polygon = subulp_strip()
        for points in (polygon.homogeneous[::-1],
                       tuple(polygon.homogeneous[i] for i in (0, 2, 1, 3)),
                       (polygon.homogeneous[0],)*3):
            with self.assertRaisesRegex(ValueError, 'winding'):
                interface.rotation_metric(ExactPolygon(points), .001)
        for radius in (0., -1., np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, 'positive radius'):
                interface.rotation_metric(polygon, radius)
        antipodal = ExactPolygon(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (-1, 0, 0, 1)))
        with self.assertRaisesRegex(ValueError, 'antipodal edge'):
            interface.rotation_metric(antipodal, .001)

    def test_well_conditioned_array_keeps_fast_path(self):
        with patch.object(interface, '_precise_rotation_integral', side_effect=AssertionError('slow path')):
            metric = interface.rotation_metric(np.eye(3), .001)
        expected = np.full((3, 3), -1/3)
        np.fill_diagonal(expected, np.pi/3)
        np.testing.assert_allclose(metric, expected, rtol=0., atol=3e-16)


if __name__ == '__main__':
    unittest.main()
