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

    def test_binary64_collinear_fan_retains_exact_zero_and_positive_work(self):
        # All vertices meet the existing unit-vector contract. The first fan
        # is exactly collinear in stored binary64 coordinates, but separate
        # Decimal normalization used to create a negative determinant.
        a = np.ones(3)/np.sqrt(3.)
        b = 2.**-24*np.array([1., -1., 0.])
        c = 2.**-21*np.array([1., 1., -2.])
        polygon = np.array([a, a+b, a+2*b, a+2*b+c, a+c])
        np.testing.assert_allclose(np.linalg.norm(polygon, axis=1), 1., rtol=0., atol=1e-12)
        exact = rational_polygon(polygon)
        from exact_polygon import determinant
        self.assertEqual(determinant(*exact.homogeneous[:3]), 0)
        metric = interface.rotation_metric(polygon, .001)
        without_collinear = polygon[[0, 2, 3, 4]]
        expected = interface.rotation_metric(without_collinear, .001)
        np.testing.assert_allclose(metric, expected, rtol=3e-15,
                                   atol=np.finfo(float).eps*np.linalg.norm(expected))
        np.testing.assert_allclose(metric, interface.rotation_metric(exact, .001),
                                   rtol=3e-15, atol=np.finfo(float).eps*np.linalg.norm(metric))
        self.assertGreater(np.linalg.eigvalsh(metric).min(), 0.)
        oracle = rational_polygon(without_collinear)
        for omega in (*np.eye(3), a, np.array([.2, -.7, .3])):
            area, work = independent_ray_quadrature(oracle, omega)
            actual = float(omega@metric@omega)
            self.assertGreater(actual, 0.)
            tolerance = 32*np.finfo(float).eps*np.linalg.norm(metric)*float(omega@omega)
            self.assertAlmostEqual(actual, work, delta=tolerance)
            self.assertAlmostEqual(float(np.trace(metric)), 2*area,
                                   delta=32*np.finfo(float).eps*area)
        # A one-ULP perturbation really inverts the formerly zero fan. It
        # must still fail; exact predicates do not introduce a tolerance band.
        inverted = polygon.copy()
        inverted[1, 2] = np.nextafter(inverted[1, 2], -np.inf)
        self.assertLess(determinant(*rational_polygon(inverted).homogeneous[:3]), 0)
        for invalid in (polygon[::-1], polygon[[0, 2, 4, 3]], inverted):
            with self.assertRaisesRegex(ValueError, 'winding'):
                interface.rotation_metric(invalid, .001)

    def test_binary64_zero_fan_still_requires_a_minor_patch(self):
        # An exact zero determinant alone does not prove a harmless collinear
        # subdivision: these rays span the entire great circle. Keep the
        # pre-existing binary64 guard-digit denominator check for zero fans.
        polygon = np.array([[1., 0., 0.], [-.5, np.sqrt(3.)/2, 0.],
                            [-.5, -np.sqrt(3.)/2, 0.]])
        with self.assertRaisesRegex(ValueError, 'minor convex spherical patch'):
            interface._precise_rotation_integral(polygon)

    def test_dyadic_zero_fan_corpus_matches_exact_rays_across_numpy_modes(self):
        # These are constructed represented geometries, not a capture of the
        # failed 268 -> 269 interval. Integer offsets retain an exactly zero
        # first fan across eight widths without renormalizing the input rays.
        from exact_polygon import determinant
        scale = 2**52
        origin = np.rint(np.array([.2, .3, np.sqrt(.87)])*scale).astype(np.int64)
        for mode in ('warn', 'raise', 'ignore'):
            with np.errstate(all=mode):
                for width in (1, 2, 4, 8, 16, 32, 64, 128):
                    with self.subTest(mode=mode, width=width):
                        interface.burial_depth.geometry_cache_reset()
                        self.addCleanup(interface.burial_depth.geometry_cache_reset)
                        offsets = np.array([[0, 0, 0], [width, -width, 0],
                                            [2*width, -2*width, 0], [0, 0, -2*width]])
                        polygon = (origin+offsets)/scale
                        np.testing.assert_allclose(np.linalg.norm(polygon, axis=1),
                                                   1., rtol=0., atol=1e-12)
                        exact = rational_polygon(polygon)
                        self.assertEqual(determinant(*exact.homogeneous[:3]), 0)
                        self.assertGreater(determinant(*(exact.homogeneous[i]
                                                        for i in (0, 2, 3))), 0)
                        actual = interface.rotation_metric(polygon, .001)
                        expected = interface.rotation_metric(exact, .001)
                        without_zero_fan = interface.rotation_metric(polygon[[0, 2, 3]], .001)
                        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.)
                        np.testing.assert_allclose(actual, without_zero_fan, rtol=3e-15, atol=0.)
                        self.assertGreater(float(np.trace(actual)), 0.)
                        self.assertAlmostEqual(float(np.trace(actual))/(2*exact.solid_angle()),
                                               1., delta=3e-15)
                        # The public cache must not let caller mutation damage
                        # a later use of the same ordinary binary64 polygon.
                        actual[:] = -1.
                        np.testing.assert_allclose(interface.rotation_metric(polygon, .001),
                                                   expected, rtol=3e-15, atol=0.)

    def test_ordinary_all_zero_fans_keep_existing_zero_integral(self):
        # Unlike ExactPolygon's stricter admission contract, the existing
        # ordinary precise path permits identical rays with zero area/work.
        # Exact winding predicates must not silently change that contract.
        metric, area = interface._precise_rotation_integral(np.tile([1., 0., 0.], (3, 1)))
        np.testing.assert_array_equal(metric, np.zeros((3, 3)))
        self.assertEqual(area, 0.)

    def test_captured_233_binary64_zero_fan_matches_surface_quadrature(self):
        # Exact binary64 coordinates captured from the failed 233 -> 234 Myr
        # interface integral, reduced to geometry only. The leading fan is
        # exactly zero; Decimal64 normalization made it -2.4121e-80 and rejected
        # the positive remaining 1.25675e-32 fan as inconsistent winding.
        polygon = np.array([
            [.24904695108520772, .5963652171177501, -.7631016603095953],
            [.24904695108520764, .59636521711775, -.7631016603095954],
            [.2490469510852078, .5963652171177503, -.7631016603095951],
            [.24904695108520788, .5963652171177503, -.7631016603095951],
        ])
        from exact_polygon import determinant
        exact = rational_polygon(polygon)
        self.assertEqual(determinant(*exact.homogeneous[:3]), 0)
        self.assertGreater(determinant(*(exact.homogeneous[i] for i in (0, 2, 3))), 0)
        radius = 6371.
        metric = interface.rotation_metric(polygon, radius)
        without_zero_fan = polygon[[0, 2, 3]]
        np.testing.assert_array_equal(metric, interface.rotation_metric(without_zero_fan, radius))
        np.testing.assert_array_equal(metric, interface.rotation_metric(exact, radius))
        # Omit only the proved exact-zero fan in the independent oracle, whose
        # own normalized Decimal Jacobian otherwise has the same sign hazard.
        oracle = rational_polygon(without_zero_fan)
        for omega in (*np.eye(3), np.array([.2, -.7, .3])):
            area, work = independent_ray_quadrature(oracle, omega)
            actual = float(omega@metric@omega)
            expected = work*(radius*1000.)**2
            self.assertGreater(actual, 0.)
            self.assertAlmostEqual(actual/expected, 1., delta=2e-14)
            self.assertAlmostEqual(float(np.trace(metric))/(2*area*(radius*1000.)**2),
                                   1., delta=2e-14)
        with self.assertRaisesRegex(ValueError, 'winding'):
            interface.rotation_metric(polygon[::-1], radius)

    def test_well_conditioned_array_keeps_fast_path(self):
        with patch.object(interface, '_precise_rotation_integral', side_effect=AssertionError('slow path')):
            metric = interface.rotation_metric(np.eye(3), .001)
        expected = np.full((3, 3), -1/3)
        np.fill_diagonal(expected, np.pi/3)
        np.testing.assert_allclose(metric, expected, rtol=0., atol=3e-16)


if __name__ == '__main__':
    unittest.main()
