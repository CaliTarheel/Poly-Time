"""Exact stack polygons stay exact in marker queries and unsupported clippers."""
from fractions import Fraction
from math import lcm
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import channel_pair_zones
import entry_phase_depth
from exact_polygon import ExactPolygon
import phase_evolution


def exact_polygon(rows):
    result = []
    for row in rows:
        values = tuple(Fraction(value) for value in row)
        denominator = lcm(*(value.denominator for value in values))
        result.append(tuple(int(value*denominator) for value in values)+(denominator,))
    return ExactPolygon(tuple(result))


def unit(values):
    values = np.asarray(values, float)
    return values/np.linalg.norm(values, axis=-1, keepdims=True)


def thin_polygon():
    low, high = Fraction(1, 4), Fraction(1, 4)+Fraction(1, 10**25)
    return exact_polygon([(1, Fraction(-3, 100), low), (1, Fraction(3, 100), low),
                          (1, Fraction(3, 100), high), (1, Fraction(-3, 100), high)])


class ExactPartitionConsumerTests(unittest.TestCase):
    def test_marker_halfspaces_preserve_existing_tolerance_exactly(self):
        polygon = exact_polygon(np.eye(3))
        tolerance = 1e-11
        with patch.object(ExactPolygon, 'represented', side_effect=AssertionError('projection')):
            self.assertTrue(phase_evolution._region_contains(polygon, unit([1., 1., 1.])))
            self.assertFalse(phase_evolution._region_contains(polygon, unit([-1., 1., 1.])))
            self.assertTrue(phase_evolution._region_contains(polygon, [1., 0., -tolerance]))
            self.assertFalse(phase_evolution._region_contains(
                polygon, [1., 0., np.nextafter(-tolerance, -np.inf)]))

    def test_subulp_marker_region_uses_exact_rays_and_preserves_region_selection(self):
        polygon = thin_polygon()
        projection = polygon.represented()
        np.testing.assert_array_equal(projection[0], projection[3])
        np.testing.assert_array_equal(projection[1], projection[2])
        point = unit([1., 0., .25])
        rows = [dict(face=0, polygon=polygon, upper=np.array([upper]), weight=.5)
                for upper in (7, 2)]
        source = SimpleNamespace(parcel_patch=np.array([42]), trace_patch=np.array([42]),
                                 trace_xyz=np.array([point]))
        before = source.trace_xyz.copy()
        with patch.object(ExactPolygon, 'represented', side_effect=AssertionError('projection')):
            selected, index, weight = phase_evolution._selected_regions(source, dict(regions=rows), True)
            self.assertIs(selected[0], rows[1])
            np.testing.assert_array_equal(index, [0])
            np.testing.assert_array_equal(weight, [1.])
            self.assertFalse(phase_evolution._region_contains(polygon, unit([1., 0., .250001])))
            for shift in range(len(polygon)):
                points = polygon.homogeneous[shift:]+polygon.homogeneous[:shift]
                scaled = ExactPolygon(tuple(tuple(value*(i+1) for value in row)
                                            for i, row in enumerate(points)))
                self.assertTrue(phase_evolution._region_contains(scaled, point))
        np.testing.assert_array_equal(source.trace_xyz, before)
        self.assertIs(rows[0]['polygon'], polygon)

    def test_ordered_entry_stack_rejects_unsupported_precision_before_float_clipping(self):
        lower = unit([[1., -.01, -.01], [1., .01, -.01], [1., 0., .01]])
        result = ([(thin_polygon(), (0,))], np.array([1.]), 0.)
        with patch.object(channel_pair_zones.burial_depth, 'partition_face', return_value=result), \
             patch.object(channel_pair_zones, '_split_entry_region') as clip:
            with self.assertRaisesRegex(ValueError, 'Ordered entry-stack.*exact-ray polygon precision'):
                channel_pair_zones.partition_entry_stack(lower, [lower], [2], 1, {2: {1}}, [0., 0., 1.])
        clip.assert_not_called()

    def test_entry_phase_rejects_unsupported_precision_before_float_clipping(self):
        with patch.object(entry_phase_depth.mesh_coverage, 'clip_triangle') as clip:
            with self.assertRaisesRegex(ValueError, 'Entry-phase.*exact-ray polygon precision'):
                entry_phase_depth._entered_region_triangles({}, 0, thin_polygon(), 6371.)
        clip.assert_not_called()

    def test_representable_entry_stack_retains_normal_clipping(self):
        lower = unit([[1., -.01, -.01], [1., .01, -.01], [1., 0., .01]])
        result = channel_pair_zones.partition_entry_stack(lower, [lower], [2], 1,
                                                        {2: {1}}, [0., 0., 1.])
        self.assertAlmostEqual(sum(row['fraction'] for row in result['zones']), 1., places=10)
        self.assertGreater(result['contact_area_km2'], 0.)
        self.assertTrue(all(type(row['polygon']) is np.ndarray for row in result['zones']))


if __name__ == '__main__':
    unittest.main()
