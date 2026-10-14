"""Small ordinary-law oracles; no world constructor or saved history is evolved.

The partial-cover fixture deliberately retains one thickness per face. These
tests establish composition accounting and diagnostic honesty, not a cure for
the spatial depletion limitation described by the deep-stack handoff.
"""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import crust_inventory as inventory
import eclogite_sink as sink
from tests.test_burial_depth import halves, prepare, triangle, triangle_area
from tests.test_crust_inventory import column


class OrdinaryFounderingHandoffTests(unittest.TestCase):
    def assert_state_exact(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def partial_stack(self):
        left, _ = halves()
        s = prepare([(left, 40., 0., False), (triangle(), 50., 0., False)], [(1, 2)])
        sink.upgrade_inventory(s)
        sink.upgrade_depth_integration(s)
        self.assertEqual(s.foundering_version, 2)
        self.assertEqual(s.foundering_depth_version, 1)
        return s

    def test_partial_cover_report_uses_union_not_double_counted_pair_area(self):
        state = column(50.)
        eligible = np.array([8.])
        attribution = dict(covered_area=np.array([.9]), covered_union_area_km2=np.array([.2]))
        report = sink.ordinary_loss_diagnostics(state, eligible, 1., area_km2=np.ones(1),
                                               attribution=attribution)
        expected = 8.*(1.-math.exp(-1./20.))
        self.assertEqual(report['partially_covered_faces'], 1)
        self.assertAlmostEqual(report['eligible_on_partially_covered_faces_km3'], 8.)
        self.assertAlmostEqual(report['removed_on_partially_covered_faces_km3'], expected)
        self.assertAlmostEqual(report['uniform_removal_assigned_to_uncovered_area_km3'], .8*expected)
        self.assertFalse(report['spatial_depletion_resolved'])
        legacy = sink.ordinary_loss_diagnostics(state, eligible, 1., area_km2=np.ones(1),
                                               attribution={'covered_area': np.array([.2])})
        self.assertFalse(legacy['exact_partial_cover_available'])
        self.assertNotIn('partially_covered_faces', legacy)

    def test_diagnostics_do_not_mutate_or_change_the_sink_result(self):
        s = self.partial_stack()
        eligible, _, attribution = sink.eligible_thickness_km(s, s.structure['thickness_km'])
        original = deepcopy(s.structure)
        eligible_before = eligible.copy()
        attribution_before = deepcopy(attribution)
        direct = deepcopy(original)
        direct_result = sink.apply(direct, eligible, 1.)
        with patch.object(burial_depth, 'integrate', side_effect=AssertionError('no diagnostic geometry pass')):
            report = sink.ordinary_loss_diagnostics(s.structure, eligible, 1.,
                area_km2=s.material_surface['area_km2'], attribution=attribution)
        self.assert_state_exact(original, s.structure)
        self.assert_state_exact(attribution_before, attribution)
        np.testing.assert_array_equal(eligible, eligible_before)
        actual = sink.apply(s.structure, eligible, 1.)
        self.assert_state_exact(direct, s.structure)
        for left, right in zip(direct_result, actual):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(report['admitted_loss_km3'], float(s.material_surface['area_km2']@actual[0]))

    def test_repeated_partial_cover_removal_obeys_composition_and_keeps_spent_inventory_spent(self):
        s = self.partial_stack()
        area = s.material_surface['area_km2']
        initial = deepcopy(s.structure)
        fraction = triangle_area(halves()[0])/triangle_area(triangle())
        self.assertAlmostEqual(fraction, .5, places=12)
        total_removed = np.zeros(2)
        expected_removed_lower = 0.
        conversion = 1.-math.exp(-1./20.)
        for _ in range(120):
            eligible, _, attribution = sink.eligible_thickness_km(s, s.structure['thickness_km'])
            expected_eligible = fraction*max(40.-expected_removed_lower, 0.)
            self.assertAlmostEqual(eligible[1], expected_eligible, places=10)
            expected_step = min(expected_eligible*conversion, max(30.-expected_removed_lower, 0.),
                                max(50.-expected_removed_lower-sink.RESIDUAL_FLOOR_KM, 0.))
            report = sink.ordinary_loss_diagnostics(s.structure, eligible, 1., area_km2=area,
                                                   attribution=attribution)
            removed, _ = sink.apply(s.structure, eligible, 1.)
            self.assertAlmostEqual(removed[1], expected_step, places=11)
            total_removed += removed
            expected_removed_lower += expected_step
            sink.record(s, removed, attribution)
            inventory.validate(s.structure)
            self.assertTrue(report['exact_partial_cover_available'])
            self.assertFalse(report['spatial_depletion_resolved'])
            self.assertAlmostEqual(report['removed_on_partially_covered_faces_km3'], area[1]*removed[1],
                                   delta=max(1., area[1])*1e-11)
        self.assertAlmostEqual(total_removed[1], 30., places=10)
        self.assertEqual(s.structure[inventory.REMAINING][1], 0.)
        np.testing.assert_allclose(initial[inventory.BASELINE]+s.structure[inventory.ADDED],
            s.structure[inventory.RETURNED]+s.structure[inventory.ERODED]+s.structure[inventory.REMAINING],
            rtol=0, atol=1e-12)
        self.assertAlmostEqual(s.mantle_return_km3['total'], float(area@total_removed),
                               delta=max(1., float(area@total_removed))*1e-12)
        # Eligibility remains positive under the face-mean law. No reset or
        # new geometry can replenish the already-spent composition allowance.
        before = deepcopy(s.structure)
        eligible, _, attribution = sink.eligible_thickness_km(s, s.structure['thickness_km'])
        self.assertGreater(eligible[1], 0.)
        report = sink.ordinary_loss_diagnostics(s.structure, eligible, 1., area_km2=area,
                                               attribution=attribution)
        self.assertGreater(report['inventory_blocked_requested_loss_km3'], 0.)
        removed, _ = sink.apply(s.structure, eligible, 1.)
        np.testing.assert_array_equal(removed, 0.)
        np.testing.assert_array_equal(s.structure[inventory.REMAINING], before[inventory.REMAINING])
        np.testing.assert_array_equal(s.structure[inventory.RETURNED], before[inventory.RETURNED])

    def test_heating_gate_is_still_ten_myr_for_the_ordinary_fifty_km_law(self):
        s = self.partial_stack()
        self.assertEqual((sink.ECLOGITE_DEPTH_KM, sink.HEATING_DELAY_MYR, sink.FOUNDERING_TAU_MYR),
                         (50., 10., 20.))
        s.parcel_burial_myr[:] = 9.999
        eligible, _, _ = sink.eligible_thickness_km(s, s.structure['thickness_km'])
        np.testing.assert_array_equal(eligible, 0.)
        s.parcel_burial_myr[:] = 10.
        eligible, _, _ = sink.eligible_thickness_km(s, s.structure['thickness_km'])
        self.assertAlmostEqual(eligible[1], 20., places=10)

    def test_floor_and_inventory_blocked_volumes_are_disjoint(self):
        state = column(sink.RESIDUAL_FLOOR_KM+1.)
        # A request of ten kilometres reaches the composition cap first, then
        # the one-kilometre physical floor. Neither blocked share double counts.
        report = sink.ordinary_loss_diagnostics(state, np.array([10.]), 20000.,
                                               area_km2=np.array([2.]))
        cap = .6*(sink.RESIDUAL_FLOOR_KM+1.)
        self.assertAlmostEqual(report['requested_loss_km3'], 20.)
        self.assertAlmostEqual(report['inventory_blocked_requested_loss_km3'], 2.*(10.-cap))
        self.assertAlmostEqual(report['residual_floor_blocked_requested_loss_km3'], 2.*(cap-1.))
        self.assertAlmostEqual(report['admitted_loss_km3'], 2.)
        self.assertAlmostEqual(report['requested_loss_km3'], report['admitted_loss_km3']+
            report['inventory_blocked_requested_loss_km3']+report['residual_floor_blocked_requested_loss_km3'])
        at_floor = column(sink.RESIDUAL_FLOOR_KM)
        report = sink.ordinary_loss_diagnostics(at_floor, np.array([1.]), 1., area_km2=np.ones(1))
        self.assertEqual(report['eligible_faces_at_residual_floor'], 1)
        self.assertEqual(report['residual_floor_limited_faces'], 1)
        self.assertEqual(report['admitted_loss_km3'], 0.)


if __name__ == '__main__':
    unittest.main()
