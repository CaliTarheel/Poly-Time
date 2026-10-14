"""Independent area checks for read-only entry/contact source polygons."""

import unittest

import numpy as np

import continental_entry
from channel_pair_zones import (
    partition_entry_pair, partition_entry_stack)
from channel_region_geometry import spherical_partition_intersections
from entry_channel_pressure import ordered_top_pressure
from ridge_geometry import rotate
from tests.test_entry_stack_work_oracle import FACES, NORMAL, SHEETS, geometry


class ChannelPairZoneTests(unittest.TestCase):
    def setUp(self):
        self.points = geometry()
        self.lower, self.upper = self.points[FACES]

    def zones(self, **kwargs):
        return partition_entry_pair(
            self.lower, self.upper, NORMAL[0], **kwargs)

    def test_pair_partition_closes_face_and_matches_existing_overlap(self):
        result = self.zones()
        kinds = {zone['kind'] for zone in result['zones']}
        self.assertEqual(kinds, {'unentered', 'entered-free', 'entered-contact'})
        self.assertAlmostEqual(sum(zone['fraction'] for zone in result['zones']),
                               1., places=9)
        direct = spherical_partition_intersections(
            self.lower, [zone['polygon'] for zone in result['zones']],
            [zone['polygon'] for zone in result['zones']])
        np.testing.assert_allclose(
            [zone['fraction'] for zone in result['zones']],
            direct['old_fractions'], rtol=0., atol=2e-10)
        fraction = continental_entry.reference_entry_fractions(
            self.points[FACES[:1]], NORMAL, radius_km=6371.)
        ledger = continental_entry.entry_stack_overlap_ledger(
            self.points, FACES, SHEETS, np.array([0]), NORMAL,
            fraction, radius_km=6371.)
        np.testing.assert_allclose(
            result['contact_area_km2'],
            sum(row['area_km2'] for row in ledger), rtol=2e-10)
        np.testing.assert_allclose(
            result['entry_reference_fraction'], fraction[0], rtol=2e-12)

    def test_finite_arc_clips_contact_without_losing_unentered_area(self):
        finite = self.zones(finite_midpoint=np.array([1., 0., 0.]),
                            finite_half_length_km=50.)
        full = self.zones()
        self.assertGreater(finite['contact_area_km2'], 0.)
        self.assertLess(finite['contact_area_km2'], full['contact_area_km2'])
        self.assertAlmostEqual(sum(zone['fraction'] for zone in finite['zones']),
                               1., places=9)
        fraction = continental_entry.reference_entry_fractions(
            self.points[FACES[:1]], NORMAL, radius_km=6371.,
            finite_midpoints=np.array([[1., 0., 0.]]),
            finite_half_lengths_km=np.array([50.]))
        ledger = continental_entry.entry_stack_overlap_ledger(
            self.points, FACES, SHEETS, np.array([0]), NORMAL,
            fraction, radius_km=6371.,
            finite_midpoints=np.array([[1., 0., 0.]]),
            finite_half_lengths_km=np.array([50.]))
        np.testing.assert_allclose(
            finite['contact_area_km2'],
            sum(row['area_km2'] for row in ledger), rtol=2e-10)

    def test_missing_entry_retains_whole_unentered_face(self):
        away = np.array([-1., 0., 0.])
        result = partition_entry_pair(self.lower, self.upper, away)
        self.assertEqual({zone['kind'] for zone in result['zones']},
                         {'unentered'})
        self.assertAlmostEqual(sum(zone['fraction'] for zone in result['zones']),
                               1., places=9)

    def test_two_upper_sheets_choose_the_immediately_adjacent_layer(self):
        upper_top = rotate(self.lower, [0., 0., -.006])
        graph = {3: {2, 1}, 2: {1}}
        direct = partition_entry_stack(
            self.lower, [self.upper, upper_top], [2, 3], 1, graph,
            NORMAL[0])
        reversed_order = partition_entry_stack(
            self.lower, [upper_top, self.upper], [3, 2], 1, graph,
            NORMAL[0])
        self.assertAlmostEqual(sum(zone['fraction'] for zone in direct['zones']),
                               1., places=9)
        np.testing.assert_allclose(
            direct['contact_area_km2'],
            reversed_order['contact_area_km2'], rtol=2e-12)
        both = [zone for zone in direct['zones']
                if zone['kind'] == 'entered-contact'
                and zone['covering_sheets_top_to_bottom'] == (3, 2)]
        self.assertTrue(both)
        self.assertTrue(all(zone['adjacent_upper_sheet_id'] == 2
                            for zone in both))
        source = ordered_top_pressure(80000., [
            dict(kind='rock', top_depth_m=0., base_depth_m=20000.,
                 sheet_id=3, mass_kg_m2=2800. * 20000.),
            dict(kind='rock', top_depth_m=20000., base_depth_m=45000.,
                 sheet_id=2, mass_kg_m2=2900. * 25000.),
            dict(kind='mantle', top_depth_m=45000., base_depth_m=80000.)],
            both[0]['covering_sheets_top_to_bottom'])
        self.assertEqual(source['rock_sheet_ids'], (3, 2))
        self.assertTrue(all(zone['adjacent_upper_sheet_id']
                            == zone['covering_sheets_top_to_bottom'][-1]
                            for zone in direct['zones']
                            if zone['kind'] == 'entered-contact'))
        with self.assertRaisesRegex(ValueError, 'unique vertical order'):
            partition_entry_stack(
                self.lower, [self.upper, upper_top], [2, 3], 1,
                {3: {1}, 2: {1}}, NORMAL[0])

    def test_ordered_stack_rejects_missing_lower_order(self):
        with self.assertRaisesRegex(ValueError, 'inherited order'):
            partition_entry_stack(
                self.lower, [self.upper], [2], 1, {}, NORMAL[0])

    def test_upper_sheet_subdivision_preserves_ordered_contact_area(self):
        upper_top = rotate(self.lower, [0., 0., -.006])
        graph = {3: {2, 1}, 2: {1}}
        whole = partition_entry_stack(
            self.lower, [self.upper, upper_top], [2, 3], 1, graph,
            NORMAL[0], upper_face_ids=[20, 30])
        a, b, c = self.upper
        ab, bc, ca = (a+b, b+c, c+a)
        ab /= np.linalg.norm(ab)
        bc /= np.linalg.norm(bc)
        ca /= np.linalg.norm(ca)
        children = [np.array(row) for row in
                    ((a, ab, ca), (ab, b, bc), (ca, bc, c),
                     (ab, bc, ca))]
        triangles = children + [upper_top]
        ids = [20, 21, 22, 23, 30]
        split = partition_entry_stack(
            self.lower, triangles, [2, 2, 2, 2, 3], 1, graph,
            NORMAL[0], upper_face_ids=ids)
        reversed_order = partition_entry_stack(
            self.lower, triangles[::-1], [3, 2, 2, 2, 2], 1, graph,
            NORMAL[0], upper_face_ids=ids[::-1])
        np.testing.assert_allclose(
            [split['contact_area_km2'], reversed_order['contact_area_km2']],
            whole['contact_area_km2'], rtol=2e-10)
        self.assertAlmostEqual(sum(zone['fraction'] for zone in split['zones']),
                               1., places=9)
        self.assertTrue(all(
            zone['adjacent_upper_face_id'] in ids
            for zone in split['zones'] if zone['kind'] == 'entered-contact'))

    def test_positive_area_self_overlap_of_one_sheet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'self-overlaps'):
            partition_entry_stack(
                self.lower, [self.upper, self.upper.copy()], [2, 2], 1,
                {2: {1}}, NORMAL[0], upper_face_ids=[20, 21])


if __name__ == '__main__':
    unittest.main()
