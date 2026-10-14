"""Transaction failure and historical-compatibility oracles for mantle returns."""
from copy import deepcopy
import unittest

import numpy as np

import eclogite_sink as sink
from tests.test_eclogite_sink import stack
from tests import test_eclogite_sink as fixture


class FounderingValidationTests(unittest.TestCase):
    def test_nonfinite_saved_reservoirs_are_rejected(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for key in ('total', 'by_plate_uid', 'by_sheet', 'by_contact'):
                frame = fixture.FrameTests().frame()
                if key == 'total':
                    frame['mantle_return_km3'][key] = bad
                else:
                    frame['mantle_return_km3'][key]['9'] = bad
                with self.subTest(value=bad, key=key), self.assertRaises(ValueError):
                    sink.validate_frame(frame)

    def test_closure_is_enforced_only_when_the_frame_promises_it(self):
        frame = fixture.FrameTests().frame()
        frame['mantle_return_km3']['by_contact'] = {'1': .5}
        sink.validate_frame(frame)  # historical attribution gap remains readable
        for key in ('by_plate_uid', 'by_sheet', 'by_contact'):
            broken = fixture.FrameTests().frame()
            broken['mantle_return_ledger_version'] = 1
            broken['mantle_return_km3'][key] = {'1': .5}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'does not close'):
                sink.validate_frame(broken)
        sink.validate_frame(dict(fixture.FrameTests().frame(), mantle_return_ledger_version=1))

    def test_unversioned_migrations_are_not_silently_ignored(self):
        for name in ('foundering_inventory_migration', 'foundering_depth_migration'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'matching version'):
                sink.validate_frame(dict(fixture.FrameTests().frame(version=0), **{name: {}}))
        for value in (True, -1, 2):
            with self.assertRaises(ValueError):
                sink.validate_frame(dict(fixture.FrameTests().frame(), mantle_return_ledger_version=value))

    def test_bad_contact_partition_cannot_partially_book_a_transaction(self):
        s = stack()
        before = deepcopy(s.mantle_return_km3), deepcopy(s.process_totals)
        n = len(s.mass)
        attribution = dict(lower=np.arange(n), weight=np.ones(n), contact=np.ones(n, int),
                           covered_area=np.ones(n), covered_share=np.full(n, 1.5))
        with self.assertRaisesRegex(ValueError, 'exceeds the volume returned'):
            sink.record(s, np.ones(n), attribution)
        self.assertEqual((s.mantle_return_km3, s.process_totals), before)

    def test_incomplete_new_partition_cannot_be_hidden_by_old_total(self):
        s = stack()
        before = deepcopy(s.mantle_return_km3), deepcopy(s.process_totals)
        n = len(s.mass)
        attribution = dict(lower=np.arange(n), weight=np.zeros(n), contact=np.ones(n, int),
                           covered_area=np.ones(n), covered_share=np.ones(n))
        with self.assertRaisesRegex(ValueError, 'does not close'):
            sink.record(s, np.ones(n), attribution)
        self.assertEqual((s.mantle_return_km3, s.process_totals), before)

    def test_invalid_removal_or_process_total_is_atomic(self):
        for removed, previous in ((np.full(4, np.nan), 0.), (np.full(4, -1.), 0.),
                                  (np.ones(3), 0.), (np.ones(4), float('nan'))):
            s = stack()
            s.process_totals['crust_returned_to_mantle_km3'] = previous
            before = deepcopy(s.mantle_return_km3)
            with self.assertRaises(ValueError):
                sink.record(s, removed)
            self.assertEqual(s.mantle_return_km3, before)
            actual = s.process_totals['crust_returned_to_mantle_km3']
            self.assertTrue(np.isnan(actual) if np.isnan(previous) else actual == previous)

    def test_legacy_contact_gap_survives_new_closed_increment_and_snapshot(self):
        s = stack()
        s.mantle_return_km3 = dict(total=7., by_plate_uid={'99': 7.}, by_sheet={'99': 7.})
        sink.ensure_fields(s)  # supported legacy repair adds an empty contact bucket
        removed = np.full(len(s.mass), .25)
        expected = float(s.material_surface['area_km2'] @ removed)
        ledger = s.mantle_return_km3
        self.assertEqual(sink.record(s, removed), expected)
        self.assertIs(s.mantle_return_km3, ledger)
        self.assertAlmostEqual(ledger['total']-sum(ledger['by_contact'].values()), 7.)
        frame = dict(material_faces=s.material_surface['faces'], **sink.snapshot_fields(s))
        self.assertEqual(frame['mantle_return_ledger_version'], 0)
        sink.validate_frame(frame)

    def test_new_complete_snapshot_promises_cumulative_closure(self):
        s = stack()
        sink.record(s, np.full(len(s.mass), .25))
        frame = dict(material_faces=s.material_surface['faces'], **sink.snapshot_fields(s))
        self.assertEqual(frame['mantle_return_ledger_version'], 1)
        sink.validate_frame(frame)
        frame['mantle_return_km3']['by_contact']['0'] += 1.
        with self.assertRaisesRegex(ValueError, 'does not close'):
            sink.validate_frame(frame)


if __name__ == '__main__':
    unittest.main()
