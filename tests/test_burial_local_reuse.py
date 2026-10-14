"""Unrelated arc additions do not invalidate unchanged burial partitions."""
from decimal import localcontext
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import budget
import burial_depth as burial
import mesh_coverage
from parallel_runtime import RuntimePool, physical_core_count
from tests.test_exact_burial_geometry import TRIANGLES_106, REFERENCE_106, RADIUS
from tests.test_burial_parallel_partition import _same, _stack


class LocalPartitionReuseTests(unittest.TestCase):
    def setUp(self):
        burial.geometry_cache_reset()
        self.triangles = np.r_[TRIANGLES_106, TRIANGLES_106]
        self.upper = np.array([1, 2, 2, 4, 5], dtype=np.int64)
        self.job = (0, np.array([0, 1], dtype=np.int64), REFERENCE_106, RADIUS)

    def tearDown(self):
        burial.geometry_cache_reset()

    def partition(self, triangles=None, upper=None, job=None):
        return burial.partition_geometry_session(
            self.triangles if triangles is None else triangles,
            self.upper if upper is None else upper).partition_face(*(self.job if job is None else job))

    def test_unrelated_faces_and_pair_appends_reuse_exact_regions(self):
        expected = self.partition()
        calls = burial.geometry_cache_statistics()['kernel_calls']
        enlarged = np.r_[self.triangles, np.tile(TRIANGLES_106, (20, 1, 1))]
        enlarged[-1] *= .9  # Unused malformed geometry must remain unread.
        upper = np.r_[self.upper, np.full(20, len(enlarged)-1, dtype=np.int64)]
        actual = self.partition(enlarged, upper)
        self.assertTrue(_same(expected, actual))
        statistics = burial.geometry_cache_statistics()
        self.assertEqual(statistics['kernel_calls'], calls)
        self.assertEqual(statistics['partition_hits'], 1)
        self.assertEqual(statistics['interned_roots'], 0)

    def test_exact_local_dependencies_and_exposed_pair_labels_all_invalidate(self):
        mutations = []
        moved = self.triangles.copy()
        moved[1, 0, 0] = np.nextafter(moved[1, 0, 0], np.inf)
        mutations.append((moved, self.upper, self.job))
        lower = self.triangles.copy()
        lower[0] = lower[0]@np.array([[np.cos(.001), -np.sin(.001), 0.],
                                     [np.sin(.001), np.cos(.001), 0.], [0., 0., 1.]])
        mutations.append((lower, self.upper,
                          (0, self.job[1], mesh_coverage._polygon_area(lower[0], RADIUS), RADIUS)))
        refs = self.upper.copy(); refs[0] = 4
        mutations.append((self.triangles, refs, self.job))
        for face, selected, reference, radius in (
                (3, self.job[1], REFERENCE_106, RADIUS),
                (0, np.array([0, 2]), REFERENCE_106, RADIUS),
                (0, np.array([1, 0]), REFERENCE_106, RADIUS),
                (0, self.job[1].astype(np.int32), REFERENCE_106, RADIUS),
                (0, self.job[1], np.nextafter(REFERENCE_106, np.inf), RADIUS),
                (0, self.job[1], REFERENCE_106, np.nextafter(RADIUS, np.inf))):
            mutations.append((self.triangles, self.upper, (face, selected, reference, radius)))
        for triangles, upper, job in mutations:
            with self.subTest(job=repr(job)):
                burial.geometry_cache_reset()
                self.partition()
                expected = burial._partition_face_uncached(triangles, job[0], job[1], upper, job[2], job[3])
                actual = self.partition(triangles, upper, job)
                self.assertTrue(_same(actual, expected))
                self.assertEqual(burial.geometry_cache_statistics()['partition_misses'], 2)

    def test_context_change_misses_and_cache_hits_preserve_context_and_clock(self):
        self.partition()
        with localcontext() as context:
            context.prec += 5
            with budget.step_budget(1.):
                clock = budget.snapshot()
                expected = self.partition()
                flags = dict(context.flags)
                actual = self.partition()
                self.assertEqual(budget.snapshot(), clock)
                self.assertEqual(dict(context.flags), flags)
                self.assertTrue(_same(actual, expected))
        self.assertEqual(burial.geometry_cache_statistics()['partition_misses'], 2)
        self.assertEqual(burial.geometry_cache_statistics()['partition_hits'], 1)

    def test_mutating_returned_arrays_cannot_poison_unrelated_scene_reuse(self):
        expected = self.partition()
        returned = self.partition()
        returned[1][:] = -1.
        for polygon, _ in returned[0]:
            if isinstance(polygon, np.ndarray):
                polygon[:] = 0.
        actual = self.partition(np.r_[self.triangles, TRIANGLES_106])
        self.assertTrue(_same(actual, expected))

    def test_invalid_indexing_keeps_the_original_exception(self):
        self.partition()
        for selected in (np.array([len(self.upper)]), np.array([0., 1.])):
            with self.subTest(selected=selected):
                with self.assertRaises(Exception) as expected:
                    burial._partition_face_uncached(self.triangles, 0, selected, self.upper, REFERENCE_106, RADIUS)
                with self.assertRaises(type(expected.exception)) as actual:
                    self.partition(job=(0, selected, REFERENCE_106, RADIUS))
                self.assertEqual(str(actual.exception), str(expected.exception))
        self.assertGreater(burial.geometry_cache_statistics()['bypasses'], 0)

    def test_negative_indices_and_empty_cover_keep_original_results(self):
        for face, selected in ((-3, np.array([-5, -4])), (0, np.array([], dtype=np.int64))):
            with self.subTest(face=face, selected=selected):
                expected = burial._partition_face_uncached(self.triangles, face, selected, self.upper, REFERENCE_106, RADIUS)
                actual = self.partition(job=(face, selected, REFERENCE_106, RADIUS))
                self.assertTrue(_same(actual, expected))

    def test_local_keys_and_results_stay_within_existing_payload_limit(self):
        cache = burial._ExactGeometryReuse(max_entries=2, max_bytes=8192, max_roots=2)
        with patch.object(burial, '_GEOMETRY_REUSE', cache):
            for selected in (np.array([0, 1]), np.array([0, 2]), np.array([3, 4])):
                self.partition(job=(0, selected, REFERENCE_106, RADIUS))
            statistics = cache.statistics()
            self.assertLessEqual(statistics['entries'], 2)
            self.assertLessEqual(statistics['accounted_payload_bytes'], 8192)
            self.assertGreater(statistics['evictions'], 0)

    def test_parallel_consumer_uses_existing_local_results_after_unrelated_append(self):
        triangles, upper, jobs = _stack(level=0)
        expected = list(burial.partition_geometry_session(triangles, upper).partition_faces(jobs))
        changed = np.r_[triangles, TRIANGLES_106]
        changed_upper = np.r_[upper, np.array([len(triangles)], dtype=upper.dtype)]
        count = min(2, physical_core_count())
        with RuntimePool(workers=count, max_workers=count):
            session = burial.partition_geometry_session(changed, changed_upper)
            with patch.object(session, '_parallel', side_effect=AssertionError('Unchanged partitions recomputed')):
                actual = list(session.partition_faces(jobs))
        self.assertTrue(all(_same(a, b) for a, b in zip(expected, actual)))

    def test_floating_point_warning_probe_does_not_cache_or_suppress_warning(self):
        original = burial._partition_face_uncached
        def warning_kernel(*args):
            np.divide(np.array([0.]), np.array([0.]))
            return original(*args)
        with patch.object(burial, '_partition_face_uncached', warning_kernel):
            with np.errstate(invalid='warn'), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                first = self.partition()
                second = self.partition(np.r_[self.triangles, TRIANGLES_106])
        self.assertEqual(len(caught), 2)
        self.assertTrue(_same(first, second))
        self.assertEqual(burial.geometry_cache_statistics()['partition_hits'], 0)


if __name__ == '__main__':
    unittest.main()
