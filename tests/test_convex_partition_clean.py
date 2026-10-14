"""Cyclic duplicate cleanup must preserve represented coordinate bytes."""
import json
from pathlib import Path
import sys
from timeit import repeat
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convex_partition as partition


def legacy_clean(points):
    """Previous implementation, retained here as an independent oracle."""
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) > 1:
        points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]
    return points


def assert_bytes(case, actual, expected):
    case.assertEqual(actual.dtype, expected.dtype)
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.tobytes(), expected.tobytes())


def unit(points):
    values = np.asarray(points, float)
    return values / np.linalg.norm(values, axis=1)[:, None]


class CleanEquivalenceTests(unittest.TestCase):
    def test_empty_singleton_and_each_cyclic_duplicate_case(self):
        a, b, c = [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
        cases = [[], [a], [a, a], [a, a, a], [a, b], [a, b, c],
                 [a, a, b, c], [a, b, b, c], [a, b, c, c],
                 [a, b, c, a], [a, a, b, b, c, a, a]]
        for points in cases:
            with self.subTest(points=points):
                assert_bytes(self, partition._clean(points), legacy_clean(points))

    def test_signed_zero_and_nonfinite_payloads_match_without_input_mutation(self):
        payload_nan = np.array([0x7ff8000000001234], dtype=np.uint64).view(np.float64)[0]
        points = np.array([[0., -0., 1.], [-0., 0., 1.],
                           [np.inf, 1., 0.], [np.inf, 1., 0.],
                           [-np.inf, 1., 0.], [payload_nan, 2., 3.],
                           [payload_nan, 2., 3.], [0., -0., 1.]])
        before = points.tobytes()
        with np.errstate(all='raise'):
            assert_bytes(self, partition._clean(points), legacy_clean(points))
        self.assertEqual(points.tobytes(), before)
        # Cleanup itself accepts them, while the public geometry preparation
        # must still reject nonfinite coordinates before defining any edge.
        with self.assertRaisesRegex(ValueError, 'finite coordinates'):
            partition.prepare(points)

    def test_strided_reversed_readonly_and_cast_inputs_are_unchanged(self):
        raw = np.arange(120., dtype=np.float64).reshape(20, 6)
        views = [raw[:, ::2], raw[::-2, ::2], raw[::3, 1::2],
                 np.asfortranarray(raw[:, ::2]), raw[:, ::2].astype(np.float32),
                 raw[:, ::2].astype(np.int64)]
        for points in views:
            points.setflags(write=False)
            before = points.tobytes()
            assert_bytes(self, partition._clean(points), legacy_clean(points))
            self.assertEqual(points.tobytes(), before)

    def test_random_polygons_with_adjacent_and_closing_repetitions_keep_all_bytes(self):
        random = np.random.default_rng(73051)
        for size in range(2, 66):
            points = random.normal(size=(size, 3))
            repeats = random.integers(1, 4, size)
            points = np.repeat(points, repeats, axis=0)
            if size % 2:
                points = np.vstack((points, points[:2]))
            assert_bytes(self, partition._clean(points), legacy_clean(points))

    def test_malformed_shape_and_unconvertible_values_still_raise(self):
        for points in ([1., 2.], [[1., 2.], [3., 4.]], [['bad', 1., 2.]]):
            with self.assertRaises(ValueError):
                legacy_clean(points)
            with self.assertRaises(ValueError):
                partition._clean(points)

    def test_actual_clipping_retains_identical_intersection_and_complement_vertices(self):
        random = np.random.default_rng(906)
        splits = 0
        for _ in range(40):
            offset = random.uniform(-.15, .15, 2)
            subject = unit([[1., -.2, -.2], [1., .2, -.2],
                            [1., .2, .2], [1., -.2, .2]])
            clipper = unit(np.column_stack((np.ones(3), offset +
                np.array([[-.12, -.1], [.17, -.08], [0., .18]]))))
            edges = partition.prepare(clipper)
            actual = partition.partition(subject, edges)
            with patch.object(partition, '_clean', legacy_clean):
                expected = partition.partition(subject, partition.prepare(clipper))
            assert_bytes(self, actual[0], expected[0])
            self.assertEqual(len(actual[1]), len(expected[1]))
            for current, previous in zip(actual[1], expected[1]):
                assert_bytes(self, current, previous)
            splits += len(actual[1])
        self.assertGreater(splits, 40)


def benchmark():
    """Observed timings only; timing is never a regression assertion."""
    random = np.random.default_rng(710)
    polygons = [random.normal(size=(n, 3)) for n in (0, 1, 2, 3, 4, 6, 8, 17, 64)]
    polygons += [points[np.r_[np.arange(len(points)), 0]] for points in polygons if len(points)]
    for points in polygons:
        actual, expected = partition._clean(points), legacy_clean(points)
        if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise AssertionError('Cleanup changed a represented polygon.')
    result = dict(polygons_per_iteration=len(polygons), iterations=2000, repeats=3,
                  result='All output coordinate bytes identical',
                  scope='0-64 vertex polygons, including cyclic duplicates; not a full-step prediction')
    for name, function in [('original', legacy_clean), ('adjacent_comparisons', partition._clean)]:
        seconds = repeat(lambda: [function(points) for points in polygons], number=2000, repeat=3)
        result[name] = dict(seconds=seconds, best_seconds=min(seconds))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
