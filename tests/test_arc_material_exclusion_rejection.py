"""Exact no-overlap certificates preserve admission and first witnesses."""
from fractions import Fraction
from pathlib import Path
import json
import sys
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arc_material_exclusion as exclusion


def triangle(xy):
    vertices = np.column_stack((np.asarray(xy, float), np.ones(3)))
    return vertices / np.linalg.norm(vertices, axis=1)[:, None]


def context(triangles, **values):
    return dict(triangles=np.asarray(triangles), **values)


def baseline(candidate, contexts, old=None):
    # The original path partitions every bin candidate. Disable only the
    # certificate, so the oracle retains all existing exact polygon arithmetic.
    with patch.object(exclusion._Index, 'reference_planes', exclusion._Index.planes):
        return exclusion.obstruction(candidate, contexts, old)


class ExactReferenceRejectionTests(unittest.TestCase):
    def test_separated_last_plane_avoids_fraction_polygon_construction(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        blocker = triangle([(.012, 0.), (.022, 0.), (.012, .01)])
        occupied = context([blocker])
        index = exclusion._index(occupied)
        self.assertEqual(index.candidates(candidate).tolist(), [0])
        with patch.object(exclusion, '_partition', wraps=exclusion._partition) as partitions:
            self.assertIsNone(baseline([candidate], [occupied]))
            self.assertGreater(partitions.call_count, 0)
        with patch.object(exclusion, '_partition', wraps=exclusion._partition) as partitions:
            self.assertIsNone(exclusion.obstruction([candidate], [occupied]))
            self.assertEqual(partitions.call_count, 0)

    def test_positive_slivers_and_boundary_tangencies_match_original_exact_witness(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        cases = [
            [(0., 0.), (.01, 0.), (0., .01)],  # identical
            [(.002, .002), (.008, .002), (.002, .008)],  # containment
            [(.005, 0.), (.015, 0.), (.005, .01)],  # partial intersection
            [(0., 0.), (0., -.01), (.01, 0.)],  # one common edge
            [(.01, 0.), (.02, 0.), (.01, .01)],  # one common vertex
            [(np.nextafter(.01, 0.), 0.), (.02, 0.), (.01, .01)],  # tiny positive sliver
        ]
        for xy in cases:
            with self.subTest(xy=xy):
                occupied = context([triangle(xy)])
                self.assertEqual(exclusion.obstruction([candidate], [occupied]),
                                 baseline([candidate], [occupied]))
        sliver = exclusion.obstruction([candidate], [context([triangle(cases[-1])])])
        self.assertIsNotNone(sliver)
        self.assertGreater(Fraction(sliver['exact_chart_area']), 0)

    def test_first_separating_plane_skips_later_exact_dot_products(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        blocker = triangle([(0., .012), (.01, .012), (0., .022)])
        index = exclusion._Index([blocker])
        basis = exclusion._rays(candidate)
        index._plane_normals(0)
        with patch.object(exclusion, '_dot', wraps=exclusion._dot) as dots:
            self.assertIsNone(index.reference_planes(0, basis))
            self.assertEqual(dots.call_count, 3)
        with patch.object(exclusion, '_dot', wraps=exclusion._dot) as dots:
            self.assertEqual(len(index.planes(0, basis)), 3)
            self.assertEqual(dots.call_count, 9)

    def test_old_footprint_subtraction_and_first_witness_are_unchanged(self):
        candidate = triangle([(0., 0.), (.02, 0.), (0., .02)])
        occupied = context([
            triangle([(.025, 0.), (.035, 0.), (.025, .01)]),
            triangle([(.002, .002), (.008, .002), (.002, .008)]),
            triangle([(.009, .002), (.015, .002), (.009, .008)]),
        ])
        for old in (None, context([occupied['triangles'][1]]), context([candidate])):
            with self.subTest(old=old is not None):
                actual = exclusion.obstruction([candidate], [occupied], old)
                expected = baseline([candidate], [occupied], old)
                self.assertEqual(actual, expected)
        self.assertEqual(exclusion.obstruction([candidate], [occupied])['material_face'], 1)
        self.assertEqual(exclusion.obstruction([candidate], [occupied],
                         context([occupied['triangles'][1]]))['material_face'], 2)

    def test_active_and_excluded_faces_and_context_order_keep_first_witness(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        first = context([candidate, candidate], active=np.array([False, True]), exclude=np.array([1]))
        second = context([candidate])
        actual = exclusion.obstruction([candidate], [first, second])
        self.assertEqual(actual, baseline([candidate], [first, second]))
        self.assertEqual((actual['material_context'], actual['material_face']), (1, 0))

    def test_deterministic_varied_triangle_corpus_matches_original(self):
        rng = np.random.default_rng(741)
        for _ in range(60):
            origin = rng.uniform(-.025, .025, 2)
            size = rng.uniform(.002, .025)
            candidate = triangle(origin + [[0., 0.], [size, 0.], [0., size]])
            blockers = []
            for _ in range(5):
                offset = origin + rng.uniform(-size, 2*size, 2)
                width = rng.uniform(.1, 1.5)*size
                blockers.append(triangle(offset + [[0., 0.], [width, 0.], [0., width]]))
            occupied = context(blockers)
            old = context([blockers[0]])
            for previous in (None, old):
                self.assertEqual(exclusion.obstruction([candidate], [occupied], previous),
                                 baseline([candidate], [occupied], previous))

    def test_zero_plane_is_no_constraint_and_exact_tiny_signs_are_retained(self):
        zero = Fraction(0)
        tiny = Fraction(1, 10**300)
        self.assertFalse(exclusion._excludes_reference([(zero, zero, zero)]))
        self.assertFalse(exclusion._excludes_reference([(-tiny, tiny, zero)]))
        self.assertTrue(exclusion._excludes_reference([(-tiny, zero, zero)]))


def benchmark():
    """Representative false-positive bin candidates; no timing assertion."""
    candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
    blockers = [np.roll(triangle([(x, 0.), (x+.01, 0.), (x, .01)]), i % 3, axis=0)
                for i, x in enumerate(np.linspace(.0105, .0145, 48))]
    occupied = context(blockers)
    # Warm only the shared immutable spatial/exact-plane index for both paths.
    exclusion.obstruction([candidate], [occupied])
    output = {}
    for name, disabled in [('original', True), ('certificate', False)]:
        calls = 0
        partition = exclusion._partition
        def count(*args):
            nonlocal calls
            calls += 1
            return partition(*args)
        start = perf_counter()
        with patch.object(exclusion, '_partition', side_effect=count):
            for _ in range(15):
                result = baseline([candidate], [occupied]) if disabled else exclusion.obstruction([candidate], [occupied])
                if result is not None:
                    raise AssertionError('Disjoint triangles were admitted as an overlap.')
        output[name] = dict(seconds=perf_counter()-start, exact_polygon_partitions=calls)
    output.update(candidate_triangles=15, blocker_triangles=len(blockers),
                  scope='Same-bin disjoint triangles; exact planes and witnesses unchanged.')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
