"""The float64 side proof in exact material exclusion never changes an answer.

`_Index.separated` may only say True when exact rational arithmetic says the same.
The oracle below is `obstruction` exactly as it was before the proof existed.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arc_material_exclusion as exclusion

F = exclusion.F


def original_obstruction(triangles, contexts, old=None):
    """Verbatim pre-filter implementation (eager old planes, no float proof)."""
    indices = [exclusion._index(c) for c in contexts]
    old_index = None if old is None else exclusion._index(old)
    selections = []
    for context, index in zip(contexts, indices):
        count = len(index.triangles)
        active = None if 'active' not in context else np.asarray(context['active'])
        if active is not None and (active.dtype.kind != 'b' or active.shape != (count,)):
            raise ValueError('Material exclusion active mask must be Boolean with one entry per face.')
        excluded = None if 'exclude' not in context else np.asarray(context['exclude'])
        if excluded is not None and (excluded.dtype.kind not in 'iu' or excluded.ndim != 1
                or np.any(excluded < 0) or np.any(excluded >= count)):
            raise ValueError('Material exclusion excluded indices must be a one-dimensional integer array in range.')
        selections.append((active, excluded))
    reference = [(F(0), F(0)), (F(1), F(0)), (F(0), F(1))]
    for candidate, triangle in enumerate(np.asarray(triangles, float)):
        basis = exclusion._rays(triangle)
        bounds = exclusion._box(triangle, rays=basis) if indices or old_index is not None else None
        old_planes = [] if old_index is None else [old_index.planes(int(i), basis)
                                                   for i in old_index.candidates(triangle, bounds=bounds)]
        for context_number, (context, index) in enumerate(zip(contexts, indices)):
            selected = index.candidates(triangle, bounds=bounds)
            active, excluded = selections[context_number]
            if active is not None:
                selected = selected[active[selected]]
            if excluded is not None:
                selected = selected[~np.isin(selected, excluded)]
            for face in selected:
                planes = index.reference_planes(int(face), basis)
                if planes is None:
                    continue
                inside, _ = exclusion._partition(reference, planes)
                if not inside:
                    continue
                parts = [inside]
                for planes in old_planes:
                    parts = [q for p in parts for q in exclusion._partition(p, planes)[1]]
                    if not parts:
                        break
                if parts:
                    polygon = parts[0]
                    return dict(candidate_face=int(candidate), material_context=context_number,
                        material_face=int(face), exact_chart_area=str(exclusion._area(polygon)),
                        chart_polygon=[[str(x), str(y)] for x, y in polygon])
    return None


def unit_rows(points):
    points = np.asarray(points, float)
    return points / np.linalg.norm(points, axis=1)[:, None]


def triangle(xy):
    """Positively wound unit rays near +x; vertices given as chart (y, z) offsets."""
    xy = np.asarray(xy, float)
    rays = unit_rows(np.column_stack((np.ones(3), xy[:, 0], xy[:, 1])))
    if np.linalg.det(rays) < 0:
        rays = rays[[0, 2, 1]]
    return rays


def random_chart_triangle(rng, centre, size):
    while True:
        xy = centre + size*rng.uniform(-1., 1., (3, 2))
        area = (xy[1, 0]-xy[0, 0])*(xy[2, 1]-xy[0, 1]) - (xy[2, 0]-xy[0, 0])*(xy[1, 1]-xy[0, 1])
        if abs(area) > 1e-3*size*size:
            return triangle(xy)


def context(triangles, **values):
    return dict(triangles=np.asarray(triangles), **values)


def corpus(seed, count):
    """Candidates paired with blockers: far, near, overlapping, touching, sharing parts."""
    rng = np.random.default_rng(seed)
    for number in range(count):
        centre = rng.uniform(-.02, .02, 2)
        size = 10.**rng.uniform(-4., -1.3)
        candidate = random_chart_triangle(rng, centre, size)
        blockers = []
        for _ in range(int(rng.integers(3, 9))):
            blockers.append(random_chart_triangle(rng, centre + rng.uniform(-3., 3., 2)*size,
                                                  size*rng.uniform(.3, 2.)))
        kind = number % 4
        if kind == 1:  # share a vertex with the candidate
            blockers[0] = triangle(np.vstack((_chart(candidate[0]), rng.uniform(-1., 1., (2, 2))*size + centre)))
        elif kind == 2:  # share an edge
            blockers[0] = triangle(np.vstack((_chart(candidate[0]), _chart(candidate[1]),
                                              centre + rng.uniform(-1., 1., 2)*size)))
        elif kind == 3:  # a vertex a few ulps from a blocker's edge plane
            moved = candidate.copy()
            for _ in range(int(rng.integers(0, 40))):
                moved[0] = np.nextafter(moved[0], moved[0] + rng.choice([-1., 1.], 3))
            blockers[0] = triangle(np.vstack((_chart(moved[0]), _chart(candidate[2]),
                                              centre + rng.uniform(-1., 1., 2)*size)))
        yield candidate, blockers


def _chart(ray):
    return np.array([ray[1]/ray[0], ray[2]/ray[0]])


def exactly_excluded(index, face, basis):
    return index.reference_planes(int(face), basis) is None


class FloatSeparationTests(unittest.TestCase):
    def test_a_proved_separation_is_always_exactly_an_exclusion(self):
        proved = excluded = total = 0
        for candidate, blockers in corpus(2026, 600):
            index = exclusion._Index(np.asarray(blockers))
            faces = np.arange(len(blockers))
            basis = exclusion._rays(candidate)
            apart = index.separated(faces, candidate)
            for face in faces:
                total += 1
                exact = exactly_excluded(index, face, basis)
                proved += bool(apart[face]); excluded += exact
                if apart[face]:
                    self.assertTrue(exact, (candidate.tolist(), blockers[face].tolist()))
        self.assertGreater(total, 3000)
        # The proof should carry most of the rejections, otherwise it is not worth its cost.
        self.assertGreater(proved, .6*excluded)

    def test_the_adversarial_corpus_would_catch_an_unsound_margin(self):
        """Drop the safety margin and the same corpus must expose a wrong 'proof'."""
        wrong = 0
        with patch.object(exclusion, '_FILTER_MARGIN', 0.):
            for candidate, blockers in corpus(2026, 600):
                index = exclusion._Index(np.asarray(blockers))
                faces = np.arange(len(blockers))
                basis = exclusion._rays(candidate)
                apart = index.separated(faces, candidate)
                wrong += sum(bool(apart[f]) and not exactly_excluded(index, f, basis) for f in faces)
        self.assertGreater(wrong, 0)

    def test_nonfinite_and_zero_scale_input_is_never_proved(self):
        blockers = np.asarray([triangle([(0., 0.), (.01, 0.), (0., .01)])])
        index = exclusion._Index(blockers)
        candidate = triangle([(.5, .5), (.51, .5), (.5, .51)])
        self.assertTrue(index.separated(np.array([0]), candidate)[0])
        with patch.object(exclusion, '_FILTER_FLOOR', 10.):
            self.assertFalse(index.separated(np.array([0]), candidate)[0])
        broken = candidate.copy(); broken[1, 1] = np.nan
        self.assertFalse(index.separated(np.array([0]), broken)[0])
        self.assertEqual(index.separated(np.empty(0, int), candidate).shape, (0,))


class ObstructionEquivalenceTests(unittest.TestCase):
    def check(self, candidates, contexts, old=None):
        expected = original_obstruction(candidates, contexts, old)
        actual = exclusion.obstruction(candidates, contexts, old)
        self.assertEqual(actual, expected)
        return actual

    def test_varied_corpus_gives_identical_witnesses(self):
        found = clear = 0
        for number, (candidate, blockers) in enumerate(corpus(77, 240)):
            occupied = context(blockers)
            old = context([blockers[0]]) if number % 3 == 0 else None
            if number % 5 == 0:
                occupied['active'] = np.arange(len(blockers)) % 4 != 1
            if number % 7 == 0:
                occupied['exclude'] = np.array([len(blockers)-1])
            result = self.check(candidate[None], [occupied], old)
            found += result is not None; clear += result is None
        self.assertGreater(found, 10)
        self.assertGreater(clear, 10)

    def test_several_candidates_and_contexts_keep_first_witness(self):
        rng = np.random.default_rng(5)
        for _ in range(25):
            centre = rng.uniform(-.01, .01, 2)
            candidates = np.asarray([random_chart_triangle(rng, centre + rng.uniform(-.01, .01, 2), .004)
                                     for _ in range(4)])
            contexts = [context([random_chart_triangle(rng, centre + rng.uniform(-.02, .02, 2), .004)
                                 for _ in range(6)]) for _ in range(3)]
            self.check(candidates, contexts, context([candidates[1]]))

    def test_bad_winding_still_raises_when_the_face_is_far_away(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        far = triangle([(.5, .5), (.51, .5), (.5, .51)])
        inverted = far[[0, 2, 1]]
        occupied = context([inverted])
        with patch.object(exclusion._Index, 'candidates', lambda self, t, bounds=None: np.array([0])):
            with self.assertRaisesRegex(ValueError, 'positive original winding'):
                original_obstruction(candidate[None], [occupied])
            with self.assertRaisesRegex(ValueError, 'positive original winding'):
                exclusion.obstruction(candidate[None], [context([inverted])])

    def test_bad_old_footprint_winding_still_raises_without_any_overlap(self):
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        inverted_old = triangle([(0., 0.), (.01, 0.), (0., .01)])[[0, 2, 1]]
        occupied = context([triangle([(.5, .5), (.51, .5), (.5, .51)])])
        with self.assertRaisesRegex(ValueError, 'positive original winding'):
            original_obstruction(candidate[None], [occupied], context([inverted_old]))
        with self.assertRaisesRegex(ValueError, 'positive original winding'):
            exclusion.obstruction(candidate[None], [context(occupied['triangles'])], context([inverted_old]))

    def test_exact_work_drops_but_old_planes_are_only_built_when_needed(self):
        # Same-bin disjoint triangles: every face is a bin false positive.
        candidate = triangle([(0., 0.), (.01, 0.), (0., .01)])
        blockers = [np.roll(triangle([(x, 0.), (x+.01, 0.), (x, .01)]), i % 3, axis=0)
                    for i, x in enumerate(np.linspace(.0105, .0145, 48))]
        old = context([candidate])

        def measure(function):
            occupied = context(blockers)
            function(candidate[None], [occupied], old)  # warm shared per-face exact caches
            with patch.object(exclusion, '_dot', wraps=exclusion._dot) as dots, \
                    patch.object(exclusion._Index, 'planes', autospec=True,
                                 side_effect=exclusion._Index.planes) as planes:
                result = function(candidate[None], [occupied], old)
            return result, dots.call_count, planes.call_count

        expected, before, old_before = measure(original_obstruction)
        actual, after, old_after = measure(exclusion.obstruction)
        with patch.object(exclusion, '_dot', wraps=exclusion._dot) as dots:
            exclusion.obstruction(candidate[None], [context(np.empty((0, 3, 3)))])
        per_triangle = dots.call_count  # the candidate's own box and winding check
        self.assertEqual(actual, expected)
        self.assertIsNone(actual)
        self.assertGreater(before, 100)
        self.assertEqual(after, per_triangle)  # no per-face exact dot products remain
        self.assertGreater(old_before, 0)
        self.assertEqual(old_after, 0)


if __name__ == '__main__':
    unittest.main()
