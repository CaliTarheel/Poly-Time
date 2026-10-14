"""Binary64 stack regions are kept when every cover's regions add up to that cover's own clip."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import collision_interface
import mesh_coverage
from tests.test_burial_clipped_regions import RADIUS, WIDE, clipped, stack
from tests.test_burial_partition_precision import RUN_52_TRIANGLES
from tests.test_small_spherical_faces import decimal_overlap, unit

def cover_areas(regions, areas, count):
    held = np.zeros(count)
    for (_, cover), area in zip(regions, areas):
        held[list(cover)] += area
    return held


def standing(triangles, regions, areas):
    count = len(triangles)-1
    return burial_depth._standing_regions(triangles, 0, np.arange(count), np.arange(1, count+1), regions, areas,
                                          mesh_coverage._polygon_area(triangles[0], RADIUS), RADIUS)


def ordered_state(triangles):
    """Every face its own sheet and plate, in one vertical order, with independent overlap areas."""
    count = len(triangles)
    areas = np.array([mesh_coverage._polygon_area(triangle, RADIUS) for triangle in triangles])
    contacts = [dict(id=1000*top+under, top_sheet=top, under_sheet=under)
                for top in range(1, count) for under in range(top)]
    state = SimpleNamespace(
        material_surface=dict(vertices=triangles.reshape(-1, 3).copy(), faces=np.arange(3*count).reshape(-1, 3),
                              area_km2=areas, radius_km=RADIUS),
        parcel_collision_sheet=np.arange(count), parcel_plate=np.arange(count),
        parcel_burial_myr=np.full(count, 100.), parcel_root_age_myr=np.full(count, 100.),
        collision_contacts=contacts)
    cached = np.array([decimal_overlap(triangles[0], cover) for cover in triangles[1:]])
    return state, (np.arange(1, count), np.zeros(count-1, int), cached, 1000*np.arange(1, count))


class ConvexRepresentedTests(unittest.TestCase):
    QUAD = unit([[1., -.01, -.01], [1., .01, -.01], [1., .01, .01], [1., -.01, .01]])

    def test_sound_polygon_is_returned_as_it_is(self):
        self.assertIs(burial_depth._convex_represented(self.QUAD), self.QUAD)

    def test_vertex_inside_the_hull_is_removed_without_a_distance_threshold(self):
        middle = unit(self.QUAD[1]+self.QUAD[2])
        inward = unit(middle+1e-15*(self.QUAD.mean(axis=0)-middle))
        self.assertFalse(np.array_equal(inward, middle))
        dented = np.insert(self.QUAD, 2, inward, axis=0)
        convex = burial_depth._convex_represented(dented)
        self.assertEqual(convex.tobytes(), self.QUAD.tobytes())
        # A vertex that turns left, however slightly, belongs to the region and stays.
        outward = unit(middle-1e-9*(self.QUAD.mean(axis=0)-middle))
        bulged = np.insert(self.QUAD, 2, outward, axis=0)
        self.assertIs(burial_depth._convex_represented(bulged), bulged)

    def test_vertices_a_few_ulps_apart_leave_a_positive_polygon(self):
        twin = np.nextafter(self.QUAD[2], self.QUAD[0])
        doubled = np.insert(self.QUAD, 3, twin, axis=0)
        convex = burial_depth._convex_represented(doubled)
        self.assertIsNotNone(convex)
        self.assertTrue(burial_depth._positive_binary64_winding(convex))
        self.assertTrue(np.isfinite(collision_interface.rotation_metric(convex, RADIUS)).all())
        self.assertAlmostEqual(mesh_coverage._polygon_area(convex, RADIUS)/mesh_coverage._polygon_area(self.QUAD, RADIUS),
                               1., places=12)

    def test_region_without_a_represented_interior_is_none(self):
        self.assertIsNone(burial_depth._convex_represented(self.QUAD[::-1]))
        line = unit([[1., -.01, 0.], [1., 0., 0.], [1., .01, 0.]])
        self.assertIsNone(burial_depth._convex_represented(line))
        collapsed = np.array([
            [0.4429049499273482, -0.7461530375224076, -0.4970823371695456],
            [0.44425351651949885, -0.7399922989763484, -0.5050249602898433],
            [0.4442535165194989, -0.7399922989763484, -0.5050249602898433],
            [0.4429049499273482, -0.7461530375224077, -0.49708233716954553],
        ])
        convex = burial_depth._convex_represented(collapsed)
        if convex is not None:
            self.assertTrue(burial_depth._positive_binary64_winding(convex))
            self.assertTrue(np.isfinite(collision_interface.rotation_metric(convex, RADIUS)).all())


class StandingRegionsTests(unittest.TestCase):
    def setUp(self):
        burial_depth.geometry_cache_reset()

    def tearDown(self):
        burial_depth.geometry_cache_reset()

    def test_doubted_regions_stand_when_every_cover_closes(self):
        triangles = stack(0, 40)
        reference = mesh_coverage._polygon_area(triangles[0], RADIUS)
        plain, plain_areas = clipped(triangles)
        self.assertTrue(any(burial_depth._uncertain_winding(polygon) for polygon, _ in plain))
        self.assertFalse(all(burial_depth._positive_binary64_winding(polygon) for polygon, _ in plain))
        with patch.object(burial_depth, '_precise_partition', side_effect=AssertionError('the regions stand')):
            regions, areas, error = burial_depth._partition_face_uncached(
                triangles, 0, np.arange(40), np.arange(1, 41), reference, RADIUS)
        self.assertLess(error, 2e-10)
        independent = np.array([decimal_overlap(triangles[0], cover) for cover in triangles[1:]])
        np.testing.assert_allclose(cover_areas(regions, areas, 40), independent, rtol=0., atol=reference*2e-13)
        self.assertLess(len(regions), len(plain))
        for (polygon, _), area in zip(regions, areas):
            # Every region handed on is positively wound in exact arithmetic: the interface
            # metric integrates it without meeting an inconsistent fan. None is a strip that
            # rounding along its own perimeter could have made.
            self.assertTrue(burial_depth._positive_binary64_winding(polygon))
            self.assertGreater(area, burial_depth._rounding_sweep(polygon, RADIUS))
            metric = collision_interface.rotation_metric(polygon, RADIUS)
            self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(), -2e-11*np.trace(metric))

    def test_standing_regions_agree_with_exact_clipping(self):
        for triangles in (stack(1, 24), RUN_52_TRIANGLES):
            count = len(triangles)-1
            reference = mesh_coverage._polygon_area(triangles[0], RADIUS)
            arguments = (triangles, 0, np.arange(count), np.arange(1, count+1), reference, RADIUS)
            with patch.object(burial_depth, '_precise_partition', side_effect=AssertionError('the regions stand')):
                regions, areas, _ = burial_depth._partition_face_uncached(*arguments)
            with patch.object(burial_depth, '_standing_regions', return_value=None):
                exact_regions, exact_areas, _ = burial_depth._partition_face_uncached(*arguments)
            totals = [{}, {}]
            for table, (some, values) in zip(totals, ((regions, areas), (exact_regions, exact_areas))):
                for (_, cover), area in zip(some, values):
                    table[cover] = table.get(cover, 0.)+float(area)
            for cover in set(totals[0]) | set(totals[1]):
                # A zero-width strip between two covers may exist in one and not in the other.
                self.assertAlmostEqual(totals[0].get(cover, 0.), totals[1].get(cover, 0.), delta=reference*2e-13)

    def test_52_myr_face_stands_with_its_cached_overlap_areas(self):
        regions, areas, error = burial_depth.partition_face(
            RUN_52_TRIANGLES, 0, np.arange(2), np.array([1, 2]), 5673.2372536289995, RADIUS)
        self.assertTrue(all(type(polygon) is np.ndarray for polygon, _ in regions))
        # Two uncovered regions and one under each upper face. The strip under both, which
        # only rounding puts there, is not handed to consumers that need a vertical order.
        self.assertEqual(sorted(cover for _, cover in regions), [(), (), (0,), (1,)])
        self.assertLess(error, 2e-10)
        np.testing.assert_allclose(cover_areas(regions, areas, 2), [139.5918501186568, 161.6081539824471],
                                   rtol=2e-9, atol=5673.2372536289995*2e-11)
        for polygon, _ in regions:
            self.assertTrue(np.isfinite(collision_interface.rotation_metric(polygon, RADIUS)).all())

    def test_cover_that_does_not_close_is_left_to_exact_clipping(self):
        triangles = stack(2, 12)
        regions, areas = clipped(triangles)
        self.assertIsNotNone(standing(triangles, regions, areas))
        largest = max((index for index, (_, cover) in enumerate(regions) if cover), key=lambda index: areas[index])
        lost = [region for index, region in enumerate(regions) if index != largest]
        self.assertIsNone(standing(triangles, lost, np.delete(areas, largest)))
        # Area moved from one cover to another keeps the whole face closed and is still refused.
        other = next(pair for pair in range(12) if pair not in regions[largest][1])
        moved = list(regions)
        moved[largest] = (regions[largest][0], (other,))
        self.assertIsNone(standing(triangles, moved, areas))

    def test_region_without_an_interior_is_left_out_only_within_the_cover_allowance(self):
        triangles = stack(2, 12)
        regions, areas = clipped(triangles)
        doubted = [index for index, (polygon, _) in enumerate(regions) if burial_depth._uncertain_winding(polygon)]
        self.assertTrue(doubted)
        with patch.object(burial_depth, '_convex_represented', return_value=None):
            kept = standing(triangles, regions, areas)
            if kept is not None:
                reference = mesh_coverage._polygon_area(triangles[0], RADIUS)
                self.assertLess(sum(areas[index] for index in doubted if regions[index][1]), 12*reference*1e-11)
        # Were a sound region of real size to lose its polygon, its cover would not close.
        with patch.object(burial_depth, '_uncertain_winding', side_effect=lambda polygon: len(polygon) > 3):
            with patch.object(burial_depth, '_convex_represented', return_value=None):
                self.assertIsNone(standing(triangles, regions, areas))

    def test_narrow_lower_face_is_left_to_exact_clipping(self):
        # Resolved source triangles, but one cut's rounding along the long perimeter is no
        # longer small beside the lower face: its clip cannot serve as the reference.
        narrow = ((1., -.02, 0.), (1., .02, 0.), (1., 0., .0006))
        cover = unit([[1., -.03, -.01], [1., .03, -.01], [1., 0., .03]])
        triangles = np.array([unit(narrow), cover])
        self.assertFalse(mesh_coverage._triangle_area_condition(triangles).any())
        regions, areas = clipped(triangles)
        self.assertIsNone(standing(triangles, regions, areas))
        wide = np.array([unit(WIDE), cover])
        self.assertIsNotNone(standing(wide, *clipped(wide)))

    def test_reversed_source_is_left_to_exact_clipping_and_rejected_there(self):
        triangles = stack(0, 6)
        triangles[3] = triangles[3, ::-1]
        regions, areas = clipped(triangles)
        self.assertIsNone(standing(triangles, regions, areas))

    def test_burial_and_interface_consume_standing_regions(self):
        triangles = stack(3, 16)
        state, pairs = ordered_state(triangles)
        thickness = np.linspace(8., 30., len(triangles))
        outcomes = []
        for forced in (False, True):
            burial_depth.geometry_cache_reset()
            exact = (patch.object(burial_depth, '_standing_regions', return_value=None) if forced else
                     patch.object(burial_depth, '_precise_partition', side_effect=AssertionError('the regions stand')))
            with exact, patch.object(collision_interface.eclogite_sink, '_depth_pairs', return_value=pairs):
                eligible, scope, attribution = burial_depth.integrate(
                    state, thickness, pairs, depth_km=20., heating_delay_myr=5.)
                rows = collision_interface.regions(state)
            contact = {}
            for row in rows:
                key = (row['top_face'], row['under_face'])
                contact[key] = contact.get(key, 0.)+row['area_km2']
            metric = sum(row['metric_m2'] for row in rows)
            outcomes.append((eligible, attribution['eligible_pair_volume_km3'], contact, metric, scope))
        (eligible, volume, contact, metric, scope), (exact_eligible, exact_volume, exact_contact, exact_metric, _) = outcomes
        self.assertLess(scope['maximum_region_area_relative_error'], 2e-10)
        # Far inside the 0.1% the model asks of its results.
        np.testing.assert_allclose(eligible, exact_eligible, rtol=1e-11)
        np.testing.assert_allclose(volume, exact_volume, rtol=1e-9, atol=1e-9)
        self.assertEqual(set(contact), set(exact_contact))
        reference = mesh_coverage._polygon_area(triangles[0], RADIUS)
        for key in contact:
            self.assertAlmostEqual(contact[key], exact_contact[key], delta=reference*2e-11)
        np.testing.assert_allclose(metric, exact_metric, rtol=1e-9, atol=1e-9*np.trace(exact_metric))


if __name__ == '__main__':
    unittest.main()
