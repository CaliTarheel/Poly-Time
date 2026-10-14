"""Regional accuracy must survive cancellation in whole-face area closure."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import collision_interface
import mesh_coverage
from tests.test_burial_partition_precision import independent_gnomonic_regions
from tests.test_small_spherical_faces import decimal_overlap, unit


# Source rays captured at the 150->151 Myr step: lower face 2436, then
# upper faces 1530, 1896, 2435, 3289, 4057. The ordinary partition passes
# whole-face closure and winding while cover 1896 misses its area contract.
TRIANGLES = np.array([
    [[.14632458798954137, .3920279472377368, -.9082418199655075],
     [.15443261963927243, .3831468314313761, -.9106860444497042],
     [.15443256127855734, .38314676427218325, -.9106860826018484]],
    [[.14967709997561576, .39068656914359556, -.9082735108070125],
     [.1567160592305791, .3532342329476554, -.9223153763508007],
     [.09819361404140678, .3464118970339224, -.9329291568789382]],
    [[.14967709997561576, .39068656914359556, -.9082735108070125],
     [.09819361404140678, .3464118970339224, -.9329291568789382],
     [.08471026718453167, .42037879047327165, -.9033857665215644]],
    [[.1527645016138371, .38330952286428926, -.9108989058771708],
     [.1873258993239045, .4008898110547446, -.8967699631650147],
     [.15767110207215687, .37790689028522473, -.9123191359641114]],
    [[.17361464473253901, .40287797406342113, -.8986363520072777],
     [.17882253275581506, .3696426476707669, -.9118041537532832],
     [.1440453984134726, .37957990608610936, -.9138763691504254]],
    [[.14967709997561576, .39068656914359556, -.9082735108070125],
     [.18189346876345444, .3750795072559282, -.9089720178629529],
     [.1567160592305791, .3532342329476554, -.9223153763508007]],
])
REFERENCE_AREA = .023706547049679352
RADIUS = 6371.
CACHED_AREAS = np.array([.0038554774794334053, .0016738456472154444,
                         .003562889050279081, .012253006258459143, .01817722392008181])


def ordinary_partition():
    """Unrefined clipping to demonstrate why the old admission checks miss this case."""
    regions = [(TRIANGLES[0], ())]
    for pair, triangle in enumerate(TRIANGLES[1:]):
        next_regions = []
        for polygon, cover in regions:
            inside, outside = burial_depth._split(
                polygon, mesh_coverage._triangle_planes(triangle), RADIUS, triangle)
            next_regions.extend((piece, cover) for piece in outside)
            if mesh_coverage._polygon_area(inside, RADIUS) > 0.:
                next_regions.append((inside, cover+(pair,)))
        regions = next_regions
    areas = np.array([mesh_coverage._polygon_area(polygon, RADIUS) for polygon, _ in regions])
    return regions, areas


def represented_pairs(regions, areas):
    return np.array([sum(weight for (_, cover), weight in zip(regions, areas) if pair in cover)
                     for pair in range(5)])


def source_state():
    count = len(TRIANGLES)
    areas = np.array([mesh_coverage._polygon_area(triangle, RADIUS) for triangle in TRIANGLES])
    areas[0] = REFERENCE_AREA
    surface = dict(vertices=TRIANGLES.reshape(-1, 3).copy(), faces=np.arange(3*count).reshape(-1, 3),
                   area_km2=areas, radius_km=RADIUS)
    # A fully ordered stack gives every overlapping region a unique adjacent
    # upper sheet; distinct owners make all retained interfaces carry work.
    contacts = [dict(id=10*top+under, top_sheet=top, under_sheet=under)
                for top in range(1, count) for under in range(top)]
    state = SimpleNamespace(material_surface=surface, parcel_collision_sheet=np.arange(count),
                            parcel_plate=np.arange(count), parcel_burial_myr=np.full(count, 100.),
                            parcel_root_age_myr=np.full(count, 100.), collision_contacts=contacts)
    pairs = (np.arange(1, count), np.zeros(count-1, int), CACHED_AREAS.copy(),
             10*np.arange(1, count))
    return state, pairs


class BurialRegionalPrecisionTests(unittest.TestCase):
    def setUp(self):
        burial_depth.geometry_cache_reset()

    def tearDown(self):
        burial_depth.geometry_cache_reset()

    def test_whole_face_closure_and_positive_winding_do_not_bound_pair_error(self):
        regions, areas = ordinary_partition()
        self.assertLess(abs(areas.sum()-REFERENCE_AREA)/REFERENCE_AREA, 2e-10)
        self.assertFalse(any(burial_depth._uncertain_winding(polygon) for polygon, _ in regions))
        represented = represented_pairs(regions, areas)
        self.assertGreater(abs(represented[1]-CACHED_AREAS[1]),
                           2e-9*CACHED_AREAS[1]+REFERENCE_AREA*2e-11)
        # material_overlaps normalizes source vertices through _triangles.
        # The resulting rays differ from burial's stored rays by roundoff;
        # keep that distinction in the independent cache oracle.
        normalized = unit(TRIANGLES)
        independent = np.array([decimal_overlap(normalized[0], upper) for upper in normalized[1:]])
        np.testing.assert_allclose(CACHED_AREAS, independent, rtol=2e-13, atol=0.)

    def test_shared_partition_refines_regions_against_independent_geometry(self):
        original = TRIANGLES.copy()
        with patch.object(burial_depth, '_precise_partition', wraps=burial_depth._precise_partition) as precise:
            regions, areas, error = burial_depth.partition_face(
                TRIANGLES, 0, np.arange(5), np.arange(1, 6), REFERENCE_AREA, RADIUS)
        precise.assert_called_once()
        expected = independent_gnomonic_regions(TRIANGLES)
        actual = {}
        for (_, cover), area in zip(regions, areas):
            actual[cover] = actual.get(cover, 0.)+float(area)
        self.assertEqual(set(actual), set(expected))
        np.testing.assert_allclose([actual[cover] for cover in sorted(expected)],
                                   [expected[cover] for cover in sorted(expected)], rtol=2e-13, atol=1e-18)
        independent_pairs = np.array([decimal_overlap(TRIANGLES[0], upper) for upper in TRIANGLES[1:]])
        np.testing.assert_allclose(represented_pairs(regions, areas), independent_pairs, rtol=2e-13, atol=0.)
        np.testing.assert_allclose(represented_pairs(regions, areas), CACHED_AREAS,
                                   rtol=2e-9, atol=REFERENCE_AREA*2e-11)
        self.assertLess(error, 2e-10)
        self.assertGreater(error, 0.)  # Independent areas are never rescaled to force closure.
        np.testing.assert_array_equal(TRIANGLES, original)

    def test_burial_threshold_and_adjacent_interface_consume_same_regions(self):
        state, pairs = source_state()
        thickness = np.array([30., 10., 11., 12., 13., 14.])
        expected = independent_gnomonic_regions(TRIANGLES)
        volume = sum(area*np.clip(thickness[0]+sum(thickness[pair+1] for pair in cover)-20.,
                                  0., thickness[0]) for cover, area in expected.items())
        eligible, scope, attribution = burial_depth.integrate(
            state, thickness, pairs, depth_km=20., heating_delay_myr=5.)
        self.assertAlmostEqual(eligible[0]*REFERENCE_AREA, volume, delta=2e-13)
        self.assertLess(scope['maximum_region_area_relative_error'], 2e-10)
        self.assertTrue(np.isfinite(attribution['eligible_pair_volume_km3']).all())
        with patch.object(collision_interface.eclogite_sink, '_depth_pairs', return_value=pairs):
            rows = collision_interface.regions(state)
        self.assertTrue(rows)
        expected_contact = sum(area for cover, area in expected.items() if cover)
        self.assertAlmostEqual(sum(row['area_km2'] for row in rows), expected_contact, delta=2e-14)
        for row in rows:
            self.assertNotEqual(row['top_owner'], row['under_owner'])
            self.assertGreaterEqual(np.linalg.eigvalsh(row['metric_m2']).min(),
                                    -2e-11*np.trace(row['metric_m2']))
            self.assertAlmostEqual(np.trace(row['metric_m2'])/2e6, row['area_km2'],
                                   delta=2e-9*row['area_km2']+REFERENCE_AREA*2e-11)

    def test_refinement_still_rejects_a_corrupted_pair_area(self):
        state, pairs = source_state()
        pairs[2][1] += 1e-8
        with self.assertRaisesRegex(ValueError, 'cached overlap geometry'):
            burial_depth.integrate(state, np.full(6, 30.), pairs,
                                   depth_km=20., heating_delay_myr=5.)
        with patch.object(collision_interface.eclogite_sink, '_depth_pairs', return_value=pairs):
            with self.assertRaisesRegex(ValueError, 'cached overlap areas'):
                collision_interface.regions(state)

    def test_thin_region_refines_even_when_both_source_triangles_are_resolved(self):
        triangles = unit([[[1., -.03, 0.], [1., .03, 0.], [1., 0., .03]],
                          [[1., .03, 1e-10], [1., -.03, 1e-10], [1., 0., -.03]]])
        self.assertFalse(mesh_coverage._triangle_area_condition(triangles).any())
        inside, outside = burial_depth._split(
            triangles[0], mesh_coverage._triangle_planes(triangles[1]), RADIUS, triangles[1])
        ordinary = outside+[inside]
        self.assertFalse(any(burial_depth._uncertain_winding(polygon) for polygon in ordinary))
        reference = mesh_coverage._polygon_area(triangles[0], RADIUS)
        self.assertLess(abs(sum(mesh_coverage._polygon_area(polygon, RADIUS)
                                for polygon in ordinary)-reference)/reference, 2e-10)
        with patch.object(burial_depth, '_precise_partition', wraps=burial_depth._precise_partition) as precise:
            regions, areas, _ = burial_depth.partition_face(
                triangles, 0, np.array([0]), np.array([1]), reference, RADIUS)
        precise.assert_called_once()
        covered = sum(weight for (_, cover), weight in zip(regions, areas) if cover)
        self.assertAlmostEqual(covered/decimal_overlap(*triangles), 1., delta=2e-13)

    def test_resolved_partition_keeps_ordinary_path(self):
        triangles = unit([[[1., -.03, -.02], [1., .03, -.02], [1., .02, .03]],
                          [[1., -.02, -.04], [1., .05, 0.], [1., -.02, .04]]])
        area = mesh_coverage._polygon_area(triangles[0], RADIUS)
        with patch.object(burial_depth, '_precise_partition', side_effect=AssertionError('resolved geometry')):
            regions, areas, error = burial_depth.partition_face(
                triangles, 0, np.array([0]), np.array([1]), area, RADIUS)
        self.assertTrue(all(type(polygon) is np.ndarray for polygon, _ in regions))
        self.assertLess(error, 2e-10)
        self.assertAlmostEqual(sum(weight for (_, cover), weight in zip(regions, areas) if cover),
                               decimal_overlap(*triangles), delta=2e-8)


if __name__ == '__main__':
    unittest.main()
