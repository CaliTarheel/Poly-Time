"""Initial continental crust coverage is a checked request, not an approximate disc count."""
import unittest

import numpy as np

from poly import sphere
from poly.build import CONTINENT_FRACTION_TOLERANCE, voronoi_world


class InitialCoverageTests(unittest.TestCase):
    def assert_partition(self, world):
        for plate, region in world.plate_polygons().items():
            pieces = [c.shape for c in world.crust if c.plate == plate]
            covered = sphere.union(*pieces)
            self.assertLess(sphere.area_km2(sphere.difference(region, covered)), 1e-3)
            self.assertLess(sphere.area_km2(sphere.difference(covered, region)), 1e-3)
            # Sum versus union detects overlaps between continent and ocean.
            self.assertLess(abs(sum(sphere.area_km2(p) for p in pieces) - sphere.area_km2(covered)), 1e-3)

    def test_requested_coverage_is_met_for_several_seeds_and_high_fractions(self):
        for seed in (1, 3, 5):
            for fraction in (.3, .6, .65):
                with self.subTest(seed=seed, fraction=fraction):
                    world, _ = voronoi_world(count=8, seed=seed, continent_fraction=fraction)
                    achieved = world.areas()['continent'] / sphere.SPHERE_KM2
                    self.assertLessEqual(abs(achieved - fraction), CONTINENT_FRACTION_TOLERANCE)
                    self.assert_partition(world)

    def test_coarse_boundaries_and_a_different_plate_count_keep_the_target(self):
        world, _ = voronoi_world(count=4, seed=3, spacing_km=500., continent_fraction=.65)
        self.assertAlmostEqual(world.areas()['continent'] / sphere.SPHERE_KM2, .65,
                               delta=CONTINENT_FRACTION_TOLERANCE)
        self.assert_partition(world)

    def test_the_same_seed_reproduces_geometry_and_rotations(self):
        first, first_rotations = voronoi_world(seed=3, continent_fraction=.6)
        second, second_rotations = voronoi_world(seed=3, continent_fraction=.6)
        self.assertEqual([(c.plate, c.kind) for c in first.crust],
                         [(c.plate, c.kind) for c in second.crust])
        for a, b in zip(first.crust, second.crust):
            self.assertLess(sphere.area_km2(sphere.difference(a.shape, b.shape)), 1e-6)
            self.assertLess(sphere.area_km2(sphere.difference(b.shape, a.shape)), 1e-6)
        for plate in first_rotations:
            np.testing.assert_array_equal(first_rotations[plate], second_rotations[plate])

    def test_zero_and_one_are_all_ocean_and_all_continent(self):
        for fraction, kind in ((0., 'ocean'), (1., 'continent')):
            with self.subTest(fraction=fraction):
                world, _ = voronoi_world(continent_fraction=fraction)
                self.assertEqual({c.kind for c in world.crust}, {kind})
                self.assertAlmostEqual(world.areas()[kind] / sphere.SPHERE_KM2, 1., places=10)
                self.assert_partition(world)

    def test_continent_count_mode_still_places_the_requested_discs(self):
        world, _ = voronoi_world(count=6, seed=5, continents=2)
        continents = [c for c in world.crust if c.kind == 'continent']
        self.assertEqual(len(continents), 2)
        self.assertEqual(len({c.plate for c in continents}), 2)
        self.assert_partition(world)

    def test_invalid_fractions_and_mixed_modes_fail_explicitly(self):
        for fraction in (-.1, 1.01, np.nan, np.inf, -np.inf, None, 'many'):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, 'continent_fraction.*finite.*\\[0, 1\\]'):
                    voronoi_world(continent_fraction=fraction)
        with self.assertRaisesRegex(ValueError, 'Choose continents or continent_fraction'):
            voronoi_world(continents=2, continent_fraction=.6)


if __name__ == '__main__':
    unittest.main()
