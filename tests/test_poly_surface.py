"""Crust identity and emergence must remain independent in maps and samples."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from poly import regions, render, sphere, surface
from poly.world import Crust
from tests.test_poly_evolve import disc


def island_world(thickness=30., radius=5.):
    island = sphere.polygon(disc((0., 0.), radius))
    return regions.RegionWorld(
        [Crust(1, 'continent', island, thickness_km=thickness),
         Crust(1, 'ocean', sphere.difference(regions.FULL, island), born_myr=-50.)],
        {1: np.eye(3)}, {1: regions.FULL}, [])


class SurfaceTests(unittest.TestCase):
    def test_continental_crust_can_be_underwater(self):
        world = island_world()
        sampled = surface.sample(world, sphere.from_lonlat([0., 90.], [0., 0.]))
        np.testing.assert_array_equal(sampled.continental_crust, [True, False])
        np.testing.assert_array_equal(sampled.land, [False, False])
        self.assertAlmostEqual(sampled.elevation_km[0], -.75)
        self.assertLess(sampled.elevation_km[1], -2.6)

    def test_sea_level_changes_emergence_without_changing_crust(self):
        world = island_world(40.)
        points = sphere.from_lonlat([0., 90.], [0., 0.])
        low = surface.sample(world, points)
        high = surface.sample(world, points, sea_level_km=1.)
        self.assertTrue(low.land[0])
        self.assertFalse(high.land[0])
        np.testing.assert_array_equal(low.elevation_km, high.elevation_km)
        np.testing.assert_array_equal(low.crust_kind, high.crust_kind)
        self.assertEqual(world.crust[0].thickness_km, 40.)

    def test_sampling_uses_plate_frame(self):
        world = island_world()
        rot = sphere.rotation(np.array([0., 0., 1.1]))
        world.frames[1] = rot
        point = sphere.rotate(sphere.from_lonlat(0., 0.), rot)
        sampled = surface.sample(world, point[None])
        self.assertTrue(sampled.continental_crust[0])
        self.assertAlmostEqual(sampled.elevation_km[0], -.75)

    def test_uncovered_points_are_not_reported_as_water(self):
        world = island_world()
        world.crust = world.crust[:1]
        sampled = surface.sample(world, sphere.from_lonlat([90.], [0.]))
        self.assertTrue(np.isnan(sampled.elevation_km[0]))
        self.assertFalse(sampled.land[0])
        self.assertFalse(sampled.water[0])

    def test_submerged_tiny_continent_does_not_receive_land_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.png'
            world = island_world(20., radius=.01)
            render.draw(world, path, width=120, boundaries=False)
            water = np.asarray(Image.open(path))
            self.assertTrue(np.all(water[:, :, 2] > water[:, :, 0]))
            world.crust[0].thickness_km = 40.
            render.draw(world, path, width=120, boundaries=False)
            land = np.asarray(Image.open(path))
            self.assertTrue(np.any(land[:, :, 0] > land[:, :, 2]))

    def test_nonfinite_sea_level_is_rejected(self):
        with self.assertRaises(ValueError):
            surface.sample(island_world(), sphere.from_lonlat([0.], [0.]), np.nan)

    def test_sampling_does_not_insert_default_frames(self):
        from poly.build import cap_world
        world, _ = cap_world()
        self.assertEqual(world.frames, {})
        surface.sample(world, sphere.from_lonlat([0., 90.], [0., 0.]))
        self.assertEqual(world.frames, {})


if __name__ == '__main__':
    unittest.main()
