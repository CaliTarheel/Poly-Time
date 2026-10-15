"""The polygon core: sphere geometry, the boundary network and the closed crust ledger."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poly import sphere
from poly.build import cap_world, voronoi_world
from poly.topology import _cut_loops


class SphereTests(unittest.TestCase):
    def test_a_ring_and_its_reverse_are_complements(self):
        ring = sphere.from_lonlat([0., 120., 240.], [10., 10., 10.])
        small, large = sphere.polygon(ring), sphere.polygon(ring[::-1])
        self.assertLess(sphere.area_km2(small), .5 * sphere.SPHERE_KM2)
        self.assertAlmostEqual((sphere.area_km2(small) + sphere.area_km2(large)) / sphere.SPHERE_KM2, 1., places=12)

    def test_holes_round_trips_and_rotation_keep_area(self):
        shell = sphere.from_lonlat([0., 120., 240.], [10., 10., 10.])
        hole = sphere.from_lonlat([0., 120., 240.], [60., 60., 60.])[::-1]
        holed = sphere.polygon(shell, [hole])
        expected = sphere.area_km2(sphere.polygon(shell)) - sphere.area_km2(sphere.polygon(hole[::-1]))
        self.assertAlmostEqual(sphere.area_km2(holed) / expected, 1., places=12)
        for shape in (holed, sphere.polygon(shell[::-1])):
            rings = sphere.polygons(shape)
            rebuilt = sphere.union(*[sphere.polygon(r[0], r[1:]) for r in rings])
            self.assertAlmostEqual(sphere.area_km2(rebuilt) / sphere.area_km2(shape), 1., places=10)
        matrix = sphere.rotation(sphere.unit([1., 2., 3.]) * .7)
        self.assertAlmostEqual(sphere.area_km2(sphere.rotate_geography(holed, matrix)) / sphere.area_km2(holed), 1., places=10)

    def test_clipping_conserves_area_even_for_a_thin_strip(self):
        lon = np.degrees(np.linspace(0., 3000. / sphere.RADIUS_KM, 50))
        width = math.degrees(1. / sphere.RADIUS_KM)
        strip = sphere.polygon(sphere.from_lonlat(np.r_[lon, lon[::-1]], np.r_[np.zeros(50), np.full(50, width)]))
        cut = sphere.polygon(sphere.from_lonlat([5., 10., 10., 5.], [-1., -1., 1., 1.]))
        pieces = sphere.area_km2(sphere.intersection(strip, cut)) + sphere.area_km2(sphere.difference(strip, cut))
        self.assertAlmostEqual(pieces / sphere.area_km2(strip), 1., places=9)
        self.assertAlmostEqual(sphere.area_km2(strip), 3000., delta=1.)

    def test_rings_that_touch_themselves_split_into_loops(self):
        points = sphere.from_lonlat([0., 1., 2., 1., 3., .5], [0., 0., 1., 0., 2., 1.])
        self.assertEqual([len(loop) for loop in sphere.simple_loops(points)], [4])
        bow = sphere.from_lonlat([0., 2., 2., 1., 1., 3.], [0., 0., 1., 1., -1., -1.])
        self.assertEqual(len(_cut_loops(bow)), 4)


class LedgerTests(unittest.TestCase):
    def test_a_cap_spinning_about_its_own_centre_makes_and_destroys_almost_nothing(self):
        centre = sphere.from_lonlat(30., 20.)
        world, omegas = cap_world(centre_lonlat=(30., 20.), radius_deg=30., cap_omega=centre * 40. / 6371.)
        self.assertEqual({s.kind for s in world.network.sections.values()}, {'transform'})
        for _ in range(3):
            entry = world.step(omegas, 1.)
        # Sections are great-circle chords, not the small circle a transform follows, so sliding
        # them past each other leaves slivers of about their sag. They are booked as slivers and
        # stay well under 1% of the area the boundary sweeps (perimeter x 20 km/Myr at 30 degrees).
        swept = 2. * math.pi * sphere.RADIUS_KM * math.sin(math.radians(30.)) * 40. * math.sin(math.radians(30.))
        self.assertLess(sum(entry['gained'].values()), .01 * swept)
        self.assertLess(sum(entry['lost'].values()), .01 * swept)
        self.assertEqual(entry['gained']['ridge'] + entry['lost']['subducted'], 0.)
        self.assertLess(abs(entry['tiling_error_km2']), 1e-3)

    def test_a_moving_cap_makes_ridge_and_subducts_and_the_books_close(self):
        world, omegas = cap_world(cap_omega=np.array([0., 0., 1.]) * 40. / 6371.)
        self.assertEqual({s.kind for s in world.network.sections.values()}, {'ridge', 'trench', 'transform'})
        before = {p: sphere.area_km2(shape) for p, shape in world.plate_polygons().items()}
        for _ in range(5):
            entry = world.step(omegas, 1.)
        # About 2 x 40 km/Myr across the cap's 8,900 km wide trailing edge, split between both plates.
        self.assertGreater(entry['gained']['ridge'], 2.5e5)
        self.assertGreater(entry['lost']['subducted'], 2.5e5)
        self.assertLess(abs(entry['balance_error_km2']), 1e-3)
        self.assertLess(entry['crust_vs_plate_error_km2'], 1e-3)
        after = world.plate_polygons()
        self.assertLess(sphere.area_km2(after[1]), before[1])  # the cap is consumed at its trench
        kinds = world.areas()
        self.assertAlmostEqual(sum(kinds.values()) / sphere.SPHERE_KM2, 1., places=9)

    def test_a_random_world_tiles_the_sphere_and_keeps_its_continents(self):
        world, omegas = voronoi_world(count=6, seed=5, continents=2)
        continent = world.areas()['continent']
        self.assertAlmostEqual(sum(sphere.area_km2(s) for s in world.plate_polygons().values()) / sphere.SPHERE_KM2, 1., places=9)
        for _ in range(3):
            entry = world.step(omegas, 1.)
            self.assertLess(abs(entry['balance_error_km2']), 1e-3)
        # No step makes continent; only ridges and openings make crust, and that crust is ocean.
        self.assertLessEqual(world.areas().get('continent', 0.), continent + 1.)


class RenderTests(unittest.TestCase):
    def test_a_map_is_written(self):
        from poly import render
        world, _ = cap_world()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'map.png'
            plates = render.draw(world, path, width=120)
            self.assertTrue(path.exists())
            self.assertEqual(set(np.unique(plates)), {1, 2})


if __name__ == '__main__':
    unittest.main()
