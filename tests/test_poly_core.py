"""The polygon core: sphere geometry, the boundary network and the closed crust ledger."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_a_hole_folded_over_itself_is_cut_out_once(self):
        # Read back after a rotation, a hole can lap the same patch both ways round; it is still
        # one hole, not a shell that the polygon is intersected with.
        shell = sphere.from_lonlat([0., 30., 30., 0.], [0., 0., 30., 30.])
        patch = sphere.from_lonlat([10., 12., 12., 10.], [10., 10., 12., 12.])
        hole = np.vstack((patch, patch[:1], patch[::-1], patch[:1], patch[::-1]))
        whole = sphere.area_km2(sphere.polygon(shell))
        cut = sphere.area_km2(sphere.polygon(patch))
        region, _ = sphere.part([shell, hole], whole_km2=whole - cut)
        self.assertAlmostEqual(sphere.area_km2(region) / (whole - cut), 1., places=6)
        # The same when S2 will not take the polygon whole and it is rebuilt loop by loop.
        real = sphere.polygon_robust

        def refuse_holes(ring, holes=(), attempts=24, minimum_km2=1e-6):
            if len(holes):
                raise ValueError('refused')
            return real(ring, holes, attempts, minimum_km2)
        with mock.patch.object(sphere, 'polygon_robust', refuse_holes):
            region, _ = sphere.part([shell, hole], whole_km2=whole - cut)
        self.assertAlmostEqual(sphere.area_km2(region) / (whole - cut), 1., places=6)

    def test_a_point_on_a_neighbours_edge_becomes_a_vertex_of_both(self):
        # Two squares sharing the edge x = 10; the left one was cut at (10, 5), the right one not.
        left = sphere.from_lonlat([0., 10., 10., 10., 0.], [0., 0., 5., 10., 10.])
        right = sphere.from_lonlat([10., 20., 20., 10.], [0., 0., 10., 10.])
        a, b = sphere.node_shared([left, right])
        self.assertEqual((len(a), len(b)), (5, 5))
        cut = sphere.from_lonlat(10., 5.)
        self.assertLess(min(sphere.arc_km(b, cut[None])), 1e-6)
        # A ring that comes back to within 0.1 m of itself (a pinch) keeps the two points apart.
        pinch = sphere.from_lonlat([0., 10., 5., 10., 0., 5. + 1e-6], [0., 0., 5., 10., 10., 5.])
        (kept,) = sphere.node_shared([pinch])
        self.assertEqual(len(kept), 6)
        self.assertGreater(sphere.arc_km(kept[2], kept[5]), 0.)

    def test_a_ring_touching_itself_keeps_both_lobes(self):
        # A figure of eight (two lobes meeting at one point), as rotation can read a strip back.
        lobe_a = sphere.from_lonlat([0., 2., 1.], [0., 0., 1.])
        lobe_b = sphere.from_lonlat([1., 2., 0.], [1., 2., 2.])
        ring = np.vstack((lobe_a, lobe_b[1:], lobe_b[:1]))
        expected = sphere.loop_area_km2(lobe_a) + sphere.loop_area_km2(lobe_b)
        for matrix in (np.eye(3), sphere.rotation(np.array([.3, -.2, .9]))):
            region, _ = sphere.part([sphere.rotate(ring, matrix)], whole_km2=expected)
            self.assertAlmostEqual(sphere.area_km2(region) / expected, 1., places=6)

    def test_a_sliver_whose_every_vertex_is_a_tooth_is_tidied_not_looped_over(self):
        # A zigzag 1 km wide drifting 1 m a step: every vertex turns back within 6 degrees and
        # cuts off less than the level of detail. The one-of-two-neighbours rule keeps none of
        # them; tidy_ring used to spin forever here.
        n = 40
        lon = np.where(np.arange(n) % 2 == 0, 0., 1. / 111.)
        lat = np.arange(n) * (0.001 / 111.)
        ring = sphere.from_lonlat(lon, lat)
        tidy = sphere.tidy_ring(ring, tooth_km2=sphere.DETAIL_KM2)
        self.assertLess(len(tidy), n)
        self.assertLessEqual(abs(sphere.loop_area_km2(tidy)), abs(sphere.loop_area_km2(ring)) + sphere.DETAIL_KM2)

    def test_the_left_area_of_a_loop_is_right_either_side_of_a_hemisphere(self):
        small = sphere.from_lonlat([0., 120., 240.], [10., 10., 10.])
        self.assertAlmostEqual(sphere.left_area_km2(small) / sphere.area_km2(sphere.polygon(small)), 1., places=9)
        self.assertAlmostEqual(sphere.left_area_km2(small[::-1]) / sphere.area_km2(sphere.polygon(small[::-1])), 1., places=9)
        equator = sphere.from_lonlat(np.arange(0., 360., 1.), np.zeros(360))
        self.assertAlmostEqual(sphere.left_area_km2(equator) / (.5 * sphere.SPHERE_KM2), 1., places=9)

    def test_a_disc_has_its_area(self):
        ring = sphere.disc(sphere.from_lonlat(30., -40.), 4.)
        self.assertAlmostEqual(sphere.area_km2(sphere.polygon(ring)) / (np.pi * 16.), 1., delta=.02)

    def test_simplifying_shared_rings_keeps_them_tiling_the_sphere(self):
        # Two hemispheres meeting along the equator, drawn every 28 km with a 1 km zig-zag.
        lon = np.arange(0., 360., .25)
        lat = np.where(np.arange(len(lon)) % 2, .009, 0.)  # 0.009 degrees is 1 km
        line = sphere.from_lonlat(lon, lat)
        north, south = line, line[::-1]
        rings, removed, _ = sphere.simplify_shared([north, south], area_km2=100.)
        self.assertGreater(removed, len(line))  # most go from both rings
        self.assertLess(len(rings[0]), len(line) // 2)
        np.testing.assert_allclose(rings[0], rings[1][::-1])  # still the same edge, both ways round
        self.assertTrue(all(sphere.arc_km(r, np.roll(r, -1, axis=0)).max() <= 150. for r in rings))
        total = sum(sphere.area_km2(sphere.polygon(r)) for r in rings)
        self.assertAlmostEqual(total / sphere.SPHERE_KM2, 1., places=9)
        # Unshared, the same zig-zag stays: dropping a vertex there would open a gap or an overlap.
        lone, _, _ = sphere.simplify_shared([north], area_km2=100.)
        self.assertEqual(len(lone[0]), len(north))


class MeshTests(unittest.TestCase):
    def test_erosion_wears_a_peak_down_and_settles_the_sediment_nearby(self):
        from poly import mesh
        from tests.test_poly_evolve import disc
        shape = sphere.polygon(disc((30., 0.), 15.))
        area = sphere.area_km2(shape)
        m = mesh.ContinentMesh.build(shape, 35.)
        m.add_volume(3e7, area, near=sphere.from_lonlat(30., 0.)[None])
        volume, peak = m.volume_km3(area), m.elevation_km().max()
        removed = settled = to_sea = 0.
        for _ in range(20):
            r, s, t = m.erode(1., area)
            removed, settled, to_sea = removed + r, settled + s, to_sea + t
        self.assertLess(m.elevation_km().max(), .7 * peak)
        self.assertGreater(settled, .5 * removed)
        self.assertAlmostEqual(settled + to_sea, removed, delta=1e-6 * removed)
        self.assertAlmostEqual(m.volume_km3(area) + to_sea, volume, delta=1e-6 * volume)
        self.assertGreaterEqual(m.thickness.min(), mesh.REFERENCE_KM - 1e-9)


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
