"""The artist scenario must measure emergence and retain its material/water state."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poly import artist, geology, sphere
from poly.checkpoint import load_checkpoint, save_checkpoint
from tests.test_poly_geology import world_with_continent


class ArtistScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.rotations = artist.make_world(seed=1, plates=8, spacing_km=250.)

    def test_land_target_is_measured_separately_from_continental_crust(self):
        measured = artist.metrics(self.world)
        self.assertAlmostEqual(measured['land_fraction'], .55, delta=1. / 32768.)
        self.assertAlmostEqual(measured['continental_fraction'], .65, delta=.001)
        self.assertEqual(measured['unresolved_fraction'], 0.)
        self.assertGreater(measured['continental_fraction'], measured['land_fraction'])

    def test_sea_level_adjusts_to_retain_water_after_crustal_uplift(self):
        import copy
        world = copy.deepcopy(self.world)
        water = world.geology['water_depth_km']
        old_level = world.geology['sea_level_km']
        for piece in world.crust:
            if piece.kind == 'continent':
                piece.thickness_km += 3.
                if piece.mesh is not None:
                    piece.mesh.thickness += 3.
        artist.update_water(world)
        measured = artist.metrics(world)
        self.assertAlmostEqual(measured['water_depth_km'], water, places=12)
        self.assertGreater(world.geology['sea_level_km'], old_level)

    def test_water_quadrature_reports_a_tiny_slit_without_hiding_the_surface_gap(self):
        import copy
        world = copy.deepcopy(self.world)
        point = artist.equal_area_points(32768)[0]
        hole = sphere.polygon(sphere.disc(point, .1))
        for piece in world.crust:
            piece.shape = sphere.difference(piece.shape, hole)
        diagnostics = {}
        artist.update_water(world, diagnostics=diagnostics)
        self.assertEqual(diagnostics['water_quadrature_shifted_points'], 1)
        self.assertLessEqual(diagnostics['water_quadrature_max_shift_km'], 1.)
        self.assertTrue(np.isnan(artist.sample(world, point[None]).elevation_km[0]))
        measured = artist.metrics(world)
        self.assertGreater(measured['unresolved_fraction'], 0.)
        self.assertAlmostEqual(measured['water_depth_km'], world.geology['water_depth_km'], places=12)

    def test_water_quadrature_cannot_hide_a_large_geometry_defect(self):
        import copy
        world = copy.deepcopy(self.world)
        point = artist.equal_area_points(32768)[0]
        hole = sphere.polygon(sphere.disc(point, 1000.))
        for piece in world.crust:
            piece.shape = sphere.difference(piece.shape, hole)
        with self.assertRaisesRegex(ValueError, 'crust_gap'):
            artist.update_water(world)

    def test_detail_follows_the_rock_during_an_actual_ownership_transfer(self):
        world = world_with_continent()
        point = sphere.from_lonlat(1., 1.)[None]
        before = artist.sample(world, point).elevation_km
        world.frames[2] = sphere.rotation(np.array([.2, -.3, .1]))
        for piece in world.crust:
            if piece.plate == 2:
                piece.shape = sphere.rotate_geography(piece.shape, world.frame(2).T)
        world._give(1, world.regions[1], 2)
        after = artist.sample(world, point).elevation_km
        np.testing.assert_allclose(after, before, atol=1e-12, rtol=0.)
        geology.validate(world)

    def test_restart_reproduces_surface_and_water_exactly(self):
        points = artist.equal_area_points(2048)
        before = artist.sample(self.world, points)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(self.world, Path(folder) / 'world.json')
            restored, _ = load_checkpoint(Path(folder) / 'world.json')
        after = artist.sample(restored, points)
        np.testing.assert_array_equal(after.elevation_km, before.elevation_km)
        self.assertEqual(restored.geology['water_depth_km'], self.world.geology['water_depth_km'])
        self.assertEqual(after.sea_level_km, before.sea_level_km)

    def test_force_driven_scenario_accepts_its_first_two_steps_and_restarts(self):
        import copy
        from dataclasses import asdict
        from poly import dynamics
        from poly.run import advance
        world = copy.deepcopy(self.world)
        config = dict(dt_myr=.5, motion='forces', parameters=asdict(dynamics.Parameters(sample_points=5000)),
                      rifting=asdict(dynamics.Rifting()), rotations_per_myr=self.rotations)
        phase = dict(completed_intervals=0, remaining_myr=0.)
        def accepted(current, phase):
            measured = artist.metrics(current)
            self.assertAlmostEqual(measured['water_depth_km'], current.geology['water_depth_km'], places=12)
        advance(world, config, phase, 2, accepted)
        self.assertAlmostEqual(world.time_myr, 1.)
        self.assertTrue(world.ledger)
        geology.validate(world)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'advanced.json'
            save_checkpoint(world, path)
            restored, _ = load_checkpoint(path)
        self.assertEqual(restored.time_myr, world.time_myr)
        self.assertEqual(len(restored.geology['features']), len(world.geology['features']))


if __name__ == '__main__':
    unittest.main()
