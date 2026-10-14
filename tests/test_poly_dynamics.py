"""Plate motions from the torque balance, and absorbing a plate."""
import unittest

import numpy as np

from poly import dynamics, sphere
from poly.build import cap_world, voronoi_world


class BalanceTests(unittest.TestCase):
    def test_a_cap_going_down_a_trench_is_pulled_towards_it(self):
        # The cap's eastern edge is a trench under the world plate; its western edge a ridge.
        world, _ = cap_world(cap_omega=np.array([0., 0., 1.]) * 40. / sphere.RADIUS_KM)
        rotations = dynamics.Balance().solve(world)
        cap = rotations[1] - rotations[2]
        # Moving east at the cap's centre (lon 0, lat 0) is a rotation about +z.
        self.assertGreater(cap[2], 0.)
        speed = float(np.linalg.norm(cap)) * sphere.RADIUS_KM
        self.assertGreater(speed, 10.)
        self.assertLess(speed, 200.)

    def test_balance_is_symmetric_and_the_drivers_set_the_motion(self):
        world, _ = voronoi_world(count=5, seed=2)
        balance = dynamics.Balance()
        first = balance.solve(world)
        doubled = dynamics.Balance(dynamics.Parameters(slab_pull_n_per_m=6e13, ridge_push_n_per_m=6e12))
        second = doubled.solve(world)
        for plate in first:
            np.testing.assert_allclose(second[plate], 2. * first[plate], rtol=1e-9, atol=1e-15)


class AbsorptionTests(unittest.TestCase):
    def test_absorbing_a_plate_keeps_the_tiling_and_the_crust(self):
        world, omegas = voronoi_world(count=6, seed=5, continents=2)
        world.step(omegas, 1.)
        before = sum(c.area_km2() for c in world.crust)
        plates = world.network.plates()
        event = world.absorb(plates[0])
        self.assertEqual(event['kind'], 'plate_absorbed')
        self.assertNotIn(plates[0], world.network.plates())
        regions = world.plate_polygons()
        self.assertAlmostEqual(sum(sphere.area_km2(r) for r in regions.values()) / sphere.SPHERE_KM2, 1., places=9)
        self.assertAlmostEqual(sum(c.area_km2() for c in world.crust) / before, 1., places=6)
        self.assertTrue(all(c.plate in regions for c in world.crust))
        entry = world.step({p: v for p, v in omegas.items() if p in regions}, 1.)
        self.assertLess(abs(entry['balance_error_km2']), 1e-3)


if __name__ == '__main__':
    unittest.main()
