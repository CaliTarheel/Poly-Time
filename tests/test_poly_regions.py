"""Plates as regions: overlaps and gaps booked by boundary type, boundaries derived every step."""
import unittest

import numpy as np

from poly import dynamics, regions, sphere
from poly.build import cap_world, voronoi_world
from poly.world import COLLISION_KM2, INITIATION_KM, Crust
from tests.test_poly_evolve import EAST, disc

TOLERANCE_KM2 = regions.CHECK_TOLERANCE * sphere.SPHERE_KM2


def kinds_of(world):
    return {k for b in world.boundaries for k in b.kinds}


def assert_checks(test, entry):
    test.assertLess(abs(entry['tiling_error_km2']), TOLERANCE_KM2)
    test.assertLess(entry['crust_vs_plate_error_km2'], TOLERANCE_KM2)
    # Gaps and overlaps are measured on the rebuilt regions, which S2 moves by ~1e-7 of the sphere.
    test.assertLess(abs(entry['balance_error_km2']), 1e-6 * sphere.SPHERE_KM2)


class RegionStepTests(unittest.TestCase):
    def test_a_moving_cap_opens_a_ridge_behind_and_subducts_ahead(self):
        world = regions.from_network(cap_world(cap_omega=EAST)[0])
        omegas = {1: EAST, 2: np.zeros(3)}
        self.assertEqual(kinds_of(world), {'ridge', 'trench', 'transform'})
        for _ in range(3):
            entry = world.step(omegas, 1.)
            assert_checks(self, entry)
        # 40 km/Myr across a 40-degree cap: a few hundred thousand km2 a step each way.
        self.assertGreater(entry['gained']['ridge'], 1e5)
        self.assertGreater(entry['lost']['subducted'], 1e5)
        self.assertLess(entry['gained']['junction'] + entry['lost']['junction'], .01 * entry['gained']['ridge'])
        # The trench stayed a trench with the world plate over it, and the cap is still one plate.
        self.assertEqual(world.plates(), [1, 2])
        trench = {o for b in world.boundaries for k, o in zip(b.kinds, b.overriding) if k == 'trench'}
        self.assertEqual(trench, {2})

    def test_a_ridge_that_starts_closing_converges_then_becomes_a_trench(self):
        world = regions.from_network(cap_world(cap_omega=EAST)[0])
        omegas = {1: EAST, 2: np.zeros(3)}
        world.step(omegas, 1.)
        reverse = {plate: -vector for plate, vector in omegas.items()}
        events, closing = [], 0.
        for _ in range(int(INITIATION_KM / 40.) + 3):
            entry = world.step(reverse, 1.)
            assert_checks(self, entry)
            events += entry['events']
            closing += entry['lost']['compression']
        born = [e for e in events if e['kind'] == 'subduction_initiated']
        self.assertTrue(born)
        self.assertGreaterEqual(born[0]['closed_km'], INITIATION_KM)
        self.assertNotIn('ridge', kinds_of(world))
        # Closing before the trench existed was booked, not deleted.
        self.assertGreater(closing, 0.)

    def test_a_continent_reaching_a_trench_makes_a_collision(self):
        world = regions.from_network(cap_world(cap_omega=EAST)[0])
        continent = sphere.polygon(disc((30., 0.), 6.))
        for piece in world.crust:
            if piece.plate == 1:
                piece.shape = sphere.difference(piece.shape, continent)
        world.crust.append(Crust(1, 'continent', continent))
        events = []
        for _ in range(25):
            entry = world.step({1: EAST, 2: np.zeros(3)}, 1.)
            assert_checks(self, entry)
            events += entry['events']
            if any(e['kind'] == 'collision' for e in events):
                break
        collisions = [e for e in events if e['kind'] == 'collision']
        self.assertTrue(collisions)
        self.assertGreater(collisions[0]['arrived_km2'], COLLISION_KM2)
        self.assertIn('collision', kinds_of(world))
        self.assertGreater(sum(e['lost']['continent_at_trench'] for e in world.ledger), COLLISION_KM2)

    def test_eight_plates_keep_their_crust_accounted_for(self):
        network_world, omegas = voronoi_world(count=8, seed=3, speed_km_myr=30., continents=3)
        world = regions.from_network(network_world)
        continent = world.areas()['continent']
        for _ in range(3):
            entry = world.step(omegas, 1.)
            assert_checks(self, entry)
        self.assertEqual(len(world.plates()), 8)
        # Every plate meets another along derived boundaries, and the crust stayed put in total.
        self.assertTrue(all(any(p in (b.left, b.right) for b in world.boundaries) for p in world.plates()))
        drift = sum(e['rebuild_drift_km2'] for e in world.ledger)
        self.assertLess(abs(drift), 1e-6 * sphere.SPHERE_KM2)
        lost = sum(e['lost']['continent_at_trench'] + e['lost']['collision'] for e in world.ledger)
        self.assertAlmostEqual(world.areas()['continent'] + lost, continent, delta=1e-4 * continent + 1.)


class RegionDynamicsTests(unittest.TestCase):
    def test_the_force_balance_pulls_a_cap_towards_its_trench_and_the_books_close(self):
        world = regions.from_network(cap_world(cap_omega=EAST)[0])
        rotations = dynamics.Balance().solve(world)
        self.assertGreater((rotations[1] - rotations[2])[2], 0.)  # east, towards the trench
        entries = []
        dynamics.run(world, 2, report=lambda w, entry: entries.append(entry))
        self.assertGreaterEqual(len(entries), 2)
        for entry in entries:
            assert_checks(self, entry)
        self.assertAlmostEqual(world.time_myr, 2., places=9)
        self.assertGreater(sum(e['lost']['subducted'] for e in entries), 0.)


if __name__ == '__main__':
    unittest.main()
