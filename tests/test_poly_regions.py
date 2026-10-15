"""Plates as regions: overlaps and gaps booked by boundary type, boundaries derived every step."""
import unittest

import numpy as np
import spherely

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

    def test_a_collision_thickens_the_continent_then_welds_the_plates(self):
        world = regions.from_network(cap_world(cap_omega=EAST)[0])
        continent = sphere.polygon(disc((30., 0.), 6.))
        for piece in world.crust:
            if piece.plate == 1:
                piece.shape = sphere.difference(piece.shape, continent)
        world.crust.append(Crust(1, 'continent', continent))
        volume = world._continent_km3()
        area = world.areas()['continent']
        events = []
        for _ in range(30):
            entry = world.step({1: EAST, 2: np.zeros(3)}, 1.)
            assert_checks(self, entry)
            events += entry['events']
            if any(e['kind'] == 'suture' for e in events):
                break
        sutures = [e for e in events if e['kind'] == 'suture']
        self.assertTrue(sutures)
        self.assertEqual((sutures[0]['into'], sutures[0]['gone']), (2, 1))  # the cap joins the larger plate
        self.assertGreaterEqual(sutures[0]['closed_km'], regions.SUTURE_KM)
        self.assertEqual(world.plates(), [2])
        self.assertTrue(all(c.plate == 2 for c in world.crust))
        # The continent shortened, and kept its volume by thickening, on its mesh, where it was shortened:
        # it was carried east into the trench, so its eastern side is the thick side.
        self.assertLess(world.areas()['continent'], area)
        self.assertAlmostEqual(world._continent_km3() / volume, 1., delta=1e-3)
        (piece,) = [c for c in world.crust if c.kind == 'continent']
        self.assertGreater(piece.thickness_km, 35.)
        self.assertIsNotNone(piece.mesh)
        self.assertAlmostEqual(piece.mesh.volume_km3(piece.area_km2()) / piece.volume_km3(), 1., delta=1e-6)
        nodes = sphere.rotate(piece.mesh.nodes, world.frame(piece.plate))
        lon = sphere.to_lonlat(nodes)[0]
        east, west = lon > np.median(lon), lon <= np.median(lon)
        self.assertGreater(piece.mesh.thickness[east].mean(), piece.mesh.thickness[west].mean() + 1.)
        # A one-plate world still steps.
        assert_checks(self, world.step({2: np.zeros(3)}, 1.))
        # The suture is remembered, in the welded plate's frame, and is where the plate is weakest.
        self.assertEqual(len(world.sutures), 1)
        (line, when), = list(world.suture_lines())
        self.assertGreater(len(line), 2)
        self.assertLess(when, world.time_myr)
        balance = dynamics.Balance()
        weak = dynamics.Rifting(minimum_plate_km2=1., minimum_part_km2=1.)
        rotations = balance.solve(world)
        with_suture = dynamics.cuts(balance, world, rotations, 2, weak)
        sutures, world.sutures = world.sutures, []
        without = dynamics.cuts(balance, world, rotations, 2, weak)
        world.sutures = sutures
        # One more cut is tried, along the suture, and every cut near it is weaker than without it.
        self.assertEqual(len(with_suture), len(without) + 1)
        self.assertLess(min(c.strength_n_per_m for c in with_suture), min(c.strength_n_per_m for c in without))

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


class RiftTests(unittest.TestCase):
    def test_a_plates_boundary_torques_and_drag_balance(self):
        world = regions.from_network(voronoi_world(count=6, seed=5, continents=2)[0])
        balance = dynamics.Balance()
        rotations = balance.solve(world)
        for plate in world.plates():
            mids, torques = dynamics.plate_torques(balance, world, rotations, plate)
            region = world.world_regions()[plate]
            inside = np.asarray(spherely.contains(region, balance.geographies))
            a = balance.points[inside]
            drag = (balance.parameters.basal_drag_pa_s_per_m * dynamics.R_M ** 2 * balance.point_area_m2
                    * (np.eye(3) * len(a) - a.T @ a))
            net = torques.sum(axis=0) - drag @ (np.asarray(rotations[plate]) / dynamics.SECONDS_PER_MYR)
            self.assertLess(np.linalg.norm(net), 1e-6 * np.abs(torques).sum())

    def test_a_weak_plate_rifts_and_the_books_close(self):
        world = regions.from_network(voronoi_world(count=6, seed=5, continents=2)[0])
        weak = dynamics.Rifting(ocean_strength_n_per_m=1e10, continent_strength_n_per_m=1e10)
        crust = sum(c.area_km2() for c in world.crust)
        entries = []
        dynamics.run(world, 2, report=lambda w, entry: entries.append(entry), rifting=weak)
        rifts = [e for entry in entries for e in entry['events'] if e['kind'] == 'rift']
        self.assertGreaterEqual(len(rifts), 1)
        self.assertGreater(rifts[0]['tension_n_per_m'], rifts[0]['strength_n_per_m'])
        self.assertEqual(len(world.plates()), 6 + len(rifts))
        for entry in entries:
            assert_checks(self, entry)
        self.assertAlmostEqual(sum(c.area_km2() for c in world.crust) / crust, 1., delta=1e-6)
        # Parent and child meet along the cut.
        first = rifts[0]
        self.assertTrue(any({b.left, b.right} == {first['plate'], first['new']} for b in world.boundaries))


if __name__ == '__main__':
    unittest.main()
