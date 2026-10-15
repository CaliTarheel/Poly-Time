"""Boundaries that follow the motion: re-typing, subduction initiation and collision."""
import math
import unittest

import numpy as np

from poly import sphere
from poly.build import cap_world
from poly.world import COLLISION_KM2, INITIATION_KM, Crust

EAST = np.array([0., 0., 1.]) * 40. / sphere.RADIUS_KM  # 40 km/Myr at the equator


def disc(centre_lonlat, radius_deg, count=48):
    centre = sphere.from_lonlat(*centre_lonlat)
    pole = sphere.unit(np.cross(centre, [0., 0., 1.]))
    start = sphere.rotate(centre, sphere.rotation(pole * math.radians(radius_deg)))
    return np.array([sphere.rotate(start, sphere.rotation(centre * 2 * math.pi * k / count)) for k in range(count)])


def kinds_of(world):
    return {s.kind for s in world.network.sections.values()}


class EvolvingBoundaryTests(unittest.TestCase):
    def test_a_ridge_that_starts_closing_converges_then_becomes_a_trench(self):
        world, omegas = cap_world(cap_omega=EAST)
        self.assertIn('ridge', kinds_of(world))
        for _ in range(2):
            world.step(omegas, 1.)
        reverse = {plate: -vector for plate, vector in omegas.items()}
        events = []
        for _ in range(int(INITIATION_KM / 40.) + 3):
            events += world.step(reverse, 1.)['events']
        retyped = [e for e in events if e['kind'] == 'retyped' and e['was'] == 'ridge']
        self.assertTrue(retyped and 'converging' in retyped[0]['now'])
        born = [e for e in events if e['kind'] == 'subduction_initiated']
        self.assertTrue(born)
        self.assertGreaterEqual(born[0]['closed_km'], INITIATION_KM)
        self.assertNotIn('ridge', kinds_of(world))
        # Closing before the trench existed was booked, not deleted.
        self.assertGreater(sum(e['lost']['compression'] for e in world.ledger), 0.)

    def test_a_continent_reaching_a_trench_makes_a_collision(self):
        world, omegas = cap_world(cap_omega=EAST)
        continent = sphere.polygon(disc((30., 0.), 6.))
        for piece in world.crust:
            if piece.plate == 1:
                piece.shape = sphere.difference(piece.shape, continent)
        world.crust.append(Crust(1, 'continent', continent))
        events = []
        for _ in range(25):
            entry = world.step(omegas, 1.)
            events += entry['events']
            if any(e['kind'] == 'collision' for e in events):
                break
        collisions = [e for e in events if e['kind'] == 'collision']
        self.assertTrue(collisions)
        self.assertGreater(collisions[0]['arrived_km2'], COLLISION_KM2)
        self.assertIn('collision', kinds_of(world))
        self.assertLess(abs(entry['balance_error_km2']), 1e-3)


if __name__ == '__main__':
    unittest.main()
