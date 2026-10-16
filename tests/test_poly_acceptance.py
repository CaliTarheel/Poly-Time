"""Independent invalid worlds and cumulative failures that scalar area sums miss."""
import pickle
import unittest
from unittest import mock

import numpy as np

from poly import acceptance, regions, sphere
from poly.build import cap_world
from poly.world import Crust


def entry(world, **changes):
    result = dict(gained={k: 0. for k in regions.GAINS}, lost={k: 0. for k in regions.LOSSES},
                  plates_km2={p: sphere.area_km2(r) for p, r in world.regions.items()},
                  rebuild_drift_km2=0., numerical_area_added_km2=0., numerical_area_removed_km2=0.,
                  arc_km3=0., sediment_to_sea_km3=0., continent_before_km3=world._continent_km3())
    result.update(changes)
    return result


def continent_world():
    land = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), 500.))
    return regions.RegionWorld(
        [Crust(1, 'continent', land), Crust(1, 'ocean', sphere.difference(regions.FULL, land))],
        {1: np.eye(3)}, {1: regions.FULL}, [])


class CoverageTests(unittest.TestCase):
    def test_equal_plate_hole_and_overlap_cannot_cancel(self):
        world = regions.from_network(cap_world()[0])
        world.regions[1] = sphere.rotate_geography(world.regions[1], sphere.rotation([0., 0., .02]))
        record = entry(world)
        self.assertLess(abs(sum(record['plates_km2'].values()) - sphere.SPHERE_KM2), 1.)
        with self.assertRaisesRegex(ValueError, 'plate_gap'):
            world._check(record)
        self.assertGreater(record['plate_gap_km2'], 10000.)
        self.assertAlmostEqual(record['plate_gap_km2'], record['plate_overlap_km2'], delta=1.)

    def test_crust_hole_and_duplicate_ownership_cannot_cancel(self):
        missing = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), 100.))
        duplicate = sphere.polygon(sphere.disc(sphere.from_lonlat(40., 0.), 100.))
        world = regions.RegionWorld(
            [Crust(1, 'ocean', sphere.difference(regions.FULL, missing)), Crust(1, 'ocean', duplicate)],
            {1: np.eye(3)}, {1: regions.FULL}, [])
        self.assertLess(abs(sum(c.area_km2() for c in world.crust) - sphere.SPHERE_KM2), 1.)
        record = entry(world)
        with self.assertRaisesRegex(ValueError, 'crust_gap'):
            world._check(record)
        self.assertGreater(record['crust_overlap_km2'], 10000.)
        self.assertAlmostEqual(record['crust_gap_km2'], record['crust_overlap_km2'], delta=1.)

    def test_unowned_crust_is_rejected(self):
        world = regions.from_network(cap_world()[0])
        world.crust[0].plate = 123456
        with self.assertRaisesRegex(ValueError, 'absent plate'):
            world._check(entry(world))


class ConservationTests(unittest.TestCase):
    def test_realized_magma_and_exported_sediment_close_the_volume_budget(self):
        world = continent_world()
        before = world._continent_km3()
        land = world.crust[0]
        source, sink = .1 * before, .03 * before
        land.thickness_km += (source - sink) / land.area_km2()
        record = entry(world, continent_before_km3=before, arc_km3=source, sediment_to_sea_km3=sink)
        world._check(record)
        self.assertAlmostEqual(record['continent_balance_error_km3'], 0., delta=1e-6)
        # Pending magma belongs to a different reservoir and does not authorize
        # an unexplained increase of realized continental crust.
        land.thickness_km += .01 * before / land.area_km2()
        with self.assertRaisesRegex(ValueError, 'continent_balance_error'):
            world._check(record)

    def test_small_systematic_volume_leak_is_not_forgiven_every_step(self):
        world = continent_world()
        initial = world._continent_km3()
        for step in range(20):
            before = world._continent_km3()
            world.crust[0].thickness_km += 2e-5 * initial / world.crust[0].area_km2()
            record = entry(world, continent_before_km3=before)
            try:
                acceptance.budgets(world, record)
            except ValueError as error:
                self.assertIn('cumulative_continent_error', str(error))
                self.assertGreater(step, 0)  # individual steps fit the budget
                break
            world.ledger.append(record)
        else:
            self.fail('Repeated small volume creation was accepted indefinitely.')

    def test_signed_and_alternating_area_drift_have_cumulative_limits(self):
        for alternating in (False, True):
            world = regions.from_network(cap_world()[0])
            for step in range(150):
                drift = .9 * acceptance.GEOMETRY_KM2 * (-1 if alternating and step % 2 else 1)
                record = entry(world, rebuild_drift_km2=drift)
                try:
                    acceptance.budgets(world, record)
                except ValueError as error:
                    expected = 'absolute_rebuild_drift' if alternating else 'cumulative_rebuild_drift'
                    self.assertIn(expected, str(error))
                    self.assertGreater(step, 0)
                    break
                world.ledger.append(record)
            else:
                self.fail('Repeated area reconstruction error was accepted indefinitely.')

    def test_recorded_numerical_repairs_also_have_a_total_budget(self):
        world = regions.from_network(cap_world()[0])
        for step in range(30):
            record = entry(world, numerical_area_added_km2=.6 * acceptance.REPAIR_STEP_KM2)
            try:
                acceptance.budgets(world, record)
            except ValueError as error:
                self.assertIn('cumulative_numerical_area_repair', str(error))
                self.assertGreater(step, 0)
                break
            world.ledger.append(record)
        else:
            self.fail('Repeated numerical area creation was accepted indefinitely.')

    def test_nonfinite_flux_is_rejected(self):
        world = continent_world()
        with self.assertRaisesRegex(ValueError, 'finite'):
            acceptance.budgets(world, entry(world, arc_km3=float('nan')))


class AtomicStepTests(unittest.TestCase):
    def test_failed_final_check_preserves_world_and_live_references(self):
        world = continent_world()
        world.step({1: np.zeros(3)}, 0.)
        before = pickle.dumps(world, protocol=5)
        crust, frames, ledger = world.crust, world.frames, world.ledger

        def corrupt(candidate, *args):
            for piece in candidate.crust:
                if piece.kind == 'continent':
                    piece.thickness_km *= 2.
                    if piece.mesh is not None:
                        piece.mesh.thickness *= 2.
            candidate.sutures.append(dict(plate=1, points=np.eye(3), time_myr=999.))
            return []

        with mock.patch.object(regions.RegionWorld, '_evolve', corrupt):
            with self.assertRaisesRegex(ValueError, 'continent_balance_error'):
                world.step({1: np.array([0., 0., .001])}, .5)
        self.assertEqual(pickle.dumps(world, protocol=5), before)
        self.assertIs(world.crust, crust)
        self.assertIs(world.frames, frames)
        self.assertIs(world.ledger, ledger)
        # The same last accepted world can still advance normally.
        accepted = world.step({1: np.zeros(3)}, .5)
        self.assertEqual(world.time_myr, .5)
        self.assertIs(world.ledger[-1], accepted)

    def test_geometry_exception_also_preserves_last_state(self):
        world = regions.from_network(cap_world()[0])
        before = pickle.dumps(world, protocol=5)
        with mock.patch.object(regions.RegionWorld, '_simplify', side_effect=RuntimeError('broken geometry')):
            with self.assertRaisesRegex(RuntimeError, 'broken geometry'):
                world.step({1: np.array([0., 0., .001])})
        self.assertEqual(pickle.dumps(world, protocol=5), before)


if __name__ == '__main__':
    unittest.main()
