"""Ownership changes carry material history; segment resampling conserves pending inventories."""
import types
import unittest

import numpy as np
import spherely

from poly import regions, sphere
from poly.regions import Boundary, RegionWorld
from poly.world import Crust
from tests.test_poly_evolve import disc


def cap_world(two=False):
    cap = sphere.polygon(disc((-50., 0.) if two else (0., 0.), 15.))
    if two:
        cap = sphere.union(cap, sphere.polygon(disc((50., 0.), 10.)))
    outside = sphere.difference(regions.FULL, cap)
    world = RegionWorld([Crust(1, 'ocean', cap, -50.), Crust(2, 'ocean', outside, -50.)],
                        {1: np.eye(3), 2: np.eye(3)}, {1: cap, 2: outside}, [])
    world.boundaries = world._derive({}, {})
    for b in world.boundaries:
        b.kinds = ['trench'] * len(b.kinds)
        b.overriding = [2] * len(b.kinds)
        b.closed_km[:] = 77.
        b.arc_km3 = np.tile([10., 1.], (len(b.kinds), 1))
    return world


def bank(world):
    return sum((b.arcs().sum(axis=0) for b in world.boundaries), np.zeros(2))


def boundary(lons):
    points = sphere.from_lonlat(lons, np.zeros(len(lons)))
    count = len(points) - 1
    return Boundary(1, 2, points, ['trench'] * count, [2] * count,
                    np.full(count, 77.), np.arange(count))


class StateTransferTests(unittest.TestCase):
    def test_natural_split_carries_trench_history_and_suture(self):
        world = cap_world(two=True)
        old_ids = {int(i) for b in world.boundaries for i in b.idents}
        before = bank(world)
        line = sphere.from_lonlat([48., 50., 52.], [0., 0., 0.])
        world.sutures = [dict(plate=1, points=line, time_myr=-20.)]
        entry = world.step({1: np.zeros(3), 2: np.zeros(3)}, 1.)
        event, = [e for e in entry['events'] if e['kind'] == 'plate_split']
        child = event['new']
        moved, = [b for b in world.boundaries if child in (b.left, b.right)]
        self.assertEqual(set(moved.kinds), {'trench'})
        self.assertEqual(set(moved.overriding), {2})
        np.testing.assert_allclose(moved.closed_km, 77.)
        self.assertTrue(set(moved.idents) <= old_ids)
        np.testing.assert_allclose(bank(world), before)
        np.testing.assert_allclose(world.retired_arc_km3, 0.)
        self.assertEqual(world.sutures[0]['plate'], child)
        np.testing.assert_allclose(world.sutures[0]['points'], line, atol=1e-14)
        self.assertAlmostEqual(entry['tiling_error_km2'], 0., delta=.01)

    def test_absorption_retires_internal_bank_and_carries_whole_suture(self):
        world = cap_world()
        before = bank(world)
        line = sphere.from_lonlat([-5., 5.], [0., 0.])
        world.sutures = [dict(plate=1, points=line, time_myr=-20.)]
        index = world._index()
        world._give(1, world.regions[1], 2, boundary_index=index)
        world.regions.pop(1)
        world.boundaries = world._derive(index, {})
        self.assertEqual(world.boundaries, [])
        np.testing.assert_allclose(world.retired_arc_km3, before)
        self.assertEqual(world.sutures[0]['plate'], 2)
        np.testing.assert_allclose(world.sutures[0]['points'], line, atol=1e-14)

    def test_absorption_preserves_the_surviving_external_contact(self):
        cap = sphere.polygon(disc((0., 2.), 4.))
        north = sphere.polygon(sphere.from_lonlat([0., 90., 180., 270.], [0., 0., 0., 0.]))
        north = sphere.difference(north, cap)
        south = sphere.difference(regions.FULL, sphere.union(north, cap))
        world = RegionWorld([Crust(p, 'ocean', shape, -50.) for p, shape in ((1, cap), (2, north), (3, south))],
                            {}, {1: cap, 2: north, 3: south}, [])
        world.boundaries = world._derive({}, {})
        for b in world.boundaries:
            b.arc_km3 = np.tile([10., 1.], (len(b.kinds), 1))
            if (b.left, b.right) == (1, 3):
                b.kinds = ['trench'] * len(b.kinds)
                b.overriding = [1] * len(b.kinds)
                b.closed_km[:] = 77.
                old_ids = set(b.idents)
        before = bank(world)
        index = world._index()
        world._give(1, cap, 2, boundary_index=index)
        world.regions.pop(1)
        world.boundaries = world._derive(index, {})
        inherited = [(b, k) for b in world.boundaries for k, ident in enumerate(b.idents) if ident in old_ids]
        self.assertTrue(inherited)
        for b, k in inherited:
            self.assertEqual((b.left, b.right), (2, 3))
            self.assertEqual(b.kinds[k], 'trench')
            self.assertEqual(b.overriding[k], 2)
            self.assertEqual(b.closed_km[k], 77.)
        np.testing.assert_allclose(bank(world) + world.retired_arc_km3, before)

    def test_old_world_without_retired_inventory_can_step(self):
        world = cap_world()
        del world.retired_arc_km3
        entry = world.step({1: np.zeros(3), 2: np.zeros(3)}, 1.)
        np.testing.assert_allclose(world.retired_arc_km3, 0.)
        np.testing.assert_allclose(entry['retired_arc_km3'], 0.)

    def test_rift_clips_suture_instead_of_assigning_whole_line_by_majority(self):
        world = cap_world()
        before = bank(world)
        world.sutures = [dict(plate=1, points=sphere.from_lonlat([-10., 0., 10.], [0., 0., 0.]), time_myr=-20.)]
        cut = types.SimpleNamespace(plate=1, centre=sphere.from_lonlat(0., 0.), pole=np.array([0., 1., 0.]),
                                    length_km=1000., tension_n_per_m=1., strength_n_per_m=1., continent_fraction=0.)
        event = world.rift(cut)
        child = event['new']
        self.assertEqual({s['plate'] for s in world.sutures}, {1, child})
        lengths = []
        for s in world.sutures:
            points = sphere.rotate(s['points'], world.frame(s['plate']))
            lon, lat = sphere.to_lonlat(points)
            self.assertTrue(np.all(lon >= -1e-8) if s['plate'] == child else np.all(lon <= 1e-8))
            lengths.append(sphere.arc_km(points[:-1], points[1:]).sum())
            self.assertEqual(s['time_myr'], -20.)
        self.assertAlmostEqual(sum(lengths), 20. * np.pi / 180. * sphere.RADIUS_KM, places=6)
        np.testing.assert_allclose(bank(world) + world.retired_arc_km3, before)

    def test_pending_inventory_is_conservative_under_subdivision_and_coarsening(self):
        world = cap_world()
        old = boundary([0., 4., 8.])
        old.arc_km3 = np.array([[30., 3.], [70., 7.]])
        index = world._index([old])
        fine = boundary(np.arange(9.))
        fine.arc_km3 = np.zeros((8, 2))
        world._remap_arcs(index, [fine])
        np.testing.assert_allclose(fine.arcs()[:4], np.tile([7.5, .75], (4, 1)))
        np.testing.assert_allclose(fine.arcs()[4:], np.tile([17.5, 1.75], (4, 1)))
        coarse = boundary([0., 8.])
        world._remap_arcs(world._index([fine]), [coarse])
        np.testing.assert_allclose(coarse.arcs(), [[100., 10.]])
        np.testing.assert_allclose(world.retired_arc_km3, 0.)

    def test_unmatched_inventory_has_an_explicit_destination(self):
        world = cap_world()
        old = boundary([0., 4., 8.])
        old.arc_km3 = np.array([[30., 3.], [70., 7.]])
        world._remap_arcs(world._index([old]), [])
        np.testing.assert_allclose(world.retired_arc_km3, [100., 10.])

    def test_welding_retires_internal_inventory_once(self):
        world = cap_world()
        before = bank(world)
        for b in world.boundaries:
            b.kinds = ['collision'] * len(b.kinds)
            b.overriding = [None] * len(b.kinds)
            b.closed_km[:] = regions.SUTURE_KM
        events = world._sutures({1: np.zeros(3), 2: np.zeros(3)})
        self.assertEqual(len(events), 1)
        self.assertEqual(world.plates(), [2])
        np.testing.assert_allclose(world.retired_arc_km3, before)
        world._sutures({2: np.zeros(3)})
        np.testing.assert_allclose(world.retired_arc_km3, before)

    def test_larger_rift_child_can_reabsorb_parent_without_motion_cycle(self):
        world = cap_world()
        centre = sphere.from_lonlat(-2., 0.)
        cut = types.SimpleNamespace(plate=1, centre=centre, pole=np.cross([0., 0., 1.], centre),
                                    length_km=1000., tension_n_per_m=1., strength_n_per_m=1., continent_fraction=0.)
        child = world.rift(cut)['new']
        self.assertGreater(sphere.area_km2(world.regions[child]), sphere.area_km2(world.regions[1]))
        for b in world.boundaries:
            if {b.left, b.right} == {1, child}:
                b.kinds = ['collision'] * len(b.kinds)
                b.overriding = [None] * len(b.kinds)
                b.closed_km[:] = regions.SUTURE_KM
        world._sutures({p: np.zeros(3) for p in world.plates()})
        self.assertNotIn(1, world.plates())
        self.assertEqual(world.parents[child], 1)
        self.assertNotIn(1, world.parents)
        original_motion = np.array([.01, .02, .03])
        np.testing.assert_allclose(world._complete({1: original_motion})[child], original_motion)
        np.testing.assert_allclose(world._complete({})[child], 0.)

    def test_malformed_motion_ancestry_raises_instead_of_hanging(self):
        world = cap_world()
        world.parents = {1: 3, 3: 1}
        with self.assertRaisesRegex(ValueError, 'Cyclic motion ancestry'):
            world._complete({})

    def test_suture_frame_changes_preserve_world_position(self):
        world = cap_world()
        line = sphere.from_lonlat([-5., 5.], [0., 0.])
        world.sutures = [dict(plate=1, points=line, time_myr=-20.)]
        world.frames[2] = sphere.rotation(np.array([0., 0., .3]))
        for c in world.crust:
            if c.plate == 2:
                c.shape = sphere.rotate_geography(c.shape, world.frame(2).T)
        world._give(1, world.regions[1], 2)
        np.testing.assert_allclose(sphere.rotate(world.sutures[0]['points'], world.frame(2)), line, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
