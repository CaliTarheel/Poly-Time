"""Rock coordinates and contact history persist independently of plate identities."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poly import geology, regions, sphere
from poly.build import cap_world
from poly.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from poly.world import Crust


def world_with_continent():
    world = regions.from_network(cap_world(spacing_km=500.)[0])
    shape = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), 500.))
    world.crust[0].shape = sphere.difference(world.crust[0].shape, shape)
    world.crust.append(Crust(1, 'continent', shape, thickness_km=40.))
    return geology.activate(world, seed=12)


class GeologicalHistoryTests(unittest.TestCase):
    def test_texture_coordinates_follow_rotation_and_transfer_to_another_plate(self):
        world = world_with_continent()
        piece = next(c for c in world.crust if c.kind == 'continent')
        point = sphere.from_lonlat(1., 1.)
        reference = geology.reference_points(piece, point)
        rotation = sphere.rotation(sphere.unit([1., 2., 3.]) * 1.1)
        properties = geology.carry(piece.geology, rotation)
        moved = copy.deepcopy(piece)
        moved.geology = properties
        np.testing.assert_allclose(geology.reference_points(moved, sphere.rotate(point, rotation)),
                                   reference, atol=1e-14)
        self.assertEqual(properties['province_id'], piece.geology['province_id'])
        np.testing.assert_array_equal(piece.geology['reference_from_plate'], np.eye(3))

    def test_actual_plate_handover_keeps_province_and_world_texture_position(self):
        world = world_with_continent()
        original = next(c for c in world.crust if c.kind == 'continent')
        ident = original.geology['province_id']
        point = sphere.from_lonlat(1., 1.)
        reference = geology.reference_points(original, point)
        world.frames[2] = sphere.rotation(np.array([.2, -.3, .1]))
        for c in world.crust:
            if c.plate == 2:
                c.shape = sphere.rotate_geography(c.shape, world.frame(2).T)
        world._give(1, world.regions[1], 2)
        given = next(c for c in world.crust if c.kind == 'continent')
        self.assertEqual(given.plate, 2)
        self.assertEqual(given.geology['province_id'], ident)
        local = sphere.rotate(point, world.frame(2).T)
        np.testing.assert_allclose(geology.reference_points(given, local), reference, atol=1e-14)

    def test_collision_above_old_fusion_threshold_keeps_two_plate_rotations_and_a_belt(self):
        world = world_with_continent()
        rotations = {1: np.array([0., 0., .005]), 2: np.zeros(3)}
        for b in world.boundaries:
            b.kinds = ['collision'] * len(b.kinds)
            b.overriding = [None] * len(b.kinds)
            b.closed_km[:] = 200.
        world._evolve(rotations, 1., dict(continent_arrived=[], arc_km3=0., gained={'arc': 0.}, lost={'arc': 0.}))
        self.assertEqual(world.plates(), [1, 2])
        self.assertTrue(any(f['kind'] == 'collision' for f in world.geology['features']))
        self.assertTrue(list(world.suture_lines()))
        geology.validate(world)

    def test_checkpoint_preserves_provinces_material_reference_and_belt_history(self):
        world = world_with_continent()
        point = sphere.from_lonlat(1., 1.)
        piece = next(c for c in world.crust if c.kind == 'continent')
        piece.geology = geology.carry(piece.geology, sphere.rotation(np.array([.2, .1, 0.])))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'world.json'
            save_checkpoint(world, path)
            restored, _ = load_checkpoint(path)
        after = next(c for c in restored.crust if c.kind == 'continent')
        np.testing.assert_array_equal(geology.reference_points(after, point),
                                      geology.reference_points(piece, point))
        self.assertEqual(restored.geology['next_province'], world.geology['next_province'])
        self.assertEqual(after.geology['formed_myr'], piece.geology['formed_myr'])

    def test_incomplete_province_state_is_rejected(self):
        world = world_with_continent()
        next(c for c in world.crust if c.kind == 'continent').geology['reference_from_plate'][0, 0] = 2.
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(CheckpointError):
                save_checkpoint(world, Path(folder) / 'bad.json')

    def test_lateral_escape_is_invariant_to_the_absolute_plate_frame(self):
        from poly.mesh import ContinentMesh
        world = world_with_continent()
        piece = next(c for c in world.crust if c.kind == 'continent')
        piece.mesh = ContinentMesh.build(piece.shape, piece.thickness_km)
        piece.mesh.lost_nodes = piece.mesh.nodes[:2].copy()
        piece.escape_km2 = 50000.
        rotated = copy.deepcopy(world)
        rotated.frames[1] = sphere.rotation(np.array([0., 0., np.pi / 2.]))
        initial_volume = piece.volume_km3()
        results = []
        for current in (world, rotated):
            piece = next(c for c in current.crust if c.kind == 'continent')
            mine = [c for c in current.crust if c.plate == 1]
            entry = dict(gained=dict(escape=0.), lost=dict(escape=0.))
            current._escape(piece, 1000., mine, entry)
            self.assertGreater(entry['gained']['escape'], 0.)
            self.assertAlmostEqual(piece.volume_km3() / initial_volume, 1., places=12)
            results.append((piece.shape.__getstate__(), entry))
        self.assertEqual(results[0], results[1])

    def test_disconnected_continents_become_separate_bodies_with_their_own_fields(self):
        from poly.mesh import ContinentMesh
        world = world_with_continent()
        piece = next(c for c in world.crust if c.kind == 'continent')
        second = sphere.polygon(sphere.disc(sphere.from_lonlat(15., 0.), 200.))
        for ocean in world.crust:
            if ocean.plate == 1 and ocean.kind == 'ocean':
                ocean.shape = sphere.difference(ocean.shape, second)
        piece.shape = sphere.union(piece.shape, second)
        piece.mesh = ContinentMesh.build(piece.shape, 40.)
        lon, _ = sphere.to_lonlat(piece.mesh.nodes)
        piece.mesh.thickness = np.where(lon > 10., 60., 40.)
        piece.thickness_km = piece.mesh.volume_km3(piece.area_km2()) / piece.area_km2()
        piece.escape_km2 = 1000.
        volume = piece.volume_km3()
        province = piece.geology['province_id']
        geology.separate_bodies(world)
        bodies = [c for c in world.crust if c.kind == 'continent']
        self.assertEqual(len(bodies), 2)
        self.assertAlmostEqual(sum(c.volume_km3() for c in bodies) / volume, 1., places=12)
        np.testing.assert_allclose(sorted(c.thickness_km for c in bodies), [40., 60.], atol=1e-8)
        self.assertEqual({c.geology['province_id'] for c in bodies}, {province})
        self.assertAlmostEqual(sum(c.escape_km2 for c in bodies), 1000., places=8)


if __name__ == '__main__':
    unittest.main()
