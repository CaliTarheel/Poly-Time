"""Sub-resolution rock is conserved without independent fragment mechanics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poly import acceptance, coarsen, geology, regions, sphere
from poly.build import cap_world
from poly.checkpoint import CheckpointError, load_checkpoint, save_checkpoint, world_from_data
from poly.mesh import ContinentMesh, MINIMUM_KM2
from poly.world import Crust


def fixture(longitude=5.):
    world = regions.from_network(cap_world(spacing_km=500.)[0])
    large = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), 500.))
    small = sphere.polygon(sphere.disc(sphere.from_lonlat(longitude, 0.), 10.))
    for ocean in world.crust:
        if ocean.plate == 1:
            ocean.shape = sphere.difference(ocean.shape, sphere.union(large, small))
    world.crust.extend([Crust(1, 'continent', large, thickness_km=40.,
                              mesh=ContinentMesh.build(large, 40.)),
                        Crust(1, 'continent', small, thickness_km=1000.)])
    geology.activate(world)
    return world


def entry():
    return dict(events=[], gained={k: 0. for k in regions.GAINS},
                lost={k: 0. for k in regions.LOSSES}, numerical_area_added_km2=0.,
                numerical_area_removed_km2=0.)


class PlanetaryCoarseningTests(unittest.TestCase):
    def test_captured_microscopic_field_cell_joins_a_resolved_patch_without_losing_rock(self):
        payload = json.loads((Path(__file__).parent / 'fixtures/poly_coastal_field_support.json').read_text())
        field = ContinentMesh(payload['nodes'], payload['triangles'], payload['thickness'])
        before = field.volume_km3(payload['polygon_km2'])
        peak = float(field.thickness.max())
        moved, support = coarsen.lump_field(field, payload['polygon_km2'])
        self.assertGreater(moved, 0.)
        self.assertGreaterEqual(support, coarsen.MINIMUM_FIELD_KM2)
        self.assertLess(float(field.thickness.max()), peak / 2.)
        self.assertAlmostEqual(field.volume_km3(payload['polygon_km2']) / before, 1., places=12)
        projected = field.thickness.copy()
        coarsen.lump_field(field, payload['polygon_km2'])
        np.testing.assert_allclose(field.thickness, projected, atol=1e-12, rtol=0.)

    def test_field_resolution_is_not_a_thickness_ceiling(self):
        payload = json.loads((Path(__file__).parent / 'fixtures/poly_coastal_field_support.json').read_text())
        field = ContinentMesh(payload['nodes'], payload['triangles'], np.full(len(payload['nodes']), 250.))
        coarsen.lump_field(field, payload['polygon_km2'])
        np.testing.assert_allclose(field.thickness, 250., atol=1e-10, rtol=0.)

    def test_field_projection_does_not_exchange_rock_between_disconnected_meshes(self):
        payload = json.loads((Path(__file__).parent / 'fixtures/poly_coastal_field_support.json').read_text())
        first = ContinentMesh(payload['nodes'], payload['triangles'], payload['thickness'])
        size = len(first.nodes)
        moved = sphere.rotate(first.nodes, sphere.rotation(np.array([0., 1.5, 0.])))
        field = ContinentMesh(np.vstack((first.nodes, moved)),
                              np.vstack((first.triangles, first.triangles + size)),
                              np.r_[first.thickness, first.thickness + 100.])
        areas = field.node_areas_km2()
        before = [float(areas[:size] @ field.thickness[:size]),
                  float(areas[size:] @ field.thickness[size:])]
        coarsen.lump_field(field, 2. * payload['polygon_km2'])
        after = [float(areas[:size] @ field.thickness[:size]),
                 float(areas[size:] @ field.thickness[size:])]
        np.testing.assert_allclose(np.asarray(after) / before, 1., atol=1e-12, rtol=0.)

    def test_terrane_accretes_only_after_its_third_plate_overlap_is_resolved(self):
        payload = json.loads((Path(__file__).parent / 'fixtures/poly_terrane_third_contact.json').read_text())
        world = world_from_data(payload['world'])
        rotations = {int(p): np.asarray(v) for p, v in payload['rotations'].items()}
        before = world._continent_km3()
        record = world.step(rotations, payload['dt_myr'])
        self.assertTrue(any(e['kind'] == 'accreted' and e['plates'] == (3, 2) for e in record['events']))
        self.assertLess(record['plate_overlap_km2'], acceptance.GEOMETRY_KM2)
        self.assertLess(record['plate_gap_km2'], acceptance.GEOMETRY_KM2)
        self.assertAlmostEqual((world._continent_km3() + record['sediment_to_sea_km3']
                               - record['arc_km3']) / before, 1., delta=acceptance.VOLUME_RELATIVE)

    def test_nearby_remnant_joins_resolved_crust_without_a_gap_or_rock_loss(self):
        world = fixture()
        before = world._continent_km3()
        provinces = [c.geology['province_id'] for c in world.crust if c.kind == 'continent']
        plates = {p: g.__getstate__() for p, g in world.regions.items()}
        record = entry()
        coarsen.consolidate(world, record)
        acceptance.geometry(world, regions.FULL, record)
        self.assertEqual({p: g.__getstate__() for p, g in world.regions.items()}, plates)
        self.assertAlmostEqual(world._continent_km3() / before, 1., places=12)
        self.assertEqual(coarsen.volume_km3(world), 0.)
        self.assertEqual(record['coarsened_bodies'], 1)
        self.assertEqual(record['coarsening_transfers'][0]['source_geology']['province_id'], provinces[1])
        self.assertEqual(record['coarsening_transfers'][0]['receiver_province'], provinces[0])
        self.assertLess(geology.maximum_thickness(world), 80.)
        self.assertTrue(all(c.area_km2() >= MINIMUM_KM2 for c in world.crust if c.kind == 'continent'))
        self.assertLess(record['crust_gap_km2'], 1e-5)
        self.assertLess(record['crust_overlap_km2'], 1e-5)

    def test_isolated_rock_is_passive_inventory_not_transport_to_a_remote_continent(self):
        world = fixture(15.)
        body = next(c for c in world.crust if c.kind == 'continent')
        old_field = body.mesh.thickness.copy()
        before = world._continent_km3()
        small_volume = world.crust[-1].volume_km3()
        record = entry()
        coarsen.consolidate(world, record)
        self.assertAlmostEqual(coarsen.volume_km3(world), small_volume, places=6)
        self.assertAlmostEqual(world._continent_km3() / before, 1., places=12)
        self.assertEqual(len(coarsen.inventory(world)), 1)
        # A volume-weighted regional projection can round a uniform field by
        # a few ulps; no material from the remote island may enter this body.
        np.testing.assert_allclose(body.mesh.thickness, old_field, atol=1e-12, rtol=0.)
        self.assertEqual(record['coarsening_deposited_km3'], 0.)
        self.assertAlmostEqual(geology.maximum_thickness(world), 40., places=12)
        self.assertTrue(any(c.kind == 'ocean' and c.born_myr is None for c in world.crust))

    def test_inventory_survives_restart_and_plate_handover_in_material_coordinates(self):
        world = fixture(15.)
        coarsen.consolidate(world)
        before = copy.deepcopy(coarsen.inventory(world)[0])
        source_point = before['geology']['reference_from_plate'] @ before['point']
        source_world = sphere.rotate(before['point'], world.frame(1))
        world.frames[2] = sphere.rotation(np.array([.1, -.2, .3]))
        world._give(1, world.regions[1], 2)
        world.regions.pop(1)
        world.boundaries = []  # the retired plate no longer has a contact
        parcel = coarsen.inventory(world)[0]
        self.assertEqual(parcel['plate'], 2)
        np.testing.assert_allclose(sphere.rotate(parcel['point'], world.frame(2)), source_world, atol=1e-13)
        np.testing.assert_allclose(parcel['geology']['reference_from_plate'] @ parcel['point'], source_point, atol=1e-13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            save_checkpoint(world, path)
            restored, _ = load_checkpoint(path)
        self.assertEqual(coarsen.volume_km3(restored), before['volume_km3'])
        np.testing.assert_array_equal(coarsen.inventory(restored)[0]['point'], parcel['point'])
        self.assertEqual(restored.plate_policy.minimum_plate_km2, 250_000.)

    def test_passive_material_is_emplaced_when_a_resolved_receiver_arrives(self):
        world = fixture(15.)
        coarsen.consolidate(world)
        parcel = coarsen.inventory(world)[0]
        body = next(c for c in world.crust if c.kind == 'continent')
        # Move the unresolved tracer as a plate would carry it toward the mainland.
        parcel['point'] = sphere.from_lonlat(5., 0.)
        before = world._continent_km3()
        field_volume = body.volume_km3()
        pending = coarsen.volume_km3(world)
        coarsen.consolidate(world)
        self.assertEqual(coarsen.volume_km3(world), 0.)
        self.assertAlmostEqual(body.volume_km3() - field_volume, pending, places=6)
        self.assertAlmostEqual(world._continent_km3() / before, 1., places=12)

    def test_invalid_inventory_is_rejected_at_checkpoint_boundary(self):
        world = fixture(15.)
        coarsen.consolidate(world)
        coarsen.inventory(world)[0]['volume_km3'] = -1.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CheckpointError, 'unresolved material'):
                save_checkpoint(world, Path(directory) / 'checkpoint.json')


if __name__ == '__main__':
    unittest.main()
