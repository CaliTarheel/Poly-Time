"""Native fracture geometry and the first finite-width continental opening."""
import unittest

import numpy as np

from poly import acceptance, dynamics, fracture, geology, regions, sphere, surface
from poly.build import cap_world, densify
from poly.mesh import ContinentMesh
from poly.world import Crust


def continent_world(radius=3000., thickness=35.):
    world = regions.from_network(cap_world(spacing_km=500.)[0])
    shape = sphere.polygon(sphere.disc(sphere.from_lonlat(0., 0.), radius, count=64))
    for piece in world.crust:
        if piece.plate == 1:
            piece.shape = sphere.difference(piece.shape, shape)
    body = Crust(1, 'continent', shape, thickness_km=thickness,
                 mesh=ContinentMesh.build(shape, thickness))
    world.crust.append(body)
    geology.activate(world)
    return world, body


def weak_belt(world):
    longitude = np.linspace(-37., 37., 151)
    points = sphere.from_lonlat(longitude, 2.8 * np.sin(longitude * np.pi / 25.))
    world.geology['features'] = [dict(
        uid=1, kind='collision', plate=1, points=points, time_myr=0., last_active_myr=0.,
        cumulative_strain_km=300., width_km=150., boundary_idents=[])]
    world.geology['next_feature'] = 2


def nominal_cut():
    return dynamics.Cut(1, sphere.from_lonlat(0., 0.), np.array([0., 0., 1.]),
                        5000., 1e99, 1., 1.)


class FractureGeometryTests(unittest.TestCase):
    def test_uniform_resistance_does_not_add_decorative_wiggles(self):
        offsets = np.arange(-fracture.REACH_KM, fracture.REACH_KM + 1., fracture.ACROSS_KM)
        displacement = fracture._route(np.ones((100, len(offsets))), offsets, 100.)
        np.testing.assert_array_equal(displacement, np.zeros(100))

    def test_native_path_follows_a_curved_weak_belt_and_is_scored_again(self):
        world, _ = continent_world()
        weak_belt(world)
        balance = dynamics.Balance(dynamics.Parameters(sample_points=3000))
        rotations = balance.solve(world)
        cut = fracture.routed_cut(balance, world, rotations, nominal_cut(), dynamics.Rifting())
        self.assertIsNotNone(cut)
        offset = np.arcsin(np.clip(cut.path @ cut.pole, -1., 1.)) * sphere.RADIUS_KM
        self.assertGreater(np.abs(offset).max(), 80.)
        self.assertLessEqual(np.abs(offset).max(), fracture.REACH_KM)
        longitude, latitude = sphere.to_lonlat(cut.path)
        central = np.abs(longitude) < 25.
        inherited = 2.8 * np.sin(longitude[central] * np.pi / 25.)
        self.assertGreater(np.corrcoef(latitude[central], inherited)[0, 1], .8)
        self.assertTrue(np.isfinite(cut.tension_n_per_m))
        self.assertLess(cut.tension_n_per_m, 1e90)
        self.assertGreater(cut.strength_n_per_m, 1e10)
        self.assertNotEqual(cut.length_km, nominal_cut().length_km)
        shape = sphere.polygon(cut.path)
        part = sphere.intersection(world.regions[1], shape)
        rest = sphere.difference(world.regions[1], shape)
        self.assertGreater(sphere.area_km2(part), 3e6)
        self.assertGreater(sphere.area_km2(rest), 3e6)
        self.assertLess(sphere.area_km2(sphere.intersection(part, rest)), 1e-4)

    def test_curved_partition_preserves_volume_coverage_and_material_references(self):
        world, body = continent_world()
        weak_belt(world)
        before = world._continent_km3()
        province = body.geology['province_id']
        balance = dynamics.Balance(dynamics.Parameters(sample_points=3000))
        cut = fracture.routed_cut(balance, world, balance.solve(world), nominal_cut(), dynamics.Rifting())
        event = world.rift(cut)
        record = {}
        acceptance.geometry(world, regions.FULL, record)
        self.assertEqual(event['path_type'], 'strength_routed')
        self.assertGreater(event['maximum_path_offset_km'], 80.)
        self.assertAlmostEqual(world._continent_km3() / before, 1., places=10)
        self.assertLess(record['plate_gap_km2'], 1e-3)
        self.assertLess(record['plate_overlap_km2'], 1e-3)
        for piece in world.crust:
            if piece.kind == 'continent':
                self.assertEqual(piece.geology['province_id'], province)
                np.testing.assert_allclose(piece.geology['reference_from_plate'], np.eye(3), atol=1e-14)
        self.assertTrue(any({b.left, b.right} == {1, event['new']} for b in world.boundaries))

    def test_first_provisional_rift_gap_stretches_continent_before_ocean_birth(self):
        world, _ = continent_world(radius=1500., thickness=40.)
        event = world.rift(nominal_cut())
        speed = 25. / sphere.RADIUS_KM
        rotations = {1: np.array([0., speed, 0.]), 2: np.zeros(3),
                     event['new']: np.array([0., -speed, 0.])}
        before = world._continent_km3()
        record = world.step(rotations, 1.)
        self.assertGreater(record['gained']['sliver'] + record['gained']['junction'], 0.)
        self.assertGreater(record['continental_extension_km2'], 0.)
        central = surface.sample(world, sphere.from_lonlat([0.], [0.]))
        self.assertEqual(central.crust_kind[0], 'continent')
        self.assertGreater(central.elevation_km[0], -1.)
        self.assertAlmostEqual((world._continent_km3() + record['sediment_to_sea_km3']
                               - record['arc_km3']) / before, 1., delta=acceptance.VOLUME_RELATIVE)

    def test_thin_and_thick_sections_of_one_coast_break_up_locally(self):
        world, body = continent_world()
        ring = sphere.from_lonlat([-10., 0., 0., -10.], [-15., -15., 15., 15.])
        body.shape = sphere.polygon(densify(ring, 100., closed=True))
        def thickness(points):
            _, latitude = sphere.to_lonlat(points)
            return 10. + 30. * np.clip((latitude + 2.) / 4., 0., 1.)
        body.mesh = ContinentMesh.build(body.shape, 0., field=thickness)
        body.thickness_km = body.mesh.volume_km3(body.area_km2()) / body.area_km2()
        np.testing.assert_allclose(body.mesh.sample(sphere.from_lonlat([0., 0.], [-10., 10.])), [10., 40.], atol=.1)
        gain = sphere.polygon(sphere.from_lonlat([0., .4, .4, 0.], [-14., -14., 14., 14.]))
        before = body.volume_km3()
        remaining = geology.extend_rift(world, [body], gain, {})
        import spherely
        points = spherely.points([.2, .2], [-10., 10.])
        np.testing.assert_array_equal(spherely.contains(remaining, points), [True, False])
        self.assertAlmostEqual(body.volume_km3() / before, 1., places=12)


if __name__ == '__main__':
    unittest.main()
