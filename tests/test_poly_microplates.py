"""Retention, geometry-based absorption and basal drag on resolved microplates."""
import unittest

import numpy as np
import spherely

from poly import dynamics, regions, sphere
from poly.policy import PlatePolicy
from poly.quadrature import basal_moment_m2
from poly.world import Crust


def cap(radius, lon=0., count=128):
    return sphere.polygon(sphere.disc(sphere.from_lonlat(lon, 0.), radius, count=count))


def box(west, south, east, north):
    return sphere.polygon(sphere.from_lonlat([west, east, east, west], [south, south, north, north]))


def world_for(shapes, policy=None):
    return regions.RegionWorld([Crust(p, 'ocean', g, -80.) for p, g in shapes.items()],
                               {p: np.eye(3) for p in shapes}, shapes, [],
                               plate_policy=policy or PlatePolicy())


class MicroplatePolicyTests(unittest.TestCase):
    def test_adriatic_scale_plate_survives_zero_motion_step(self):
        small = cap(300.)
        world = world_for({1: small, 2: sphere.difference(regions.FULL, small)})
        world.boundaries = world._derive({}, {})
        # Disable scheduled outline simplification: this test isolates retention.
        world.simplified_myr = 0.
        world.simplified_count = sum(len(r) for g in world.regions.values()
                                     for polygon in sphere.polygons(g) for r in polygon)
        before = sphere.area_km2(small)
        for _ in range(3):
            world.step({1: np.zeros(3), 2: np.zeros(3)}, dt=0.)
        self.assertEqual(world.plates(), [1, 2])
        self.assertAlmostEqual(sphere.area_km2(world.regions[1]), before, delta=.01)

    def test_policy_is_per_world_and_legacy_threshold_is_available(self):
        small = cap(300.)
        shapes = {1: small, 2: sphere.difference(regions.FULL, small)}
        retained = world_for(dict(shapes))
        legacy = world_for(dict(shapes), PlatePolicy.legacy())
        self.assertEqual(retained._regroup(), [])
        self.assertTrue(legacy._regroup())
        self.assertEqual(legacy.plates(), [2])
        self.assertEqual(retained.plates(), [1, 2])
        self.assertAlmostEqual(sum(c.area_km2() for c in legacy.crust), sphere.SPHERE_KM2, places=5)
        smaller = cap(100.)
        local = world_for({1: smaller, 2: sphere.difference(regions.FULL, smaller)},
                          PlatePolicy(minimum_plate_km2=20_000.))
        self.assertEqual(local._regroup(), [])

    def test_numerical_crumbs_and_resolved_detached_microplates_differ(self):
        large, small, crumb = cap(1000., -60.), cap(300.), cap(2., 60.)
        donor = sphere.union(large, small, crumb)
        world = world_for({1: donor, 2: sphere.difference(regions.FULL, donor)})
        events = world._regroup()
        split = [e for e in events if e['kind'] == 'plate_split']
        crumbs = [e for e in events if e['kind'] == 'crumbs_given']
        self.assertEqual(len(split), 1)
        self.assertEqual(len(crumbs), 1)
        self.assertEqual(crumbs[0]['into'], 2)
        self.assertAlmostEqual(split[0]['area_km2'], sphere.area_km2(small), places=6)
        self.assertEqual(world.plates(), [1, 2, 3])

    def test_recipient_uses_shared_boundary_not_insertion_order(self):
        piece = box(0., 0., 4., 1.)
        short = box(-1., 0., 0., 1.)
        long = box(0., 1., 4., 3.)
        for shapes in ({1: piece, 2: short, 3: long}, {3: long, 2: short, 1: piece}):
            world = world_for(shapes)
            self.assertEqual(world._main_neighbour(piece, 1), 3)
            self.assertEqual(world._main_neighbours([piece, piece], 1), [3, 3])

    def test_shared_contact_survives_frame_roundoff_and_subdivision(self):
        piece = box(0., 0., 4., 1.)
        short = box(-1., 0., 0., 1.)
        long = box(0., 1., 4., 3.)
        rotation = sphere.rotation(np.array([.2, .3, .4]))
        returned = sphere.rotate_geography(sphere.rotate_geography(long, rotation), rotation.T)
        for order in ({1: piece, 2: short, 3: returned}, {3: returned, 2: short, 1: piece}):
            self.assertEqual(world_for(order)._main_neighbour(piece, 1), 3)
        ring = sphere.polygons(long)[0][0]
        subdivided = np.vstack([point for a, b in zip(ring, np.roll(ring, -1, axis=0))
                                for point in (a, sphere.unit(a + b))])
        long2 = sphere.polygon(subdivided)
        self.assertEqual(world_for({1: piece, 2: short, 3: long2})._main_neighbour(piece, 1), 3)

    def test_nearby_separate_arc_is_not_a_contact(self):
        from poly.policy import _boundary_edges, _shared_boundary_km
        piece = box(0., 0., 4., 1.)
        # A 1 m gap exceeds the 1 cm matching tolerance by two orders of magnitude.
        separate = box(0., 1. + .001 / sphere.RADIUS_KM * 180. / np.pi, 4., 3.)
        self.assertAlmostEqual(_shared_boundary_km(_boundary_edges(piece), _boundary_edges(separate)), 0.)

    def test_polygon_line_debris_does_not_change_the_recipient(self):
        piece = box(0., 0., 4., 1.)
        short = box(-1., 0., 0., 1.)
        long = box(0., 1., 4., 3.)
        debris = spherely.create_linestring([(60., 0.), (61., 0.)])
        mixed = spherely.create_collection([long, debris])
        world = world_for({1: piece, 2: short, 3: mixed})
        self.assertEqual(world._main_neighbour(piece, 1), 3)

    def test_equal_shared_boundaries_and_detached_gaps_have_stable_ties(self):
        piece = box(0., 0., 4., 1.)
        west, east = box(-1., 0., 0., 1.), box(4., 0., 5., 1.)
        for shapes in ({1: piece, 7: east, 2: west}, {2: west, 7: east, 1: piece}):
            self.assertEqual(world_for(shapes)._main_neighbour(piece, 1), 2)
        detached = cap(2.)
        near, far = cap(20., 1.), cap(20., 5.)
        self.assertEqual(world_for({1: detached, 3: far, 2: near})._main_neighbour(detached, 1), 2)
        self.assertIsNone(world_for({1: piece})._main_neighbour(piece, 1))

    def test_policy_validation(self):
        for kwargs in (dict(minimum_plate_km2=0.), dict(numerical_crumb_km2=0.),
                       dict(numerical_crumb_km2=300_000.), dict(minimum_plate_km2=float('nan'))):
            with self.assertRaises(ValueError):
                PlatePolicy(**kwargs)


class MicroplateQuadratureTests(unittest.TestCase):
    def test_refined_polygon_converges_to_analytic_spherical_cap(self):
        radius = 250.
        c = np.cos(radius / sphere.RADIUS_KM)
        radius_m = sphere.RADIUS_KM * 1000.
        area = 2. * np.pi * radius_m ** 2 * (1. - c)
        axial = 2. * np.pi * radius_m ** 2 * (1. - c) ** 2 * (c + 2.) / 3.
        exact = np.diag([axial, area - .5 * axial, area - .5 * axial])
        errors = []
        for n in (32, 64, 128, 256):
            moment = basal_moment_m2(cap(radius, count=n))
            errors.append(np.linalg.norm(moment - exact) / np.linalg.norm(exact))
            self.assertTrue(np.all(np.linalg.eigvalsh(moment) > 0.))
            # Both the very small axial eigenvalue and large transverse terms converge.
            np.testing.assert_allclose(np.diag(moment) / np.diag(exact), 1., rtol=.014)
        self.assertTrue(all(b < .26 * a for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], .00011)

    def test_moments_rotate_and_holes_complement_and_components_add(self):
        shape = cap(300.)
        rotation = sphere.rotation(np.array([.3, -.5, .1]))
        moment = basal_moment_m2(shape)
        np.testing.assert_allclose(basal_moment_m2(sphere.rotate_geography(shape, rotation)),
                                   rotation @ moment @ rotation.T, rtol=1e-10, atol=.1)
        outside = sphere.difference(regions.FULL, shape)
        full = np.eye(3) * (2. / 3.) * sphere.SPHERE_KM2 * 1e6
        np.testing.assert_allclose(moment + basal_moment_m2(outside), full, rtol=1e-12, atol=.1)
        another = cap(100., 30.)
        np.testing.assert_allclose(basal_moment_m2(sphere.union(shape, another)),
                                   moment + basal_moment_m2(another), rtol=1e-10, atol=.1)

    def test_zero_global_samples_still_give_finite_small_plate_drag(self):
        shape = cap(100.)
        world = world_for({1: shape, 2: sphere.difference(regions.FULL, shape)})
        world.boundaries = world._derive({}, {})
        for boundary in world.boundaries:
            boundary.kinds = ['trench' if m[1] > 0. else 'ridge' for m in boundary.mids()]
            boundary.overriding = [2 if k == 'trench' else None for k in boundary.kinds]
        balance = dynamics.Balance(dynamics.Parameters(sample_points=4))
        rotations = balance.solve(world)
        self.assertEqual(balance.last['drag_quadrature'][1], dict(method='polygon_boundary', samples=0))
        self.assertGreater(np.linalg.norm(rotations[1]), 0.)
        for omega in rotations.values():
            self.assertTrue(np.isfinite(omega).all())
        # Exact integration for both plates makes this independent of global density.
        denser = dynamics.Balance(dynamics.Parameters(sample_points=100, minimum_drag_samples=128))
        for plate, omega in denser.solve(world).items():
            np.testing.assert_allclose(omega, rotations[plate], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
