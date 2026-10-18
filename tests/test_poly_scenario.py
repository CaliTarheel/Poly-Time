"""Chosen starting geometry must still tile a sphere and use the actual crust."""
import unittest

import numpy as np

from poly import acceptance, artist, scenario, sphere
from poly.build import voronoi_world
from poly.regions import FULL, from_network


class StartingGeometryTests(unittest.TestCase):
    def test_spherical_twists_are_deterministic_and_invertible(self):
        points = artist.equal_area_points(1024)
        warp = scenario.SphericalWarp(17)
        mapped = warp(points)
        np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1., atol=1e-14, rtol=0.)
        np.testing.assert_array_equal(mapped, scenario.SphericalWarp(17)(points))
        np.testing.assert_allclose(warp(mapped, inverse=True), points, atol=1e-12, rtol=0.)

    def test_curved_shared_network_still_tiles_the_sphere(self):
        template, _ = voronoi_world(count=6, seed=3, spacing_km=400.)
        scenario.curve_network(template, seed=3, spacing_km=400.)
        world = from_network(template)
        entry = {}
        acceptance.geometry(world, FULL, entry)
        for key in ('plate_gap_km2', 'plate_overlap_km2', 'crust_gap_km2',
                    'crust_outside_km2', 'crust_overlap_km2'):
            self.assertLess(entry[key], .01, key)
        for ident in template.network.sections:
            line = template.network.points(ident)
            self.assertLessEqual(float(sphere.arc_km(line[:-1], line[1:]).max()), 400. + 1e-8)

    def test_starting_warp_cannot_replace_preexisting_continental_material(self):
        template, _ = voronoi_world(count=6, seed=3, spacing_km=400.)
        template.crust[0].kind = 'continent'
        with self.assertRaisesRegex(ValueError, 'before continents are placed'):
            scenario.curve_network(template, seed=3, spacing_km=400.)


if __name__ == '__main__':
    unittest.main()
