"""The continental field must follow its material domain, including narrow seas and holes."""
import unittest
import json
from pathlib import Path

import numpy as np

from poly import sphere
from poly.mesh import ContinentMesh
from poly.checkpoint import _unshape


def polygon(lon, lat):
    return sphere.polygon(sphere.from_lonlat(lon, lat))


def slit():
    return polygon([0, 10, 10, 5.1, 5.1, 4.9, 4.9, 0], [0, 0, 10, 10, 2, 2, 10, 10])


class MeshDomainTests(unittest.TestCase):
    def test_clipped_microscopic_component_keeps_the_resolved_field_and_full_volume(self):
        for name in ['poly_remesh_microscopic.json', 'poly_remesh_duplicate.json',
                     'poly_remesh_collapsed_triangle.json', 'poly_remesh_fallback.json',
                     'poly_remesh_touching.json', 'poly_remesh_wide.json', 'poly_remesh_ear_clip.json',
                     'poly_remesh_small_hole.json']:
            with self.subTest(fixture=name):
                fixture = Path(__file__).parent / 'fixtures' / name
                shape = _unshape(json.loads(fixture.read_text())['geography'])
                area = sphere.area_km2(shape)
                original = shape.__getstate__()
                mesh = ContinentMesh.build(shape, 41.)
                self.assertIsNotNone(mesh)
                self.assertGreater(len(mesh.triangles), 10)
                self.assertAlmostEqual(mesh.volume_km3(area) / area, 41., places=10)
                self.assertEqual(shape.__getstate__(), original)
                self.assertAlmostEqual(mesh.area_km2() / area, 1., places=9)

    def test_tidy_ring_removes_repeats_even_in_a_three_vertex_component(self):
        point = sphere.from_lonlat(10., 30.)
        other = sphere.from_lonlat(12., 32.)
        ring = np.array([point, point.copy(), other])
        self.assertEqual(len(sphere.tidy_ring(ring, tolerance=1e-13)), 2)
        proper = sphere.from_lonlat([0., 2., 0.], [0., 0., 2.])
        np.testing.assert_array_equal(sphere.tidy_ring(proper), proper)

    def test_wide_continent_has_connected_charts_and_preserves_its_field(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        shape = sphere.polygon(sphere.disc(sphere.from_lonlat(23., 14.),
                                           sphere.RADIUS_KM * np.radians(85.), count=48))
        mesh = ContinentMesh.build(shape, 40., spacing_km=800.)
        self.assertIsNotNone(mesh)
        self.assertAlmostEqual(mesh.area_km2() / sphere.area_km2(shape), 1., places=9)
        edges = np.vstack((mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]], mesh.triangles[:, [2, 0]]))
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(mesh.nodes), len(mesh.nodes)))
        self.assertEqual(connected_components(graph, directed=False)[0], 1)
        self.assertTrue(np.all(mesh.node_areas_km2() > 0.))
        mesh.thickness += 5. * mesh.nodes[:, 0]
        before = mesh.volume_km3(sphere.area_km2(shape))
        mesh.spread(50.)
        self.assertAlmostEqual(mesh.volume_km3(sphere.area_km2(shape)) / before, 1., places=12)

    def assert_domain(self, shape, mesh):
        self.assertIsNotNone(mesh)
        faces = []
        for tri in mesh.triangles:
            ring = mesh.nodes[tri]
            face = sphere.polygon(ring)
            if sphere.area_km2(face) > sphere.SPHERE_KM2 / 2:
                face = sphere.polygon(ring[::-1])
            self.assertLess(sphere.area_km2(sphere.difference(face, shape)), 1e-5)
            faces.append(face)
        covered = sphere.union(*faces)
        self.assertLess(sphere.area_km2(sphere.difference(shape, covered)), 1e-5)
        self.assertAlmostEqual(mesh.area_km2() / sphere.area_km2(shape), 1., places=10)
        self.assertTrue(np.all(mesh.node_areas_km2() > 0.))

    def test_reentrant_coast_is_covered_without_faces_crossing_the_slit(self):
        shape = slit()
        mesh = ContinentMesh.build(shape, 35.)
        self.assert_domain(shape, mesh)
        self.assertAlmostEqual(mesh.volume_km3(sphere.area_km2(shape)) / sphere.area_km2(shape), 35.)

    def test_holes_remain_outside_the_mesh(self):
        shell = sphere.from_lonlat([0, 10, 10, 0], [0, 0, 10, 10])
        hole = sphere.from_lonlat([4, 6, 6, 4], [2, 2, 8, 8])[::-1]
        shape = sphere.polygon(shell, [hole])
        self.assert_domain(shape, ContinentMesh.build(shape, 35.))

    def test_disconnected_components_do_not_need_a_shared_projection(self):
        first = polygon([0, 5, 5, 0], [0, 0, 5, 5])
        second = polygon([175, 180, 180, 175], [0, 0, 5, 5])
        shape = sphere.union(first, second)
        self.assert_domain(shape, ContinentMesh.build(shape, 35.))

    def test_sampling_and_transport_do_not_mix_across_a_narrow_seaway(self):
        first = polygon([0, 1, 1, 0], [0, 0, 5, 5])
        second = polygon([1.2, 2.2, 2.2, 1.2], [0, 0, 5, 5])
        shape = sphere.union(first, second)
        mesh = ContinentMesh.build(shape, 35.)
        self.assert_domain(shape, mesh)
        mesh.thickness = np.where(sphere.to_lonlat(mesh.nodes)[0] < 1.1, 40., 80.)
        # Interior points and new ground just offshore both sample their own shore, even
        # where the nearest three old nodes would have included the opposite component.
        points = sphere.from_lonlat([.99, 1.21, 1.05, 1.15], [2.5] * 4)
        np.testing.assert_allclose(mesh.sample(points), [40., 80., 40., 80.], atol=1e-9)
        before = mesh.thickness.copy()
        mesh.spread(1.)
        np.testing.assert_allclose(mesh.thickness, before, atol=1e-9)

    def test_samples_reproduce_the_nodal_field_and_rotate_with_it(self):
        mesh = ContinentMesh.build(slit(), 35.)
        mesh.thickness = np.linspace(35., 70., len(mesh.nodes))
        np.testing.assert_allclose(mesh.sample(mesh.nodes), mesh.thickness, atol=1e-7)
        points = sphere.unit(mesh.nodes[mesh.triangles].sum(axis=1))
        rotation = sphere.rotation(sphere.unit([1., 2., 3.]) * 1.2)
        np.testing.assert_allclose(mesh.rotated(rotation).sample(sphere.rotate(points, rotation)),
                                   mesh.sample(points), atol=1e-7)

    def test_indexed_sampling_matches_spherical_containment_and_coast_extrapolation(self):
        mesh = ContinentMesh.build(slit(), 35.)
        mesh.thickness = np.linspace(35., 70., len(mesh.nodes))
        lon, lat = np.meshgrid(np.linspace(-1., 11., 37), np.linspace(-1., 11., 31))
        points = sphere.from_lonlat(lon.ravel(), lat.ravel())
        np.testing.assert_allclose(mesh.sample(points), mesh._sample_fallback(points), atol=1e-8, rtol=0.)

    def test_old_zero_width_face_cannot_make_the_resolved_field_singular(self):
        nodes = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        mesh = ContinentMesh(nodes, [[0, 1, 2], [0, 2, 3]], [40., 900., 50., 60.])
        point = sphere.unit([1., 1., 1.])[None]
        np.testing.assert_allclose(mesh.sample(point), [50.], atol=1e-12)

    def test_clipped_mesh_diffusion_preserves_volume(self):
        shape = slit()
        mesh = ContinentMesh.build(shape, 35.)
        mesh.thickness = np.linspace(35., 70., len(mesh.nodes))
        volume = mesh.volume_km3(sphere.area_km2(shape))
        mesh.spread(2.)
        self.assertAlmostEqual(mesh.volume_km3(sphere.area_km2(shape)) / volume, 1., places=12)

    def test_diffusion_with_a_microscopic_cell_preserves_real_volume_and_bounds(self):
        # A tiny boundary face shares an edge with a resolved face. The former 1-km2 mass floor
        # moved material using an area different from the actual volume integral.
        nodes = sphere.from_lonlat([0., 1., 0., 0.], [0., 0., 1., -1e-9])
        mesh = ContinentMesh(nodes, [[0, 1, 2], [1, 0, 3]], [35., 35., 70., 90.])
        area = mesh.area_km2()
        before = mesh.volume_km3(area)
        mesh.spread(50.)
        self.assertAlmostEqual(mesh.volume_km3(area) / before, 1., places=12)
        self.assertGreaterEqual(float(mesh.thickness.min()), 35. - 1e-9)
        self.assertLessEqual(float(mesh.thickness.max()), 90. + 1e-9)

    def test_erosion_books_the_real_volume_of_microscopic_cells(self):
        nodes = sphere.from_lonlat([0., 1., 0., 0.], [0., 0., 1., -1e-9])
        mesh = ContinentMesh(nodes, [[0, 1, 2], [1, 0, 3]], [35., 35., 70., 90.])
        area = mesh.area_km2()
        before = mesh.volume_km3(area)
        removed, settled, exported = mesh.erode(1., area)
        after = mesh.volume_km3(area)
        self.assertAlmostEqual((before - after) / before, exported / before, places=12)
        self.assertAlmostEqual((removed - settled) / before, exported / before, places=12)

    def test_erosion_does_not_remove_submerged_continental_rock(self):
        nodes = sphere.from_lonlat([0., 1., 0.], [0., 0., 1.])
        mesh = ContinentMesh(nodes, [[0, 1, 2]], [36., 36., 36.])
        before = mesh.thickness.copy()
        self.assertEqual(mesh.erode(10., mesh.area_km2(), sea_level_km=.5), (0., 0., 0.))
        np.testing.assert_array_equal(mesh.thickness, before)

    def test_sediment_cannot_make_a_mountain_in_a_microscopic_lowland_cell(self):
        nodes = sphere.from_lonlat([0., 1., 0., 0.], [0., 0., 1., -1e-8])
        mesh = ContinentMesh(nodes, [[0, 1, 2], [1, 0, 3]], [70., 70., 70., 35.])
        area = mesh.area_km2()
        before = mesh.volume_km3(area)
        removed, settled, exported = mesh.erode(1., area, bounded_deposition=True)
        self.assertGreater(exported, .99 * removed)
        self.assertLessEqual(mesh.thickness[3], 35. + .2 / .15 + 1e-10)
        self.assertLess(mesh.thickness.max(), 70.)
        self.assertAlmostEqual((mesh.volume_km3(area) + exported) / before, 1., places=12)
        self.assertAlmostEqual((settled + exported) / removed, 1., places=12)

    def test_submarine_sediment_does_not_rebuild_a_thinned_continental_basement(self):
        nodes = sphere.from_lonlat([0., 1., 0., 0.], [0., 0., 1., -1e-8])
        mesh = ContinentMesh(nodes, [[0, 1, 2], [1, 0, 3]], [70., 70., 70., 30.])
        removed, settled, exported = mesh.erode(1., mesh.area_km2(), bounded_deposition=True)
        self.assertEqual(settled, 0.)
        self.assertEqual(exported, removed)
        self.assertEqual(mesh.thickness[3], 30.)

    def test_thicker_crust_spreads_faster_without_a_height_clamp_or_volume_loss(self):
        import copy
        nodes = sphere.from_lonlat([0., 4., 4., 0.], [0., 0., 4., 4.])
        constant = ContinentMesh(nodes, [[0, 1, 2], [0, 2, 3]], [35., 35., 120., 35.])
        gravity = copy.deepcopy(constant)
        area = gravity.area_km2()
        volume = gravity.volume_km3(area)
        constant.spread(2.)
        gravity.spread(2., gravity_exponent=3.)
        self.assertLess(gravity.thickness.max(), constant.thickness.max())
        self.assertGreaterEqual(gravity.thickness.min(), 35. - 1e-10)
        self.assertAlmostEqual(gravity.volume_km3(area) / volume, 1., places=12)

    def test_thin_margins_relax_more_slowly_than_a_constant_mobility_crust(self):
        import copy
        nodes = sphere.from_lonlat([0., 4., 4., 0.], [0., 0., 4., 4.])
        constant = ContinentMesh(nodes, [[0, 1, 2], [0, 2, 3]], [28., 28., 38., 38.])
        margin = copy.deepcopy(constant)
        constant.spread(10.)
        margin.spread(10., gravity_exponent=3.)
        self.assertLess(np.linalg.norm(margin.thickness - [28., 28., 38., 38.]),
                        np.linalg.norm(constant.thickness - [28., 28., 38., 38.]))

    def test_large_diffusion_step_does_not_transfer_volume_between_components(self):
        shape = sphere.union(polygon([0, 5, 5, 0], [0, 0, 5, 5]),
                             polygon([30, 35, 35, 30], [0, 0, 5, 5]))
        mesh = ContinentMesh.build(shape, 35.)
        lon, _ = sphere.to_lonlat(mesh.nodes)
        left = lon < 15.
        mesh.thickness = np.where(left, np.linspace(35., 65., len(lon)),
                                  np.linspace(70., 100., len(lon)))
        weights = mesh.node_areas_km2()
        volumes = [float((weights[mask] * mesh.thickness[mask]).sum()) for mask in (left, ~left)]
        mesh.spread(100.)
        for mask, volume in zip((left, ~left), volumes):
            self.assertAlmostEqual(float((weights[mask] * mesh.thickness[mask]).sum()) / volume,
                                   1., places=12)

    def test_recut_preserves_target_volume_and_new_domain(self):
        shape = slit()
        mesh = ContinentMesh.build(shape, 35.)
        mesh.thickness = np.linspace(35., 70., len(mesh.nodes))
        target = mesh.volume_km3(sphere.area_km2(shape))
        cut = sphere.intersection(shape, polygon([-1, 9, 9, -1], [-1, -1, 11, 11]))
        recut = mesh.recut(cut, sphere.area_km2(cut), target)
        self.assert_domain(cut, recut)
        self.assertAlmostEqual(recut.volume_km3(sphere.area_km2(cut)) / target, 1., places=12)


if __name__ == '__main__':
    unittest.main()
