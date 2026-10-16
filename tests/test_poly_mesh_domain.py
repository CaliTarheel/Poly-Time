"""The continental field must follow its material domain, including narrow seas and holes."""
import unittest

import numpy as np

from poly import sphere
from poly.mesh import ContinentMesh


def polygon(lon, lat):
    return sphere.polygon(sphere.from_lonlat(lon, lat))


def slit():
    return polygon([0, 10, 10, 5.1, 5.1, 4.9, 4.9, 0], [0, 0, 10, 10, 2, 2, 10, 10])


class MeshDomainTests(unittest.TestCase):
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

    def test_clipped_mesh_diffusion_preserves_volume(self):
        shape = slit()
        mesh = ContinentMesh.build(shape, 35.)
        mesh.thickness = np.linspace(35., 70., len(mesh.nodes))
        volume = mesh.volume_km3(sphere.area_km2(shape))
        mesh.spread(2.)
        self.assertAlmostEqual(mesh.volume_km3(sphere.area_km2(shape)) / volume, 1., places=12)

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
