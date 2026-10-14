"""Exact regions retain positive fans even when their binary64 projection fails."""
import unittest

import numpy as np

import burial_depth
import collision_interface
import mesh_coverage
from tests.test_small_spherical_faces import decimal_overlap
from tests.test_collision_interface import independent_patch_work


# Fixed binary64 source pair from seeded orientation20 of the combined PR203/
# strict-physical-work/PR204 covariance sweep. Independent of private saves.
TRIANGLES=np.array([
    [[-.28256626158347503,-.5540472328576319,-.783063197690672],
     [-.21795486178323611,-.6121069435575915,-.7601452281463286],
     [-.28044107817040365,-.6042185128315223,-.7458369730888149]],
    [[-.15797067978372062,-.6172215602009927,-.7707676757309699],
     [-.2213241044474705,-.6117492210177828,-.7594593678233624],
     [-.22356992091375333,-.5614171554742231,-.7967604834590193]],
])


class BurialRoundoffFanTests(unittest.TestCase):
    def test_large_uncovered_region_survives_a_tiny_unrepresentable_fan(self):
        source_area=mesh_coverage._polygon_area(TRIANGLES[0],6371.)
        regions,areas,error=burial_depth.partition_face(
            TRIANGLES,0,np.array([0]),np.array([1]),source_area,6371.)
        self.assertLess(error,2e-10)  # unchanged production conservation gate
        covered=sum(area for (_,cover),area in zip(regions,areas) if cover)
        expected=decimal_overlap(TRIANGLES[0],TRIANGLES[1])
        self.assertAlmostEqual(covered,expected,delta=2e-8)
        self.assertGreater(sum(area for (_,cover),area in zip(regions,areas) if not cover),81000.)
        for polygon,cover in regions:
            if isinstance(polygon,burial_depth.ExactPolygon):
                self.assertGreater(polygon.solid_angle(),0.)
            else:self.assertTrue(burial_depth._positive_binary64_winding(polygon))
            metric=collision_interface.rotation_metric(polygon,6371.)
            self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(),0.)

    def test_covered_subdivision_preserves_stack_and_additive_interface_work(self):
        # Admit a complete upper cover before the cutting cover. The same
        # formerly discarded large polygon now carries a real interface.
        triangles=np.concatenate((TRIANGLES,TRIANGLES[[0]]))
        source_area=mesh_coverage._polygon_area(triangles[0],6371.)
        regions,areas,error=burial_depth.partition_face(
            triangles,0,np.array([0,1]),np.array([2,1]),source_area,6371.)
        self.assertLess(error,2e-10)
        self.assertTrue(all(0 in cover for _,cover in regions))
        self.assertGreater(sum(area for (_,cover),area in zip(regions,areas)
                               if cover==(0,)),81000.)
        second_cover=sum(area for (_,cover),area in zip(regions,areas) if 1 in cover)
        self.assertAlmostEqual(second_cover,decimal_overlap(*TRIANGLES),delta=2e-8)
        metric=sum(collision_interface.rotation_metric(polygon,6371.)
                   for polygon,_ in regions)
        self.assertGreater(np.linalg.eigvalsh(metric).min(),0.)
        centroid=triangles[0].sum(axis=0);centroid/=np.linalg.norm(centroid)
        for omega in (*np.eye(3),centroid):
            # Independent positive surface quadrature uses a 70-digit
            # Jacobian, without the implementation's boundary-moment formula.
            expected=independent_patch_work(triangles[0],omega)*(6371.e3)**2
            actual=float(omega@metric@omega)
            tolerance=32*np.finfo(float).eps*np.linalg.norm(metric)*float(omega@omega)
            self.assertAlmostEqual(actual,expected,delta=tolerance)


if __name__=='__main__':unittest.main()
