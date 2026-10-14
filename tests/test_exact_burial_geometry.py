"""Exact physical polygons survive coordinate collapse, reuse, and workers."""
import copy
from dataclasses import FrozenInstanceError
import pickle
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import collision_interface
from exact_polygon import ExactPolygon
import mesh_coverage
from parallel_runtime import RuntimePool
from tests.test_burial_partition_precision import independent_gnomonic_regions


# Lower face2436 and upper faces2098/2099 during the105->106 Myr step.
# Cover1 survives exact clipping but its binary64 area misses the unchanged
# regional agreement bound, even though the full exact partition conserves.
TRIANGLES_106=np.array([
    [[.27217024750971547,.43049126126532106,-.8605815651898974],
     [.2801525563384552,.42137651289896205,-.8625290601215357],
     [.28015250043655804,.42137645057379336,-.8625291087267583]],
    [[.2798265393261918,.4500440486369277,-.8480315219231004],
     [.28422609132204396,.4324657892858658,-.8556803551029708],
     [.2725595831245633,.4227349916692275,-.864295320168623]],
    [[.2725595831245633,.4227349916692275,-.864295320168623],
     [.25651611766846716,.43061426186856794,-.8653154562653355],
     [.2798265393261918,.4500440486369277,-.8480315219231004]],
])
REFERENCE_106=.023706547138341614
RADIUS=6371.


def captured_partition():
    return burial_depth.partition_face(
        TRIANGLES_106,0,np.array([0,1]),np.array([1,2]),REFERENCE_106,RADIUS)


def collapsed_polygon(width=1):
    scale=10**30
    return ExactPolygon(((scale,scale,scale,scale),
                         (scale,scale+10**25,scale,scale),
                         (scale,scale+10**25,scale+width,scale)))


class ExactBurialGeometryTests(unittest.TestCase):
    def setUp(self):
        burial_depth.geometry_cache_reset()

    def tearDown(self):
        burial_depth.geometry_cache_reset()

    def test_106_preserves_unrepresentable_region_and_conserves_source(self):
        source=TRIANGLES_106.copy()
        regions,areas,error=captured_partition()
        expected=independent_gnomonic_regions(source)
        actual={cover:float(area) for (_,cover),area in zip(regions,areas)}
        self.assertEqual(set(actual),set(expected))
        np.testing.assert_allclose([actual[key] for key in sorted(expected)],
                                   [expected[key] for key in sorted(expected)],rtol=2e-13,atol=0.)
        self.assertLess(error,2e-10)
        self.assertGreater(error,0.)  # The source area is never forced to close.
        exact=[(polygon,area) for (polygon,cover),area in zip(regions,areas) if cover==(1,)]
        self.assertEqual(len(exact),1)
        polygon,area=exact[0]
        self.assertIsInstance(polygon,ExactPolygon)
        rounded=mesh_coverage._polygon_area(polygon.represented(),RADIUS)
        self.assertGreater(abs(rounded-area),2e-9*area+REFERENCE_106*2e-11)
        metric=collision_interface.rotation_metric(polygon,RADIUS)
        self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(),0.)
        self.assertAlmostEqual(np.trace(metric)/2e6,area,delta=8*np.finfo(float).eps*area)
        # The other regions retain their established binary64 representation.
        self.assertTrue(all(isinstance(p,np.ndarray) for p,cover in regions if cover!=(1,)))
        np.testing.assert_array_equal(source,TRIANGLES_106)

    def test_immutable_rays_survive_pickle_without_implicit_projection(self):
        polygon=collapsed_polygon()
        self.assertGreater(polygon.solid_angle(),0.)
        self.assertFalse(burial_depth._positive_binary64_winding(polygon.represented()))
        with self.assertRaises(FrozenInstanceError):polygon.homogeneous=()
        with self.assertRaises(TypeError):polygon.homogeneous[0][0]=0
        with self.assertRaisesRegex(TypeError,'diagnostic only'):np.asarray(polygon,float)
        for restored in (copy.deepcopy(polygon),pickle.loads(pickle.dumps(polygon))):
            self.assertEqual(restored.homogeneous,polygon.homogeneous)
            self.assertEqual(restored.solid_angle(),polygon.solid_angle())
        projected=polygon.represented();projected[:]=0
        self.assertTrue(np.any(polygon.represented()!=0))

    def test_cache_keys_distinguish_rays_with_identical_binary64_projection(self):
        first,second=collapsed_polygon(),collapsed_polygon(2)
        np.testing.assert_array_equal(first.represented(),second.represented())
        one=collision_interface.rotation_metric(first,RADIUS)
        two=collision_interface.rotation_metric(second,RADIUS)
        np.testing.assert_allclose(two,2*one,rtol=2e-14,atol=0.)
        stats=burial_depth.geometry_cache_statistics()
        self.assertEqual(stats['metric_misses'],2)
        repeated=collision_interface.rotation_metric(pickle.loads(pickle.dumps(first)),RADIUS)
        np.testing.assert_array_equal(repeated,one)
        self.assertEqual(burial_depth.geometry_cache_statistics()['metric_hits'],1)
        repeated[:]=0
        np.testing.assert_array_equal(collision_interface.rotation_metric(first,RADIUS),one)

    def test_cache_counts_integer_payload_and_invalidates_changed_integral(self):
        polygon=collapsed_polygon()
        expected=sum(max(16,(value.bit_length()+7)//8) for point in polygon.homogeneous for value in point)
        self.assertEqual(burial_depth._geometry_payload(polygon),expected)
        huge=ExactPolygon(tuple(tuple(value*10**1000 for value in point) for point in polygon.homogeneous))
        self.assertGreater(burial_depth._geometry_payload(huge),4096)
        one=collision_interface.rotation_metric(polygon,RADIUS)
        original=ExactPolygon.solid_angle
        def refreshed(self):return original(self)
        with patch.object(ExactPolygon,'solid_angle',refreshed):
            np.testing.assert_array_equal(collision_interface.rotation_metric(polygon,RADIUS),one)
        self.assertEqual(burial_depth.geometry_cache_statistics()['source_invalidations'],1)

    def test_exact_regions_cross_worker_boundary_and_are_reused(self):
        expected=captured_partition()
        burial_depth.geometry_cache_reset()
        jobs=[(0,np.array([0,1]),REFERENCE_106,RADIUS)]*burial_depth.MIN_PARALLEL_PARTITION_FACES
        with RuntimePool(workers=2,max_workers=2):
            session=burial_depth.partition_geometry_session(TRIANGLES_106,np.array([1,2]))
            results=list(session.partition_faces(jobs))
        for regions,areas,error in results:
            np.testing.assert_array_equal(areas,expected[1])
            self.assertEqual(error,expected[2])
            for (polygon,cover),(want,want_cover) in zip(regions,expected[0]):
                self.assertEqual(cover,want_cover)
                if isinstance(want,ExactPolygon):self.assertEqual(polygon.homogeneous,want.homogeneous)
                else:np.testing.assert_array_equal(polygon,want)
        reused=burial_depth.partition_geometry_session(TRIANGLES_106,np.array([1,2])).partition_face(*jobs[0])
        np.testing.assert_array_equal(reused[1],expected[1])
        self.assertGreater(burial_depth.geometry_cache_statistics()['partition_hits'],0)


if __name__=='__main__':unittest.main()
