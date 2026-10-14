"""Source-ray geometry survives partitions of long, narrow stack regions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import burial_depth
import collision_interface
import mesh_coverage
from tests.test_small_spherical_faces import decimal_area, decimal_overlap


# Lower face 2436 and upper face 2098 at the 104->105 Myr coarse step.
# Winding stays positive, but independently rounded cut vertices shift the
# area of this long, narrow face beyond the whole-footprint conservation gate.
TRIANGLES=np.array([
    [[.275983334858385,.43050711189432017,-.8593583801237153],
     [.2839539501061702,.42137760936269714,-.8612845433112573],
     [.28395389424617573,.4213775471819377,-.861284592149056]],
    [[.2813999826907794,.4514352942925299,-.8467704676053872],
     [.2858233949312348,.4338704422332845,-.8544363207789551],
     [.27417929814656405,.42414735452946445,-.8630902236229503]],
])
SOURCE_AREA=.023706547125274265
RADIUS=6371.


def partition(reference=SOURCE_AREA):
    return burial_depth._partition_face_uncached(
        TRIANGLES,0,np.array([0]),np.array([1]),reference,RADIUS)


class BurialAreaPrecisionTests(unittest.TestCase):
    def test_good_winding_area_failure_uses_source_ray_integrals(self):
        source=TRIANGLES.copy()
        inside,outside=burial_depth._split(TRIANGLES[0],
            mesh_coverage._triangle_planes(TRIANGLES[1]),RADIUS,TRIANGLES[1])
        ordinary=outside+[inside]
        self.assertFalse(any(burial_depth._uncertain_winding(p) for p in ordinary))
        rounded=sum(mesh_coverage._polygon_area(p,RADIUS) for p in ordinary)
        self.assertGreater(abs(rounded-SOURCE_AREA)/SOURCE_AREA,2e-10)

        with patch.object(burial_depth,'_precise_partition',wraps=burial_depth._precise_partition) as exact:
            regions,areas,error=partition()
        exact.assert_called_once()
        expected_source=decimal_area(TRIANGLES[0])
        expected_cover=decimal_overlap(*TRIANGLES)
        self.assertEqual([cover for _,cover in regions],[(),(0,)])
        np.testing.assert_allclose(areas,[expected_source-expected_cover,expected_cover],
                                   rtol=2e-13,atol=0.)
        self.assertLess(error,2e-10)
        self.assertGreater(error,0.)  # no rescaling to force exact source closure
        np.testing.assert_array_equal(TRIANGLES,source)

        # Safe binary64 projections retain the established regional contract.
        # Regions which cannot meet it must retain exact rays for their metric.
        for (polygon,_),area in zip(regions,areas):
            self.assertTrue(burial_depth._positive_binary64_winding(polygon))
            metric=collision_interface.rotation_metric(polygon,RADIUS)
            self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(),0.)
            metric_area=float(np.trace(metric))/2e6
            self.assertAlmostEqual(metric_area,float(area),
                delta=2e-9*float(area)+SOURCE_AREA*2e-11)

    def test_exact_fallback_does_not_repair_a_wrong_reference_area(self):
        with self.assertRaisesRegex(ValueError,'conserve their lower footprint'):
            partition(SOURCE_AREA*1.001)

    def test_bad_binary64_region_cannot_hide_behind_exact_area(self):
        regions,areas=burial_depth._precise_partition(
            TRIANGLES,0,np.array([0]),np.array([1]),radius=RADIUS)
        altered=[]
        for polygon,cover in regions:
            # A positive but shrunken return polygon no longer represents the
            # exact source region; an exact area sidecar cannot legitimize it.
            if isinstance(polygon,burial_depth.ExactPolygon):polygon=polygon.represented()
            center=polygon.mean(axis=0)
            moved=mesh_coverage._unit(.99*polygon+.01*center)
            self.assertTrue(burial_depth._positive_binary64_winding(moved))
            altered.append((moved,cover))
        with patch.object(burial_depth,'_precise_partition',return_value=(altered,areas)):
            with self.assertRaisesRegex(ValueError,'binary64 area representation'):
                partition()

    def test_burial_and_interface_agree_with_the_overlap_cache(self):
        vertices=TRIANGLES.reshape(-1,3)
        faces=np.arange(6).reshape(-1,3)
        source_area=np.array([mesh_coverage._polygon_area(t,RADIUS) for t in TRIANGLES])
        overlap=mesh_coverage.material_overlaps(vertices,faces,np.array([0,1]))
        self.assertEqual(len(overlap['area_km2']),1)
        pairs=(np.array([1]),np.array([0]),overlap['area_km2'],np.array([7]))
        s=SimpleNamespace(material_surface=dict(vertices=vertices,faces=faces,
            area_km2=source_area,radius_km=RADIUS),parcel_collision_sheet=np.array([0,1]),
            parcel_plate=np.array([0,1]),parcel_burial_myr=np.full(2,100.),
            parcel_root_age_myr=np.full(2,100.),collision_contacts=[])
        eligible,scope,_=burial_depth.integrate(s,np.array([30.,10.]),pairs,
            depth_km=20.,heating_delay_myr=5.)
        expected_cover=decimal_overlap(*TRIANGLES)
        expected_lower_volume=decimal_area(TRIANGLES[0])*10.+expected_cover*10.
        self.assertAlmostEqual(float(eligible[0]*SOURCE_AREA),expected_lower_volume,delta=1e-13)
        self.assertLess(scope['maximum_region_area_relative_error'],2e-10)
        with patch.object(collision_interface.eclogite_sink,'_depth_pairs',return_value=pairs):
            rows=collision_interface.regions(s)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]['area_km2'],expected_cover,delta=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(rows[0]['metric_m2']).min(),0.)


if __name__=='__main__':
    unittest.main()
