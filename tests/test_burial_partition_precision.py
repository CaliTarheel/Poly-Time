"""Guard-digit burial partition checked against independent planar clipping."""
from decimal import Decimal,localcontext
import math
from types import SimpleNamespace
import unittest
import numpy as np

import burial_depth
import collision_interface
import mesh_coverage

TRIANGLES = np.array([
    [
        [0.5778844589152441, -0.5751905995830088, -0.5789691928726214],
        [0.5785437574872386, -0.5739978668760287, -0.5794942359457221],
        [0.5786290538034823, -0.5749296437802208, -0.5784845052352183],
    ],
    [
        [0.5829835677783053, -0.5730763713227064, -0.5759458588548733],
        [0.5716651305483145, -0.5865590469809336, -0.5737137464973256],
        [0.577060546391007, -0.5684599244214354, -0.5863910300522072],
    ],
    [
        [0.5782005387688327, -0.5737656594380767, -0.5800664660338681],
        [0.5790132991308522, -0.5741975657326923, -0.5788270509714081],
        [0.57678324868433, -0.574905127230622, -0.580349186887595],
    ],
    [
        [0.5764172928077288, -0.5752173774009198, -0.5804033711895692],
        [0.5780781655343524, -0.5749043877851449, -0.5790600827529262],
        [0.5772779139082869, -0.5754988812244631, -0.5792678549885951],
    ],
    [
        [0.5780781655343524, -0.5749043877851449, -0.5790600827529262],
        [0.5798732752684231, -0.5745625011395965, -0.5776027327789254],
        [0.578209546247736, -0.5757991766385411, -0.5780389509456656],
    ],
    [
        [0.5780781655343524, -0.5749043877851449, -0.5790600827529262],
        [0.578209546247736, -0.5757991766385411, -0.5780389509456656],
        [0.5772779139082869, -0.5754988812244631, -0.5792678549885951],
    ],
    [
        [0.5766559288417431, -0.574516796692021, -0.5808600434273349],
        [0.5789873130763183, -0.573769277817947, -0.5792775734731467],
        [0.5776527993660187, -0.5745220007838338, -0.5798635304965668],
    ],
    [
        [0.5764172928077288, -0.5752173774009198, -0.5804033711895692],
        [0.5776527993660187, -0.5745220007838338, -0.5798635304965668],
        [0.5780781655343524, -0.5749043877851449, -0.5790600827529262],
    ],
    [
        [0.5776527993660187, -0.5745220007838338, -0.5798635304965668],
        [0.5789873130763183, -0.573769277817947, -0.5792775734731467],
        [0.5798732752684231, -0.5745625011395965, -0.5776027327789254],
    ],
    [
        [0.5776527993660187, -0.5745220007838338, -0.5798635304965668],
        [0.5798732752684231, -0.5745625011395965, -0.5776027327789254],
        [0.5780781655343524, -0.5749043877851449, -0.5790600827529262],
    ],
])

# Minimal source set around material face 5717 in the stopped 52 Myr run.
# The two upper faces share a represented vertex; guard-digit clipping creates
# three extra regions whose winding collapses only when returned to binary64.
RUN_52_TRIANGLES = np.array([
    [[.4478538142863678,-.7380890190219179,-.5046301229895164],
     [.4488601717737763,-.7474283039885316,-.48977084089607004],
     [.4337261049791398,-.7454645935346123,-.5061266695659293]],
    [[.44498579335191296,-.7365232520404762,-.5094321769565502],
     [.4429049499273482,-.7461530375224076,-.4970823371695456],
     [.4440083829586744,-.736993145909962,-.5096053951285845]],
    [[.4429049499273482,-.7461530375224076,-.4970823371695456],
     [.4407135657447927,-.755596597218517,-.4846084349661162],
     [.4440083829586744,-.736993145909962,-.5096053951285845]],
])


def independent_gnomonic_regions(triangles):
    """Great circles become straight lines: use rational planar cuts at 80 digits."""
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def area2(poly):return sum(cross(a,b) for a,b in zip(poly,poly[1:]+poly[:1]))
    def clip(poly,a,b,side):
        if not poly:return []
        edge=sub(b,a);d=[side*cross(edge,sub(p,a)) for p in poly];result=[]
        inside=[value>=0 for value in d]
        for i,p in enumerate(poly):
            previous=(i-1)%len(poly)
            if inside[i]!=inside[previous]:
                fraction=d[previous]/(d[previous]-d[i])
                result.append([x*(1-fraction)+y*fraction for x,y in zip(poly[previous],p)])
            if inside[i]:result.append(p)
        return result
    with localcontext() as context:
        context.prec=80
        projected=[]
        for triangle in triangles:
            points=[]
            for point in triangle:
                x,y,z=[Decimal.from_float(float(v)) for v in point]
                points.append([x/(-z),y/z])
            projected.append(points)
        regions=[(projected[0],())]
        for pair,triangle in enumerate(projected[1:]):
            result=[]
            for polygon,cover in regions:
                inside=polygon
                for a,b in zip(triangle,triangle[1:]+triangle[:1]):
                    outside=clip(inside,a,b,-1)
                    if area2(outside)>Decimal('1e-65'):result.append((outside,cover))
                    inside=clip(inside,a,b,1)
                if area2(inside)>Decimal('1e-65'):result.append((inside,cover+(pair,)))
            regions=result
        totals={}
        for polygon,cover in regions:
            points=[]
            for u,v in polygon:
                length=(u*u+v*v+1).sqrt();points.append([u/length,-v/length,-1/length])
            area=0.
            a=points[0]
            for b,c in zip(points[1:-1],points[2:]):
                determinant=(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])
                             +a[2]*(b[0]*c[1]-b[1]*c[0]))
                denominator=1+sum(a[i]*b[i]+b[i]*c[i]+c[i]*a[i] for i in range(3))
                area+=2*math.atan2(float(determinant),float(denominator))*6371.**2
            totals[cover]=totals.get(cover,0.)+area
        return totals


class BurialPartitionPrecisionTests(unittest.TestCase):
    def test_guard_digit_return_discards_only_unrepresentable_winding(self):
        collapsed=np.array([
            [0.4429049499273482,-0.7461530375224076,-0.4970823371695456],
            [0.44425351651949885,-0.7399922989763484,-0.5050249602898433],
            [0.4442535165194989,-0.7399922989763484,-0.5050249602898433],
            [0.4429049499273482,-0.7461530375224077,-0.49708233716954553],
        ])
        self.assertFalse(burial_depth._positive_binary64_winding(collapsed))
        with self.assertRaisesRegex(ValueError,'winding'):
            collision_interface.rotation_metric(collapsed,6371.)

        # No geometric epsilon is used: a much smaller but representably
        # positive patch remains available to the exact rotation integral.
        width=1e-14
        thin=np.array([[1.,-.03,0.],[1.,.03,0.],[1.,.03,width],[1.,-.03,width]])
        thin/=np.linalg.norm(thin,axis=1)[:,None]
        self.assertTrue(burial_depth._positive_binary64_winding(thin))
        self.assertTrue(np.all(np.linalg.eigvalsh(
            collision_interface.rotation_metric(thin,.001))>0.))
        self.assertFalse(burial_depth._positive_binary64_winding(thin[::-1]))

    def test_52_myr_guard_digit_partition_retains_every_exact_positive_region(self):
        regions,areas,error=burial_depth.partition_face(
            RUN_52_TRIANGLES,0,np.arange(2),np.array([1,2]),
            5673.2372536289995,6371.)
        # True halfspaces retain a representable positive uncovered sliver
        # that the former -5e-14 band swallowed. Its interface remains passive.
        self.assertEqual(len(regions),6)
        tiny=int(np.argmin(areas))
        self.assertEqual(regions[tiny][1],())
        self.assertGreater(areas[tiny],0.)
        self.assertLess(areas[tiny],2e-11)
        # The formerly unrepresentable positive region now keeps exact rays;
        # no vertex cast discards area, even below the source rounding budget.
        self.assertIsInstance(regions[tiny][0],burial_depth.ExactPolygon)
        self.assertLess(error,16*np.finfo(float).eps)
        represented=np.zeros(2)
        for (polygon,cover),area in zip(regions,areas):
            if isinstance(polygon,burial_depth.ExactPolygon):self.assertGreater(polygon.solid_angle(),0.)
            else:self.assertTrue(burial_depth._positive_binary64_winding(polygon))
            self.assertTrue(np.isfinite(collision_interface.rotation_metric(polygon,6371.)).all())
            represented[list(cover)]+=area
        np.testing.assert_allclose(represented,[139.5918501186568,161.6081539824471],
                                   rtol=2e-13,atol=2e-11)

    def test_52_myr_partition_remains_valid_for_shared_burial_integration(self):
        vertices=RUN_52_TRIANGLES.reshape(-1,3)
        faces=np.arange(len(vertices)).reshape(-1,3)
        areas=np.array([mesh_coverage._polygon_area(triangle,6371.)
                        for triangle in RUN_52_TRIANGLES])
        s=SimpleNamespace(material_surface=dict(vertices=vertices,faces=faces,
                          area_km2=areas,radius_km=6371.),
            parcel_collision_sheet=np.arange(3),parcel_burial_myr=np.full(3,100.),
            parcel_root_age_myr=np.full(3,100.))
        pair_area=np.array([139.5918501186568,161.6081539824471])
        pairs=(np.array([1,2]),np.array([0,0]),pair_area,np.array([10,10]))
        eligible,scope,attribution=burial_depth.integrate(
            s,np.array([30.,10.,10.]),pairs,depth_km=20.,heating_delay_myr=5.)
        self.assertEqual(scope['local_stack_regions'],6)
        self.assertLess(scope['maximum_region_area_relative_error'],16*np.finfo(float).eps)
        self.assertAlmostEqual(scope['covered_union_area_km2'],float(pair_area.sum()),places=9)
        self.assertTrue(np.isfinite(eligible).all())
        self.assertTrue(np.isfinite(attribution['eligible_pair_volume_km3']).all())

    def test_saved_shared_vertex_partition_matches_independent_planar_geometry(self):
        triangles=TRIANGLES.copy();original=triangles.copy()
        selected=np.arange(9);upper=np.arange(1,10)
        regions,areas,error=burial_depth.partition_face(triangles,0,selected,upper,25.094560173568368,6371.)
        expected=independent_gnomonic_regions(triangles)
        actual={}
        for (polygon,cover),area in zip(regions,areas):
            actual[cover]=actual.get(cover,0.)+area
            metric=collision_interface.rotation_metric(polygon,6371.)
            # A valid interface must retain nonnegative resisting work.
            self.assertGreaterEqual(float(np.linalg.eigvalsh(metric).min()),-2e-11*float(np.trace(metric)))
        self.assertEqual(set(actual),set(expected))
        np.testing.assert_allclose([actual[key] for key in sorted(expected)],
                                   [expected[key] for key in sorted(expected)],rtol=2e-9,atol=8e-10)
        self.assertEqual(len(regions),13)
        self.assertGreater(float(areas.min()),2e-7)
        self.assertLess(error,2e-10)
        np.testing.assert_array_equal(triangles,original)
        # No interface reorientation or weakened winding rejection is used.
        with self.assertRaisesRegex(ValueError,'winding'):
            collision_interface.rotation_metric(regions[0][0][::-1],6371.)

    def test_reversed_source_material_is_rejected_instead_of_reoriented(self):
        triangles=TRIANGLES.copy();triangles[0]=triangles[0,::-1]
        with self.assertRaisesRegex(ValueError,'positively wound'):
            burial_depth.partition_face(triangles,0,np.arange(9),np.arange(1,10),25.094560173568368,6371.)

    def test_resolved_small_positive_patch_is_retained(self):
        scale=1e-7
        small=np.array([[1.,-scale,-scale],[1.,scale,-scale],[1.,0.,scale]])
        small/=np.linalg.norm(small,axis=1)[:,None]
        triangles=np.array([small,small])
        area=mesh_coverage._polygon_area(small,6371.)
        regions,areas,error=burial_depth.partition_face(triangles,0,np.array([0]),np.array([1]),area,6371.)
        self.assertGreater(area,0.)
        self.assertAlmostEqual(float(areas.sum())/area,1.,places=13)
        self.assertEqual([cover for _,cover in regions],[(0,)])
        self.assertLess(error,2e-10)


if __name__=='__main__':unittest.main()
