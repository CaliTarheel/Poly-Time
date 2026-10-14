"""Physical halfspaces preserve real slivers and reject tolerance-only area."""
from fractions import Fraction
import unittest

import numpy as np

import mesh_coverage as coverage
from spherical_predicates import edge_distances,plane_distances
from tests.test_small_spherical_faces import decimal_overlap,unit


def exact_distance(a,b,p):
    a,b,p=[[Fraction(float(v)) for v in point] for point in (a,b,p)]
    return sum((a[i]*(b[(i+1)%3]*p[(i+2)%3]-b[(i+2)%3]*p[(i+1)%3])
                for i in range(3)),Fraction(0))


class SphericalPredicateTests(unittest.TestCase):
    def test_broadcast_sign_matches_exact_original_edges_and_skips_padding(self):
        rng=np.random.default_rng(205)
        a=unit(rng.normal(size=(20,3)))
        b=unit(a+1e-6*rng.normal(size=(20,3)))
        p=unit(a[:,None,:]+rng.normal(size=(20,7,3))*1e-6)
        p[:,0]=a;p[:,1]=b;p[:,2]=0.
        valid=np.ones((20,7),bool);valid[:,2]=False
        stats={};actual=edge_distances(p,a[:,None,:],b[:,None,:],valid=valid,stats=stats)
        for row in range(20):
            for slot in range(7):
                expected=exact_distance(a[row],b[row],p[row,slot]) if valid[row,slot] else 0
                self.assertEqual(np.sign(actual[row,slot]),(expected>0)-(expected<0))
        self.assertEqual(stats['incident_shortcuts'],40)
        self.assertLessEqual(stats['exact_fallbacks'],80)

    def test_exact_fallback_retains_distinct_near_plane_signs(self):
        a=unit([.4,-.1,1.]);b=unit([.6,.2,1.])
        midpoint=unit(a+b)
        p=np.array([midpoint,np.nextafter(midpoint,np.inf),np.nextafter(midpoint,-np.inf)])
        stats={};actual=edge_distances(p,a,b,stats=stats)
        expected=np.array([float(exact_distance(a,b,row)) for row in p])
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(stats['exact_fallbacks'],3)

    def test_exact_fallback_values_are_the_rationals_rounded_once(self):
        rng=np.random.default_rng(31)
        a=unit(rng.normal(size=(40,3)));b=unit(a+1e-3*rng.normal(size=(40,3)))
        # Points of each edge's own great circle, rounded: every distance is in doubt.
        weights=rng.uniform(.1,.9,size=(40,9,1))
        p=unit(weights*a[:,None,:]+(1-weights)*b[:,None,:])
        stats={};actual=edge_distances(p,a[:,None,:],b[:,None,:],stats=stats)
        expected=np.array([[float(exact_distance(a[row],b[row],p[row,slot])) for slot in range(9)]
                           for row in range(40)])
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(stats['exact_fallbacks'],360)
        self.assertTrue(np.any(actual>0.) and np.any(actual<0.))

    def test_plane_only_predicate_describes_stored_normal(self):
        normal=np.array([1.,1.,-1.])
        p=np.array([[1.,2.,3.],[1.,2.,np.nextafter(3.,np.inf)]])
        np.testing.assert_array_equal(plane_distances(p,normal),[0.,-np.spacing(3.)])

    def test_physical_scalar_and_batch_area_agree_below_old_inside_band(self):
        other=unit([[-.2,0.,1.],[.2,0.,1.],[0.,.2,1.]])
        for offsets in [(-4e-14,-6e-14),(-6e-14,-4e-14),(4e-14,6e-14),(-1e-16,-2e-16)]:
            first=unit([[-.05,offsets[0],1.],[.05,offsets[1],1.],[0.,.03,1.]])
            expected=decimal_overlap(first,other)
            for subject,clipper in [(first,other),(other,first)]:
                scalar=coverage._polygon_area(coverage.clip_triangle(subject,clipper),6371.)
                batch=coverage._intersection_areas(subject[None],clipper[None],6371.)[0]
                self.assertLess(abs(scalar-expected),1e-9)
                self.assertLess(abs(batch-expected),1e-9)

    def test_force_stencil_retains_tiny_and_zero_pairs_without_contact_inventory(self):
        first=unit([[1.,-.1,0.],[1.,.1,0.],[1.,0.,.1]])
        # Opposite halfspaces share one edge, with no current buried area.
        touching=first[[1,0,2]].copy();touching[2,2]*=-1.
        for width in (0.,1e-16):
            other=touching.copy();other[:2,2]=width;other=unit(other)
            vertices=np.vstack((first,other));faces=np.arange(6).reshape(2,3);sheets=np.arange(2)
            ledger=coverage.material_overlaps(vertices,faces,sheets)
            stencil=coverage.material_overlaps(vertices,faces,sheets,include_touching=True)
            self.assertEqual(len(ledger['first']),0)
            self.assertEqual(len(stencil['first']),1)
            self.assertTrue(stencil['complete_force_stencil'])
            self.assertGreaterEqual(stencil['area_km2'][0],0.)
            if width:self.assertGreater(stencil['area_km2'][0],0.)


if __name__=='__main__':unittest.main()
