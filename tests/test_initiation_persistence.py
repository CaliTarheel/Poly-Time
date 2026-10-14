"""Finite fault history survives motion without training new support.

The compact fixture contains only saved 173/174 Myr candidate arc coordinates
from Asein35 seed14 run 20261007-182358-d0c22f. The older retired record was
already carried to174 before the failed match; no live checkpoint is needed.
"""
from copy import deepcopy
from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import effective_subduction_initiation as initiation
import checkpoint
from tests.test_effective_subduction_initiation import world,set_pieces,tick,motion,points


def records(starts,ends,uids=(11,22)):
    result=[]
    for i,(a,b) in enumerate(zip(starts,ends)):
        middle=a+b;middle/=np.linalg.norm(middle)
        length=6371.*math.atan2(float(np.linalg.norm(np.cross(a,b-a))),float(a@b))
        result.append(dict(start=a,end=b,mid=middle,index=i,parent=i,uids=uids,slots=(0,1),
            length=length,normal_speed=-12.,shear=12.,water=(1.,1.),age=(80.,40.)))
    return result


def carried(group):
    return dict(segments_start=[r['start'].tolist() for r in group],
                segments_end=[r['end'].tolist() for r in group])


def polar(x,y):
    value=np.array([x,y,1.]);return value/np.linalg.norm(value)


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent/'fixtures/initiation_persistence_174.npz') as data:
            cls.saved={key:data[key] for key in data.files}

    def test_captured_short_curved_arcs_cover_themselves_and_reversal(self):
        g=records(self.saved['self_start'],self.saved['self_end'])
        self.assertTrue(initiation._same_support(g,carried(g),120.))
        reverse=records(self.saved['self_end'],self.saved['self_start'])
        self.assertTrue(initiation._same_support(reverse,carried(g),120.))

    def test_captured_173_to174_core_keeps_clock_but_not_unobserved_ends(self):
        old=records(self.saved['old_start'],self.saved['old_end'])
        new=records(self.saved['new_start'],self.saved['new_end'])
        core=initiation._retained_core(new,carried(old),120.)
        length=sum(r['length'] for r in core)
        self.assertGreater(length,1490.);self.assertLess(length,1500.)
        self.assertLess(length,sum(r['length'] for r in new))
        s=world();s.omega[:]=0.;initiation.upgrade(s)
        with patch.object(initiation,'_pieces',return_value=old):tick(s)
        with patch.object(initiation,'_pieces',return_value=new):tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertEqual(row['phase'],'weakening');self.assertEqual(row['consecutive_myr'],1.)
        self.assertEqual(row['shortening_km'],12.);self.assertTrue(row['bounded_history'])
        self.assertEqual(len(s.effective_subduction_initiation['candidates']),1)
        self.assertTrue(initiation.enabled(s))

    def test_hole_between_endpoint_and_midpoint_is_not_covered(self):
        g=records(points([-.06]),points([.06]))
        support=records(points([-.06,-.025,.025]),points([-.03,.02,.06]))
        self.assertFalse(initiation._same_support(g,carried(support),120.))
        core=initiation._retained_core(g,carried(support),120.)
        self.assertEqual(len(core),3)
        self.assertAlmostEqual(sum(r['length'] for r in core),6371.*.11,places=7)

    def test_connected_bend_fills_only_its_missing_joint_sector(self):
        joint=polar(0.,0.);a=polar(-.05,0.);b=polar(0.,.05)
        support=records([a,joint],[joint,b])
        target=records([polar(.0002,-.001)],[polar(.001,-.0002)])
        with patch.object(initiation,'_bend_regions',return_value=[]):
            self.assertFalse(initiation._arcs_covered(np.array([target[0]['start']]),np.array([target[0]['end']]),
                np.array([a,joint]),np.array([joint,b]),120.))
        core=initiation._retained_core(target,carried(support),120.)
        self.assertEqual(len(core),1);self.assertTrue(core[0]['complete_piece'])

    def test_short_terminal_and_hairpin_ends_do_not_get_joint_caps(self):
        joint=polar(0.,0.)
        for far in (polar(0.,.05),polar(-.0004,.0002)):
            support=records([polar(-.0005,0.),joint],[joint,far])
            target=records([polar(-.003,-.001)],[polar(-.002,-.001)])
            self.assertFalse(initiation._retained_core(target,carried(support),120.))

    def test_duplicate_reversed_and_overlapping_arcs_do_not_cap_real_ends(self):
        end=polar(0.,0.);far=polar(.05,0.);near=polar(.025,0.)
        target=records([polar(-.002,0.)],[polar(-.001,0.)])
        for support in (records([end,far],[far,end]),records([end,end],[far,near])):
            self.assertFalse(initiation._retained_core(target,carried(support),120.))

    def test_rotated_short_overlaps_cannot_be_mistaken_for_internal_bends(self):
        for vector in ([1.,2.,3.],[-2.,.5,1.]):
            joint=np.asarray(vector);joint/=np.linalg.norm(joint)
            tangent=np.array([2.,-1.,.2]);tangent-=joint*(joint@tangent);tangent/=np.linalg.norm(tangent)
            def point(angle):return joint*math.cos(angle)+tangent*math.sin(angle)
            for scale in (.05,1e-4,1e-6,1e-8):
                with self.subTest(vector=vector,scale=scale):
                    starts=np.array([joint,joint]);ends=np.array([point(scale),point(scale/2.)])
                    self.assertFalse(initiation._bend_regions(starts,ends,120.))
                    _,_,intervals=initiation._arc_intervals(point(-scale/10.),point(-scale/20.),starts,ends,120.)
                    self.assertFalse(intervals)

    def test_short_resolved_rotated_bend_is_not_rejected_by_conditioning_guard(self):
        joint=np.array([1.,2.,3.]);joint/=np.linalg.norm(joint)
        first=np.array([2.,-1.,.2]);first-=joint*(joint@first);first/=np.linalg.norm(first)
        second=np.cross(joint,first)
        starts=np.array([joint,joint])
        ends=np.array([joint*math.cos(1e-4)+direction*math.sin(1e-4) for direction in (first,second)])
        self.assertEqual(len(initiation._bend_regions(starts,ends,120.)),1)

    def test_tiny_clipped_endpoint_does_not_create_invalid_checkpoint_arc(self):
        g=records(points([-.05,.01]),points([0.,.02]))
        old=records(points([-.05,.01]),points([0.,.01+5e-13]))
        core=initiation._retained_core(g,carried(old),120.)
        self.assertEqual(len(core),1)
        self.assertGreater(np.linalg.norm(np.cross(core[0]['start'],core[0]['end'])),1e-12)

    def test_clipped_wet_fraction_is_a_lower_bound_not_parent_fraction(self):
        g=records(points([-.1]),points([.1]));g[0]['water']=(.8,1.)
        old=records(points([.08]),points([.1]))
        core=initiation._retained_core(g,carried(old),120.)
        self.assertAlmostEqual(core[0]['water'][0],0.)
        self.assertAlmostEqual(core[0]['water'][1],1.)

    def test_rotating_arc_bounds_survive_restriction_to_slow_end(self):
        a,b=points([-.6,.6]);g=records([a],[b])[0]
        g.update(normal=np.array([0.,1.,0.]),relative_omega=np.array([0.,.003,-.004]))
        v=np.cross(g['relative_omega'],g['mid'])*6371.
        g['normal_speed']=float(v@g['normal']);g['shear']=float(np.linalg.norm(v-g['normal_speed']*g['normal']))
        closing,shear=initiation._rate_bounds(g)
        for r in points(np.linspace(-.6,.6,101)):
            v=np.cross(g['relative_omega'],r)*6371.
            self.assertLessEqual(closing,max(0.,-float(v@g['normal']))+1e-12)
            self.assertLessEqual(shear,float(np.linalg.norm(v-(v@g['normal'])*g['normal']))+1e-12)
        self.assertLess(closing,-g['normal_speed'])

    def test_removed_fast_end_cannot_transfer_its_historical_mean(self):
        s=world();set_pieces(s,points_phi:=np.linspace(-.12,.12,7));s.omega[:]=0.
        group=records(points(points_phi[:-1]),points(points_phi[1:]))
        group[-1].update(normal_speed=-120.,shear=120.)
        initiation.upgrade(s)
        with patch.object(initiation,'_pieces',return_value=group):
            tick(s);tick(s);tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertGreater(row['shortening_km'],50.)
        with patch.object(initiation,'_pieces',return_value=group[:-1]):tick(s)
        self.assertEqual(row['last_seen_myr'],28.)  # transactions replace row objects
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertEqual(row['consecutive_myr'],3.)
        self.assertAlmostEqual(row['shortening_km'],36.)
        self.assertAlmostEqual(row['shear_slip_km'],36.)

    def test_split_and_merge_do_not_join_independent_clocks(self):
        for merge in (False,True):
            with self.subTest(merge=merge):
                s=world();s.omega[:]=0.;initiation.upgrade(s)
                joined=records(points([-.12,-.08,-.04,0.,.04,.08]),points([-.08,-.04,0.,.04,.08,.12]))
                divided=joined[:2]+joined[3:]
                before,after=(divided,joined) if merge else (joined,divided)
                with patch.object(initiation,'_pieces',return_value=before):tick(s);tick(s)
                old_ids=[r['id'] for r in s.effective_subduction_initiation['candidates']]
                with patch.object(initiation,'_pieces',return_value=after):tick(s)
                rows=s.effective_subduction_initiation['candidates']
                self.assertTrue(all(r['phase']=='retired' for r in rows if r['id'] in old_ids))
                self.assertTrue(all(r['shortening_km']==0. for r in rows if r['phase']=='weakening'))

    def test_partial_parent_birth_uses_only_trained_complete_interior(self):
        s=world();set_pieces(s,np.linspace(-.1,.1,9));initiation.upgrade(s);tick(s)
        # Extend the last parent beyond observed support; six interior parents
        # remain fully trained and connected, while the fringe is excluded.
        set_pieces(s,np.linspace(-.1,.125,10),list(range(8))+[7])
        for _ in range(10):tick(s)
        self.assertEqual(len(s.trench_systems),1)
        row=s.trench_systems[0]
        self.assertTrue(row['effective_subduction']['activation_pending'])
        self.assertLess(np.arcsin(np.asarray(row['geometry_xyz'])[:,2]).max(),.1)
        self.assertFalse(np.any(s.trench_maturity))

    def test_legacy_means_have_no_invented_spatial_lower_bound(self):
        s=world();initiation.upgrade(s);tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        row.update(consecutive_myr=20.,shortening_km=1000.,shear_slip_km=1000.)
        for key in ('consecutive_lower_bound_myr','shortening_lower_bound_km','shear_lower_bound_km','bounded_history'):
            row.pop(key)
        set_pieces(s,np.linspace(-.03,.04,8));tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertEqual(row['consecutive_myr'],1.)
        self.assertLess(row['shortening_km'],12.);self.assertLess(row['shear_slip_km'],12.)
        self.assertFalse(s.trench_systems);self.assertTrue(initiation.enabled(s))

    def test_new_bound_metadata_rejects_invalid_numbers_and_policy(self):
        for key,value in (('shortening_lower_bound_km',float('nan')),
                          ('shear_lower_bound_km',True),('consecutive_lower_bound_myr',-1.),
                          ('shortening_lower_bound_km',1.),('bounded_history',1)):
            with self.subTest(key=key,value=value):
                s=world();initiation.upgrade(s);tick(s)
                s.effective_subduction_initiation['candidates'][0][key]=value
                with self.assertRaisesRegex(ValueError,'spatial history'):initiation.enabled(s)

    def test_bounded_weakening_checkpoint_replays_identical_future_history(self):
        s=world();initiation.upgrade(s);tick(s);tick(s)
        set_pieces(s,np.linspace(-.03,.04,8));tick(s)
        self.assertTrue(s.effective_subduction_initiation['candidates'][0]['bounded_history'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bounded-initiation.npz'
            checkpoint.write_checkpoint(path,s,{'config':s.config},{})
            restored,_=checkpoint.read_checkpoint(path,{},type(s))
        self.assertEqual(restored.effective_subduction_initiation,s.effective_subduction_initiation)
        for subject in (s,restored):tick(subject)
        self.assertEqual(restored.effective_subduction_initiation,s.effective_subduction_initiation)
        self.assertTrue(initiation.enabled(restored))


if __name__=='__main__':unittest.main()
