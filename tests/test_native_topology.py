"""Native split commits preserve material, histories and force-derived motion."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from native_engine import Simulation
import native_topology as topology
import material_surface
import structure_engine
import plate_balance
import force_rifting


def world(kind=0,level=2):
    initial=dict(width=48,height=24,crust=np.full(48*24,kind,np.uint8))
    return Simulation(dict(width=96,height=48,mesh_level=level,seed=37,plate_count=4,erosion=0.),initial)


def ocean_partition():
    s=world()
    lon=np.arctan2(s.xyz[:,1],s.xyz[:,0]);lat=np.arcsin(s.xyz[:,2])
    parent=(np.abs(lon)<.8)&(np.abs(lat)<.55)
    s.plate[:]=np.where(parent,0,np.where(lon<0,1,2))
    s.active[:]=False;s.active[:3]=True
    s.plate_uid[:3]=[1,2,3];s.next_plate_uid=4
    s.support[:]=0;s.support[s.plate,np.arange(s.n)]=1.
    # Hand-authored evolved support replaces the initializer's single-owner
    # source partition; this controlled loading experiment uses its own graph.
    s.native_initial_ownership_consumed=True
    s.omega[:]=0;s.mantle[:]=.002
    region=parent&(lon>0)
    a,b=s.native_mesh['edge_faces'].T
    edges=np.flatnonzero(parent[a]&parent[b]&(region[a]!=region[b]))
    s._boundaries()
    proposal=dict(parent_slot=0,parent_uid=1,parent_faces=np.flatnonzero(parent),
        child_faces=np.flatnonzero(region),failed_edge_indices=edges,
        delta_omega=np.array([[0.,0.,-.001],[0.,0.,.001]]),mean_opening_km_myr=10.,
        cause='controlled developed ocean tensile belt')
    return s,proposal


class NativeTopologyTests(unittest.TestCase):
    def test_native_icosphere_loading_develops_and_commits_an_actual_ocean_split(self):
        s,_=ocean_partition()
        s.omega[1,2]=-.008;s.omega[2,2]=.008;s._boundaries()
        lon=np.arctan2(s.xyz[:,1],s.xyz[:,0])
        history=topology.ocean_rifting.initialize(s.n,weakness=np.exp(-(lon/.18)**2)*(s.plate==0))
        boundary={name:getattr(s,name) for name in ('ba','bb','bp','bq','bmid','bn','bl')}
        boundary['valid']=s._valid_loading_edges()
        material=dict(owner=s.plate,ocean=s.crust==0,age_myr=np.full(s.n,80.))
        plates=dict(omega=s.omega,uids=s.plate_uid,active=s.active)
        proposal=None
        # Fixed geometry with prescribed boundary motion is a controlled
        # mechanical loading experiment on real closed native triangle edges.
        for step in range(100):
            result=topology.ocean_rifting.update(s.native_mesh,material,plates,boundary,history,2.)
            history=result['history']
            if step==0:self.assertEqual(result['proposals'],[])
            if result['proposals']:
                proposal=result['proposals'][0];break
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal['parent_uid'],1)
        self.assertGreater(step,10)
        before=int(s.active.sum())
        self.assertTrue(topology._commit_ocean(s,proposal))
        self.assertEqual(int(s.active.sum()),before+1)

    def test_primordial_ocean_commit_transfers_native_region_without_rejuvenation_or_mantle_kick(self):
        s,proposal=ocean_partition()
        s.t=125.;s.born[0]=7.
        age=s.age.copy();primordial=s.primordial_fraction.copy();mantle=s.mantle[0].copy()
        parent=s.plate==0;support=s.support.sum(axis=0);old=s.omega[0].copy()
        self.assertTrue(topology._commit_ocean(s,proposal))
        q=int(s.plate[proposal['child_faces'][0]])
        self.assertNotEqual(q,0)
        self.assertEqual(s.born[0],7.)
        self.assertEqual(s.born[q],125.)
        np.testing.assert_array_equal(s.mantle[[0,q]],np.tile(mantle,(2,1)))
        np.testing.assert_array_equal(s.omega[[0,q]],old+proposal['delta_omega'])
        np.testing.assert_array_equal(s.age,age)
        np.testing.assert_array_equal(s.primordial_fraction,primordial)
        np.testing.assert_array_equal(s.support.sum(axis=0),support)
        self.assertTrue(np.all(np.isin(s.plate[parent],[0,q])))
        self.assertEqual(s.events[-1]['type'],'rift')
        self.assertEqual(s.events[-1]['details']['rift_belt_relief_reduction_m'],0.)

    def test_actual_underresolved_buoyant_triangle_blocks_an_ocean_cut(self):
        s,proposal=ocean_partition()
        edge=int(proposal['failed_edge_indices'][0])
        a,b=s.native_mesh['vertices'][s.native_mesh['edge_vertices'][edge]]
        centre=topology._unit(.69*a+.31*b)
        tangent=topology._unit(b-a);normal=topology._unit(np.cross(a,b))
        vertices=topology._unit(np.array([centre+.001*tangent+.001*normal,
                                         centre-.001*tangent+.001*normal,
                                         centre-.0002*tangent-.001*normal]))
        s.material_surface=material_surface.initialize_surface(vertices,np.array([[0,1,2]]),np.array([0]),np.array([2]))
        self.assertFalse(np.any(s.crust))
        self.assertFalse(topology._ocean_cut_clear(s,proposal['failed_edge_indices']))
        active=s.active.copy()
        self.assertFalse(topology._commit_ocean(s,proposal))
        np.testing.assert_array_equal(s.active,active)

    def test_minor_arc_triangle_intersection_including_endcaps_and_no_false_far_hit(self):
        a=topology._unit([1.,-.1,0.]);b=topology._unit([1.,.1,0.])
        crossing=topology._unit(np.array([[1.,-.002,-.002],[1.,.003,-.002],[1.,0.,.003]]))
        far=topology._unit(crossing+np.array([0.,0.,.08]))
        self.assertTrue(topology._arc_hits_triangles(a,b,crossing[None]))
        self.assertFalse(topology._arc_hits_triangles(a,b,far[None]))
        self.assertTrue(topology._arc_hits_triangles(crossing[0],crossing[1],crossing[None]))

    def test_continental_commit_preserves_full_columns_and_zero_residual_motion(self):
        s=world(kind=1)
        s.t=125.;s.born[0]=7.
        s.parcel_plate[:]=0;s.trace_plate[:]=0;s.plate[:]=0
        s.active[:]=False;s.active[0]=True;s.support[:]=0;s.support[0]=1.
        s._sync_material();s._boundaries()
        crack=SimpleNamespace(center=np.array([1.,0.,0.]),normal=np.array([0.,1.,0.]),
                              signed_distance=lambda points:np.asarray(points)[:,1])
        chosen=dict(crack=crack,parcel_side=s.pos[:,1]>0,trace_side=s.trace_xyz[:,1]>0,
                    grid_distance=s.xyz[:,1],rotations=np.zeros((2,3)))
        fields={name:deepcopy(getattr(s,name)) for name in ('mass','parcel_patch','relief','structure','trace_relief_m')}
        mantle=s.mantle[0].copy();omega=s.omega[0].copy()
        vertex_count=len(s.material_surface['vertices'])
        self.assertTrue(topology.split_continent(s,0,chosen))
        q=int(s.parcel_plate[chosen['parcel_side']][0])
        self.assertEqual(s.born[0],7.)
        self.assertEqual(s.born[q],125.)
        np.testing.assert_array_equal(s.omega[[0,q]],np.tile(omega,(2,1)))
        np.testing.assert_array_equal(s.mantle[[0,q]],np.tile(mantle,(2,1)))
        self.assertGreater(len(s.material_surface['vertices']),vertex_count)
        for name,old in fields.items():
            if name=='structure':
                for key,value in old.items():np.testing.assert_array_equal(s.structure[key],value)
            else:np.testing.assert_array_equal(getattr(s,name),old)
        np.testing.assert_array_equal(s.material_surface['face_owner'],s.parcel_plate)

    def test_protected_craton_and_disconnected_cut_are_rejected_before_identity_allocation(self):
        s=world(kind=1);s.parcel_plate[:]=0;s.trace_plate[:]=0;s.plate[:]=0;s._sync_material()
        s.parcel_craton[:]=99
        crack=SimpleNamespace(center=np.array([1.,0.,0.]),normal=np.array([0.,1.,0.]),signed_distance=lambda p:p[:,1])
        chosen=dict(crack=crack,parcel_side=s.pos[:,1]>0,trace_side=s.trace_xyz[:,1]>0,
                    grid_distance=s.xyz[:,1],rotations=np.zeros((2,3)))
        uid=s.next_plate_uid
        self.assertFalse(topology.split_continent(s,0,chosen))
        self.assertEqual(s.next_plate_uid,uid)
        s.parcel_craton[:]=-1
        chosen['grid_distance']=np.where(np.arange(s.n)%2,1.,-1.)
        self.assertFalse(topology.split_continent(s,0,chosen))
        self.assertEqual(s.next_plate_uid,uid)

    def test_continental_proposal_has_priority_and_only_one_cut_commits(self):
        s,proposal=ocean_partition()
        s.ocean_diagnostics=dict(consumed_area_km2=1234.,generated_area_km2=456.,
                                 unresolved_overlap_area_km2=789.,checked=False)
        returned=dict(history=deepcopy(s.ocean_history),diagnostics={'checked':True},proposals=[proposal])
        with patch.object(topology.ocean_rifting,'update',return_value=returned), \
             patch.object(topology.progressive_rifting,'commit',return_value=True), \
             patch.object(topology,'_commit_ocean') as commit:
            self.assertTrue(topology.update_topology(s,2.))
        commit.assert_not_called()
        self.assertTrue(s.ocean_diagnostics['checked'])
        self.assertEqual(s.ocean_diagnostics['consumed_area_km2'],1234.)
        self.assertEqual(s.ocean_diagnostics['generated_area_km2'],456.)
        self.assertEqual(s.ocean_diagnostics['unresolved_overlap_area_km2'],789.)

    def test_tiny_empty_ocean_remnant_retires_without_new_plates_or_lost_support(self):
        s=world(level=3)
        s.active[:]=False;s.active[:2]=True;s.plate[:]=0;s.plate[0]=1
        s.plate_uid[:2]=[1,2];s.support[:]=0;s.support[0]=1.;s.support[0,0]=0.;s.support[1,0]=1.
        s.steps=20;uid=s.next_plate_uid;support=s.support.sum(axis=0)
        topology._retire_remnants(s)
        self.assertFalse(s.active[1]);self.assertTrue(np.all(s.plate==0))
        self.assertEqual(s.next_plate_uid,uid)
        np.testing.assert_array_equal(s.support.sum(axis=0),support)

    def test_consumed_plate_retires_before_post_transport_force_balance(self):
        s=world(level=2)
        s.active[:]=False;s.active[:2]=True;s.plate[:]=0
        s.support[:]=0.;s.support[0]=1.
        s._boundaries()
        support=s.support.copy();uids=s.plate_uid.copy();omega=s.omega.copy()
        returned=dict(history=deepcopy(s.ocean_history),diagnostics={},proposals=[])
        def current_force(state,dt):
            self.assertFalse(state.active[1])
            # The real rigid solver must succeed with the represented plate,
            # without regularizing an empty slot's three zero-drag modes.
            balance=plate_balance.Balance(state,dt)
            balance.solve()
            self.assertEqual(balance.plates,[0])
        with patch.object(topology.ocean_rifting,'update',return_value=returned), \
             patch.object(force_rifting,'update',side_effect=current_force) as current, \
             patch.object(topology.backarc,'commit_pending',return_value=True):
            self.assertTrue(topology.update_topology(s,1.))
        current.assert_called_once_with(s,1.)
        np.testing.assert_array_equal(s.support,support)
        np.testing.assert_array_equal(s.plate_uid,uids)
        np.testing.assert_array_equal(s.omega,omega)

    def test_empty_retirement_keeps_all_represented_and_invalid_slots(self):
        s=SimpleNamespace(active=np.ones(8,bool),plate=np.array([0,1]),
            support=np.zeros((8,2)),parcel_plate=np.array([2]),trace_plate=np.array([3]))
        s.support[0]=1.
        s.support[4,0]=1e-30
        s.support[5,0]=np.nan
        s.support[6,0]=-1.
        topology._retire_empty_plates(s)
        np.testing.assert_array_equal(s.active,[True,True,True,True,True,True,True,False])

    def test_remnant_cleanup_preserves_positive_fractional_only_plate(self):
        s=world(level=2)
        s.active[:]=False;s.active[:2]=True;s.plate[:]=0
        s.support[:]=0.;s.support[0]=1.;s.support[1,0]=1e-15
        before=s.support.copy()
        topology._retire_empty_plates(s)
        topology._retire_remnants(s)
        self.assertTrue(s.active[1])
        np.testing.assert_array_equal(s.support,before)


if __name__=='__main__':unittest.main()
