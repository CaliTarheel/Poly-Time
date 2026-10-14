"""Accepted finite-fault history, pending source activation and exact rollback."""
from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import adaptive_timestepping
import budget
import checkpoint
import effective_subduction as effective
import effective_subduction_initiation as initiation
import native_engine
import native_subduction
import plate_balance
from ridge_geometry import rotate
from tests.test_effective_subduction_forces import fixture as force_fixture


def points(phi):
    return np.column_stack((np.cos(phi),np.zeros(len(phi)),np.sin(phi)))


def world():
    s=force_fixture(edges=(127.42,127.42,127.42))
    s.t=25.;s.steps=100;s.next_trench_id=1;s.trench_systems=[]
    s.effective_subduction_diagnostics={};s.polarity=np.full((2,2),-1,int)
    s.trench_id=np.zeros(3,int);s.trench_maturity=np.zeros(3)
    s.w=1024;s.h=512;s.cell_area[:]=10000.
    s.age[:6]=80.;s.age[6:]=40.;s.rng=np.random.default_rng(17)
    set_pieces(s,np.array([-.03,-.01,.01,.03]))
    motion(s,12.,12.)
    return s


def set_pieces(s,phi,parents=None):
    xyz=points(phi);n=len(phi)-1
    if parents is None:parents=np.arange(n)
    count=int(max(parents))+1
    s.ba=np.zeros(count,int);s.bb=np.full(count,6,int)
    s.bp=np.zeros(count,int);s.bq=np.ones(count,int);s.bcode=np.full(count,3,np.uint8)
    s.down=np.full(count,-1,int);s.normal_speed=np.full(count,-12.);s.trench_id=np.zeros(count,int)
    mid=xyz[:-1]+xyz[1:];mid/=np.linalg.norm(mid,axis=1)[:,None]
    length=6371.*np.diff(phi)
    s.bl=np.array([length[np.asarray(parents)==i].sum() for i in range(count)])
    s.bmid=np.array([mid[np.asarray(parents)==i][0] for i in range(count)])
    s.bn=np.tile([0.,1.,0.],(count,1));s.trench_maturity=np.zeros(count)
    s.native_boundary_geometry=dict(segments_start=xyz[:-1].copy(),segments_end=xyz[1:].copy(),
        segment_normals=np.tile([0.,1.,0.],(n,1)),contact_index=np.asarray(parents,int))


def motion(s,closing,shear):
    # The fixed overriding chart keeps this fixture's front stationary.
    s.omega[:]=0.;s.omega[0]=-np.array([0.,shear,-closing])/6371.


def tick(s,dt=1.):
    s.t+=dt;effective.update(s,dt)


def qualify(s):
    initiation.upgrade(s)
    tick(s)  # first observation receives no retrospective loading
    for _ in range(10):tick(s)
    return s.trench_systems[0]


class InitiationTests(unittest.TestCase):
    def test_public_native_next_solved_interval_activates_new_force_and_intake(self):
        s=native_engine.Simulation(dict(width=48,height=24,mesh_level=2,coast_geometry_level=2,
            plate_count=4,mechanics_nodes=128,seed=37,physics_profile='reviewed_v1',
            primordial_subduction={'enabled':True},effective_subduction={'enabled':True}))
        # Fixture replaces the initial declarations, then observes an actual native fault.
        s.trench_systems=[];s.t=1.;s.steps=1;s.age[:]=80.;s.age[s.bb]=40.
        s.omega[:]=0.;s.bcode[:]=3;s.down[:]=-1;s.trench_id[:]=0
        initiation.upgrade(s,dict(enabled=True,minimum_consecutive_myr=.001,
            minimum_shortening_km=.001,minimum_shear_slip_km=.001,minimum_length_km=1.,
            minimum_ocean_age_myr=1.,minimum_ocean_fraction=.1))
        contour=s.native_boundary_geometry
        for index,(a,b,parent) in enumerate(zip(contour['segments_start'],contour['segments_end'],contour['contact_index'])):
            mid=a+b;mid/=np.linalg.norm(mid);normal=contour['segment_normals'][index]
            normal=normal-mid*(normal@mid);normal/=np.linalg.norm(normal)
            s.omega[:]=0.;s.omega[int(s.bp[parent])]=(np.cross(mid,normal)*12.+normal*12.)/6371.
            if any(r['minimum_convergence_km_myr']>2. for r in initiation.prepare(s)):break
        for _ in range(2):tick(s,.01)
        self.assertEqual(len(s.trench_systems),1)
        trace=s.trench_systems[0]
        self.assertTrue(trace['effective_subduction']['activation_pending'])
        self.assertFalse(np.any(effective.line_state(s)[1]))
        before=float(s.process_totals['ocean_consumed_km2'])
        self.assertEqual(native_subduction.capture_polygons(s,.001),[])
        # This checks source activation and conservation, not the 1.8-second
        # production wall limit implied by a .001 Myr interval. Geometry
        # startup exceeds that limit on an otherwise passing baseline too.
        with budget.step_budget(.001,seconds_per_myr=300000):s.step(.001)
        self.assertAlmostEqual(s.t,1.021)
        self.assertFalse(trace['effective_subduction']['activation_pending'])
        self.assertEqual(trace['effective_subduction']['activated_myr'],1.02)
        self.assertGreater(s.process_totals['ocean_consumed_km2']-before,0.)
        self.assertGreater(s.snapshot()['effective_subduction_diagnostics']['integrated_incoming_force_n'],0.)
        ledger=s.plate_balance_diagnostics
        self.assertGreater(ledger['effective_subduction_integrated_force_n'],0.)
        self.assertLessEqual(ledger['scaled_force_residual'],ledger['force_relative_tolerance'])
        self.assertEqual(ledger['slab_power_w']['overriding'],0.)
        self.assertEqual(s.effective_subduction_initiation['births'][0]['time_myr'],1.02)
        self.assertEqual(s.effective_subduction_initiation['events'][-1]['kind'],'source_activated')

    def test_actual_native_front_birth_force_and_finite_intake_start_next_source(self):
        from tests.test_native_subduction_oracle import fixture,strip_area
        s,axis,transported=fixture(split=4,level=2)
        s.t=25.;s.steps=100;s.next_trench_id=1
        s.active[2]=False
        s.age=np.where(s.plate==0,80.,40.)
        s.config={'effective_subduction':{'enabled':True,'force_n_per_m':3e13}}
        s.effective_subduction_version=1;s.effective_subduction_settings=effective.normalize(s.config['effective_subduction'])
        s.effective_subduction_diagnostics={};s.slab_memory_version=0;s.subduction_response_version=0
        s.trench_id=np.zeros(1,int);s.polarity=np.full((3,3),-1,int)
        s.bmid=np.array([axis.sum(axis=0)/np.linalg.norm(axis.sum(axis=0))]);s.bn=np.array([[0.,1.,0.]])
        s.bl=np.array([6371.*np.arccos(axis[0]@axis[1])]);s.names=['Incoming','Upper','Unused']
        # Incoming moves; the overriding chart is stationary in the observation fixture.
        s.omega[:]=0.;s.omega[0]=[0.,-.002,.002]
        initiation.upgrade(s)
        for _ in range(11):tick(s)
        self.assertEqual(len(s.trench_systems),1)
        row=s.trench_systems[0]
        self.assertEqual(row['effective_subduction']['force_n_per_m'],3e13)
        self.assertAlmostEqual(row['length_km'],s.bl[0],delta=2e-8)
        np.testing.assert_array_equal(native_subduction.removal(s,transported,.25),0.)
        self.assertEqual(native_subduction.capture_polygons(s,.25),[])
        initiation.activate_pending(s)
        before=transported.copy();removed=native_subduction.removal(s,transported,.25)
        self.assertGreater(float(removed[0]@s.cell_area),0.)
        np.testing.assert_array_equal(removed[1:],0.);np.testing.assert_array_equal(transported,before)
        self.assertGreater(sum(p['raw_area_km2'] for p in native_subduction.capture_polygons(s,.25)),0.)
        s.plate_resistance_version=1
        model=plate_balance.Balance(s,.25);model.solve()
        self.assertGreater(model.drivers['slab'][2],0.)
        np.testing.assert_array_equal(model.drivers['slab'][3:],0.)

    def test_default_disabled_and_activation_changes_only_new_policy(self):
        s=world();before=pickle.dumps(vars(s))
        self.assertNotIn('initiation',effective.normalize({'enabled':True}))
        self.assertFalse(initiation.enabled(s));self.assertEqual(initiation.prepare(s),[])
        self.assertFalse(initiation.activate_pending(s));self.assertEqual(initiation.snapshot(s),{})
        self.assertEqual(pickle.dumps(vars(s)),before)
        old=deepcopy(vars(s));receipt=initiation.upgrade(s)
        self.assertEqual(set(vars(s))-set(old),{'effective_subduction_initiation'})
        for key,value in old.items():
            if key!='config':self.assertEqual(pickle.dumps(getattr(s,key)),pickle.dumps(value),key)
        self.assertEqual(receipt['epoch_myr'],25.);self.assertEqual(receipt['candidates'],[])
        self.assertEqual(receipt['retrospective_time_myr'],0.)
        self.assertNotIn('initiation',s.effective_subduction_settings)

    def test_configuration_and_checkpoint_errors_fail_before_mutation(self):
        for raw in ({'enabled':1},{'version':True},{'minimum_ocean_fraction':1.1},
                    {'minimum_convergence_km_myr':0.},{'unknown':1}):
            with self.subTest(raw=raw),self.assertRaises(ValueError):initiation.normalize(raw)
        s=world();initiation.upgrade(s);tick(s)
        before=pickle.dumps(vars(s));s.t+=.5
        staged=pickle.dumps(vars(s))
        with self.assertRaisesRegex(ValueError,'clock'):initiation.update(s,1.)
        self.assertEqual(pickle.dumps(vars(s)),staged)
        s.t-=.5;s.effective_subduction_initiation['candidates'][0]['incoming_plate_uid']=True
        with self.assertRaisesRegex(ValueError,'identity'):initiation.enabled(s)

    def test_pending_birth_has_no_force_then_next_source_has_unchanged_traction(self):
        s=world();initiation.upgrade(s);tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['shortening_km'],0.)
        for _ in range(9):tick(s)
        self.assertFalse(s.trench_systems)
        tick(s);row=s.trench_systems[0]
        self.assertEqual(row['born_myr'],36.);self.assertEqual(row['active_myr'],0.)
        self.assertTrue(row['effective_subduction']['activation_pending'])
        np.testing.assert_array_equal(effective.line_state(s)[1],0.)
        np.testing.assert_array_equal(s.down,-1);np.testing.assert_array_equal(s.trench_maturity,0.)
        self.assertFalse(np.any(plate_balance.Balance(s,1.).drivers['slab']))
        before=pickle.dumps(vars(s))
        for _ in range(3):initiation.prepare(s);effective.snapshot(s)
        self.assertEqual(pickle.dumps(vars(s)),before)
        with patch('world_design.record_transitions'),patch('collision_contacts.weld_enabled',return_value=False):
            native_engine.Simulation._prepare_step(s)
        self.assertFalse(row['effective_subduction']['activation_pending'])
        self.assertEqual(row['effective_subduction']['activated_myr'],36.)
        owners,force,live=effective.line_state(s)
        np.testing.assert_array_equal(owners,0);np.testing.assert_array_equal(force,5e12)
        self.assertTrue(live.all());np.testing.assert_array_equal(s.trench_maturity,1.)
        model=plate_balance.Balance(s,1.);model.solve()
        self.assertGreater(model.drivers['slab'][2],0.)
        np.testing.assert_array_equal(model.drivers['slab'][3:],0.)
        self.assertEqual(effective.snapshot(s)['effective_subduction_initiation_diagnostics']['activated_trenches'],1)
        np.testing.assert_allclose(row['geometry_xyz'],np.stack((s.native_boundary_geometry['segments_start'],
            s.native_boundary_geometry['segments_end']),axis=1).reshape(-1,3),rtol=0,atol=0)

    def test_opening_resets_compression_and_low_slip_cannot_birth(self):
        s=world();initiation.upgrade(s);tick(s)
        for _ in range(5):tick(s)
        motion(s,-3.,12.);tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertEqual(row['consecutive_myr'],0.);self.assertEqual(row['shortening_km'],0.)
        self.assertGreater(row['shear_slip_km'],60.)
        slow=world();motion(slow,12.,0.);initiation.upgrade(slow)
        for _ in range(14):tick(slow)
        self.assertFalse(slow.trench_systems)

    def test_young_dry_short_and_local_opening_gates(self):
        for change in ('young','dry','short'):
            s=world()
            if change=='young':s.age[:]=10.
            elif change=='dry':s.crust[:]=1
            else:set_pieces(s,np.array([-.01,0.,.01]))
            initiation.upgrade(s)
            for _ in range(12):tick(s)
            self.assertFalse(s.effective_subduction_initiation['candidates'],change)
        s=world();initiation.upgrade(s);tick(s)
        # A real local normal, unlike a mean parent speed, exposes one opening piece.
        s.native_boundary_geometry['segment_normals'][0]*=-1
        tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['consecutive_myr'],0.)

    def test_exact_piece_water_does_not_borrow_parent_ocean_fraction(self):
        s=world();set_pieces(s,np.array([-.03,-.01,.01,.03]),[0,0,0])
        initiation.upgrade(s)
        def intervals(a,b,owner):
            return [] if a[2]<-.02 else [(0.,1.)]
        with patch('native_subduction.enabled',return_value=True),patch('native_subduction._material',side_effect=lambda s,p:p),\
                patch('native_subduction.exact.ocean_intervals',side_effect=intervals):
            pieces=initiation._pieces(s)
        self.assertEqual(pieces[0]['water'],(0.,0.))
        self.assertIsNone(initiation._measure(pieces,s.effective_subduction_initiation['parameters']))

    def test_tangent_normal_and_frozen_local_polarity(self):
        s=world();initiation.upgrade(s)
        mid=s.native_boundary_geometry['segments_start']+s.native_boundary_geometry['segments_end']
        mid/=np.linalg.norm(mid,axis=1)[:,None]
        normals=np.tile([0.,1.,0.],(3,1))+mid*.75
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        s.native_boundary_geometry['segment_normals']=normals
        pieces=initiation._pieces(s)
        self.assertAlmostEqual(pieces[1]['normal_speed'],-12.)
        tick(s);s.age[6:]=200.;tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['incoming_plate_uid'],11)
        np.testing.assert_array_equal(s.polarity,-1)

    def test_remesh_retains_and_extension_cannot_borrow_core_history(self):
        s=world();initiation.upgrade(s);tick(s);tick(s)
        set_pieces(s,np.linspace(-.03,.03,7));tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['consecutive_myr'],2.)
        set_pieces(s,np.linspace(-.03,.04,8));tick(s)
        row=s.effective_subduction_initiation['candidates'][0]
        self.assertEqual(row['phase'],'weakening')
        self.assertEqual(row['consecutive_myr'],3.)
        self.assertTrue(row['bounded_history'])
        self.assertEqual(len(s.effective_subduction_initiation['candidates']),1)
        self.assertLessEqual(np.arcsin(np.asarray(row['segments_end'])[:,2]).max(),.03+1e-13)

    def test_candidate_advection_and_owner_change_cannot_borrow_history(self):
        s=world();initiation.upgrade(s);tick(s);tick(s)
        s.omega[:]=[0.,.001,0.]
        for name in ('segments_start','segments_end','segment_normals'):
            s.native_boundary_geometry[name]=rotate(s.native_boundary_geometry[name],s.omega[1])
        tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['consecutive_myr'],0.)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['phase'],'weakening')
        s.plate_uid[1]=33;tick(s)
        self.assertEqual(s.effective_subduction_initiation['candidates'][0]['phase'],'retired')
        self.assertEqual(s.effective_subduction_initiation['candidates'][1]['consecutive_myr'],0.)

    def test_parent_complete_birth_and_force_halo_rejected(self):
        s=world();set_pieces(s,np.array([-.03,-.01,.01,.03,.05]),[0,0,0,0])
        # Opening sibling is a real part of this parent's support.
        s.native_boundary_geometry['segment_normals'][-1]*=-1
        initiation.upgrade(s)
        for _ in range(12):tick(s)
        self.assertFalse(s.trench_systems)
        s=world();row=qualify(s);initiation.activate_pending(s)
        set_pieces(s,np.array([-.03,-.01,.01,.03,.04]))
        # Broad identity match deliberately assigns the untrained adjacent parent.
        s.trench_id[:]=row['id']
        owners,force,live=effective.line_state(s)
        np.testing.assert_array_equal(force[:3],5e12)
        self.assertEqual(force[-1],0.);self.assertEqual(owners[-1],-1)
        pieces=initiation._pieces(s)
        self.assertEqual([p['parent'] for p in pieces],[3])

    def test_checkpoint_replays_pending_clock_and_source_activation(self):
        s=world();qualify(s)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'initiation.npz'
            checkpoint.write_checkpoint(path,s,{'config':s.config},{})
            restored,_=checkpoint.read_checkpoint(path,{},type(s))
        self.assertEqual(restored.effective_subduction_initiation,s.effective_subduction_initiation)
        self.assertEqual(restored.trench_systems,s.trench_systems)
        for subject in (s,restored):initiation.activate_pending(subject);tick(subject,.25)
        self.assertEqual(restored.effective_subduction_initiation,s.effective_subduction_initiation)
        self.assertEqual(restored.trench_systems,s.trench_systems)

    def test_next_source_activation_loading_rejection_rolls_back_every_field(self):
        s=world();qualify(s);before=pickle.dumps(vars(s));seen=[]
        def rejected(dt):
            with patch('world_design.record_transitions'),patch('collision_contacts.weld_enabled',return_value=False):
                native_engine.Simulation._prepare_step(s)
            seen.append(np.any(plate_balance.Balance(s,dt).drivers['slab']))
            s.rng.random();raise ValueError('force loading rejected')
        with self.assertRaisesRegex(ValueError,'loading rejected'):
            adaptive_timestepping.advance(s,.25,rejected)
        self.assertEqual(seen,[True]);self.assertEqual(pickle.dumps(vars(s)),before)


if __name__=='__main__':unittest.main()
