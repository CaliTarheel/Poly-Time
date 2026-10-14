"""Tiny fixed-interface continuity tests; no simulation construction or steps."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SUPPORT=ROOT if (ROOT/'cohesive_failure.py').is_file() else Path(json.loads((ROOT/'candidate-manifest.json').read_text())['source_runtime'])
sys.path.insert(0,str(SUPPORT))
import cohesive_failure
import material_geometry

def functions(path,names=None):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and
           (names is None or isinstance(n,ast.FunctionDef) and n.name in names)]
    env=dict(np=np,math=math,hashlib=hashlib,deepcopy=copy.deepcopy)
    exec(compile(ast.Module(nodes,type_ignores=[]),str(path),'exec'),env)
    return env

f=functions(ROOT/'force_rifting.py')
mode=functions(SUPPORT/'breakup_mode.py',{'_basis','_cut'})
pb=types.SimpleNamespace(SECONDS_PER_MYR=31557600.e6,CM_YR_M_S=.01/31557600.)
f['plate_balance']=pb;mode['pb']=pb;mode['cohesive_failure']=cohesive_failure

def fixture():
    xyz=np.array([[1.,0.,0.],[.8,.6,0.],[-.8,.6,0.],[-1.,0.,0.],[0.,0.,1.]])
    s=types.SimpleNamespace(xyz=xyz,n=5,plate=np.array([0,0,0,0,1]),plate_uid=np.array([7,8]),
        active=np.array([True,True]),parcel_plate=np.zeros(4,int),parcel_patch=np.array([101,102,103,104]),
        parcel_craton=np.full(4,-1),parcel_cell=np.arange(4),mass=np.ones(4),pos=xyz[:4].copy(),
        cell_area=np.array([1.,2.,3.,4.,1.]),support=np.array([[1.,1.,1.,1.,0.],[0.,0.,0.,0.,1.]]),
        edge_a=np.array([0,1]),edge_b=np.array([2,3]),edge_mid=np.array([[0.,1.,0.],[0.,1.,0.]]),
        edge_length=np.array([.1,.2]))
    s._indices=lambda points:np.argmax(np.asarray(points)@s.xyz.T,axis=1)
    balance=types.SimpleNamespace(_support=lambda p:s.support[p],slot={0:0},size=3,
        uses_force_ledger=True,effective_subduction=True,solve=lambda:None,x=np.zeros(3))
    found=dict(cells=np.arange(4),piece=np.array([True,True,False,False]),axis=np.array([0.,0.,1.]),
        cut_edges=np.array([0,1]),ratio=2.,predicted_opening_cm_yr_mean=.1,current_capacity_n=2.,
        opening_mode_supported=True,common_motion_relaxed=True,mode_work=dict(driving_power_w=2000./pb.SECONDS_PER_MYR,
        resisting_power_w=0.,cut_power_w=2000./pb.SECONDS_PER_MYR),cut_length_km=.3,cut_continental_fraction=1.,
        piece_rotation_rad_myr=[0.,0.,1.],rest_rotation_rad_myr=[0.,0.,-1.],relative_rotation_rad_myr=2.,
        search_policy='bounded_axes_v1',searched_axis_count=2,search_scope='held prior axis pair')
    return s,balance,found

def proof(s,found):
    out={};legacy=f['_stable_cut'](s,0,found,continuity=out)
    return dict(support_identity=legacy,fixed_interface_continuity=out,axis=found['axis'].tolist())

def extend(s,found):
    s.plate[4]=0;s.support[0,4]=.25
    return dict(copy.deepcopy(found),cells=np.arange(5),piece=np.array([True,True,False,False,False]))

class IdentityTests(unittest.TestCase):
    def test_owner_churn_changes_legacy_but_preserves_proved_interface(self):
        s,b,a=fixture();old=proof(s,a);z=extend(s,a);new=proof(s,z)
        self.assertNotEqual(old['support_identity'],new['support_identity'])
        self.assertEqual(old['fixed_interface_continuity'],new['fixed_interface_continuity'])
        self.assertTrue(f['_same_fixed_interface'](old,new['fixed_interface_continuity'],z['axis']))

    def test_legacy_digest_bytes_unchanged(self):
        s,b,a=fixture();d=hashlib.sha256();cells=a['cells'].astype('<i8');side=cells[a['piece']];other=cells[~a['piece']]
        for value in (np.array([7],dtype='<i8'),cells,side,other,a['cut_edges'].astype('<i8')):
            d.update(np.array([len(value)],dtype='<i8').tobytes());d.update(value.tobytes())
        pairs=np.column_stack((s.parcel_patch,[1,1,0,0])).astype('<i8');d.update(pairs.tobytes())
        self.assertEqual(f['_stable_cut'](s,0,a),d.hexdigest())
        self.assertEqual(proof(s,a)['support_identity'],d.hexdigest())

    def test_actual_basis_differs_only_by_free_common_rotation(self):
        s,b,a=fixture();old,_alpha=mode['_basis'](s,b,{'plates':{0:{'cells':a['cells']}}},0,a['cells'],a['piece'],a['axis'])
        z=extend(s,a);new,alpha=mode['_basis'](s,b,{'plates':{0:{'cells':z['cells']}}},0,z['cells'],z['piece'],z['axis'])
        q=.7;u=np.array([.2,-.3,.5]);before=np.r_[u,q];after=np.r_[u+(alpha-_alpha)*q*a['axis'],q]
        np.testing.assert_allclose(np.einsum('ijk,k->ij',old[0],before),np.einsum('ijk,k->ij',new[0],after)[:4],rtol=0.,atol=2e-16)
        np.testing.assert_allclose(new[0][4]@after,u-_alpha*q*a['axis'],rtol=0.,atol=2e-16)

    def test_actual_cut_signed_rates_weights_and_capacity_unchanged(self):
        s,b,a=fixture();z=extend(s,a)
        geometry=dict(edges=np.column_stack((s.edge_a,s.edge_b)),mid=s.edge_mid,length_m=s.edge_length*1e3,cut=np.ones(2,bool))
        strength=(np.array([2.,3.]),np.array([7.,8.]),np.array([3.,4.]))
        x=mode['_cut'](s,a['cells'],a['piece'],a['axis'],geometry,strength)
        y=mode['_cut'](s,z['cells'],z['piece'],z['axis'],geometry,strength)
        for key in x:np.testing.assert_array_equal(x[key],y[key])

    def test_axis_and_selected_side_change_rejected(self):
        s,b,a=fixture();old=proof(s,a);z=extend(s,a)
        self.assertFalse(f['_same_fixed_interface'](old,proof(s,z)['fixed_interface_continuity'],-z['axis']))
        axis=z['axis'].copy();axis[0]=1e-8
        self.assertFalse(f['_same_fixed_interface'](old,proof(s,z)['fixed_interface_continuity'],axis))
        z['piece']=~z['piece']
        self.assertFalse(f['_same_fixed_interface'](old,proof(s,z)['fixed_interface_continuity'],z['axis']))

    def test_owner_patch_birth_loss_and_side_change_rejected(self):
        for change in ['owner','patch','loss','side']:
            with self.subTest(change=change):
                s,b,a=fixture();old=proof(s,a);z=extend(s,a)
                if change=='owner':s.plate_uid[0]=70
                if change=='patch':s.parcel_patch[0]=999
                if change=='loss':s.parcel_plate[0]=1
                if change=='side':s.pos[0]=s.xyz[3]
                self.assertFalse(f['_same_fixed_interface'](old,proof(s,z)['fixed_interface_continuity'],z['axis']))

    def test_exact_cut_geometry_or_edge_change_rejected(self):
        for change in ['xyz','mid','length','edge','orientation']:
            with self.subTest(change=change):
                s,b,a=fixture();old=proof(s,a);z=extend(s,a)
                if change=='xyz':s.xyz[0,2]=1e-9
                if change=='mid':s.edge_mid[0,2]=1e-9
                if change=='length':s.edge_length[0]+=1e-9
                if change=='edge':z['cut_edges']=np.array([1])
                if change=='orientation':s.edge_a[0],s.edge_b[0]=s.edge_b[0],s.edge_a[0]
                self.assertFalse(f['_same_fixed_interface'](old,proof(s,z)['fixed_interface_continuity'],z['axis']))

    def test_missing_invalid_or_degenerate_geometry_proof_unavailable(self):
        for change in ['absent','zero_mid','zero_length','nan','same_endpoints','axis_parallel']:
            with self.subTest(change=change):
                s,b,a=fixture()
                if change=='absent':del s.edge_mid
                if change=='zero_mid':s.edge_mid[:]=0.
                if change=='zero_length':s.edge_length[:]=0.
                if change=='nan':s.edge_mid[0,0]=np.nan
                if change=='same_endpoints':s.edge_b[:]=s.edge_a
                if change=='axis_parallel':a['axis']=np.array([0.,1.,0.])
                self.assertEqual(proof(s,a)['fixed_interface_continuity'],{})

    def test_material_must_have_complete_single_side_assignments(self):
        s,b,a=fixture();del s.parcel_patch
        with self.assertRaisesRegex(ValueError,'complete represented'):proof(s,a)
        s,b,a=fixture();pairs=np.array([[101,0],[101,1]],dtype='<i8')
        self.assertEqual(f['_fixed_interface_proof'](s,0,a,a['cells'][a['piece']],pairs),{})

    def test_unavailable_or_fixed_common_motion_cannot_claim_gauge_proof(self):
        for field,value in [('opening_mode_supported',False),('common_motion_relaxed',False),('common_motion_relaxed',None)]:
            s,b,a=fixture();a[field]=value
            self.assertEqual(proof(s,a)['fixed_interface_continuity'],{})

    def test_missing_future_proof_or_wrong_version_rejected(self):
        s,b,a=fixture();p=proof(s,a);now=p['fixed_interface_continuity']
        for previous in [None,{'axis':a['axis']},dict(p,fixed_interface_continuity={}),
                         dict(p,fixed_interface_continuity=dict(now,version=True)),
                         dict(p,fixed_interface_continuity=dict(now,version=2))]:
            self.assertFalse(f['_same_fixed_interface'](previous,now,a['axis']))

class WorkHistoryTests(unittest.TestCase):
    def setUp(self):
        self.s,self.b,self.found=fixture();s=self.s
        settings=f['normalize'](dict(enabled=True,search_policy=dict(f['BOUNDED_SEARCH'])))
        s.config={'force_limit_rifting':settings};s.t=0.;s.force_rifting_version=1
        s.force_rifting_state={};s.force_rifting_diagnostics=dict(version=1,checks=0,commits=0,refusals=[])
        s.force_rifting_policy=dict(version=1,mode='bounded_axes',parameters=settings,activation_myr=0.,epoch_myr=0.,history='Future matched cut observations only; virtual development, not realized material extension.')
        self.searches=[]
        def search(*args,**kwargs):self.searches.append(kwargs.get('edge_strength_scale'));return copy.deepcopy(self.found)
        f['plate_limit_analysis']=types.SimpleNamespace(active_balance=lambda *args:self.b,worst_mechanism=search)
        f['_loaded_plates']=lambda *args,**kwargs:[0]
        self.previous=sys.modules.get('balance_force_ledger');sys.modules['balance_force_ledger']=types.SimpleNamespace(export=lambda *args:{})
    def tearDown(self):
        if self.previous is None:sys.modules.pop('balance_force_ledger',None)
        else:sys.modules['balance_force_ledger']=self.previous
    def advance(self):
        self.s.t+=1.;f['update'](self.s,1.)
        frame=f['snapshot'](self.s);frame['time_myr']=self.s.t;f['validate_frame'](frame)
        return self.s.force_rifting_state['7']
    def test_churn_continues_future_work_without_intact_retry(self):
        first=self.advance();self.assertEqual(first['opened_km'],0.);self.assertEqual(first['paid_cut_work_j'],0.)
        second=self.advance();paid=second['paid_cut_work_j'];self.assertAlmostEqual(second['opened_km'],1.)
        self.found=extend(self.s,self.found);count=len(self.searches)
        third=self.advance();self.assertEqual(len(self.searches),count+1)
        self.assertEqual(third['support_match_kind'],'fixed_interface');self.assertAlmostEqual(third['opened_km'],2.)
        self.assertGreater(third['paid_cut_work_j'],paid);self.assertEqual(third['since_myr'],1.)
        self.assertEqual(third['supported_elapsed_myr'],2.)
        self.assertEqual(self.s.force_rifting_diagnostics.get('retired_candidate_paths',[]),[])
    def test_current_capacity_is_repriced_not_reused(self):
        self.advance();second=self.advance();prior=second['paid_cut_work_j']
        self.found=extend(self.s,self.found);self.found['current_capacity_n']=4.
        self.found['mode_work']=dict(driving_power_w=4000./pb.SECONDS_PER_MYR,resisting_power_w=0.,cut_power_w=4000./pb.SECONDS_PER_MYR)
        third=self.advance();width=150000.;start=1000.;expected=cohesive_failure.necking_work_j(4.*(1.+start/width),start,1000.,width)
        self.assertAlmostEqual(third['paid_cut_work_j']-prior,expected)
        self.assertGreater(third['paid_cut_work_j']-prior,prior)
    def test_old_row_cannot_reconstruct_new_proof_or_old_credit(self):
        self.advance();second=self.advance();paid=second['paid_cut_work_j']
        del self.s.force_rifting_state['7']['fixed_interface_continuity']
        self.found=extend(self.s,self.found);row=self.advance()
        self.assertEqual(row['opened_km'],0.);self.assertEqual(row['paid_cut_work_j'],0.)
        self.assertEqual(self.s.force_rifting_diagnostics['retired_candidate_paths'][0]['paid_cut_work_j'],paid)
    def test_changed_material_resets_and_returned_old_material_does_not_revive(self):
        self.advance();paid=self.advance()['paid_cut_work_j'];self.found=extend(self.s,self.found)
        self.s.parcel_patch[0]=901;reset=self.advance();self.assertEqual(reset['opened_km'],0.)
        self.assertEqual(self.s.force_rifting_diagnostics['retired_candidate_paths'][0]['paid_cut_work_j'],paid)
        self.s.parcel_patch[0]=101;row=self.advance();self.assertEqual(row['opened_km'],0.)
        self.assertEqual(row['paid_cut_work_j'],0.);self.assertEqual(self.s.force_rifting_diagnostics['accepted_cut_work_j'],paid)
    def test_gap_and_duplicate_epoch_do_not_earn_work(self):
        self.advance();self.s.force_rifting_state['7']['last_observation_myr']=0.;self.found=extend(self.s,self.found)
        row=self.advance();self.assertEqual(row['opened_km'],0.)
        old=copy.deepcopy(self.s.force_rifting_state)
        with self.assertRaisesRegex(ValueError,'one new matching'):f['update'](self.s,1.)
        self.assertEqual(old,self.s.force_rifting_state)
    def test_optional_frame_proof_schema_and_legacy_compatibility(self):
        self.advance();self.advance();frame=f['snapshot'](self.s);frame['time_myr']=self.s.t
        legacy=copy.deepcopy(frame);legacy['force_rifting_plates']['7'].pop('fixed_interface_continuity');legacy['force_rifting_plates']['7'].pop('support_match_kind')
        f['validate_frame'](legacy)
        for field,value in [('version',True),('version',2),('scope','other'),('identity','Z'*64),
                            ('piece_count',0),('material_patch_count',-1),('owner_uid',8)]:
            broken=copy.deepcopy(frame);broken['force_rifting_plates']['7']['fixed_interface_continuity'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):f['validate_frame'](broken)
        for proof,kind in [({},'fixed_interface'),(None,'exact'),({},'unknown')]:
            broken=copy.deepcopy(frame);broken['force_rifting_plates']['7'].update(fixed_interface_continuity=proof,support_match_kind=kind)
            with self.assertRaises(ValueError):f['validate_frame'](broken)

    def test_healing_does_not_refund_and_insufficient_current_work_is_atomic(self):
        self.advance();paid=self.advance()['paid_cut_work_j'];self.found=extend(self.s,self.found)
        self.found['ratio']=.5;row=self.advance();self.assertLess(row['opened_km'],1.);self.assertEqual(row['paid_cut_work_j'],paid)
        self.found['ratio']=2.;self.found['mode_work']=dict(driving_power_w=1./pb.SECONDS_PER_MYR,resisting_power_w=0.,cut_power_w=1./pb.SECONDS_PER_MYR)
        old=copy.deepcopy(self.s.force_rifting_state);epoch=self.s.force_rifting_policy['epoch_myr'];self.s.t+=1.
        with self.assertRaisesRegex(ValueError,'insufficient cut work'):f['update'](self.s,1.)
        self.assertEqual(old,self.s.force_rifting_state);self.assertEqual(epoch,self.s.force_rifting_policy['epoch_myr'])

if __name__=='__main__':unittest.main(verbosity=2)
