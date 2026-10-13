"""Production pre-force ordering and local mechanical contact regressions."""
from unittest.mock import patch
import unittest
import numpy as np
import collision_contacts as contacts
import native_engine
import plate_balance as balance
import material_surface
from ridge_geometry import rotate
from tests.test_collision_local_fronts import fixture
from tests.test_suture_weld import collided,sheet

def weld_model(s):
    s.suture_weld_version=1
    model=balance.Balance.__new__(balance.Balance)
    model.s=s;model.plates=list(range(len(s.omega)))
    model.slot={p:p for p in model.plates};model.size=3*len(model.plates)
    model.bp=model.bq=np.empty(0,int)
    model.strength=model.length_m=np.empty(0)
    model.elements=[];model._weld()
    return model

class AtForce(Exception):pass

class CollisionForceContinuityTests(unittest.TestCase):
    def local_case(self,**kwargs):
        s=fixture(**kwargs)
        selected=s.material_surface['vertex_owner']==1
        s.material_surface['vertices'][selected]=rotate(s.material_surface['vertices'][selected],np.array([0.,.000037,0.]))
        material_surface.refresh_geometry(s.material_surface)
        s.pos=material_surface.face_centres(s.material_surface)
        contacts.refresh(s)
        return s,weld_model(s)

    def test_disconnected_patches_keep_opposite_area_rates(self):
        s,model=self.local_case()
        operator=model.elements[0]['rows'][0]
        self.assertEqual(len(model.weld_rows),2)
        x=s.omega.ravel()/(balance.CM_YR_M_S/balance.RADIUS_M*balance.SECONDS_PER_MYR)
        rates=operator@x*balance.SECONDS_PER_MYR/1e6
        self.assertGreater(rates.max(),3000.)
        self.assertLess(rates.min(),-3000.)
        self.assertLess(abs(rates.sum()),1e-6)
        self.assertGreater(np.maximum(-rates,0).sum(),3000.)

    def test_local_weld_is_invariant_to_either_sheet_subdivision(self):
        _,base=self.local_case(coarse=True)
        def rows(model):
            order=np.argsort([r['front_center'][1] for r in model.weld_rows])
            element=model.elements[0]
            return tuple(element[key][order] if key!='rows' else element[key][0][order]
                         for key in ('rows','coefficient','scale'))
        expected=rows(base)
        for a,b in ((True,False),(False,True),(True,True)):
            with self.subTest(top=a,under=b):
                _,model=self.local_case(coarse=True,refine_top=a,refine_under=b)
                for actual,reference in zip(rows(model),expected):
                    np.testing.assert_allclose(actual,reference,rtol=2e-8,atol=1e-9)

    def test_local_weld_has_no_common_rigid_rotation_response(self):
        _,model=self.local_case(coarse=True)
        operator=model.elements[0]['rows'][0]
        np.testing.assert_allclose(operator[:,:3]+operator[:,3:],0.,atol=1e-12)

    def test_actual_step_keeps_contact_after_consolidation_without_aging(self):
        s=collided(sheet(-3000,1200,0),sheet(-1200,3000,1))
        before=weld_model(s)
        self.assertGreater(len(before.weld_rows),0)
        mass=s.mass.copy();thickness=s.structure['thickness_km'].copy()
        burial=s.parcel_burial_myr.copy();suture=s.parcel_collision_suture.copy()
        clocks={r['id']:(r['convergent_myr'],r['cumulative_convergence_km'],r['weld_cohesion']) for r in s.collision_contacts}
        captured={}
        def forces(dt):
            captured['welds']=len(weld_model(s).weld_rows)
            captured['rows']=s.collision_contacts
            raise AtForce()
        s._forces=forces
        # step() hands over to these through the instance.
        s._budgeted_step=lambda dt:native_engine.Simulation._budgeted_step(s,dt)
        s._step_once=lambda dt:native_engine.Simulation._step_once(s,dt)
        with patch.object(native_engine.world_design,'next_transition',return_value=None), \
             patch.object(native_engine.lip_events,'next_transition',return_value=None), \
             patch.object(native_engine.world_design,'record_transitions'), \
             self.assertRaises(AtForce):
            native_engine.Simulation.step(s,2.)
        self.assertTrue(s.consolidation_diagnostics['moved'])
        self.assertGreater(captured['welds'],0,'Ownership transfer removed the next force solve\'s weld.')
        self.assertEqual(s.t,0.)
        for actual,expected in ((s.mass,mass),(s.structure['thickness_km'],thickness),
                                (s.parcel_burial_myr,burial),(s.parcel_collision_suture,suture)):
            np.testing.assert_array_equal(actual,expected)
        for row in captured['rows']:
            if row['id'] in clocks:
                self.assertEqual((row['convergent_myr'],row['cumulative_convergence_km'],row['weld_cohesion']),clocks[row['id']])

if __name__=='__main__':unittest.main()
