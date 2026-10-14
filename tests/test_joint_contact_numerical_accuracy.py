"""Joint endpoint constraints and persistent numerical admission regressions."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import bounded_gravity as bg
import contact_response as contact
import deforming_regions as deform
import viscous_sheet as sheet
from numerical_accuracy import NumericalPolicy,allowance
from material_surface import spherical_face_areas

# A reviewed capture of the contact problem at 105 Myr. tmp/ is git-ignored and the capture was
# not published with the source, so the tests that load it skip where it is absent.
CAPTURE=Path(__file__).resolve().parents[1]/'tmp/contact105-original-capture-reviewed'
NO_CAPTURE='The contact-105 capture is not part of the published source.'

def captured():
    with np.load(CAPTURE/'contact-input.npz',allow_pickle=False) as z:
        return [z[k].copy() for k in ('points','faces','area','initial_velocity','rigid','minimum','maximum')]

def policy(A,lo,hi):return NumericalPolicy(A,allowance(A),np.zeros_like(A),A,lo,hi,reference_identity='independent_local_fixture')

@unittest.skipUnless((CAPTURE/'contact-input.npz').is_file(),NO_CAPTURE)
class JointContactTests(unittest.TestCase):
    def test_captured_joint_area_quality_stationarity_and_original_guards(self):
        args=captured();p,f,A,v,rigid,lo,hi=args;before=[x.copy() for x in args]
        q,result,solver,d=contact.redistribute(*args,1.,numerical_policy=policy(A,lo,hi))
        signed,Q=deform._quality(q,f);_,Q0=deform._quality(p,f)
        self.assertTrue(np.all(signed>1e-15));self.assertTrue(np.all(Q>=np.minimum(.15,Q0*.5)))
        self.assertGreater(d['quality_reaction_norm'],0.);self.assertIn('quality',d['active_sides'])
        self.assertLessEqual(solver['relative_residual'],1e-6);self.assertLessEqual(d['normalized_complementarity'],1e-6)
        self.assertLessEqual(d['sqp_iterations'],32);self.assertLessEqual(np.linalg.norm(result,axis=1).max(),100.)
        np.testing.assert_array_equal(q[rigid],p[rigid]);np.testing.assert_array_equal(result[rigid],0.)
        for original,current in zip(before,args):np.testing.assert_array_equal(original,current)
        measured=spherical_face_areas(q,f);self.assertTrue(policy(A,lo,hi).feasible(measured,lo,hi))
        # Independent geometry-derived endpoint reaction, finite differences
        # of both physical inequalities; never trust saved KKT scalars.
        basis=[]
        for point in p:
            e=np.eye(3)[np.argmin(abs(point))];b=np.cross(point,e);b/=np.linalg.norm(b);basis.append((b,np.cross(point,b)))
        ctx=sheet.prepare(p,f,A,100.);rhs=sheet.apply(ctx,v);rhs[rigid]=0.;work=float(np.sum(rhs*v));reaction=np.zeros_like(p)
        for face,side,m in zip(d['active_faces'],d['active_sides'],d['multipliers_scaled']):
            gradient=np.zeros_like(p)
            for node in np.unique(f[face]):
                if rigid[node]:continue
                for direction in basis[node]:
                    dv=np.zeros_like(result);dv[node]=direction*.001
                    qp=bg.endpoint(p,result+dv,1.,6371.);qm=bg.endpoint(p,result-dv,1.,6371.);qp[rigid]=p[rigid];qm[rigid]=p[rigid]
                    if side=='quality':value=(deform._quality(qp,f)[1][face]-deform._quality(qm,f)[1][face])/(2*.001*d['quality_minimum'][face])
                    else:value=(1 if side=='minimum' else -1)*(spherical_face_areas(qp,f)[face]-spherical_face_areas(qm,f)[face])/(2*.001*A[face])
                    gradient[node]+=direction*value
            reaction+=work*m*gradient
        residual=sheet.apply(ctx,result)-rhs-reaction;residual-=p*np.sum(p*residual,axis=1)[:,None];residual[rigid]=0.
        self.assertLess(np.linalg.norm(residual)/np.linalg.norm(rhs),1e-6)

    def test_false_preferred_certificate_is_recomputed(self):
        p,f,A,v,rigid,lo,hi=captured();ctx=sheet.prepare(p,f,A,100.);rhs=sheet.apply(ctx,v);rhs[rigid]=0.;body=rhs/ctx['data'][:,None]
        q,result,solver,d=bg.solve_endpoint(p,f,A,body,np.zeros_like(v),{'converged':True,'relative_residual':0.},rigid,lo,hi,1.,numerical_policy=policy(A,lo,hi),quality_minimum=np.minimum(.15,deform._quality(p,f)[1]*.5))
        self.assertTrue(d['inner_solves']);self.assertLessEqual(solver['relative_residual'],1e-6);self.assertGreater(np.linalg.norm(result),0.)

    def test_speed_failure_retains_endpoint_history_without_scaling(self):
        args=captured();A,lo,hi=args[2],args[5],args[6]
        with self.assertRaises(bg.SpeedBoundError) as caught:contact.redistribute(*args,1.,max_speed_km_myr=.1,numerical_policy=policy(A,lo,hi))
        evidence=caught.exception.diagnostics
        self.assertTrue(evidence['history']);self.assertIsNotNone(evidence['last_endpoint']);self.assertIsNotNone(evidence['last_velocity'])
        self.assertTrue(evidence['no_posthoc_geometry_correction'])

    def test_outer_exhaustion_retains_actual_failed_iterate(self):
        p,f,A,v,rigid,lo,hi=captured();ctx=sheet.prepare(p,f,A,100.);rhs=sheet.apply(ctx,v);rhs[rigid]=0.
        with self.assertRaises(bg.ConstraintSolveError) as caught:bg.solve_endpoint(p,f,A,rhs/ctx['data'][:,None],v,{'converged':True},rigid,lo,hi,1.,max_sqp=1,numerical_policy=policy(A,lo,hi),quality_minimum=np.minimum(.15,deform._quality(p,f)[1]*.5))
        self.assertEqual(len(caught.exception.diagnostics['history']),1);self.assertIsNotNone(caught.exception.diagnostics['last_endpoint'])

    def test_quality_formula_matches_exact_original_caller(self):
        p,f,A,v,rigid,lo,hi=captured()
        np.testing.assert_array_equal(bg._joint_quality(p,f)[0],deform._quality(p,f)[1])
        q=bg.endpoint(p,v,1.,6371.);np.testing.assert_array_equal(bg._joint_quality(q,f)[0],deform._quality(q,f)[1])

    def test_exact_zero_certificate_retains_original_physical_tightness(self):
        import gravity_constraints_schema
        p,f,A,v,rigid,lo,hi=captured();ctx=sheet.prepare(p,f,A,100.)
        rhs=np.zeros_like(p);np.add.at(rhs,f[0],bg.endpoint_area_jacobian(p,np.zeros_like(p),f,1.,6371.)[0])
        rhs-=p*np.sum(p*rhs,axis=1)[:,None];rhs[rigid]=0.
        body=rhs/ctx['data'][:,None];start=rhs.copy();floor=np.minimum(.15,deform._quality(p,f)[1]*.5)
        q,result,solver,d=bg.solve_endpoint(p,f,A,body,start,{'iterations':1,'maximum_iterations':2},rigid,
            A*.5,A.copy(),1.,numerical_policy=policy(A,A*.5,A),quality_minimum=floor)
        np.testing.assert_array_equal(q,p);np.testing.assert_array_equal(result,0.)
        self.assertEqual(solver['stationarity_kind'],'current_geometry_zero_kkt')
        self.assertTrue(d['stationary_current_geometry']);self.assertLessEqual(solver['relative_residual'],1e-8)
        row=dict(dt_myr=1.,solver=solver,bound_constraints=d,stationary_within_constraint_kkt=True,
            stationary_within_energy_roundoff=True,virtual_power_km4_myr=0.,
            numerical_accuracy_policy=policy(A,A*.5,A).evidence(A,A*.5,A,include_arrays=False))
        gravity_constraints_schema.validate_accepted_row(row)
        # An area excursion admitted by the user's policy is not a tight
        # physical bound and cannot be relabeled an exact-zero equilibrium.
        self.assertIsNone(bg.current_stationary_certificate(p,f,A,rhs,rigid,A*.5,A-1e-5,
            1.,6371.,1e-8,start,feasibility_tolerance=1e-12))

    def test_serial_accepted_substeps_cannot_renew_spent_credit(self):
        p,f,A,v,rigid,lo,hi=captured();stage=policy(A,A,A);axis=np.zeros_like(p);gradient=v.copy()*1000.
        # Independent controlled area observations isolate bookkeeping. The
        # original nonrenewing policy must reject the third excursion after
        # an intermediate return to nominal area, even though final miss fits
        # the initially allocated allowance. Geometry/forces are not tested here.
        observations=[A+1.5,A.copy(),A+1.5]
        with patch.object(deform,'spherical_face_areas',side_effect=observations):
            result=deform._advance_region(p,f,axis,gradient,6371.,3.,A,np.ones_like(A),np.ones_like(A),deform._quality(p,f)[1],3,1,numerical_policy=stage)
        self.assertAlmostEqual(result[1],2/3);self.assertAlmostEqual(result[-1].spent_km2[0],1.5);self.assertTrue(result[4][0])
        np.testing.assert_array_equal(stage.spent_km2,0.)

    def test_rigid_actual_area_ratio_matches_exact_volume_policy_only(self):
        import numerical_accuracy as accuracy
        import crustal_structure as columns
        p,f,A,v,rigid,lo,hi=captured()
        mesh=dict(vertices=p,faces=f,vertex_owner=np.zeros(len(p),np.int32),
            face_owner=np.zeros(len(f),np.int32),face_kind=np.ones(len(f),np.uint8),radius_km=6371.)
        boundaries=dict(bmid=np.array([[1.,0.,0.]]),bn=np.array([[0.,1.,0.]]),bl=np.array([100.]),
            bp=np.array([0]),bq=np.array([1]))
        omega=np.tile([.4,-.2,.3],(2,1));limits=dict(face_min_area_ratio=np.ones_like(A),
            face_max_area_ratio=np.ones_like(A)*75/8,mechanics_version=1,require_material_contact=False,
            vertex_protected=np.ones(len(p),bool))
        legacy=deform.deform(mesh,omega,boundaries,1.,**limits)
        np.testing.assert_array_equal(legacy['areal_strain'],0.)
        active=deform.deform(mesh,omega,boundaries,1.,numerical_policy=policy(A,A,A*75/8),**limits)
        actual=spherical_face_areas(active['vertices'],f)
        np.testing.assert_array_equal(active['area_ratio'],actual/A)
        np.testing.assert_array_equal(active['areal_strain'],np.log(actual/A))
        np.testing.assert_array_equal(active['vertices'],legacy['vertices'])
        state=columns.initialize_structure(np.ones(len(A),np.uint8),np.ones(len(A))*220.)
        state['thickness_km'][:]=75.;accuracy.initialize_columns(state,A)
        endpoint=accuracy.endpoint_columns(state,actual,active['diagnostics']['numerical_accuracy_final_spent_km2'])
        new,_=columns.evolve_structure(state,1.,geometric_log_area=active['areal_strain'],geometric_accuracy=endpoint)
        np.testing.assert_allclose(actual*new['thickness_km'],A*75.,rtol=2e-15,atol=0.)

if __name__=='__main__':unittest.main()
