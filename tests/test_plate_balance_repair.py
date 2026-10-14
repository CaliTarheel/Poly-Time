"""Regression cases for active remnants, solver failure, and convex friction."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import plate_balance as balance
from mesh_geometry import icosphere
import mesh_transport


class PlateBalanceRepairTests(unittest.TestCase):
    def test_fractional_remnant_has_drag_without_dominant_cells(self):
        mesh = icosphere(1)
        n = len(mesh['faces'])
        s = SimpleNamespace(plate=np.zeros(n, int), support=np.vstack((np.full(n,.9),np.full(n,.1))),
            cell_area=mesh['area_km2'], xyz=mesh['xyz'], crust=np.zeros(n,int))
        model = balance.Balance.__new__(balance.Balance)
        model.s=s; model.plates=[0,1]; model.slot={0:0,1:1}; model.size=6
        model.bp=np.zeros(0,int);model.live=np.zeros(0,int);model.trench=np.zeros(0,bool);model.notes={}
        model._viscous()
        self.assertGreater(np.linalg.eigvalsh(model.basal[1]).min(), 0.)
        np.testing.assert_allclose(model.basal[1], model.basal[0]/9, rtol=1e-14, atol=1e-6)

    def test_unresisted_mode_is_rejected_without_diagonal_drag(self):
        with self.assertRaisesRegex(ValueError, 'unresisted'):
            balance.Balance._solve_spd(np.diag([1.,1.,0.]), np.array([0.,0.,1.]))

    def test_extreme_area_contrast_does_not_change_linear_solution(self):
        matrix=np.diag([1e-20,1.,1e20]); expected=np.array([2.,-3.,4.])
        np.testing.assert_allclose(balance.Balance._solve_spd(matrix,matrix@expected),expected,rtol=1e-14)

    def fixture(self):
        model=balance.Balance.__new__(balance.Balance)
        model.s=SimpleNamespace()
        model.size=3; model.stiffness=np.eye(3); model.torque=np.array([-4.,3.,1.])
        model.hinge=np.zeros((0,3)); model.hinge_coefficient=np.zeros(0)
        model.elements=[dict(kind='megathrust',shape='cone', rows=(np.array([[1.,0.,0.]]),
            np.array([[0.,1.,0.]])), coefficient=np.array([5.]),scale=np.ones(1))]
        return model

    def test_cone_has_psd_curvature_and_correct_derivatives(self):
        model=self.fixture(); model.stiffness[:]=0.;model.torque[:]=0.
        for opening in (-5.,-.1,0.,.1,1.,5.):
            x=np.array([opening,.3,0.]); h=1e-5
            _,g,H,_,_=model._evaluate(x,1.)
            self.assertGreaterEqual(np.linalg.eigvalsh(H).min(), -1e-12)
            numeric=np.column_stack([(model._evaluate(x+np.eye(3)[i]*h,1.)[1]
                -model._evaluate(x-np.eye(3)[i]*h,1.)[1])/(2*h) for i in range(3)])
            np.testing.assert_allclose(H,numeric,atol=1e-8,rtol=1e-7)

    def test_converged_force_and_exhausted_solver_rejection(self):
        model=self.fixture();model.solve()
        residual=model._evaluate(model.x,balance.HUBER_CONTINUATION_KM_MYR[-1]*balance.KM_MYR_CM_YR)[1]
        self.assertLessEqual(model._relative_residual(residual),balance.FORCE_RELATIVE_TOLERANCE)
        with patch.object(balance,'MAX_NEWTON_ITERATIONS',0):
            with self.assertRaisesRegex(ValueError,'did not converge'):
                self.fixture().solve()

    def test_runaway_transport_fails_before_touching_input(self):
        mesh=icosphere(1); values=np.arange(len(mesh['faces']),dtype=float); before=values.copy()
        with self.assertRaisesRegex(ValueError,'resource limit'):
            mesh_transport.advect(mesh,values,[-25494069.,-166480598.,60123841.],2.)
        np.testing.assert_array_equal(values,before)

    def test_energy_roundoff_cannot_block_force_convergence(self):
        model=self.fixture(); model.elements=[]; model.torque=np.array([0.,0.,1.])
        model._warm_start=lambda: (np.array([0.,0.,1.-1e-6]),np.eye(3))
        evaluate=model._evaluate
        def rounded(x,delta,want_derivatives=True):
            _,g,h,parts,yielded=evaluate(x,delta,want_derivatives)
            # A one-ulp energy barrier conceals the actual decrease near the
            # known quadratic optimum. Forces remain independently resolved.
            value=np.nextafter(1e20,np.inf) if x[2]>1.-2e-7 else 1e20
            return value,g,h,parts,yielded
        model._evaluate=rounded
        model.solve()
        np.testing.assert_allclose(model.x,[0.,0.,1.],atol=1e-12,rtol=0.)

    def test_roundoff_band_still_rejects_unimproved_forces(self):
        model=self.fixture(); model.elements=[]; model.torque=np.array([0.,0.,1.])
        model._warm_start=lambda: (np.zeros(3),np.eye(3))
        def no_progress(x,delta,want_derivatives=True):
            value=np.nextafter(1e20,np.inf) if np.any(x) else 1e20
            return value,np.array([0.,0.,-1e-6]),np.eye(3),{},{}
        model._evaluate=no_progress
        with self.assertRaisesRegex(ValueError,'line search failed'):
            model.solve()

    def test_zero_external_driver_has_no_smoothing_generated_motion(self):
        model=self.fixture();model.torque[:]=0.;model.resistance_version=1
        model.solve()
        delta=balance.HUBER_CONTINUATION_KM_MYR[-1]*balance.KM_MYR_CM_YR
        np.testing.assert_array_equal(model.x,0.)
        gradient=model._evaluate(model.x,delta)[1]
        self.assertLessEqual(model._relative_residual(gradient),balance.FORCE_RELATIVE_TOLERANCE)

    def test_large_plate_driver_cannot_hide_unbalanced_force_on_weak_plate(self):
        model=self.fixture();model.size=6;model.elements=[]
        model.stiffness=np.eye(6);model.torque=np.array([1e20,0.,0.,1e-20,0.,0.])
        model.hinge=np.empty((0,6))
        wrong=np.array([1e20,0.,0.,0.,0.,0.])
        gradient=model._evaluate(wrong,.01)[1]
        self.assertEqual(model._relative_residual(gradient),1.)
        model.solve()
        np.testing.assert_array_equal(model.x,model.torque)

    def test_tiny_resistive_potential_matches_high_precision_without_cancellation(self):
        from decimal import Decimal, localcontext
        speed=1e-10
        with localcontext() as context:
            context.prec=60
            v=Decimal.from_float(speed)
            expected=float((1+v*v).sqrt()-1)
        self.assertGreater(expected,0.)
        self.assertAlmostEqual(float(balance._abs_terms(np.array([speed]),1.)[0][0])/expected,1.,places=14)
        model=self.fixture();model.stiffness[:]=0.;model.torque[:]=0.
        model.resistance_version=1
        value,_,_,_,_=model._evaluate(np.array([1.,speed,0.]),1.)
        self.assertAlmostEqual(value/(5.*expected),1.,places=14)


if __name__=='__main__':
    unittest.main()
