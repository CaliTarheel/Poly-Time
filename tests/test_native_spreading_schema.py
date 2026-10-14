"""Saved exact ridge pieces validate strictly and retain real replay fields."""
from copy import deepcopy
import unittest
import numpy as np
import native_spreading
import native_frame_sampling
from tectonics import Simulation


class NativeSpreadingSchemaTests(unittest.TestCase):
    def test_normal_spreading_policy_is_optional_and_strictly_versioned(self):
        for value in (None,0,1):
            frame=deepcopy(self.frame)
            frame.pop('normal_spreading_version',None)
            if value is not None:
                frame['normal_spreading_version']=value
            native_spreading.validate_frame(frame)
        for value in (True,1.,-1,2,'1'):
            frame=deepcopy(self.frame)
            frame['normal_spreading_version']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                native_spreading.validate_frame(frame)
        with self.assertRaises(ValueError):
            native_spreading.validate_frame({'normal_spreading_version':1})

    @classmethod
    def setUpClass(cls):
        cls.simulation=Simulation(dict(mesh_level=2,coast_geometry_level=2,adaptive_refinement=0,
            width=48,height=24,plate_count=4,seed=37))
        cls.simulation.step(2.)
        cls.frame=cls.simulation.snapshot()

    def test_real_step_saved_geometry_and_length_statistics_prepare(self):
        native_spreading.validate_frame(self.frame)
        native_frame_sampling.prepare(self.frame)
        self.assertEqual(self.frame['native_spreading_version'],1)
        self.assertGreater(len(self.frame['boundary_segments']),0)
        # A corrected initial owner partition need not contain an opening
        # ocean/ocean contact. Explicit moving-front tests cover positive birth.
        self.assertGreaterEqual(self.frame['spreading_diagnostics']['generated_area_km2'],0.)
        self.assertAlmostEqual(self.frame['spreading_diagnostics']['side_p_area_km2'],
                               self.frame['spreading_diagnostics']['side_q_area_km2'],places=9)

    def test_version_and_parent_owners_must_be_strict_integers(self):
        mutations=[lambda f:f.update(native_spreading_version=1.),
                   lambda f:f.update(native_spreading_version=True),
                   lambda f:f['boundary_segments'][0].update(geometry_version=np.bool_(True)),
                   lambda f:f['boundary_segments'][0].update(geometry_version=1.),
                   lambda f:f.update(native_boundary_owner_a=f['native_boundary_owner_a'].astype(float)+.2)]
        for mutate in mutations:
            f=deepcopy(self.frame);mutate(f)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                native_frame_sampling.prepare(f)

    def test_local_pairing_policy_and_area_budgets_are_strict(self):
        self.assertEqual(self.frame['spreading_diagnostics']['local_pairing_version'],1)
        mutations=[lambda d:d.pop('local_pairing_version'),
                   lambda d:d.update(local_pairing_version=True),
                   lambda d:d.update(local_pairing_version=1.),
                   lambda d:d.update(local_paired_patches=.5),
                   lambda d:d.update(local_unpaired_area_km2=float('nan')),
                   lambda d:d.update(local_unpaired_area_km2=complex(0.,1.)),
                   lambda d:d.update(candidate_paired_area_km2=d['candidate_paired_area_km2']+1.),
                   lambda d:d.update(maximum_water_capacity_residual_km2=.01),
                   lambda d:d.pop('maximum_water_capacity_residual_km2')]
        for mutate in mutations:
            f=deepcopy(self.frame);mutate(f['spreading_diagnostics'])
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                native_frame_sampling.prepare(f)

    def test_old_area_pooling_diagnostics_remain_readable(self):
        f=deepcopy(self.frame)
        for name in native_spreading.LOCAL_PAIRING_MARKERS:
            f['spreading_diagnostics'].pop(name,None)
        native_frame_sampling.prepare(f)

    def test_capacity_roundoff_metadata_is_optional_but_validated(self):
        legacy=deepcopy(self.frame)
        legacy['spreading_diagnostics'].pop('capacity_roundoff_removed_area_km2',None)
        native_spreading.validate_frame(legacy)
        for value in (-1.,float('nan'),float('inf'),True,'0'):
            f=deepcopy(self.frame)
            f['spreading_diagnostics']['capacity_roundoff_removed_area_km2']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                native_spreading.validate_frame(f)

    def test_the_converging_label_is_a_valid_segment_code_only_under_label_version_two(self):
        f=deepcopy(self.frame)
        row=next(r for r in f['boundary_segments'] if r['code'] not in (1,5))
        row['code']=6
        with self.assertRaisesRegex(ValueError,'classification'):native_spreading.validate_frame(f)
        f['boundary_label_version']=2
        native_spreading.validate_frame(f)
        row['code']=7
        with self.assertRaisesRegex(ValueError,'classification'):native_spreading.validate_frame(f)

    def test_local_policy_markers_require_spreading_version_even_without_segments(self):
        f=deepcopy(self.frame);f.pop('native_spreading_version');f['boundary_segments']=[]
        with self.assertRaises(ValueError):native_frame_sampling.prepare(f)

    def test_malformed_nested_geometry_is_rejected_by_sampler(self):
        mutations=[lambda r:r.update(material_side_classification='shore-ish'),
                   lambda r:r.update(normal=[float('nan'),0.,0.]),
                   lambda r:r.update(geometry_xyz=[[1.,0.,0.],[2.,0.,0.]]),
                   lambda r:r.update(contact_index=10000000),
                   lambda r:r.update(contact_index=0.5),
                   lambda r:r.update(owner_a=-1),
                   lambda r:r.update(code=1,material_side_classification='continental')]
        for mutate in mutations:
            f=deepcopy(self.frame);mutate(f['boundary_segments'][0])
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                native_frame_sampling.prepare(f)

    def test_piece_length_tampering_cannot_change_reported_ridge_statistics(self):
        f=deepcopy(self.frame);f['stats']['ridge_length_km']+=1.
        with self.assertRaisesRegex(ValueError,'length statistic'):
            native_frame_sampling.prepare(f)

    def test_unversioned_history_preserves_original_sampler_behavior(self):
        f=deepcopy(self.frame);f.pop('native_spreading_version')
        f['boundary_segments']=[{'legacy_review_metadata':'not authoritative geometry'}]
        for name in native_spreading.LOCAL_PAIRING_MARKERS:
            f['spreading_diagnostics'].pop(name,None)
        native_frame_sampling.prepare(f)

    def test_stripping_version_cannot_silently_downgrade_exact_metadata(self):
        for value in (None,0):
            f=deepcopy(self.frame)
            if value is None:f.pop('native_spreading_version')
            else:f['native_spreading_version']=value
            with self.subTest(version=value),self.assertRaisesRegex(ValueError,'requires its spreading version'):
                native_frame_sampling.prepare(f)

    def test_rasterization_keeps_partial_ocean_history_and_is_idempotent(self):
        s=deepcopy(self.simulation)
        partial=np.flatnonzero((s.crust>0)&(s.land_mass<s.cell_area*.999))
        self.assertGreater(len(partial),0)
        cell=int(partial[0]);owner=int(s.plate[cell]);other=(owner+1)%4
        s.support[:,cell]=0.;s.support[owner,cell]=.63;s.support[other,cell]=.37;s.age[cell]=123.
        expected=s.support[:,cell].copy()
        pure=np.flatnonzero((s.crust>0)&(np.abs(s.land_mass-s.cell_area)<s.cell_area*1e-11))
        self.assertGreater(len(pure),0)
        dry=int(pure[0]);s.age[dry]=91.;s.support[:,dry]=0.;s.support[:4,dry]=.25
        for _ in range(2):
            s._rasterize()
            np.testing.assert_array_equal(s.support[:,cell],expected)
            self.assertEqual(s.age[cell],123.)
            self.assertEqual(s.age[dry],0.)
            self.assertEqual(s.support[s.plate[dry],dry],1.)
            self.assertEqual(np.count_nonzero(s.support[:,dry]),1)
        s._boundaries();f=s.snapshot()
        native_spreading.validate_frame(f)
        native_frame_sampling.prepare(f)
        np.testing.assert_array_equal(f['native_boundary_owner_a'],s.plate[s.ba])
        np.testing.assert_array_equal(f['native_boundary_owner_b'],s.plate[s.bb])
        self.assertTrue(np.isfinite(f['native_boundary_normal_speed_km_myr']).all())


if __name__=='__main__': unittest.main()
