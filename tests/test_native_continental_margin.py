"""Engine, saved history and checkpoint contracts for versioned coast profiles."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from native_engine import Simulation
from checkpoint import write_checkpoint, read_checkpoint
import mesh_history
import continental_margin
import native_frame_sampling


def world(**changes):
    config=dict(seed=37,width=96,height=48,duration_myr=4,mesh_level=2,
                coast_geometry_level=2,mechanics_nodes=128,plate_count=4)
    config.update(changes)
    return Simulation(config)


class NativeContinentalMarginTests(unittest.TestCase):
    def test_ocean_only_initial_snapshot_is_valid_without_a_continental_outline(self):
        s=Simulation(dict(width=96,height=48,mesh_level=2,coast_geometry_level=2,plate_count=4),
                     dict(width=48,height=24,crust=np.zeros(48*24,np.uint8)))
        frame=s.snapshot()
        mesh_history.arrays(frame)
        self.assertEqual(len(frame['material_faces']),0)
        self.assertTrue(np.all(frame['crust']==0))
        self.assertTrue(np.isfinite(frame['elevation']).all())

    def test_profile_choices_and_snapshot_cadence_do_not_change_physical_evolution(self):
        a=world(continental_margin_width_km=75.)
        b=world(continental_margin_width_km=250.,continental_shelf_depth_m=300.)
        for stage in range(2):
            for name in ('plate','omega','mantle','mass','pos','parcel_patch','kind'):
                np.testing.assert_array_equal(getattr(a,name),getattr(b,name),err_msg=name)
            for key in a.structure:
                np.testing.assert_array_equal(a.structure[key],b.structure[key],err_msg=key)
            for key in a.material_lineage:
                np.testing.assert_array_equal(a.material_lineage[key],b.material_lineage[key],err_msg=key)
            if stage==0:
                a.snapshot();a.snapshot();a.step();b.step()
        self.assertIs(a._forces.__func__,b._forces.__func__)

    def test_snapshot_sampler_schema_and_checkpoint_preserve_margin_version(self):
        s=world();frame=s.snapshot()
        self.assertEqual(frame['continental_margin_version'],1)
        mesh_history.arrays(frame)
        lon,lat=np.meshgrid((np.arange(s.w)+.5)*2*np.pi/s.w-np.pi,
                           np.pi/2-(np.arange(s.h)+.5)*np.pi/s.h)
        points=np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                                np.cos(lat).ravel()*np.sin(lon).ravel(),np.sin(lat).ravel()))
        sampled=native_frame_sampling.sample_frame(frame,points)
        np.testing.assert_array_equal(frame['elevation'],sampled['elevation'].astype(np.float32))
        np.testing.assert_array_equal(frame['plate'],sampled['plate'])
        for patch in (dict(continental_margin_version=3),
                      dict(continental_margin_parameters=dict(width_km=-1.,shelf_depth_m=180.))):
            invalid=dict(frame,**patch)
            with self.assertRaises(ValueError):mesh_history.arrays(invalid)
        old=dict(frame)
        for name in list(old):
            if name.startswith('continental_margin_'):del old[name]
        mesh_history.arrays(old)
        derived=continental_margin.upgrade_frame(old)
        # Optional array families this world does not use are in neither frame.
        saved=set(mesh_history.DTYPES).intersection(frame)
        self.assertEqual(set(mesh_history.DTYPES).intersection(derived),saved)
        for name in saved:
            np.testing.assert_array_equal(derived[name],frame[name])
        # Snapshot is read-only and all new state consists of ordinary typed
        # checkpoint fields, with no hidden locator or nonserializable cache.
        compatibility=dict(test='continental-margin-v1')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint'
            write_checkpoint(path,s,dict(config=s.config),compatibility)
            restored,_=read_checkpoint(path,compatibility,Simulation)
        self.assertEqual(restored.continental_margin_version,1)
        self.assertEqual(restored.continental_margin_parameters,s.continental_margin_parameters)
        actual=restored.snapshot()
        np.testing.assert_array_equal(actual['elevation'],frame['elevation'])


if __name__=='__main__':unittest.main()
