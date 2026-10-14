"""Read-only current-source checkpoint audit; never construct or advance a world."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import native_engine
import server
import crust_inventory
import dense_crust
from checkpoint import read_checkpoint, checkpoint_header

ARRAYS = ('elevation', 'plate', 'crust', 'age', 'domain', 'mesh_vertices', 'mesh_faces',
    'mesh_plate', 'mesh_crust', 'mesh_age_myr', 'mesh_area_km2', 'mesh_owner_slots',
    'mesh_vertex_support', 'material_vertices', 'material_faces', 'material_face_id',
    'material_owner', 'material_kind', 'material_height_m', 'material_reference_area_km2',
    'material_reference_corners', 'material_domain', 'material_crustal_thickness_km',
    'material_refinement_level')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def stable_checkpoint(path, expected):
    for attempt in range(4):
        prior = sha(path)
        header = checkpoint_header(path)
        state, committed = read_checkpoint(path, expected, native_engine.Simulation)
        after = sha(path)
        if prior == after:
            return state, committed, header, prior
        time.sleep(.1)
    raise RuntimeError('Checkpoint changed during the bounded read; retry this audit later.')


def speeds(state):
    values = np.linalg.norm(np.cross(state.omega[state.plate], state.xyz), axis=1) * 6371. * .1
    material_slots = set(map(int, np.unique(state.parcel_plate[state.kind != 3])))
    # This groups the entire plate, including attached ocean, by continental material ownership.
    continental = np.isin(state.plate, list(material_slots))
    result = dict(scope='Native control-cell rigid Euler velocity; cm/yr; continental group includes attached ocean',
        global_area_weighted_mean=float(np.average(values, weights=state.cell_area)),
        global_control_max=float(values.max()))
    for name, mask in (('continental_bearing_plates', continental), ('pure_ocean_plates', ~continental)):
        result[name] = dict(area_km2=float(state.cell_area[mask].sum()),
            area_weighted_mean=float(np.average(values[mask], weights=state.cell_area[mask])) if mask.any() else None,
            control_max=float(values[mask].max()) if mask.any() else None)
    result['plates'] = [dict(uid=int(state.plate_uid[p]), name=state.names[p],
        continental_bearing=p in material_slots, area_km2=float(state.cell_area[state.plate == p].sum()),
        mean_speed_cm_yr=float(np.average(values[state.plate == p], weights=state.cell_area[state.plate == p])),
        max_speed_cm_yr=float(values[state.plate == p].max())) for p in np.unique(state.plate)]
    return result


def initial_identity(run):
    paths = {'new': run / 'frame_0000.npz',
        'weak_lite': ROOT / 'output/runs/20261002-101850-e80ab8/frame_0000.npz',
        'original_detailed': ROOT.parent / 'Deep-Time-asein-startup/output/runs/20261001-153852-27d072/frame_0000.npz'}
    files = {name: sha(path) for name, path in paths.items()}
    rows = {}
    with np.load(paths['new'], allow_pickle=False) as actual:
        for name in ('weak_lite', 'original_detailed'):
            with np.load(paths[name], allow_pickle=False) as prior:
                rows[name] = {key: dict(equal=bool(actual[key].dtype == prior[key].dtype and
                    actual[key].shape == prior[key].shape and actual[key].tobytes() == prior[key].tobytes()),
                    shape=list(actual[key].shape), dtype=actual[key].dtype.str,
                    sha256=digest(actual[key].tobytes())) for key in ARRAYS}
    return dict(array_count=len(ARRAYS), files_sha256=files, comparisons=rows,
        all_equal=all(row['equal'] for comparison in rows.values() for row in comparison.values()),
        scope='Geometry/material/ownership/artwork only; detailed and Lite trench initialization laws differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', default='reviews/asein-integration/lite-3e13-production-verification.json')
    parser.add_argument('--port', type=int, default=8769)
    args = parser.parse_args()
    run = ROOT / 'output/runs' / args.run_id
    output = ROOT / args.output
    expected = server.SimulationManager.compatibility()
    # The supported reader uses __new__; these sentinels make an accidental model action explicit.
    native_engine.Simulation.__init__ = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Constructor forbidden.'))
    native_engine.Simulation.step = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Step forbidden.'))
    state, committed, header, checkpoint_sha = stable_checkpoint(run / 'checkpoint.npz', expected)
    report = dict(kind='asein_lite_current_source_read_only_production_verification',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), run_id=args.run_id,
        read_only=True, constructor_called=False, simulation_stepped=False, server_mutated=False,
        compatibility_scope='Exact CURRENT runtime closure and both current native backend fingerprints; no compatibility substitution',
        checkpoint=dict(sha256=checkpoint_sha, time_myr=float(state.t), steps=int(state.steps),
            embedded_next_output_myr=committed.get('next_output_myr'), compatibility=header['compatibility']),
        state='values_recorded_before_audit_checks', config=state.config)
    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/status', timeout=10) as response:
        status = json.load(response)
    report['api'] = {key: status.get(key) for key in ('run_id', 'state', 'error', 'time_myr', 'integration_time_myr', 'checkpoint_time_myr')}
    report['initial_identity'] = initial_identity(run)
    report['source_closure'] = dict(helper_count=len(expected['auxiliary_sources_sha256']),
        runtime_matches_checkpoint=expected == header['compatibility'],
        saved_helpers_match=all(sha(run / name) == value for name, value in expected['auxiliary_sources_sha256'].items()),
        core_candidate_sha256=sha(ROOT / 'bounded_gravity.py'))
    mass, area, structure = state.mass, state.material_surface['area_km2'], state.structure
    budget = dict(getattr(state, 'material_column_budget', {}))
    denominator = max(abs(float(budget.get('before_motion_volume_km3', 0.))), 1.)
    relative = float(budget.get('residual_km3', 0.)) / denominator
    report['column_budget'] = dict(ledger=budget, residual_km3=budget.get('residual_km3'),
        relative_residual=relative, relative_denominator_km3=denominator, audit_relative_tolerance=1e-10,
        tolerance_source='tests/test_collision_contacts.py:291 before_motion_volume_km3 * 1e-10; this is an extra audit, not a production guard',
        scope='Last accepted mechanical/column source interval BEFORE separate juvenile arc emplacement; includes magma, erosion, foundering and phase contraction',
        physical_current_volume_km3=float(area @ structure['thickness_km']),
        reference_current_volume_km3=float(mass @ (structure['area_factor'] * structure['thickness_km'])),
        geometric_vs_reference_volume_residual_km3=float(area @ structure['thickness_km'] - mass @ (structure['area_factor'] * structure['thickness_km'])),
        reference_area_equals_mass=bool(np.array_equal(state.material_surface['reference_area_km2'], mass)),
        last_geometric_shortening_volume_residual_km3=getattr(state, 'material_geometry_volume_residual_km3', None))
    totals = dict(state.process_totals)
    created_reference = float(totals.get('arc_added_km2', 0.)) - float(totals.get('arc_deposited_source_area_km2', 0.))
    expected_reference = float(state.original_mass) + created_reference
    report['reference_material_accounting'] = dict(original_reference_area_km2=float(state.original_mass),
        retained_reference_area_km2=float(mass.sum()), juvenile_created_reference_area_km2=created_reference,
        retained_minus_original_and_created_residual_km2=float(mass.sum() - expected_reference),
        relative_residual=float((mass.sum()-expected_reference)/max(expected_reference, 1.)),
        continental_retained_fraction=float((mass.sum()-created_reference)/state.original_mass),
        craton_retained_fraction=float(mass[state.kind == 2].sum()/state.original_craton_mass),
        reference_area_by_material_kind={str(kind): float(mass[state.kind == kind].sum()) for kind in np.unique(state.kind)},
        process_totals=totals, material_face_count=len(mass),
        scope='Reference material area follows parcels; geographic area may strain. Ocean created/consumed is a separate finite-volume inventory, not a continental mass loss.',
        arc_emplacement_ledger=getattr(state, 'arc_material_diagnostics', None),
        adaptation_ledger=getattr(state, 'material_adaptivity_diagnostics', None))
    report['composition_inventory'] = dict(present=crust_inventory.present(structure),
        weighted_fields={name: float(mass @ structure[name]) for name in crust_inventory.FIELDS if name in structure})
    if crust_inventory.present(structure):
        fields = report['composition_inventory']['weighted_fields']
        supplied = fields[crust_inventory.BASELINE] + fields[crust_inventory.ADDED]
        accounted = fields[crust_inventory.ERODED] + fields[crust_inventory.RETURNED] + fields[crust_inventory.REMAINING]
        report['composition_inventory'].update(supplied_km3=supplied, accounted_km3=accounted,
            residual_km3=supplied-accounted, relative_residual=(supplied-accounted)/max(supplied, 1.))
    report['retained_dense_phase'] = dict(present=dense_crust.present(structure),
        weighted_fields={name: float(mass @ structure[name]) for name in dense_crust.FIELDS if name in structure})
    report['speeds'] = speeds(state)
    report['plate_balance'] = getattr(state, 'plate_balance_diagnostics', {})
    report['timestep_diagnostics'] = getattr(state, 'timestep_diagnostics', {})
    report['deformation_diagnostics'] = getattr(state, 'deformation_diagnostics', {})
    report['collision_contacts'] = getattr(state, 'collision_contacts', [])
    report['saved_epochs'] = committed.get('frames', [])
    # Publish measured values before evaluating any assertions or state validators.
    write(output, report)
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    balance = report['plate_balance']
    check('accepted_checkpoint_at_least_1myr', state.t >= 1.)
    check('exact_24_initial_arrays', report['initial_identity']['all_equal'])
    check('current_source_closure', report['source_closure']['runtime_matches_checkpoint'] and report['source_closure']['saved_helpers_match'])
    check('requested_effective_force', state.config['effective_subduction']['force_n_per_m'] == 3e13)
    check('column_volume_relative_audit', abs(relative) < 1e-10)
    check('reference_material_area_closure', abs(report['reference_material_accounting']['relative_residual']) < 1e-12)
    check('reference_area_alignment', report['column_budget']['reference_area_equals_mass'])
    check('continental_retention', abs(report['reference_material_accounting']['continental_retained_fraction']-1.) < 1e-12)
    check('craton_retention', abs(report['reference_material_accounting']['craton_retained_fraction']-1.) < 1e-12)
    check('force_balance', balance.get('scaled_force_residual', float('inf')) <= balance.get('force_relative_tolerance', 0.))
    check('passivity', balance.get('negative_resisting_work_elements', -1) == 0)
    power_scale = max(abs(float(balance.get('driver_work_w', 0.))), 1.)
    report['power_relative_residual'] = float(balance.get('power_balance_error_w', float('inf'))) / power_scale
    check('power_balance', abs(report['power_relative_residual']) <= 1e-8)
    nonpenetration = report['deformation_diagnostics'].get('same_sheet_nonpenetration', {})
    check('same_sheet_nonpenetration', nonpenetration.get('final_overlap_pairs', -1) == 0)
    for name, validator in (('composition_valid', crust_inventory.validate), ('dense_phase_valid', dense_crust.validate)):
        try:
            validator(structure)
            checks[name] = True
        except ValueError as error:
            checks[name] = False
            report[name + '_error'] = str(error)
    report.update(checks=checks, state='extra_read_only_audit_passed' if all(checks.values()) else 'extra_read_only_audit_has_discrepancy',
        audit_failure_is_production_guard_failure=False,
        current_api_reports_production_error=bool(status.get('error')))
    write(output, report)
    print(json.dumps(dict(state=report['state'], run_id=args.run_id, checkpoint_myr=state.t,
        checkpoint_sha256=checkpoint_sha, column_volume_residual_km3=budget.get('residual_km3'),
        column_relative_residual=relative, global_mean_speed_cm_yr=report['speeds']['global_area_weighted_mean'],
        global_max_speed_cm_yr=report['speeds']['global_control_max'], force_residual=balance.get('scaled_force_residual'),
        failed_checks=[name for name, passed in checks.items() if not passed], output=str(output), output_sha256=sha(output))), flush=True)


if __name__ == '__main__':
    main()
