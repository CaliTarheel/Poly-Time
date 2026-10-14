"""Read-only accepted continuation proof for the numerical boundary repair.

No constructor, force solve, advection or boundary rebuild is permitted. The
permanent source boundary is decoded under the new closure and compared with
the entire predecessor state decoded under its own recorded closure.
"""
import argparse
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
PARENT = '20261002-162235-9da60a'
EPOCH = 54.


def audit(args):
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    base = importlib.import_module('lite_checkpoint_verification')
    old = importlib.import_module('lite_feedback_verification')
    transition = importlib.import_module('lite_boundary_geometry_transition')
    force = importlib.import_module('force_rifting')
    carrier = importlib.import_module('effective_subduction_carrier_traction')
    initiation = importlib.import_module('effective_subduction_initiation')
    def forbidden(*a, **k):
        raise AssertionError('Read-only audit forbids constructing or evolving a model.')
    for name in ('__init__', 'step', '_step_once', '_advance_step', '_boundaries', '_forces'):
        setattr(base.native_engine.Simulation, name, forbidden)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run_id) or args.run_id == PARENT:
        raise ValueError('A separate successor identity is required.')
    run = ROOT/'output/runs'/args.run_id
    preservation = ROOT/'reviews/asein-integration'/('preserved-lite-'+PARENT+'-before-boundary-fallback')
    expected = base.server.SimulationManager.compatibility()
    state, manifest, header, cp_sha = base.stable_checkpoint(run/'checkpoint.npz', expected)
    receipt = old.read_json(run/'source-transition.json')
    parent_header = base.checkpoint_header(preservation/'run/checkpoint.npz')
    parent, parent_manifest = base.read_checkpoint(preservation/'run/checkpoint.npz', parent_header['compatibility'], base.native_engine.Simulation)
    boundary, boundary_manifest, boundary_header, boundary_sha = base.stable_checkpoint(run/'checkpoint.source-boundary.npz', expected)
    seal = old.read_json(preservation/'preservation.json')
    sealed = {row['path']:row['sha256'] for row in seal['files']}
    status = old.read_json(args.status) if args.status else __import__('json').load(
        urllib.request.urlopen('http://127.0.0.1:8769/api/status', timeout=10))
    report = dict(kind='lite_boundary_geometry_current_source_read_only_continuation_audit',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), run_id=args.run_id,
        read_only=True, constructor_called=False, simulation_stepped=False, server_mutated=False,
        checkpoint=dict(time_myr=float(state.t), sha256=cp_sha, next_output_myr=manifest.get('next_output_myr')),
        qualifications=['All predecessor arrays, clocks, RNG, history and accepted geometry cache remain exact at source import.',
            'The first successor force solve uses the predecessor cache; geometry repair applies during normal subsequent advection.',
            'Represented source power exchanges work with the declared omitted slab/mantle reservoir; no closed mantle budget is claimed.'])
    checks = {}
    def check(name, value): checks[name] = bool(value)
    def validate(name, fn):
        try: fn(); checks[name] = True
        except (ValueError, TypeError, KeyError, AssertionError) as error:
            checks[name] = False; report[name+'_error'] = str(error)
    closure = expected['auxiliary_sources_sha256']
    bad = [name for name,value in closure.items() if old.sha(ROOT/name) != value or old.sha(run/name) != value]
    check('accepted_progress_beyond_source_boundary', state.t >= args.minimum_myr > EPOCH)
    check('current_live_and_durable_identity', status.get('run_id') == manifest.get('run_id') == args.run_id
        and not status.get('error') and float(status.get('checkpoint_time_myr', -1)) >= state.t)
    check('exact_current_173_helper_backend_closure', len(closure) == 173 and not bad
        and header['compatibility'] == boundary_header['compatibility'] == receipt['child_compatibility'] == expected
        and old.sha(ROOT/'tectonics.py') == old.sha(run/'engine.py') == expected['engine_sha256'])
    check('sealed_current_runtime', old.read_json(run/'runtime-source-seal.json') == closure
        and old.sha(run/'runtime-source-seal.json') == receipt['runtime_source_seal_sha256'])
    prior = parent_header['compatibility']
    delta = {name:dict(parent_sha256=prior['auxiliary_sources_sha256'][name], child_sha256=closure[name])
        for name in closure if prior['auxiliary_sources_sha256'].get(name) != closure[name]}
    required = {transition.HELPER:dict(parent_sha256=transition.PARENT_HELPER_SHA256, child_sha256=transition.CHILD_HELPER_SHA256)}
    check('only_reviewed_geometry_helper_changed', set(closure) == set(prior['auxiliary_sources_sha256'])
        and delta == required == receipt['source_delta']['changed_helpers']
        and receipt['source_delta']['added_helpers'] == {} and receipt['source_delta']['removed_helpers'] == []
        and all(expected[key] == prior[key] for key in ('engine_sha256','numpy_version','flow_backend','material_backend')))
    comparisons = {name:old.file_differences(sealed, old.files(folder)) for name,folder in
        (('original',ROOT/'output/runs'/PARENT),('preserved',preservation/'run'),('child_copy',run/'inherited-parent'))}
    report['parent_preservation'] = comparisons
    check('all_684_predecessor_files_exact', len(sealed) == seal['file_count'] == receipt['parent_file_count'] == 684
        and receipt['parent_files_sha256'] == sealed and all(row['equal'] for row in comparisons.values())
        and old.sha(run/'parent-preservation.json') == old.sha(preservation/'preservation.json') == receipt['parent_preservation_receipt_sha256'])
    check('paused_parent54_and_exact_next56', parent.t == EPOCH and parent_manifest['state'] == 'paused'
        and not parent_manifest.get('error') and parent_manifest['next_output_myr'] == receipt['next_output_myr'] == 56.
        and old.sha(preservation/'run/checkpoint.npz') == receipt['parent_checkpoint_sha256'])
    parent_fingerprint = transition._fingerprint(vars(parent))
    boundary_fingerprint = transition._fingerprint(vars(boundary))
    check('entire_accepted_state_and_cache_exact_at_import', parent_fingerprint == boundary_fingerprint
        and parent_fingerprint['sha256'] == receipt['full_state_fingerprint']
        and receipt['entire_typed_state_exact'] and receipt['configuration_unchanged'] and receipt['rng_unchanged']
        and receipt['all_existing_policy_clocks_and_history_unchanged'] and receipt['cached_boundary_geometry_unchanged']
        and not receipt['geometry_rebuilt_at_import'] and not receipt['model_constructed'] and not receipt['model_stepped']
        and receipt['physical_time_advanced_myr'] == 0. and not receipt['physical_policies_added'])
    check('permanent_source_boundary_sealed', boundary_sha == receipt['permanent_source_boundary_sha256'] == receipt['child_checkpoint_sha256']
        and receipt['permanent_source_boundary_file'] == 'checkpoint.source-boundary.npz'
        and boundary.t == EPOCH and boundary_manifest['run_id'] == args.run_id and boundary_manifest['next_output_myr'] == 56.)
    check('configuration_remains_exact', parent.config == boundary.config == state.config)
    check('new_human_authority_and_reviews_exact', receipt['kind'] == transition.KIND
        and receipt['parent_run_id'] == PARENT and receipt['child_run_id'] == args.run_id and receipt['epoch_myr'] == EPOCH
        and old.sha(run/'application-authority.json') == receipt['authority_sha256']
        and old.sha(run/'repair-review.json') == receipt['repair_review_sha256']
        and old.sha(run/'source-delta-review.json') == receipt['source_delta_review_sha256'])
    evidence = receipt['validation_evidence']
    check('separate_retained_regression_interval_evidence_exact', len(evidence) >= 2 and all(
        row['storage'] == 'child_relative_exact_copy' and old.sha(run/row['child_file']) == row['sha256'] for row in evidence))
    count = receipt['inherited_frame_count']
    transition_meta = manifest['source_transition']
    lineage = manifest.get('frame_sources', [])
    exact = count == parent_manifest['frame_count'] == 29 and manifest['frames'][:count] == parent_manifest['frames']
    exact &= transition_meta['inherited_frame_count'] == count and transition_meta['kind'] == transition.KIND
    for index,row in enumerate(parent_manifest['frames']):
        expected_source = dict(index=row['index'],time_myr=row['time_myr'],run_id=PARENT,
            source_manifest='inherited-parent/manifest.json',source_manifest_sha256=receipt['parent_manifest_sha256'],
            engine_sha256=prior['engine_sha256'],auxiliary_sources_sha256=prior['auxiliary_sources_sha256'],
            files_sha256={suffix:sealed[f'frame_{index:04d}{suffix}'] for suffix in ('.npz','.json')},
            inherited=True,parent_frame_source=parent_manifest.get('frame_sources',[])[index]
                if index < len(parent_manifest.get('frame_sources',[])) else None)
        exact &= lineage[index] == expected_source
        for suffix in ('.npz','.json'):
            name=f'frame_{index:04d}{suffix}'
            exact &= old.sha(run/name) == sealed[name] == transition_meta['inherited_frame_files_sha256'][name]
    check('all_29_inherited_frames_and_recursive_provenance_exact', exact)
    future_valid = True
    for index,row in enumerate(manifest['frames'][count:],start=count):
        metadata=old.read_json(run/f'frame_{index:04d}.json')
        future_valid &= row['time_myr'] > EPOCH and row['index'] == metadata['index'] == index and metadata['time_myr'] == row['time_myr']
        try: force.validate_frame(metadata); carrier.validate_frame(metadata)
        except (ValueError,TypeError,KeyError): future_valid=False
    check('future_frame_indices_and_policy_metadata_valid', future_valid)
    scheduled=56.
    while scheduled <= state.t+1e-8 and scheduled < state.config['duration_myr']:
        scheduled=min(state.config['duration_myr'],scheduled+state.config['snapshot_myr'])
    check('unchanged_two_myr_output_scheduler', manifest['next_output_myr'] == scheduled)
    initial=base.initial_identity(run)
    report['initial_identity']=dict(all_equal=initial['all_equal'],array_count=initial['array_count'],files_sha256=initial['files_sha256'])
    check('all_24_original_initial_arrays_exact', initial['all_equal'] and initial['array_count'] == 24)
    check('force_coefficient_unchanged', state.config['effective_subduction']['force_n_per_m'] == state.effective_subduction_settings['force_n_per_m'] == 3e13)
    validate('force_breakup_policy_valid',lambda:force.bounded_policy(state))
    validate('carrier_policy_valid',lambda:carrier.validate(state))
    validate('initiation_policy_valid',lambda:initiation.enabled(state))
    check('established_activation_clocks_unchanged', state.force_rifting_policy['activation_myr'] == parent.force_rifting_policy['activation_myr'] == 43.
        and state.effective_subduction_carrier_traction['activation_myr'] == parent.effective_subduction_carrier_traction['activation_myr'] == 43.
        and state.effective_subduction_initiation['activation_myr'] == parent.effective_subduction_initiation['activation_myr'] == 25.
        and state.effective_subduction_carrier_traction['retrospective_work_j'] == 0.)
    balance,budget=state.plate_balance_diagnostics,state.material_column_budget
    power_relative=float(balance['power_balance_error_w'])/max(abs(float(balance['driver_work_w'])),1.)
    column_relative=float(budget['residual_km3'])/max(abs(float(budget['before_motion_volume_km3'])),abs(float(budget['after_columns_volume_km3'])),1.)
    created=(float(state.process_totals.get('arc_added_km2',0.))
             -float(state.process_totals.get('arc_deposited_source_area_km2',0.)))
    reference_relative=(float(state.mass.sum())-float(state.original_mass)-created)/max(float(state.original_mass)+created,1.)
    geometric=float(state.material_surface['area_km2'] @ state.structure['thickness_km'])
    reference_volume=float(state.mass @ (state.structure['area_factor']*state.structure['thickness_km']))
    source=balance.get('effective_subduction_carrier_power',{})
    slab=balance.get('slab_power_w',{})
    scale=max(abs(float(source.get('represented_source_power_w',0.))),abs(float(source.get('downgoing_power_w',0.))),abs(float(source.get('carrier_power_w',0.))),1.)
    finite=all(type(source.get(key)) in (int,float) and math.isfinite(source[key]) for key in
        ('downgoing_power_w','carrier_power_w','represented_source_power_w','reservoir_exchange_power_w','power_partition_residual_w'))
    partition=finite and abs(source['represented_source_power_w']-source['downgoing_power_w']-source['carrier_power_w']) <= 1e-10*scale
    partition &= finite and abs(source['reservoir_exchange_power_w']+source['represented_source_power_w']) <= 1e-10*scale
    partition &= finite and abs(source['power_partition_residual_w']-(source['represented_source_power_w']
        -source['downgoing_power_w']-source['carrier_power_w'])) <= 1e-10*scale
    partition &= all(abs(source.get(key,math.inf)-slab.get(other,math.inf)) <= 1e-10*scale for key,other in
        (('downgoing_power_w','downgoing'),('carrier_power_w','overriding'),('represented_source_power_w','total')))
    check('force_balance',balance['scaled_force_residual'] <= balance['force_relative_tolerance'])
    check('passive_resistance',balance['negative_resisting_work_elements'] == 0 and balance['plate_resistance_version'] == 1)
    check('represented_power_balance',abs(power_relative) <= 1e-8)
    check('carrier_source_partition',partition and source.get('alpha') == .1 and source.get('law') == carrier.POLICY
        and source.get('passive_operators_changed') is False and source.get('unrepresented_force_reaction_fraction') == .9)
    check('column_volume_relative_audit',abs(column_relative) <= 1e-11)
    check('reference_area_with_juvenile_sources',abs(reference_relative) < 1e-12)
    check('reference_area_alignment',base.np.array_equal(state.material_surface['reference_area_km2'],state.mass))
    check('geometric_reference_column_alignment',abs(geometric-reference_volume)/max(abs(geometric),abs(reference_volume),1.) < 1e-12)
    check('continental_retention',abs((float(state.mass.sum())-created)/float(state.original_mass)-1.) < 1e-12)
    check('craton_retention',abs(float(state.mass[state.kind==2].sum())/float(state.original_craton_mass)-1.) < 1e-12)
    check('same_sheet_nonpenetration',state.deformation_diagnostics['same_sheet_nonpenetration']['final_overlap_pairs'] == 0)
    validate('crust_inventory_valid',lambda:base.crust_inventory.validate(state.structure))
    validate('dense_phase_valid',lambda:base.dense_crust.validate(state.structure))
    report['physical_audits']=dict(plate_balance=balance,carrier_source_power=source,slab_power=slab,
        source_power_comparison_scale_w=scale,force_residual=balance['scaled_force_residual'],power_relative=power_relative,
        material_column_budget=budget,column_residual_km3=budget['residual_km3'],column_relative=column_relative,
        original_reference_area_km2=float(state.original_mass),current_reference_area_km2=float(state.mass.sum()),
        juvenile_created_reference_area_km2=created,reference_relative=reference_relative,
        original_craton_area_km2=float(state.original_craton_mass),current_craton_area_km2=float(state.mass[state.kind==2].sum()),
        geometric_current_volume_km3=geometric,reference_current_volume_km3=reference_volume,
        same_sheet_nonpenetration=state.deformation_diagnostics['same_sheet_nonpenetration'],
        current_geometry_diagnostics=state.native_boundary_geometry.get('diagnostics',{}))
    report['audit_utility_sha256']=old.sha(Path(__file__))
    check('source_proofs_unchanged_during_audit',all(old.sha(ROOT/name) == value for name,value in closure.items())
        and old.sha(run/'checkpoint.source-boundary.npz') == boundary_sha)
    report.update(checks=checks,failed_checks=[name for name,value in checks.items() if not value],
        state='passed' if all(checks.values()) else 'audit_discrepancy',production_guards_changed=False)
    output=ROOT/args.output
    base.write(output,report)
    print(__import__('json').dumps(dict(state=report['state'],run_id=args.run_id,checkpoint_myr=state.t,
        check_count=len(checks),failed_checks=report['failed_checks'],output=str(output),sha256=old.sha(output))))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--minimum-myr',type=float,default=55.)
    parser.add_argument('--status',type=Path)
    parser.add_argument('--output',default='reviews/asein-integration/lite-boundary-fallback-production-verification.json')
    raise SystemExit(audit(parser.parse_args()))
