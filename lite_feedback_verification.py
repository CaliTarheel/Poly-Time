"""Read-only current-source feedback-child audit; no constructor or model step.

The permanent boundary proves zero credit on activation. Later checkpoints are
checked against that sealed proof, without redoing the full old physical-state
fingerprint or pretending an old checkpoint has current compatibility.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.request

DEFAULT_ROOT = Path(__file__).resolve().parent
PARENT = '20261002-140444-3e196a'
TRANSITION_KIND = 'lite_mechanical_feedback_source_transition'
ACTIVATION_MYR = 43.
INITIATION_MYR = 25.
EXPECTED_PARENT_FILES = 436
EXPECTED_HELPERS = 173


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-id', required=True)
    result.add_argument('--parent-run-id', default=PARENT)
    result.add_argument('--minimum-myr', type=float, default=44.)
    result.add_argument('--preservation', type=Path, required=True,
        help='Original failed-parent preservation directory containing preservation.json and run/.')
    result.add_argument('--workspace', '--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--port', type=int, choices=(8769,), default=8769)
    result.add_argument('--output', type=Path,
        default=Path('reviews/asein-integration/lite-feedback-production-continuation-verification.json'))
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while data := source.read(1024*1024):
            digest.update(data)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def files(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():
            raise ValueError('A sealed history cannot contain a symlink.')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha(path)
    return result


def file_differences(expected, actual):
    differences = sorted(name for name in expected.keys() | actual.keys()
        if expected.get(name) != actual.get(name))
    return dict(equal=not differences, mismatch_count=len(differences),
                first_mismatches=differences[:20], actual_file_count=len(actual))


def source_depth(row):
    depth = 0
    while isinstance(row, dict):
        depth += 1
        row = row.get('parent_frame_source')
    return depth


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def normalize_controls(raw, force, carrier):
    """Restore each API's typed parameter ordering after sorted JSON evidence.

    Later audit comparisons are semantic dict equality. The returned ordering
    also remains suitable for a producer's strict typed zero-policy fingerprint;
    the continuation audit itself never reruns that complete activation proof.
    """
    if type(raw) is not dict or set(raw) != {'force_limit_rifting', 'carrier_traction'}:
        raise ValueError('Feedback evidence must name exactly the two selected policies.')
    return dict(force_limit_rifting=force.normalize(raw['force_limit_rifting']),
                carrier_traction=carrier.normalize(raw['carrier_traction']))


def review_checks(root, run, receipt, source_review, numerical, controls):
    """Bind actual old/new helper hashes and each retained validation receipt."""
    delta = receipt.get('source_delta', {})
    repair = delta.get('changed_helpers', {}).get('bounded_gravity.py', {})
    prior, current = receipt['parent_compatibility'], receipt['child_compatibility']
    original_review = root/'reviews/asein-integration/lite-feedback-numerical-contact-repair-review.json'
    # A later recovery may replace the canonical root review. Only the exact
    # review sealed into this child can authorize that root-relative anchor;
    # otherwise use the child's self-contained retained interval evidence.
    anchor = (original_review.parent if original_review.is_file()
        and sha(original_review) == receipt['numerical_repair_review_sha256'] else run)
    evidence = numerical.get('validation_receipts', [])
    evidence_exact = isinstance(evidence, list) and len(evidence) >= 2
    distinct_paths = set()
    for row in evidence if isinstance(evidence, list) else ():
        if not isinstance(row, dict) or set(row) != {'path', 'sha256'}:
            evidence_exact = False
            continue
        path = Path(row['path'])
        path = (path if path.is_absolute() else anchor/path).resolve()
        evidence_exact &= path not in distinct_paths
        distinct_paths.add(path)
        evidence_exact &= path.is_file() and sha(path) == row['sha256']
    return dict(source_delta_and_numerical_review_sealed=
        sha(run/'source-delta-review.json') == receipt['source_delta_review_sha256']
        and source_review.get('kind') == 'lite_feedback_runtime_source_delta_review'
        and source_review.get('version') == 1 and source_review.get('ready') is True
        and source_review.get('policy') == controls
        and source_review.get('parent_run_id') == receipt['parent_run_id']
        and source_review.get('parent_checkpoint_sha256') == receipt['parent_checkpoint_sha256']
        and source_review.get('boundary_kind') == 'durable_stopped'
        and source_review.get('parent_compatibility_sha256') == canonical_sha(prior)
        and source_review.get('child_compatibility_sha256') == canonical_sha(current)
        and source_review.get('changed_helpers') == delta.get('changed_helpers')
        and source_review.get('added_helpers') == delta.get('added_helpers')
        and source_review.get('removed_helpers') == delta.get('removed_helpers') == []
        and source_review.get('numerical_repair_review_sha256') == receipt['numerical_repair_review_sha256']
        and sha(run/'numerical-repair-review.json') == receipt['numerical_repair_review_sha256'],
        exact_reviewed_numerical_parent_and_child_helper=
        numerical.get('kind') == 'lite_feedback_numerical_contact_repair_review' and numerical.get('version') == 1
        and numerical.get('ready') is True and numerical.get('parent_run_id') == receipt['parent_run_id']
        and numerical.get('parent_checkpoint_sha256') == receipt['parent_checkpoint_sha256']
        and numerical.get('helper') == 'bounded_gravity.py'
        and numerical.get('parent_sha256') == repair.get('parent_sha256') == prior['auxiliary_sources_sha256']['bounded_gravity.py']
        and numerical.get('child_sha256') == repair.get('child_sha256') == current['auxiliary_sources_sha256']['bounded_gravity.py']
        and numerical.get('physical_guards_unchanged') is True
        and numerical.get('original_policy_interval_passed') is True and numerical.get('feedback_policy_interval_passed') is True,
        retained_original_and_feedback_validation_receipts_exact=evidence_exact)


def history_checks(run, parent, manifest, receipt, sealed_files, *, frame_validators=(), controls=None):
    """Check exact inherited files and recursively retained source metadata."""
    count = receipt['inherited_frame_count']
    parent_frames = parent['frames']
    inherited_files = {name: value for name, value in sealed_files.items()
        if re.fullmatch(r'frame_\d+\.(npz|json)', name)}
    transition = manifest.get('source_transition', {})
    sources = manifest.get('frame_sources', [])
    checks = dict(inherited_frame_count=type(count) is int and count == parent['frame_count'] == len(parent_frames),
        inherited_frame_records_exact=manifest.get('frames', [])[:count] == parent_frames,
        inherited_source_frame_map=transition.get('inherited_frame_files_sha256') == inherited_files,
        source_transition_metadata=transition.get('kind') == TRANSITION_KIND
            and transition.get('version') == 1 and transition.get('epoch_myr') == ACTIVATION_MYR
            and transition.get('parent_run_id') == receipt['parent_run_id']
            and transition.get('inherited_frame_count') == count
            and transition.get('inherited_source_compatibility') == receipt['parent_compatibility'],
        inherited_frame_bytes=True, recursive_frame_provenance=True,
        all_frame_indices_and_times=True, future_frames_after_policy_boundary=True,
        future_frame_policy_receipts_valid=True)
    maximum_depth = 0
    for index, row in enumerate(parent_frames):
        expected_source = dict(index=row['index'], time_myr=row['time_myr'], run_id=receipt['parent_run_id'],
            source_manifest='inherited-parent/manifest.json', source_manifest_sha256=receipt['parent_manifest_sha256'],
            engine_sha256=receipt['parent_compatibility']['engine_sha256'],
            auxiliary_sources_sha256=receipt['parent_compatibility']['auxiliary_sources_sha256'],
            files_sha256={suffix: sealed_files[f'frame_{index:04d}{suffix}'] for suffix in ('.npz', '.json')},
            inherited=True, parent_frame_source=parent.get('frame_sources', [])[index]
                if len(parent.get('frame_sources', [])) > index else None)
        checks['recursive_frame_provenance'] &= len(sources) > index and sources[index] == expected_source
        maximum_depth = max(maximum_depth, source_depth(expected_source))
        for suffix in ('.npz', '.json'):
            name = f'frame_{index:04d}{suffix}'
            checks['inherited_frame_bytes'] &= sha(run/name) == sealed_files[name]
    prior = -math.inf
    for index, row in enumerate(manifest.get('frames', [])):
        metadata = read_json(run/f'frame_{index:04d}.json')
        epoch = row.get('time_myr', -math.inf)
        checks['all_frame_indices_and_times'] &= (row.get('index') == metadata.get('index') == index
            and epoch == metadata.get('time_myr') and math.isfinite(epoch) and prior <= epoch)
        prior = epoch
        if index >= count:
            checks['future_frames_after_policy_boundary'] &= epoch > ACTIVATION_MYR
            try:
                for operation in frame_validators:
                    operation(metadata)
                if controls is not None:
                    checks['future_frame_policy_receipts_valid'] &= (
                        metadata['force_rifting_policy']['parameters'] == controls['force_limit_rifting']
                        and metadata['force_rifting_policy']['activation_myr'] == ACTIVATION_MYR
                        and metadata['effective_subduction_carrier_traction_parameters'] == controls['carrier_traction']
                        and metadata['effective_subduction_carrier_traction_diagnostics']['activation_myr'] == ACTIVATION_MYR)
            except (ValueError, TypeError, KeyError):
                checks['future_frame_policy_receipts_valid'] = False
    return checks, dict(inherited_frame_count=count, inherited_frame_file_count=len(inherited_files),
        maximum_recursive_source_depth=maximum_depth,
        inherited_sources_preserved_exactly=checks['recursive_frame_provenance'],
        future_frame_source='Current sealed run compatibility; inherited prefix retains exact recursive predecessor provenance.')


def scheduler_checks(state, manifest, receipt):
    pending = manifest.get('next_output_myr')
    duration, cadence = state.config['duration_myr'], state.config['snapshot_myr']
    # Match the native server's snapshot threshold; this is scheduler
    # bookkeeping, not a numerical or physical acceptance tolerance.
    epsilon = 1e-8
    valid_pending = (type(pending) in (int, float) and math.isfinite(pending)
        and (state.t < pending <= min(duration, state.t+cadence)+epsilon if state.t < duration else pending == duration))
    scheduled = float(receipt['next_output_myr'])
    while scheduled <= float(state.t)+1e-8 and scheduled < duration:
        scheduled = min(duration, scheduled+cadence)
    return dict(valid_durable_output_scheduler=valid_pending,
        inherited_scheduler_uses_embedded_44myr=receipt['next_output_myr'] == 44.
            and receipt.get('durable_recovery', {}).get('embedded_next_output_myr') == 44.
            and receipt.get('durable_recovery', {}).get('external_stale_next_output_myr') == 26.
            and receipt.get('durable_recovery', {}).get('historical_manifests_rewritten') is False,
        original_repeated_addition_scheduler=pending == scheduled)


def load_runtime(root):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    base = importlib.import_module('lite_checkpoint_verification')
    if base.ROOT.resolve() != root:
        raise ValueError('Read-only utility module belongs to a different selected runtime workspace.')
    modules = {name: importlib.import_module(name) for name in
        ('force_rifting', 'effective_subduction', 'effective_subduction_carrier_traction',
         'effective_subduction_initiation', 'lite_feedback_transition')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Production continuation verification forbids constructors and model steps.')
    for name in ('__init__', 'step', '_step_once', '_advance_step'):
        setattr(base.native_engine.Simulation, name, forbidden)
    return base, modules


def audit(args):
    root = args.workspace.resolve()
    if (not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run_id)
            or args.parent_run_id != PARENT or args.run_id == args.parent_run_id):
        raise ValueError('The audited child and explicitly selected original failed-parent identity are required.')
    if not math.isfinite(args.minimum_myr) or args.minimum_myr < 44.:
        raise ValueError('This production recovery proof must require accepted progress through at least 44 Myr.')
    preservation = args.preservation.resolve()
    base, modules = load_runtime(root)
    force = modules['force_rifting']; carrier = modules['effective_subduction_carrier_traction']
    initiation = modules['effective_subduction_initiation']
    run = root/'output/runs'/args.run_id
    original_run = root/'output/runs'/args.parent_run_id
    expected = base.server.SimulationManager.compatibility()
    closure = expected['auxiliary_sources_sha256']
    state, manifest, header, checkpoint_sha = base.stable_checkpoint(run/'checkpoint.npz', expected)
    receipt = read_json(run/'source-transition.json')
    receipt_sha = sha(run/'source-transition.json')
    seal = read_json(preservation/'preservation.json')
    sealed_files = {row['path']: row['sha256'] for row in seal['files']}
    parent_header = base.checkpoint_header(preservation/'run/checkpoint.npz')
    parent_manifest = parent_header['manifest']
    # Historical state is decoded with its OWN exact declaration, never the
    # current closure. Only the inherited initiation metadata is compared;
    # complete physical-state exactness remains the immutable producer proof.
    parent_state, _ = base.read_checkpoint(preservation/'run/checkpoint.npz',
        parent_header['compatibility'], base.native_engine.Simulation)
    external_parent = read_json(preservation/'run/manifest.json')
    boundary_path = run/receipt['permanent_policy_boundary_file']
    if receipt['permanent_policy_boundary_file'] != 'checkpoint.policy-boundary.npz':
        raise ValueError('The declared permanent policy boundary must be the producer\'s fixed child checkpoint file.')
    boundary, boundary_manifest, boundary_header, boundary_sha = base.stable_checkpoint(boundary_path, expected)
    controls = normalize_controls(dict(force_limit_rifting=dict(enabled=True, inherited_weakness=True,
        search_policy=dict(version=1, mode='bounded_axes', max_axes=10)),
        carrier_traction=dict(version=1, enabled=True, alpha=.1)), force, carrier)
    receipt_controls = normalize_controls(receipt['activated_controls'], force, carrier)
    initiation_controls = initiation.normalize(dict(enabled=True))
    source_seal = read_json(run/'runtime-source-seal.json')
    source_review = read_json(run/'source-delta-review.json')
    numerical = read_json(run/'numerical-repair-review.json')
    status = json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/status', timeout=10))
    report = dict(kind='Asein_Lite_feedback_current_source_constructorless_continuation_audit',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), run_id=args.run_id,
        read_only=True, constructor_called=False, simulation_stepped=False, server_mutated=False,
        state='measured_values_before_audit_checks', selected_controls=controls,
        checkpoint=dict(sha256=checkpoint_sha, time_myr=state.t, steps=state.steps,
            embedded_run_id=manifest.get('run_id'), next_output_myr=manifest.get('next_output_myr')),
        api={key: status.get(key) for key in ('run_id', 'state', 'error', 'time_myr', 'integration_time_myr', 'checkpoint_time_myr')},
        qualifications=[
            'Exact current compatibility is required for child and permanent boundary. Preserved predecessors retain their own declared closure.',
            'Whole inherited physical-state exactness is the sealed producer proof; this later audit does not redo that complete fingerprint.',
            'Force-rift development/work is an independent virtual candidate path, not realized extension, material heating or simultaneous world energy.',
            'Prescribed carrier source exchanges power and unrepresented reaction with Lite\'s omitted slab/mantle reservoir; no closed mantle energy budget is claimed.',
            'Column and reference inventory tolerances are extra read-only audits, not modified production guards.'])
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    def validate(name, operation):
        try:
            operation(); checks[name] = True
        except (ValueError, TypeError, KeyError, AssertionError) as error:
            checks[name] = False; report[name+'_error'] = str(error)
    check('accepted_progress_beyond_failed_interval', state.t >= args.minimum_myr)
    check('embedded_and_live_child_identity', manifest.get('run_id') == args.run_id == status.get('run_id'))
    check('api_no_error_and_covers_durable_checkpoint', not status.get('error')
        and float(status.get('checkpoint_time_myr', -1)) >= state.t)
    source_bad = [name for name, value in closure.items() if sha(root/name) != value or sha(run/name) != value]
    check('exact_current_173_helper_and_backend_closure', len(closure) == EXPECTED_HELPERS
        and expected == header['compatibility'] == boundary_header['compatibility'] == receipt['child_compatibility']
        and not source_bad and sha(run/'engine.py') == expected['engine_sha256'] == sha(root/'tectonics.py'))
    check('sealed_runtime_source_capture', source_seal == closure
        and sha(run/'runtime-source-seal.json') == receipt['runtime_source_seal_sha256'])
    report['source_closure'] = dict(helper_count=len(closure), helper_mismatches=source_bad,
        current_matches_checkpoint=header['compatibility'] == expected,
        saved_and_runtime_helpers_exact=not source_bad,
        engine_sha256=expected['engine_sha256'], flow_backend=expected['flow_backend'], material_backend=expected['material_backend'])
    preservation_comparisons = {name: file_differences(sealed_files, files(folder)) for name, folder in
        (('preserved_run', preservation/'run'), ('original_failed_run', original_run), ('child_inherited_parent', run/'inherited-parent'))}
    check('preserved_failed_parent_436_files_bit_exact', seal.get('run_id') == args.parent_run_id
        and len(sealed_files) == seal.get('file_count') == EXPECTED_PARENT_FILES
        and all(row['equal'] for row in preservation_comparisons.values())
        and sealed_files.get('checkpoint.npz') == seal['checkpoint_sha256'] == receipt['parent_checkpoint_sha256'])
    check('parent_preservation_receipt_sealed', sha(preservation/'preservation.json') ==
        sha(run/'parent-preservation.json') == receipt['parent_preservation_receipt_sha256'])
    check('receipt_names_exact_parent_file_map_and_closure', receipt.get('parent_files_sha256') == sealed_files
        and receipt.get('parent_file_count') == EXPECTED_PARENT_FILES
        and receipt.get('parent_compatibility') == parent_header['compatibility']
        and receipt.get('parent_manifest_sha256') == sealed_files.get('manifest.json'))
    report['parent_preservation'] = dict(run_id=args.parent_run_id, file_count=len(sealed_files),
        checkpoint_sha256=seal['checkpoint_sha256'], comparisons=preservation_comparisons)
    check('truthful_failed_external_and_valid_durable_parent', parent_header['time_myr'] == ACTIVATION_MYR
        and parent_manifest.get('state') == 'interrupted' and not parent_manifest.get('error')
        and parent_manifest.get('checkpoint_time_myr') == parent_manifest.get('integration_time_myr') == ACTIVATION_MYR
        and external_parent.get('state') == 'error' and bool(external_parent.get('error'))
        and external_parent.get('time_myr') == 42. and external_parent.get('integration_time_myr') ==
            external_parent.get('checkpoint_time_myr') == ACTIVATION_MYR
        and sha(run/'stop-evidence.json') == receipt['stop_evidence_sha256'])
    proof = receipt.get('preservation', {})
    check('sealed_producer_inherited_state_exact_proof', receipt.get('kind') == TRANSITION_KIND
        and receipt.get('version') == 1 and receipt.get('boundary_kind') == 'durable_stopped'
        and receipt.get('parent_run_id') == args.parent_run_id and receipt.get('child_run_id') == args.run_id
        and receipt.get('epoch_myr') == ACTIVATION_MYR and receipt.get('activated_controls') == receipt_controls == controls
        and receipt.get('child_decoded_state_exact') is True and receipt.get('inherited_parent_copy_exact') is True
        and all(proof.get(key) is True for key in ('inherited_state_exact', 'rng_unchanged', 'clocks_unchanged', 'initiation_history_unchanged'))
        and proof.get('physical_time_advanced_myr') == 0. and proof.get('force_solve_performed') is False
        and proof.get('past_fracture_loading_reconstructed') is False and proof.get('past_carrier_work_reconstructed') is False
        and isinstance(proof.get('inherited_state_fingerprint'), str) and re.fullmatch(r'[0-9a-f]{64}', proof['inherited_state_fingerprint'])
        and receipt.get('model_constructed') is False and receipt.get('model_stepped') is False
        and receipt.get('physical_time_advanced_myr') == 0. and receipt.get('published_state') == 'paused')
    checks.update(review_checks(root, run, receipt, source_review, numerical, controls))
    check('permanent_policy_boundary_sealed', boundary_sha == receipt['permanent_policy_boundary_sha256'] == receipt['child_checkpoint_sha256']
        and boundary.t == ACTIVATION_MYR and boundary_manifest.get('run_id') == args.run_id
        and boundary_manifest.get('next_output_myr') == 44. and boundary.config == state.config)
    preserved_config = deepcopy(parent_state.config)
    preserved_config['force_limit_rifting'] = deepcopy(controls['force_limit_rifting'])
    preserved_config['effective_subduction']['carrier_traction'] = deepcopy(controls['carrier_traction'])
    check('only_two_declared_config_policies_changed', boundary.config == preserved_config == state.config)
    validate('permanent_force_policy_valid', lambda: force.bounded_policy(boundary))
    validate('permanent_carrier_policy_valid', lambda: carrier.validate(boundary))
    check('activation_receipt_matches_permanent_metadata', receipt.get('activation') == dict(
        force_limit_rifting=boundary.force_rifting_policy,
        carrier_traction=boundary.effective_subduction_carrier_traction))
    check('initiation_metadata_exact_at_policy_boundary', boundary.effective_subduction_initiation ==
        parent_state.effective_subduction_initiation)
    check('zero_initial_candidate_work_and_clocks', boundary.force_rifting_state == {}
        and boundary.force_rifting_diagnostics == dict(version=1, checks=0, commits=0, refusals=[])
        and boundary.force_rifting_policy['activation_myr'] == boundary.force_rifting_policy['epoch_myr'] == ACTIVATION_MYR
        and boundary.effective_subduction_carrier_traction['retrospective_work_j'] == 0.
        and boundary.effective_subduction_carrier_traction['activation_myr'] == ACTIVATION_MYR)
    report['permanent_policy_boundary'] = dict(sha256=boundary_sha, time_myr=boundary.t,
        source_transition_receipt_sha256=receipt_sha, inherited_state_exact_proof=proof,
        initial_candidate_state=boundary.force_rifting_state, initial_candidate_diagnostics=boundary.force_rifting_diagnostics,
        scope='Separate immutable activation read and sealed producer proof; no later historical full-state fingerprint recomputation.')
    inherited_checks, lineage = history_checks(run, parent_manifest, manifest, receipt, sealed_files,
        frame_validators=(force.validate_frame, carrier.validate_frame), controls=controls)
    checks.update(inherited_checks); report['inherited_history'] = lineage
    checks.update(scheduler_checks(state, manifest, receipt))
    initial = base.initial_identity(run); report['initial_identity'] = initial
    check('original_24_initial_arrays', initial['all_equal'] and initial['array_count'] == 24)
    check('force_coefficient_and_selected_policies_exact', state.config['effective_subduction']['force_n_per_m'] ==
        state.effective_subduction_settings['force_n_per_m'] == 3e13
        and state.config['force_limit_rifting'] == controls['force_limit_rifting']
        and state.config['effective_subduction']['carrier_traction'] == controls['carrier_traction']
        and state.force_rifting_policy['parameters'] == controls['force_limit_rifting']
        and state.effective_subduction_carrier_traction['parameters'] == controls['carrier_traction'])
    validate('force_policy_valid_and_future_only', lambda: force.bounded_policy(state))
    validate('carrier_policy_valid_and_future_only', lambda: carrier.validate(state))
    validate('bounded_candidate_frame_receipts_valid', lambda: force.validate_frame(dict(time_myr=float(state.t), **force.snapshot(state))))
    check('feedback_activation_epoch_and_no_retrospective_carrier_work', state.force_rifting_policy['activation_myr'] == ACTIVATION_MYR
        and state.force_rifting_policy['epoch_myr'] == state.t
        and state.effective_subduction_carrier_traction['activation_myr'] == ACTIVATION_MYR
        and state.effective_subduction_carrier_traction['retrospective_work_j'] == 0.)
    validate('initiation_policy_valid', lambda: initiation.enabled(state))
    policy = state.effective_subduction_initiation
    epsilon = 128*base.np.finfo(float).eps*max(float(state.t), 1.)
    traces = {row['id']: row for row in state.trench_systems}
    future = all(row['created_myr'] >= INITIATION_MYR and row['last_seen_myr'] <= state.t
        and row['consecutive_myr'] <= row['last_seen_myr']-row['created_myr']+epsilon for row in policy['candidates'])
    birth_gates = all(row['time_myr'] >= INITIATION_MYR
        and row['consecutive_myr'] >= initiation_controls['minimum_consecutive_myr']
        and row['shortening_km'] >= initiation_controls['minimum_shortening_km']
        and row['shear_slip_km'] >= initiation_controls['minimum_shear_slip_km']
        and row['force_n_per_m'] == 3e13 and row['trench_id'] in traces for row in policy['births'])
    for event in policy['events']:
        if event['kind'] == 'source_activated':
            law = traces[event['trench_id']]['effective_subduction']
            birth_gates &= event['time_myr'] >= law['eligible_myr'] >= INITIATION_MYR and event['force_n_per_m'] == 3e13
    check('initiation_25myr_law_and_future_gates_preserved', policy['activation_myr'] == INITIATION_MYR
        and policy['epoch_myr'] == state.t and policy['parameters'] == initiation_controls
        and state.config['effective_subduction']['initiation'] == initiation_controls and future and birth_gates
        and all(policy[key] == 0. for key in ('retrospective_time_myr', 'retrospective_shortening_km', 'retrospective_shear_slip_km')))
    report['policy_history'] = dict(force_policy=force.snapshot(state), carrier_policy=state.effective_subduction_carrier_traction,
        initiation_activation_myr=policy['activation_myr'], initiation_epoch_myr=policy['epoch_myr'],
        initiation_candidate_count=len(policy['candidates']), initiation_birth_count=len(policy['births']),
        initiation_future_clocks=future, initiation_birth_and_source_gates=birth_gates)
    balance, budget = state.plate_balance_diagnostics, state.material_column_budget
    power_relative = float(balance['power_balance_error_w'])/max(abs(float(balance['driver_work_w'])), 1.)
    column_scale = max(abs(float(budget['before_motion_volume_km3'])), abs(float(budget['after_columns_volume_km3'])), 1.)
    column_relative = float(budget['residual_km3'])/column_scale
    created = (float(state.process_totals.get('arc_added_km2', 0.))
               - float(state.process_totals.get('arc_deposited_source_area_km2', 0.)))
    reference_expected = float(state.original_mass)+created
    reference_relative = (float(state.mass.sum())-reference_expected)/max(reference_expected, 1.)
    geometric = float(state.material_surface['area_km2'] @ state.structure['thickness_km'])
    reference_volume = float(state.mass @ (state.structure['area_factor']*state.structure['thickness_km']))
    source = balance.get('effective_subduction_carrier_power', {})
    slab = balance.get('slab_power_w', {})
    source_scale = max(abs(float(source.get('represented_source_power_w', 0.))),
        abs(float(source.get('downgoing_power_w', 0.))), abs(float(source.get('carrier_power_w', 0.))), 1.)
    source_finite = all(type(source.get(name)) in (float, int) and math.isfinite(source[name]) for name in
        ('downgoing_power_w', 'carrier_power_w', 'represented_source_power_w', 'reservoir_exchange_power_w', 'power_partition_residual_w'))
    partition = (source_finite and abs(source['represented_source_power_w']-source['downgoing_power_w']-source['carrier_power_w']) <= 1e-10*source_scale
        and abs(source['reservoir_exchange_power_w']+source['represented_source_power_w']) <= 1e-10*source_scale
        and all(abs(source[key]-slab.get(other, math.inf)) <= 1e-10*source_scale for key, other in
            (('downgoing_power_w', 'downgoing'), ('carrier_power_w', 'overriding'), ('represented_source_power_w', 'total'))))
    check('force_balance', balance['scaled_force_residual'] <= balance['force_relative_tolerance'])
    check('passive_resistance', balance['negative_resisting_work_elements'] == 0 and balance.get('plate_resistance_version') == 1)
    check('represented_power_balance', abs(power_relative) <= 1e-8)
    check('carrier_source_work_partition', partition and source.get('version') == 1 and source.get('alpha') == .1
        and source.get('law') == carrier.POLICY and source.get('passive_operators_changed') is False
        and source.get('accumulated_work_reported') is False and source.get('unrepresented_force_reaction_fraction') == .9)
    check('column_volume_relative_audit', abs(column_relative) <= 1e-11)
    check('reference_area_including_juvenile_arc_sources', abs(reference_relative) < 1e-12)
    check('reference_area_alignment', base.np.array_equal(state.material_surface['reference_area_km2'], state.mass))
    check('geometric_reference_column_volume_alignment', abs(geometric-reference_volume)/max(abs(geometric), abs(reference_volume), 1.) < 1e-12)
    check('continental_material_retention', abs((float(state.mass.sum())-created)/float(state.original_mass)-1.) < 1e-12)
    check('intact_craton_inventory', abs(float(state.mass[state.kind == 2].sum())/float(state.original_craton_mass)-1.) < 1e-12)
    check('same_sheet_nonpenetration', state.deformation_diagnostics['same_sheet_nonpenetration']['final_overlap_pairs'] == 0)
    validate('composition_inventory_valid', lambda: base.crust_inventory.validate(state.structure))
    validate('dense_phase_inventory_valid', lambda: base.dense_crust.validate(state.structure))
    report['physical_audits'] = dict(plate_balance=balance, material_column_budget=budget,
        power_relative_residual=power_relative, column_relative_residual=column_relative,
        reference_material_relative_residual=reference_relative, juvenile_created_reference_area_km2=created,
        geometric_column_volume_km3=geometric, reference_column_volume_km3=reference_volume,
        geometric_reference_volume_residual_km3=geometric-reference_volume,
        carrier_instantaneous_source_power=source, same_sheet_nonpenetration=state.deformation_diagnostics['same_sheet_nonpenetration'],
        timestep_diagnostics=getattr(state, 'timestep_diagnostics', {}), speeds=base.speeds(state))
    check('immutable_evidence_unchanged_during_audit', sha(boundary_path) == boundary_sha
        and sha(run/'source-transition.json') == receipt_sha and all(sha(root/name) == value for name, value in closure.items()))
    report.update(checks=checks, failed_checks=[name for name, passed in checks.items() if not passed],
        state='audit_passed' if all(checks.values()) else 'audit_discrepancy',
        audit_failure_is_production_guard_failure=False)
    return report, base


def main():
    args = parser().parse_args()
    output = args.output if args.output.is_absolute() else args.workspace.resolve()/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, base = audit(args)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError, AttributeError, AssertionError) as error:
        raced = isinstance(error, RuntimeError) and 'Checkpoint changed' in str(error)
        failure = dict(kind='Asein_Lite_feedback_current_source_constructorless_continuation_audit',
            recorded_at_utc=datetime.now(timezone.utc).isoformat(), run_id=args.run_id,
            state='retry_later' if raced else 'audit_read_or_schema_error', error_type=type(error).__name__,
            error=str(error), read_only=True, constructor_called=False, simulation_stepped=False,
            server_mutated=False, audit_failure_is_production_guard_failure=False)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(failure, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output)
        print(json.dumps(dict(state=failure['state'], run_id=args.run_id, error=str(error), output=str(output))), flush=True)
        raise SystemExit(2 if raced else 1)
    base.write(output, report)
    print(json.dumps(dict(state=report['state'], run_id=args.run_id,
        checkpoint_myr=report['checkpoint']['time_myr'], checkpoint_sha256=report['checkpoint']['sha256'],
        failed_checks=report['failed_checks'], output=str(output), output_sha256=sha(output))), flush=True)
    if report['failed_checks']:
        raise ValueError('Saved continuation audit discrepancies; see the measured receipt.')


if __name__ == '__main__':
    main()
