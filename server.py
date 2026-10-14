"""Deep Time: loopback-only UI, background simulation, and portable scientific exports."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import budget
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import uuid
import zipfile

from parallel_runtime import RuntimePool, configure_worker_environment
from worker_commands import WorkerCommands
from run_profiler import RunProfiler
configure_worker_environment()

import numpy as np
from PIL import Image

from tectonics import DEFAULT_CONFIG, Simulation, make_initial, validate_config
import fresh_world  # server-only: outside the hashed engine helper closure
from history import TRACE_DTYPES, history_schema, read_history, read_record
import mesh_display
import mesh_history
from terrain_jobs import TerrainManager, read_saved_auxiliary_sources
from gospl_jobs import GosplManager
from gospl_results import inspect_result
from local_dialogs import choose_gospl_result
from orientation import (normalize_orientation, rotation_matrix, orient_frame,
                         orient_metadata, orient_initial, inverse_cells)
from progress_timing import ProgressTiming
from checkpoint import (VERSION as CHECKPOINT_VERSION, checkpoint_header,
                        read_checkpoint, write_checkpoint, verify_compatibility,
                        previous_checkpoint)

ROOT = Path(__file__).resolve().parent
ENGINE_SOURCE = (ROOT / 'tectonics.py').read_bytes()
ORIENTATION_SOURCE = (ROOT / 'orientation.py').read_bytes()


def capture_auxiliary_sources(engine_source, root=ROOT):
    """Freeze the local Python import closure with the engine at server import.

    Only modules present beside the engine are included; NumPy and standard
    library imports remain external dependencies.  Unimported workspace helpers
    are never attached to a run merely because they happen to exist on disk.
    """
    root = Path(root)
    captured = {}
    pending = [engine_source]
    visited = {'tectonics'}
    while pending:
        source = pending.pop()
        for node in ast.walk(ast.parse(source)):
            modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                       else [])
            for module in modules:
                module = module.split('.')[0]
                if module in visited:
                    continue
                visited.add(module)
                path = root / (module + '.py')
                if path.is_file():
                    payload = path.read_bytes()
                    captured[path.name] = payload
                    pending.append(payload)
    return captured


AUXILIARY_SOURCES = capture_auxiliary_sources(ENGINE_SOURCE)
RUNS = ROOT / "output" / "runs"
FIELDS = ("elevation", "plate", "crust", "age", "boundary")
OPTIONAL_GRID_DTYPES = {"domain": np.int32, "trench": np.int32,
    **{name: np.float32 for name in ('crustal_thickness_km', 'crustal_root_km',
        'rift_thermal_support_m', 'rift_cooling_age_myr', 'foreland_deflection_m',
        'erosion_rate_m_myr', 'rift_damage', 'rift_strength_relative',
        'deformation_weight', 'geometric_strain_percent', 'refinement_level',
        'lip_deposited_km', 'lip_thermal_support_m')}}
DTYPES = {"elevation": np.float32, "plate": np.int32, "crust": np.uint8,
          "age": np.float32, "boundary": np.uint8}
BOUNDARIES = {0: "none", 1: "spreading ridge", 2: "subduction", 3: "transform",
              4: "continental collision", 5: "continental rift", 6: "converging, no trench"}


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def json_bytes(value):
    return json.dumps(native(value), allow_nan=False, separators=(",", ":")).encode("utf-8")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    # Windows readers can briefly hold the destination without delete sharing.
    # Keep the same fully written temporary file and retry only that atomic
    # replacement; never truncate the existing manifest or hide a lasting error.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.02 * 2**attempt, .2))


def validate_initial(initial, config):
    if initial is None:
        return None
    if not isinstance(initial, dict):
        raise ValueError("Initial condition must be an object with width, height and crust.")
    w, h = initial.get("width"), initial.get("height")
    if w != config["width"] or h != config["height"]:
        raise ValueError("Starting map dimensions must match the selected resolution. Regenerate or import a matching map.")
    values = np.asarray(initial.get("crust", []))
    if values.size != w * h or not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Starting crust must contain exactly width × height values: 0 ocean, 1 continent, 2 craton.")
    result = {"width": w, "height": h, "crust": values.astype(np.uint8).reshape(-1).tolist()}
    if 'world_design' in initial:
        import world_design
        result['world_design'] = world_design.normalize(initial['world_design'])
    if 'initial_plate_topology' in initial:
        import initial_plate_topology
        result['initial_plate_topology'] = initial_plate_topology.normalize(initial['initial_plate_topology'])
    if 'initial_subduction' in initial:
        import primordial_subduction
        result['initial_subduction'] = primordial_subduction.normalize(initial['initial_subduction'])
    if 'continental_lifecycle' in initial:
        import continental_lifecycle
        result['continental_lifecycle'] = continental_lifecycle.normalize(initial['continental_lifecycle'])
    if 'rift_traction' in initial:
        import rift_traction
        result['rift_traction'] = rift_traction.normalize(initial['rift_traction'])
    return result


def color_image(elevation, width, height):
    z = np.asarray(elevation).reshape(height, width)
    stops = np.array([-11000, -6500, -4000, -1500, -1, 0, 400, 1200, 2400, 4200, 6500, 9000])
    colors = np.array([[5, 18, 34], [10, 38, 65], [19, 70, 96], [37, 115, 135],
                       [94, 162, 166], [121, 154, 117], [132, 157, 111],
                       [165, 160, 114], [164, 139, 103], [154, 126, 107],
                       [207, 201, 185], [247, 245, 234]])
    rgb = np.stack([np.interp(z, stops, colors[:, k]) for k in range(3)], axis=-1)
    # Presentation shading only; exported elevations remain unmodified.
    gy, gx = np.gradient(z)
    shade = np.clip(1 - .000055 * gx - .000075 * gy, .65, 1.2)
    return Image.fromarray(np.uint8(np.clip(rgb * shade[..., None], 0, 255)))


def png_bytes(frame, colored=False):
    if colored:
        picture = color_image(frame["elevation"], frame["width"], frame["height"])
    else:
        # Fixed scale shared by every frame; absolute elevation = sample - 12000 m.
        z = np.asarray(frame["elevation"], dtype=np.float64).reshape(frame["height"], frame["width"])
        picture = Image.fromarray(np.rint(np.clip(z + 12000, 0, 65535)).astype(np.uint16))
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def boundary_geojson(frame):
    w, h = frame["width"], frame["height"]
    boundary = np.asarray(frame["boundary"]).reshape(-1)
    plates = np.asarray(frame["plate"]).reshape(-1)
    domains = np.asarray(frame["domain"]).reshape(-1) if "domain" in frame else None
    domain_names = {int(row["uid"]): row.get("name") for row in frame.get("domains", [])}
    features = []
    for i in np.flatnonzero(boundary):
        x, y = int(i % w), int(i // w)
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [
            -180 + (x + .5) * 360 / w, 90 - (y + .5) * 180 / h]},
            "properties": {"type": BOUNDARIES[int(boundary[i])], "code": int(boundary[i]),
                           "plate_id": int(plates[i]), "time_myr": frame["time_myr"]}})
        if domains is not None:
            uid = int(domains[i])
            north = (y-1)*w+x if y else (x+w//2) % w
            south = (y+1)*w+x if y < h-1 else y*w+(x+w//2) % w
            neighbors = (y*w+(x-1) % w, y*w+(x+1) % w, north, south)
            features[-1]["properties"].update(
                domain_uid=uid, domain_name=domain_names.get(uid),
                adjacent_domain_uids=sorted({int(domains[j]) for j in neighbors if int(domains[j]) != uid}))
    return {"type": "FeatureCollection", "description": "Boundary raster cell centers; approximate locations, not resolved fault traces.",
            "features": features}


def fresh_world_defaults():
    """Form defaults for a new world: the reviewed profile and its reviewed laws.

    DEFAULT_CONFIG stays the engine's missing-key contract for saved configs
    (legacy, fixed_trench, disabled, enhanced rifting off). Its law keys and
    None placeholders are dropped here so fresh_world_request and the
    normalizers supply explicit fresh-world values.
    """
    base = {key: value for key, value in DEFAULT_CONFIG.items()
            if value is not None and key not in fresh_world.REVIEWED_FRESH_LAWS
            and key != 'enhanced_rifting'}
    base['physics_profile'] = fresh_world.REVIEWED_PROFILE
    return validate_config(fresh_world.fresh_world_request(base))


@dataclass(frozen=True)
class _LoadValidation:
    """A checkpoint generation fully validated earlier in the *same* load call.

    Validation contract.  ``_recover_manifest_validated`` decodes every array
    of the selected checkpoint (including zip CRCs), checks its compatibility
    record, run identity, configuration, exact snapshot scheduler and every
    referenced frame file.  ``load`` passes this record directly to
    ``_set_resume_status`` so that the same bytes are not decoded and the same
    frame files are not reread a second time within that one call.

    The record lives only in ``load``'s local scope.  It is never stored on the
    manager, never shared across requests and never persisted, so a later
    save, recovery-generation change, run selection, branch or resume request
    always validates the files afresh.  It is used only when the manager still
    points at the same run and checkpoint file and that file's filesystem
    signature is unchanged.  The signature is a cheap tripwire, not a guarantee:
    neither it nor the manager lock protects against arbitrary external
    modification of run files.  ``can_resume`` therefore describes the files as
    they were during this load; ``resume`` and ``branch`` rely on their own
    fresh ``_checkpoint_for_current`` check before acting on a checkpoint.
    """
    run_id: str
    checkpoint: Path
    signature: tuple
    # {'manifest': committed manifest, 'time_myr': decoded simulation time}.
    # The freeze is shallow: nothing may mutate ``header['manifest']``, or
    # ``_check_current_matches`` could end up comparing a manifest with itself.
    header: dict


def _file_signature(path):
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class SimulationManager:
    def __init__(self, root=RUNS, recovery_interval_myr=10., *, workers=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.pause_requested = threading.Event()
        self.shutting_down = False
        self.worker = None
        self.parallel = RuntimePool(workers)
        self.worker_commands = WorkerCommands(self.parallel.policy)
        self.timing = None
        if not math.isfinite(recovery_interval_myr) or recovery_interval_myr <= 0:
            raise ValueError('Recovery interval must be a positive finite number of Myr.')
        self.recovery_interval_myr = float(recovery_interval_myr)
        self._resume_checkpoint_name = 'checkpoint.npz'
        self.current = {"state": "idle", "frames": [], "frame_count": 0, "progress": 0,
                        "time_myr": 0, "run_id": None}
        recent = self.list_runs()
        if recent:
            self.load(recent[0]["run_id"])

    def status(self):
        with self.lock:
            result = native(self.current)
            if self.timing is not None:
                result.update(self.timing.snapshot())
            else:
                result.setdefault('elapsed_seconds', 0.)
                result['eta_seconds'] = 0. if result.get('state') == 'complete' else None
            result.setdefault('can_resume', False)
            result['shutting_down'] = self.shutting_down
            result['workers'] = self.workers_status()
            return result

    def workers_status(self):
        return dict(self.worker_commands.status(), **self.parallel.status())

    def set_workers(self, body):
        if self.shutting_down:
            raise ValueError('The app is shutting down.')
        self.worker_commands.apply(body)
        return self.workers_status()

    def list_runs(self):
        results = []
        for path in sorted(self.root.glob("*/manifest.json"), reverse=True):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("frame_count", 0) or item.get('state') in ('running', 'pausing', 'paused', 'resuming'):
                    results.append({k: item.get(k) for k in ("run_id", "title", "created", "state", "duration_myr", "time_myr", "frame_count", "config", 'can_resume')})
            except (OSError, ValueError):
                pass
        return results

    def path(self, run_id=None):
        run_id = run_id or self.current.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("Invalid run identifier.")
        return self.root / run_id

    def load(self, run_id):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("Pause the current simulation before opening another run.")
            path = self.path(run_id) / "manifest.json"
            if not path.is_file():
                raise ValueError("Saved run not found.")
            result = json.loads(path.read_text(encoding="utf-8"))
            self._resume_checkpoint_name = 'checkpoint.npz'
            # ``validated`` is local to this call; see ``_LoadValidation``.
            result, validated = self._recover_manifest_validated(result, run_id)
            self.current = result
            self.timing = None
            self._set_resume_status(validated)
            return self.status()

    @staticmethod
    def compatibility():
        import flow_acceleration
        import material_acceleration
        return dict(engine_sha256=hashlib.sha256(ENGINE_SOURCE).hexdigest(),
                    auxiliary_sources_sha256={name: hashlib.sha256(payload).hexdigest()
                                              for name, payload in AUXILIARY_SOURCES.items()},
                    numpy_version=np.__version__, flow_backend=flow_acceleration.provenance(),
                    material_backend=material_acceleration.provenance())

    def _validate_checkpoint_history(self, header, run_id, config):
        saved = header['manifest']
        if saved.get('run_id') != run_id or saved.get('config') != config:
            raise ValueError('Checkpoint belongs to a different experiment or configuration.')
        count = int(saved.get('frame_count', 0))
        if count < 1 or len(saved.get('frames', [])) != count:
            raise ValueError('Checkpoint history index is incomplete.')
        pending = saved.get('next_output_myr')
        if (not isinstance(pending, (int, float)) or not math.isfinite(pending)
                or (pending <= header['time_myr'] and not (header['time_myr'] == pending == saved['config']['duration_myr']))
                or pending > saved['config']['duration_myr']):
            raise ValueError('Checkpoint has no valid exact snapshot scheduler; it cannot resume without changing the integration steps.')
        for index, entry in enumerate(saved['frames']):
            frame = self.path(run_id)/f'frame_{index:04d}'
            if not all(frame.with_suffix(suffix).is_file() for suffix in ('.npz', '.json')):
                raise ValueError('A frame referenced by the checkpoint is missing.')
            metadata = json.loads(frame.with_suffix('.json').read_text(encoding='utf-8'))
            if (entry.get('index') != index or metadata.get('index') != index
                    or metadata.get('time_myr') != entry.get('time_myr')
                    or entry['time_myr'] > header['time_myr'] + 1e-6):
                raise ValueError('A frame referenced by the checkpoint does not match its history index.')

    def _guarded_checkpoint(self, result, run_id):
        """Choose only a contiguous independently closed immutable checkpoint chain."""
        directory = getattr(self, '_guardian_directory', None) or result.get('guardian_acceptance_directory')
        if directory is None:
            return None
        directory = Path(directory).resolve()
        control_pin = getattr(self, '_guardian_control_sha256', None) or result.get('guardian_control_sha256')
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        def read(path):
            return json.loads(Path(path).read_bytes())
        def pinned(reference, path):
            path = Path(path).resolve()
            require(Path(reference['path']).resolve() == path and self._output_reference(path) == reference,
                    'Accepted interval file pin differs: '+str(path))
            return read(path)
        require(run_id == '20261007-coarse-burial-speed-history', 'Guarded checkpoint run differs')
        gate = read(directory/'launch-gate.json') if (directory/'launch-gate.json').is_file() else None
        paths = list(directory.glob('closed-interval-*.json'))
        if gate is None:
            require(not paths, 'Closed interval lacks launch gate')
            return None
        require(gate['run_id'] == run_id and gate['control_sha256'] == control_pin, 'Guardian launch binding differs')
        final_path = directory/'guardian-completion.json'
        final = read(final_path) if final_path.is_file() else {}
        incomplete = final.get('incomplete_interval', {}).get('token')
        rows = []
        for path in paths:
            closed = read(path)
            token = closed.get('token', '')
            require(re.fullmatch(r'[0-9a-f]{32}', token) and path.name == 'closed-interval-'+token+'.json',
                    'Invalid closed interval token')
            if token == incomplete:
                continue
            transition_path = directory/('transitioned-interval-'+token+'.json')
            if not transition_path.is_file():
                continue  # Publication alone does not prove the hard guard advanced in time.
            transition = read(transition_path)
            require(transition.get('kind') == 'guardian_interval_transition' and transition.get('version') == 1
                and transition.get('run_id') == run_id and transition.get('token') == token
                and transition.get('control_sha256') == control_pin and transition.get('budget') == closed['budget']
                and transition.get('source_time_myr') == closed['source_time_myr']
                and transition.get('destination_time_myr') == closed['destination_time_myr']
                and transition.get('closed_interval') == self._output_reference(path), 'Guardian transition witness differs')
            advanced = transition['transitioned_at_perf_counter']
            require(type(advanced) in (int,float) and math.isfinite(advanced)
                and closed['closed_at_perf_counter'] <= advanced < closed['budget']['deadline'], 'Late guardian transition')
            closed['_transition'] = transition
            rows.append((closed['source_time_myr'], path, closed))
        epoch, previous = 121., None
        expected_token, expected_budget = gate['initial_interval_token'], gate['initial_budget']
        selected = None
        for _, path, closed in sorted(rows, key=lambda row: row[0]):
            token = closed['token']
            base = self.path(run_id)/'numerical-budget'/token
            ack = pinned(closed['guardian_acceptance'], directory/('accepted-interval-'+token+'.json'))
            proof = pinned(closed['server_acceptance'], base/'accepted.json')
            completion = pinned(ack['completion'], base/'completion.json')
            require(closed.get('kind') == 'guardian_closed_interval' and closed.get('version') == 1
                and ack.get('kind') == 'guardian_accepted_interval' and ack.get('version') == 1
                and proof.get('kind') == 'server_observed_guardian_acceptance' and proof.get('version') == 1
                and completion.get('kind') == 'lite_budgeted_history_interval' and completion.get('version') == 2,
                'Wrong interval acceptance schema')
            for row in (closed, ack, proof, completion):
                require(row.get('run_id') == run_id and row.get('token') == expected_token == token
                    and row.get('budget') == expected_budget, 'Accepted interval identity/clock changed')
            require(closed['control_sha256'] == ack['control_sha256'] == control_pin
                and ack['child'] == gate['child'] and ack['guardian'] == gate['guardian']
                and ack['previous_acceptance_token'] == previous, 'Accepted interval authority or chain differs')
            dt = expected_budget['dt_myr']
            destination = epoch+dt
            require(dt == min(1.,1000.-epoch) and ack['requested_dt_myr'] == completion['requested_dt_myr'] == dt
                and all(row['source_time_myr'] == epoch and row['destination_time_myr'] == destination
                        for row in (closed,ack,completion)), 'Accepted interval is discontinuous')
            started = expected_budget['started']
            require(expected_budget == dict(policy='budgeted_history_1800s_per_myr_v1',dt_myr=dt,
                started=started,deadline=started+1800.*dt,physical_deadline=started+1680.*dt,
                force_refinement_deadline=started+900.*dt), 'Accepted interval budget policy differs')
            times = [started, completion['completed_at_perf_counter'], ack['accepted_at_perf_counter'],
                     proof['observed_at_perf_counter'], closed['closed_at_perf_counter']]
            require(all(type(value) in (int,float) and math.isfinite(value) for value in times)
                and times == sorted(times) and times[-1] < expected_budget['deadline'], 'Late interval acceptance')
            checkpoint = completion['checkpoint']
            require(completion.get('interval_and_required_output_completed') is True
                and checkpoint == ack['checkpoint'] == proof['checkpoint'] == closed['checkpoint']
                and checkpoint['time_myr'] == destination
                and Path(checkpoint['path']).resolve() == (base/'checkpoint.npz').resolve()
                and proof['completion'] == ack['completion']
                and proof['guardian_acceptance'] == closed['guardian_acceptance'], 'Accepted checkpoint evidence differs')
            frames = completion['frames']
            require(frames == ack['frames'] and len(frames) == int(destination % 2 == 0), 'Accepted frame cadence differs')
            for frame in frames:
                index = 62+int((destination-122.)/2.)
                require(frame['index'] == index and frame['time_myr'] == destination, 'Accepted frame time differs')
                for extension in ('npz','json'):
                    frame_path = self.path(run_id)/('frame_%04d.%s'%(index,extension))
                    require(self._output_reference(frame_path) == frame[extension], 'Accepted frame hash differs')
            transition = closed['_transition']
            terminal = bool(closed.get('paused') or closed.get('complete'))
            require(transition['next_interval_token'] == (None if terminal else ack['next_interval_token'])
                and transition['next_budget'] == (None if terminal else ack['next_budget']), 'Transition next clock differs')
            following = ack['next_budget']
            if destination < 1000.:
                require(re.fullmatch(r'[0-9a-f]{32}',ack['next_interval_token'])
                    and following['started'] == ack['accepted_at_perf_counter'], 'Next interval clock was renewed')
            else:
                require(following is None and ack['next_interval_token'] is None, 'Final interval issued another token')
            selected = Path(checkpoint['path'])
            epoch, previous = destination, token
            expected_token, expected_budget = transition['next_interval_token'], transition['next_budget']
        if selected is not None:
            reference = self._output_reference(selected)
            require(reference['sha256'] == checkpoint['sha256'], 'Latest accepted checkpoint hash differs')
        return selected

    def _recover_manifest_validated(self, result, run_id):
        """Choose an entire committed generation; never merge two histories.

        Returns ``(manifest, validation)``, where ``validation`` is the
        ``_LoadValidation`` of the selected generation (or None) for use only
        within the same load call."""
        guarded = self._guarded_checkpoint(result, run_id)
        if result.get('state') not in ('running', 'pausing', 'resuming', 'paused', 'interrupted', 'error') and guarded is None:
            return result, None  # An unguarded terminal history remains view-only.
        original_state = result['state']
        if original_state in ('running', 'pausing', 'resuming'):
            result = dict(result, state='interrupted')
        try:
            verify_compatibility(result, self.compatibility())
        except ValueError:
            return result, None
        candidates = []
        primary = self.path(run_id)/'checkpoint.npz'
        choices = (guarded,) if guarded is not None else (primary, previous_checkpoint(primary))
        for candidate in choices:
            try:
                # Taken before reading: any later replacement changes it.
                signature = _file_signature(candidate)
                simulation, committed = read_checkpoint(candidate, self.compatibility(), Simulation)
                header = dict(manifest=committed, time_myr=float(simulation.t))
                del simulation
                self._validate_checkpoint_history(header, run_id, result['config'])
                candidates.append((header['time_myr'], str(candidate.relative_to(self.path(run_id))).replace('\\','/'), committed,
                                   _LoadValidation(run_id, candidate, signature, header)))
            except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
                continue
        if not candidates:
            if guarded is not None:
                raise ValueError('The accepted immutable checkpoint failed full validation')
            return result, None
        # Stable sort prefers primary if both generations have the same time.
        epoch, name, committed, validated = max(candidates, key=lambda candidate: candidate[0])
        recovered = native(committed)  # deep copy; ``committed`` stays untouched (see _LoadValidation.header)
        recovered['state'] = ('paused' if committed.get('state') == 'paused'
                              and original_state in ('paused', 'pausing') else 'interrupted')
        if epoch >= float(recovered['config']['duration_myr'])-1e-8:
            recovered['state'] = 'complete'
        recovered.update(durable_accepted_checkpoint_time_myr=epoch, checkpoint_acceptance_pending=False)
        self._resume_checkpoint_name = name
        changed = (original_state != 'paused' or name != primary.name
                   or result.get('time_myr') != epoch
                   or result.get('frame_count') != committed.get('frame_count'))
        if changed:
            recovered['recovery'] = dict(checkpoint_file=name, checkpoint_time_myr=epoch,
                previously_saved_time_myr=result.get('time_myr', 0),
                previously_integrated_time_myr=result.get('integration_time_myr', result.get('time_myr', 0)),
                reverted_frame_count=max(0, result.get('frame_count', 0)-committed.get('frame_count', 0)))
        if result.get('error'):
            recovered['error'] = result['error']
        return recovered, validated

    def _checkpoint_for_current(self):
        """Fresh check of the files on disk; used by every resume and branch."""
        guarded = self._guarded_checkpoint(self.current, self.current['run_id'])
        if guarded is not None and guarded.resolve() != (self.path()/self._resume_checkpoint_name).resolve():
            raise ValueError('Selected checkpoint is not the latest independently closed interval')
        if guarded is None and self._resume_checkpoint_name not in ('checkpoint.npz', 'checkpoint.previous.npz'):
            raise ValueError('Selected checkpoint lacks independent closure')
        header = checkpoint_header(self.path()/self._resume_checkpoint_name, self.compatibility())
        simulation, _ = read_checkpoint(self.path()/self._resume_checkpoint_name, self.compatibility(), Simulation)
        del simulation
        self._check_current_matches(header)
        self._validate_checkpoint_history(header, self.current['run_id'], self.current['config'])
        return header

    def _check_current_matches(self, header):
        verify_compatibility(self.current, self.compatibility())
        saved = header['manifest']
        if (saved.get('run_id') != self.current.get('run_id')
                or saved.get('frame_count') != self.current.get('frame_count')
                or abs(header['time_myr']-self.current.get('time_myr', -1)) > 1e-6
                or saved.get('config') != self.current.get('config')):
            raise ValueError('The available checkpoint does not match the latest saved history; it cannot safely resume this experiment.')

    def _validation_applies(self, validated):
        """Whether a same-load validation still names the current checkpoint."""
        if validated is None:
            return False
        try:
            checkpoint = self.path()/self._resume_checkpoint_name
            return (validated.run_id == self.current.get('run_id')
                    and validated.checkpoint == checkpoint
                    and validated.signature == _file_signature(checkpoint))
        except (ValueError, OSError):
            return False

    def _set_resume_status(self, validated=None):
        """Recompute ``can_resume``.  ``validated`` may be passed only by
        ``load`` for the generation it validated in that same call."""
        self.current['can_resume'] = False
        if self.current.get('state') not in ('paused', 'interrupted'):
            self.current.pop('resume_reason', None)
            return
        try:
            if self._validation_applies(validated):
                # The checkpoint decode, compatibility record and full history
                # pass already succeeded for this generation in this call, for
                # the same run id and configuration.  Only the in-memory
                # comparison with the recovered manifest remains.
                self._check_current_matches(validated.header)
            else:
                self._checkpoint_for_current()
            self.current['can_resume'] = True
            self.current.pop('resume_reason', None)
        except (ValueError, OSError) as exc:
            self.current['resume_reason'] = str(exc)

    def _check_selected(self, run_id):
        if run_id is not None and run_id != self.current.get('run_id'):
            raise ValueError('The selected experiment changed; select it again before pausing or resuming.')

    def pause(self, run_id=None):
        with self.lock:
            self._check_selected(run_id)
            if self.current.get('state') in ('paused', 'complete'):
                return self.status()
            if (not self.worker or not self.worker.is_alive() or self.stop.is_set()
                    or self.current.get('state') not in ('running', 'resuming', 'pausing')):
                raise ValueError('There is no running simulation to pause.')
            self.pause_requested.set()
            self.current.update(state='pausing', can_resume=False)
            self.persist()
            return self.status()

    def resume(self, run_id=None):
        with self.lock:
            self._check_selected(run_id)
            if self.shutting_down:
                raise ValueError('The app is saving and shutting down. Reopen it before resuming.')
            if self.worker and self.worker.is_alive():
                if self.current.get('state') in ('running', 'resuming'):
                    return self.status()
                raise ValueError('The current simulation is still finishing its pause.')
            if self.current.get('state') not in ('paused', 'interrupted'):
                raise ValueError('Only a paused experiment with an exact checkpoint can be resumed. Older cancelled histories are view-only.')
            self._checkpoint_for_current()
            self._preserve_abandoned_frames()
            self.stop.clear()
            self.pause_requested.clear()
            self.current.update(state='resuming', can_resume=False)
            self.current.pop('error', None)
            self.current.pop('resume_reason', None)
            self.timing = ProgressTiming(total_units=self.current['duration_myr'],
                                         completed_units=self.current['time_myr'],
                                         elapsed_seconds=self.current.get('elapsed_seconds', 0.))
            self.persist()
            self.worker = threading.Thread(target=self._run, args=(self.current['config'], None, True), daemon=True)
            self.worker.start()
            return self.status()

    def _preserve_abandoned_frames(self):
        """Keep interrupted output before resumed frame numbers can reuse it."""
        root = self.path().resolve()
        count = self.current['frame_count']
        abandoned = [path for path in root.glob('frame_*')
                     if (match := re.fullmatch(r'frame_(\d+)\.(npz|json)', path.name))
                     and int(match[1]) >= count]
        if not abandoned:
            return
        archive = root/'recovery-abandoned'/uuid.uuid4().hex
        archive.mkdir(parents=True)
        for source in abandoned + [root/'manifest.json']:
            # Copy, do not move: a failure cannot invalidate the saved history.
            shutil.copy2(source, archive/source.name)
        self.current.setdefault('recovery', {})['abandoned_artifacts'] = str(archive.relative_to(root)).replace('\\', '/')

    def cancel(self, run_id=None):
        with self.lock:
            self._check_selected(run_id)
            self.stop.set()
            self.pause_requested.clear()
            if self.current.get('state') in ('running', 'pausing', 'resuming', 'paused', 'interrupted'):
                # Acknowledging cancellation is a durable terminal decision.
                # Recovery must never replace it with a concurrently committed
                # pause checkpoint if shutdown/crash occurs immediately after.
                self.current.update(state='cancelled', can_resume=False)
                self.current.pop('resume_reason', None)
                self.persist()
            return {'ok': True}

    def prepare_shutdown(self):
        with self.lock:
            self.shutting_down = True
            if (self.worker and self.worker.is_alive() and not self.stop.is_set()
                    and self.current.get('state') in ('running', 'resuming', 'pausing')):
                self.pause_requested.set()
                self.current.update(state='pausing', can_resume=False)
                self.persist()

    def start(self, config, initial):
        import world_design
        # The only place a new world is created: reviewed_v1 requests receive
        # the reviewed law choices explicitly before validation, so config.json
        # records them. Resume and branch re-read saved configs untouched.
        config = fresh_world.fresh_world_request(dict(config or {}), initial)
        if isinstance(initial, dict) and 'world_design' in initial:
            design = world_design.normalize(initial['world_design'])
            if 'world_design' in config and world_design.normalize(config['world_design']) != design:
                raise ValueError('The starting map and run configuration specify different world designs.')
            config['world_design'] = design
        config = validate_config(config)
        initial = validate_initial(initial, config)
        if initial is None:
            initial = validate_initial(native(make_initial(config)), config)
        if 'world_design' in config:
            initial['world_design'] = world_design.normalize(config['world_design'])
        with self.lock:
            if self.shutting_down:
                raise ValueError('The app is shutting down; reopen it before starting a world.')
            if self.worker and self.worker.is_alive():
                raise ValueError("A simulation is already running.")
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            run_path = self.path(run_id)
            run_path.mkdir()
            (run_path / 'engine.py').write_bytes(ENGINE_SOURCE)
            for name, payload in AUXILIARY_SOURCES.items():
                (run_path / name).write_bytes(payload)
            self.stop.clear()
            self.pause_requested.clear()
            self._resume_checkpoint_name = 'checkpoint.npz'
            self.current = {"state": "running", "run_id": run_id,
                            "created": datetime.now(timezone.utc).isoformat(), "config": config,
                            "engine_sha256": hashlib.sha256(ENGINE_SOURCE).hexdigest(),
                            "auxiliary_sources_sha256": {name: hashlib.sha256(payload).hexdigest()
                                                         for name, payload in AUXILIARY_SOURCES.items()},
                            "numpy_version": np.__version__,
                            "flow_backend": self.compatibility()['flow_backend'],
                            "material_backend": self.compatibility()['material_backend'],
                            "duration_myr": config["duration_myr"], "frames": [], "frame_count": 0,
                            "progress": 0, "time_myr": 0, "elapsed_seconds": 0,
                            "can_resume": False, "recovery_interval_myr": self.recovery_interval_myr}
            self.timing = ProgressTiming(total_units=config['duration_myr'])
            write_json(run_path / "initial.json", initial)
            write_json(run_path / "config.json", config)
            self.persist()
            self.worker = threading.Thread(target=self._run, args=(config, initial), daemon=True)
            self.worker.start()
            return {"run_id": run_id}

    def branch(self, run_id=None, world_design=None):
        """Clone a compatible pause and author only its future regional design."""
        import world_design as design_api
        with self.lock:
            self._check_selected(run_id)
            if self.shutting_down:
                raise ValueError('The app is shutting down; reopen it before branching.')
            if self.worker and self.worker.is_alive():
                raise ValueError('Wait for the simulation to finish pausing before branching.')
            if self.current.get('state') != 'paused':
                raise ValueError('Branching requires a compatible paused checkpoint.')
            self._checkpoint_for_current()
            parent_path = self.path()
            parent_checkpoint = parent_path/self._resume_checkpoint_name
            simulation, saved = read_checkpoint(parent_checkpoint, self.compatibility(), Simulation)
            design = design_api.validate_branch(simulation.config.get('world_design'), world_design, simulation.t)
            config = validate_config(dict(simulation.config, world_design=design))
            initial = json.loads((parent_path/'initial.json').read_text(encoding='utf-8'))
            initial['world_design'] = design
            initial = validate_initial(initial, config)
            engine = (parent_path/'engine.py').read_bytes()
            if hashlib.sha256(engine).hexdigest() != saved['engine_sha256']:
                raise ValueError('The parent saved engine source is damaged; branching cannot preserve its provenance.')
            helpers = read_saved_auxiliary_sources(parent_path, saved.get('auxiliary_sources_sha256', {}))
            branch_id = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            branch_path = self.path(branch_id)
            origin = dict(run_id=saved['run_id'], time_myr=float(simulation.t),
                          checkpoint_file=parent_checkpoint.name,
                          checkpoint_sha256=hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest(),
                          world_design_revision=design['revision'])
            # Only authored metadata changes in the decoded copy. The parent's
            # physical arrays, source files and committed frame bytes stay intact.
            simulation.config = config
            simulation.world_design = design_api.normalize(design)
            simulation._record('authored_branch', 'The author branched this paused experiment with future regional controls.',
                details=dict(origin='authored', parent=origin, world_design=design))
            design_api.record_transitions(simulation)
            manifest = native(saved)
            manifest.update(run_id=branch_id, created=datetime.now(timezone.utc).isoformat(),
                            title='Branch of '+saved['run_id'], config=config, state='paused',
                            can_resume=True, branch_origin=origin,
                            checkpoint_version=CHECKPOINT_VERSION, checkpoint_time_myr=float(simulation.t))
            for key in ('error', 'resume_reason', 'recovery'):
                manifest.pop(key, None)
            branch_path.mkdir()
            (branch_path/'engine.py').write_bytes(engine)
            for name, payload in helpers.items():
                (branch_path/name).write_bytes(payload)
            for index in range(saved['frame_count']):
                for suffix in ('.npz', '.json'):
                    filename = f'frame_{index:04d}{suffix}'
                    shutil.copy2(parent_path/filename, branch_path/filename)
            write_json(branch_path/'initial.json', initial)
            write_json(branch_path/'config.json', config)
            write_json(branch_path/'branch-parent-manifest.json', saved)
            write_checkpoint(branch_path/'checkpoint.npz', simulation, manifest, self.compatibility())
            del simulation
            # Publishing the manifest last makes an interrupted partial copy
            # invisible to the saved-experiment list.
            write_json(branch_path/'manifest.json', manifest)
            self.current = manifest
            self.timing = None
            self.worker = None
            self._resume_checkpoint_name = 'checkpoint.npz'
            self._set_resume_status()
            return self.status()

    def persist(self):
        with self.lock:
            if self.timing is not None:
                self.current.update(self.timing.snapshot())
            write_json(self.path() / "manifest.json", self.current)

    def _append_snapshot(self, simulation, pause_frame=False):
        budget.require_finalization_time('snapshot')
        snapshot = simulation.snapshot()
        if budget.current() is not None:
            snapshot['numerical_budget'] = budget.evidence()
        budget.require_finalization_time('snapshot:rendered')
        with self.lock:
            if self.current['frames'] and abs(self.current['frames'][-1]['time_myr']-simulation.t) < 1e-6:
                return
            index = len(self.current['frames'])
        if pause_frame:
            snapshot['pause_frame'] = True
        self.save_frame(index, snapshot)
        with self.lock:
            entry = dict(index=index, time_myr=snapshot['time_myr'], stats=native(snapshot.get('stats', {})))
            if pause_frame:
                entry['pause_frame'] = True
            self.current['frames'].append(entry)
            self.current.update(frame_count=index+1, time_myr=snapshot['time_myr'],
                                progress=min(1, snapshot['time_myr']/simulation.config['duration_myr']))
            self.persist()

    def _run(self, config, initial, resuming=False):
        with RunProfiler(self.path() / 'profiling'), self.parallel.activate():
            return self._run_owned(config, initial, resuming)

    def _open_numerical_budget(self, source_time, requested_dt):
        active = dict(kind='coarse_history_active_interval', version=2, active=True,
            phase='physical', run_id=self.current['run_id'], pid=os.getpid(),
            token=self._next_numerical_token, source_time_myr=float(source_time),
            requested_dt_myr=float(requested_dt), budget=budget.snapshot(),
            started_at_utc=datetime.now(timezone.utc).isoformat())
        if budget.snapshot() != self._next_numerical_budget:
            raise ValueError('Controller interval clock was replaced')
        self._active_numerical_budget = active
        self._interval_first_frame = len(self.current['frames'])
        self._pending_checkpoint = None
        write_json(self.path()/'numerical-budget-active.json', active)

    @staticmethod
    def _output_reference(path):
        path = Path(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(path=str(path.resolve()), sha256=digest)

    def _finish_numerical_budget(self, source_time, requested_dt, simulation):
        active = dict(self._active_numerical_budget)
        token = active['token']
        pending = self._pending_checkpoint
        if not pending or pending['manifest']['checkpoint_time_myr'] != float(simulation.t):
            raise ValueError('Interval has no complete immutable checkpoint')
        checkpoint_ref = self._output_reference(pending['path'])
        checkpoint_ref['time_myr'] = float(simulation.t)
        frames = []
        for row in self.current['frames'][self._interval_first_frame:]:
            index = row['index']
            frames.append(dict(index=index, time_myr=row['time_myr'],
                npz=self._output_reference(self.path()/('frame_%04d.npz'%index)),
                json=self._output_reference(self.path()/('frame_%04d.json'%index))))
        budget.require_finalization_time('immutable_outputs_hashed')
        report = dict(kind='lite_budgeted_history_interval', version=2,
            run_id=self.current['run_id'], token=token, source_time_myr=source_time,
            destination_time_myr=float(simulation.t), requested_dt_myr=requested_dt,
            interval_and_required_output_completed=True, budget=budget.snapshot(),
            checkpoint=checkpoint_ref, frames=frames, evidence=budget.evidence(),
            completed_at_perf_counter=time.perf_counter(), error=None)
        path = self.path()/'numerical-budget'/token/'completion.json'
        if path.exists():
            raise ValueError('Interval completion already exists; no retry or clock renewal')
        write_json(path, report)
        completion = self._output_reference(path)
        budget.require_finalization_time('completion_receipt_written')
        active.update(phase='awaiting_guardian', completion=completion)
        write_json(self.path()/'numerical-budget-active.json', active)
        acknowledgement = Path(self._guardian_directory)/('accepted-interval-'+token+'.json')
        while not acknowledgement.is_file():
            budget.require_finalization_time('guardian_acceptance_wait')
            time.sleep(.025)
        accepted = json.loads(acknowledgement.read_bytes())
        if (accepted.get('kind') != 'guardian_accepted_interval' or accepted.get('token') != token
                or accepted.get('run_id') != self.current['run_id']
                or accepted.get('control_sha256') != self._guardian_control_sha256
                or accepted.get('budget') != budget.snapshot()
                or accepted.get('completion') != completion or accepted.get('checkpoint') != checkpoint_ref):
            raise ValueError('Guardian acceptance identity differs')
        budget.require_finalization_time('guardian_acceptance_received')
        with self.lock:
            terminal = simulation.t >= float(simulation.config['duration_myr'])-1e-8
            paused = bool(self.pause_requested.is_set() or self.stop.is_set()) and not terminal
            self._resume_checkpoint_name = str(Path(pending['path']).relative_to(self.path())).replace('\\','/')
            self.current.update(checkpoint_version=CHECKPOINT_VERSION,
                checkpoint_time_myr=source_time, integration_time_myr=float(simulation.t),
                next_output_myr=pending['manifest']['next_output_myr'],
                state='complete' if terminal else ('paused' if paused else 'running'),
                can_resume=paused, checkpoint_acceptance_pending=True,
                durable_accepted_checkpoint_time_myr=source_time,
                accepted_checkpoint_file=self._resume_checkpoint_name,
                guardian_acceptance_directory=str(self._guardian_directory),
                guardian_control_sha256=self._guardian_control_sha256,
                guardian_acceptance_file=str(acknowledgement), numerical_budget_record=str(path.relative_to(self.path())),
                numerical_budget=report, eta_seconds=None)
            self.current.update(self.timing.update(min(float(simulation.config['duration_myr']),float(simulation.t))))
            if terminal or paused:
                self.current.update(self.timing.finish() if terminal else self.timing.pause())
            self.current['eta_seconds'] = None
            self.persist()
        active.update(active=False, phase='accepted', interval_and_required_output_completed=True,
            guardian_acceptance=self._output_reference(acknowledgement),
            paused=paused, complete=terminal)
        budget.require_finalization_time('accepted_pointer_written')
        acceptance_path = path.with_name('accepted.json')
        write_json(acceptance_path,dict(kind='server_observed_guardian_acceptance',version=1,
            run_id=self.current['run_id'],token=token,budget=budget.snapshot(),completion=completion,
            guardian_acceptance=active['guardian_acceptance'],checkpoint=checkpoint_ref,
            observed_at_perf_counter=time.perf_counter()))
        budget.require_finalization_time('acceptance_readback')
        active['server_acceptance'] = self._output_reference(acceptance_path)
        write_json(self.path()/'numerical-budget-active.json', active)
        budget.require_finalization_time('server_acceptance_published')
        release_path = Path(self._guardian_directory)/('closed-interval-'+token+'.json')
        while not release_path.is_file():
            budget.require_finalization_time('guardian_closure_wait')
            time.sleep(.025)
        release = json.loads(release_path.read_bytes())
        if (release.get('kind') != 'guardian_closed_interval' or release.get('token') != token
                or release.get('run_id') != self.current['run_id']
                or release.get('control_sha256') != self._guardian_control_sha256
                or release.get('budget') != budget.snapshot()
                or release.get('guardian_acceptance') != active['guardian_acceptance']
                or release.get('server_acceptance') != active['server_acceptance']
                or release.get('checkpoint') != checkpoint_ref):
            raise ValueError('Guardian interval closure differs')
        transition_path = Path(self._guardian_directory)/('transitioned-interval-'+token+'.json')
        while not transition_path.is_file():
            budget.require_finalization_time('guardian_transition_wait')
            time.sleep(.025)
        transition = json.loads(transition_path.read_bytes())
        if (transition.get('kind') != 'guardian_interval_transition' or transition.get('token') != token
                or transition.get('run_id') != self.current['run_id']
                or transition.get('control_sha256') != self._guardian_control_sha256
                or transition.get('budget') != budget.snapshot()
                or transition.get('closed_interval') != self._output_reference(release_path)
                or not release['closed_at_perf_counter'] <= transition['transitioned_at_perf_counter'] < budget.snapshot()['deadline']):
            raise ValueError('Guardian hard-deadline transition differs')
        with self.lock:
            self.current.update(checkpoint_time_myr=float(simulation.t), durable_accepted_checkpoint_time_myr=float(simulation.t), checkpoint_acceptance_pending=False)
        self._next_numerical_budget = accepted.get('next_budget')
        self._next_numerical_token = accepted.get('next_interval_token')
        return accepted

    def _run_owned(self, config, initial, resuming=False):
        initial_scope = budget.installed(self._initial_numerical_budget)
        initial_scope.__enter__()
        self._next_numerical_budget = self._initial_numerical_budget
        self._next_numerical_token = self._initial_numerical_token
        try:
            if not resuming:
                raise ValueError('Guarded successor requires its prepared checkpoint')
            budget.require_physical_time('resume_checkpoint_decode')
            simulation, resume_manifest = read_checkpoint(self.path()/self._resume_checkpoint_name,
                self.compatibility(), Simulation)
            budget.require_physical_time('resume_checkpoint_decode:completed')
            with self.lock:
                self.current['state'] = 'running'
            stop = float(config['duration_myr'])
            cadence = float(config['snapshot_myr'])
            next_output = resume_manifest['next_output_myr']
            while simulation.t < stop-1e-8 and not self.stop.is_set() and not self.pause_requested.is_set():
                dt = min(1., config['dt_myr'], next_output-simulation.t, stop-simulation.t)
                source_time = float(simulation.t)
                supplied = self._next_numerical_budget
                if supplied is None or supplied['dt_myr'] != dt:
                    raise ValueError('Controller interval length differs from native scheduler')
                with budget.installed(supplied):
                    self._open_numerical_budget(source_time, dt)
                    budget.require_physical_time('server_interval')
                    budget.record('server_interval', event='entry', source_time_myr=source_time, dt_myr=dt)
                    simulation.step(dt)
                    if not math.isclose(float(simulation.t)-source_time,dt,rel_tol=1e-10,abs_tol=1e-10):
                        raise ValueError('Interval did not complete its requested physical time')
                    budget.require_finalization_time('required_output')
                    if simulation.t >= next_output-1e-8:
                        self._append_snapshot(simulation)
                        next_output = min(stop,next_output+cadence)
                    # A pause during the interval uses this same immutable checkpoint;
                    # it never creates an off-cadence frame or a fresh output clock.
                    self._commit_checkpoint(simulation,next_output,
                        paused=bool(self.pause_requested.is_set() or self.stop.is_set()))
                    if simulation.t >= stop-1e-8 and self.current['frame_count']:
                        budget.require_finalization_time('final_output')
                        self.save_final(self.frame(self.current['frame_count']-1))
                    budget.require_finalization_time('required_output:completed')
                    self._finish_numerical_budget(source_time,dt,simulation)
                if initial_scope is not None:
                    initial_scope.__exit__(None,None,None)
                    initial_scope = None
            # Required terminal/pause output and status were closed inside _finish.
        except (Exception,budget.BudgetExhausted) as error:
            import traceback
            traceback.print_exc()
            active = dict(getattr(self,'_active_numerical_budget',{}))
            active.update(active=False,phase='incomplete',interval_and_required_output_completed=False,
                error=dict(type=type(error).__name__,message=str(error)))
            write_json(self.path()/'numerical-budget-active.json',active)
            with self.lock:
                self.current.update(state='error',can_resume=False,error=str(error),eta_seconds=None)
                self.current.update(self.timing.pause())
                self.persist()
        finally:
            if initial_scope is not None:
                initial_scope.__exit__(None,None,None)

    def _commit_checkpoint(self, simulation, next_output, paused=False):
        budget.require_finalization_time('checkpoint')
        active = self._active_numerical_budget
        directory = self.path()/'numerical-budget'/active['token']
        directory.mkdir(exist_ok=True,parents=True)
        path = directory/'checkpoint.npz'
        if path.exists():
            raise ValueError('Immutable checkpoint already exists')
        with self.lock:
            committed = native(self.current)
            committed.update(state='paused' if paused else 'interrupted',can_resume=True,
                checkpoint_version=CHECKPOINT_VERSION,checkpoint_time_myr=float(simulation.t),
                time_myr=float(simulation.t),integration_time_myr=float(simulation.t),
                progress=min(1.,simulation.t/simulation.config['duration_myr']),next_output_myr=next_output,
                accepted_checkpoint_file=str(path.relative_to(self.path())).replace('\\','/'),
                guardian_acceptance_directory=str(self._guardian_directory),
                guardian_control_sha256=self._guardian_control_sha256,
                guardian_acceptance_file=str(Path(self._guardian_directory)/('accepted-interval-'+active['token']+'.json')),
                eta_seconds=None)
            committed.pop('error',None)
        write_checkpoint(path,simulation,committed,self.compatibility())
        budget.require_finalization_time('checkpoint:written')
        self._pending_checkpoint = dict(path=str(path),manifest=committed)
        return committed

    def save_frame(self, index, snapshot):
        arrays = {key: np.asarray(snapshot[key], dtype=DTYPES[key]).reshape(-1) for key in FIELDS}
        expected = snapshot["width"] * snapshot["height"]
        if any(a.size != expected or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Engine returned an invalid or non-finite raster.")
        for key, dtype in OPTIONAL_GRID_DTYPES.items():
            if key not in snapshot:
                continue
            values = np.asarray(snapshot[key]).reshape(-1)
            integer = np.issubdtype(dtype, np.integer)
            if (values.size != expected or not np.isfinite(values).all()
                    or (integer and (np.any(values < 0) or np.any(values > np.iinfo(dtype).max)
                                     or np.any(values != np.floor(values))))
                    or (not integer and key != 'geometric_strain_percent'
                        and np.any(values < (-1 if key in ('rift_cooling_age_myr', 'deformation_weight', 'refinement_level') else 0)))
                    or (key == 'rift_damage' and np.any(values > 1))
                    or (key == 'geometric_strain_percent' and np.any(values <= -100.))
                    or (key == 'deformation_weight' and np.any((values != -1.) & ((values < 0.) | (values > 1.))))
                    or (key == 'refinement_level' and np.any((values != np.floor(values)) | (values > 3.)))
                    or (key in ('deformation_weight', 'refinement_level')
                        and np.any((values == -1.) != (arrays['crust'] == 0)))
                    or (key == 'rift_strength_relative' and np.any(values <= 0))):
                raise ValueError(f"Engine returned an invalid {key} raster.")
            arrays[key] = values.astype(dtype)
        if "domain" in arrays:
            rows = snapshot.get("domains", [])
            ids = [int(row["uid"]) for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("Domain identities must be unique in a frame.")
            by_uid = {int(row["uid"]): row for row in rows}
            unknown = arrays['domain'] == 0
            if np.any(unknown):
                # A reconstructed ocean owner can have diffuse support but no
                # nearby winning control cell. Domain 0 explicitly means that
                # its local domain is unresolved; it must never borrow the
                # name of a distant component belonging to the same plate.
                reported = snapshot.get('mesh_diagnostics', {}).get('unresolved_ocean_domain_queries')
                if (snapshot.get('mesh_version') != 1 or snapshot.get('owner_reconstruction_version') != 1
                        or reported != int(np.count_nonzero(unknown)) or np.any(arrays['crust'][unknown] != 0)):
                    raise ValueError('Unresolved domain 0 requires declared native ocean coverage.')
            known = ~unknown
            unique, inverse = np.unique(arrays["domain"][known], return_inverse=True)
            if not all(int(uid) in by_uid for uid in unique):
                raise ValueError("Domain raster identities need matching domain metadata.")
            owners = np.array([int(by_uid[int(uid)]["plate_id"]) for uid in unique], np.int32)
            if not np.array_equal(owners[inverse], arrays["plate"][known]):
                raise ValueError("Domain metadata must match the underlying plate owner.")
        traces = {key: np.asarray(snapshot[key], dtype=dtype)
                  for key, dtype in TRACE_DTYPES.items() if key in snapshot}
        if traces:
            trace_count = len(traces.get("trace_id", []))
            if not all(k in traces for k in ("trace_id", "trace_xyz", "trace_plate")):
                raise ValueError("Engine returned incomplete material histories.")
            if any(a.shape != ((trace_count, 3) if key == "trace_xyz" else (trace_count,))
                   or not np.isfinite(a).all() for key, a in traces.items()):
                raise ValueError("Engine returned invalid material histories.")
            if len(np.unique(traces["trace_id"])) != trace_count:
                raise ValueError("Material marker identities must be unique in a frame.")
            arrays.update(traces)
        arrays.update(mesh_history.arrays(snapshot))
        metadata = {k: v for k, v in snapshot.items()
                    if k not in FIELDS and k not in OPTIONAL_GRID_DTYPES and k not in TRACE_DTYPES and k not in mesh_history.DTYPES}
        metadata["index"] = index
        target = self.path() / f"frame_{index:04d}.npz"
        np.savez_compressed(target, **arrays)
        write_json(target.with_suffix(".json"), metadata)

    def frame(self, index, orientation=None, *, include_native=False):
        with self.lock:
            count = self.current["frame_count"]
            if index < 0:
                index += count
            if index < 0 or index >= count:
                raise ValueError("That history frame is not available yet.")
            target = self.path() / f"frame_{index:04d}.npz"
        metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(target, allow_pickle=False) as archive:
            metadata.update({k: archive[k] for k in FIELDS})
            metadata.update({k: archive[k] for k in OPTIONAL_GRID_DTYPES if k in archive.files})
            if include_native:
                metadata.update({k: archive[k] for k in mesh_history.DTYPES if k in archive.files})
        if any(normalize_orientation(orientation).values()):
            metadata = orient_frame(metadata, orientation)
        metadata['orientation'] = normalize_orientation(orientation)
        return metadata

    def display_fields(self, index, names, width, orientation=None):
        """Resample saved mesh fields for display. Never touches the running simulation."""
        with self.lock:
            count = self.current["frame_count"]
            if index < 0:
                index += count
            if index < 0 or index >= count:
                raise ValueError("That history frame is not available yet.")
            target = self.path() / f"frame_{index:04d}.npz"
        angles = normalize_orientation(orientation)
        matrix = rotation_matrix(angles) if any(angles.values()) else None
        with np.load(target, allow_pickle=False) as archive:
            return mesh_display.sample(archive, names, width=width, rotation=matrix)

    def native_frame(self, index, orientation=None):
        """Explicit geometry reader; ordinary browser frames stay compact."""
        result = self.frame(index, orientation, include_native=True)
        mesh_history.arrays(result)
        return result

    def _history_context(self, run_id=None):
        with self.lock:
            if run_id is not None and run_id != self.current.get("run_id"):
                raise ValueError("The selected experiment changed. Select a region again.")
            # Read immutable frames outside the lock. A running world can keep
            # saving, and a different loaded run cannot mix into this request.
            return self.path(), self.status()

    def record(self, run_id=None, orientation=None):
        path, manifest = self._history_context(run_id)
        return orient_metadata(read_record(path, manifest), orientation)

    def history(self, index, cell, run_id=None, orientation=None):
        path, manifest = self._history_context(run_id)
        orientation = normalize_orientation(orientation)
        source_cell = cell
        if any(orientation.values()):
            count = int(manifest['frame_count'])
            resolved = index if index >= 0 else index + count
            if not 0 <= resolved < count:
                raise ValueError('That history frame is not available yet.')
            header = json.loads((path / f'frame_{resolved:04d}.json').read_text(encoding='utf-8'))
            source_cell = int(inverse_cells([cell], header['width'], header['height'], orientation)[0])
        result = orient_metadata(read_history(path, manifest, index, source_cell), orientation)
        result['selection'].update(cell=cell, source_cell=source_cell)
        if any(orientation.values()):
            result['selection'].update(
                lon=-180 + (cell % header['width'] + .5)*360/header['width'],
                lat=90 - (cell // header['width'] + .5)*180/header['height'])
        result['orientation'] = orientation
        return result

    def save_final(self, frame):
        path = self.path()
        np.save(path / "elevation_m.npy", np.asarray(frame["elevation"], dtype=np.float32).reshape(frame["height"], frame["width"]))
        (path / "heightmap_16bit.png").write_bytes(png_bytes(frame))
        (path / "terrain_preview.png").write_bytes(png_bytes(frame, True))
        write_json(path / "boundaries.geojson", boundary_geojson(frame))
        write_json(path / "raster_metadata.json", self.raster_metadata(frame))

    @staticmethod
    def raster_metadata(frame):
        metadata = {"width": frame["width"], "height": frame["height"], "time_myr": frame["time_myr"],
                "projection": "equirectangular geographic longitude / latitude (degrees), spherical Earth radius 6371 km",
                "bounds": [-180, -90, 180, 90], "row_order": "north to south", "column_order": "west to east",
                "sampling": "pixel centers", "elevation_unit": "metres relative to model sea level",
                "heightmap_decode": "elevation_metres = uint16_pixel - 12000 (1 metre per code)",
                "heightmap_encode": "round(clip(elevation_metres + 12000, 0, 65535)); use NPY for full precision",
                "plate_codes": "integer plate ID", "crust_codes": {0: "ocean", 1: "continental", 2: "craton", 3: "island arc"},
                "boundary_codes": BOUNDARIES, "age_unit": "Myr", "model_status": "reduced-complexity, not predictive geodynamics"}
        if "domain" in frame:
            metadata["domain_codes"] = "Stable connected-domain UID; domain metadata identifies its underlying kinematic plate. A domain identity alone does not imply independent motion."
        metadata['orientation'] = normalize_orientation(frame.get('orientation'))
        return metadata

    def export_file(self, index, orientation=None):
        orientation = normalize_orientation(orientation)
        rotated = any(orientation.values())
        with self.lock:
            frame = self.frame(index, orientation)
            run_path = self.path()
            manifest = self.status()
            # Paths are captured while locked; an unrelated load cannot mix histories.
            history_paths = [(run_path / f"frame_{i:04d}.npz", run_path / f"frame_{i:04d}.json") for i in range(manifest["frame_count"])]
            auxiliary_sources = read_saved_auxiliary_sources(run_path, manifest.get('auxiliary_sources_sha256', {}))
        output = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode='w+b')
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            archive.writestr("heightmap_16bit.png", png_bytes(frame))
            archive.writestr("terrain_preview.png", png_bytes(frame, True))
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(frame["elevation"], dtype=np.float32).reshape(frame["height"], frame["width"]))
            archive.writestr("elevation_m.npy", buffer.getvalue())
            buffer = io.StringIO()
            np.savetxt(buffer, np.asarray(frame["elevation"]).reshape(frame["height"], frame["width"]), delimiter=",", fmt="%.3f")
            archive.writestr("elevation_m.csv", buffer.getvalue())
            archive.writestr("raster_metadata.json", json_bytes(self.raster_metadata(frame)))
            archive.writestr("boundaries.geojson", json_bytes(boundary_geojson(frame)))
            exported_manifest = {**manifest, 'orientation': orientation}
            if rotated:
                # Geographic authored controls follow the same output globe as
                # the initial artwork. Keep exact original config separately.
                exported_config = orient_metadata(json.loads((run_path/'config.json').read_text(encoding='utf-8')), orientation)
                exported_manifest['config'] = exported_config
            archive.writestr("manifest.json", json_bytes(exported_manifest))
            archive.writestr("events.json", json_bytes(orient_metadata(read_record(run_path, manifest), orientation)))
            archive.writestr("history_schema.json", json_bytes(history_schema()))
            if rotated:
                archive.writestr('config.json', json_bytes(exported_config))
                archive.write(run_path / 'config.json', 'source/config.json')
                archive.writestr('initial.json', json_bytes(orient_initial(json.loads((run_path / 'initial.json').read_text(encoding='utf-8')), orientation)))
                archive.write(run_path / 'initial.json', 'source/initial.json')
                archive.writestr('source/manifest.json', json_bytes(manifest))
            else:
                archive.write(run_path / 'config.json', 'config.json')
                archive.write(run_path / 'initial.json', 'initial.json')
            archive.writestr('orientation.json', json_bytes({
                'orientation': orientation, 'matrix': rotation_matrix(orientation),
                'convention': 'Original row-vector XYZ @ matrix = output XYZ. Yaw Z, then pitch Y, then roll X, degrees.',
                'source_run_id': manifest['run_id'],
                'orientation_sha256': hashlib.sha256(ORIENTATION_SOURCE).hexdigest(),
                'history': 'All exported grids, traces, angular velocities and spatial event coordinates share this orientation. Source run on disk is unchanged.',
                'sampling': 'Inverse spherical sampling; categorical fields nearest, continuous fields bilinear. Saved statistics describe the original spherical world.'}))
            for binary, metadata in history_paths:
                if rotated:
                    header = json.loads(metadata.read_text(encoding='utf-8'))
                    with np.load(binary, allow_pickle=False) as saved:
                        array_keys = saved.files
                        header.update({key: saved[key] for key in array_keys})
                    transformed = orient_frame(header, orientation)
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer, **{key: transformed.pop(key) for key in array_keys})
                    transformed['orientation'] = orientation
                    archive.writestr('history/' + binary.name, buffer.getvalue(), compress_type=zipfile.ZIP_STORED)
                    archive.writestr('history/' + metadata.name, json_bytes(transformed))
                else:
                    archive.write(binary, "history/" + binary.name, compress_type=zipfile.ZIP_STORED)
                    archive.write(metadata, "history/" + metadata.name)
            for name in ("SCIENCE.md", "README.md", "VALIDATION.md"):
                if (ROOT / name).exists():
                    archive.write(ROOT / name, name)
            # Capture the source that interprets this saved history for reproducibility.
            if (run_path / 'engine.py').exists():
                archive.write(run_path / 'engine.py', 'model/tectonics.py')
            for name, payload in auxiliary_sources.items():
                archive.writestr('model/' + name, payload)
            archive.write(ROOT / 'requirements.txt', 'model/requirements.txt')
            archive.writestr('view/orientation.py', ORIENTATION_SOURCE)
        output.seek(0)
        return output

    def export(self, index, orientation=None):
        """Convenience bytes API for small exports; HTTP uses the streamed file."""
        with self.export_file(index, orientation) as output:
            return output.read()


class Handler(SimpleHTTPRequestHandler):
    manager: SimulationManager
    terrain_manager: TerrainManager | None = None
    gospl_manager: GosplManager | None = None
    gospl_init_lock = threading.Lock()
    terrain_init_lock = threading.Lock()

    def gospl_jobs(self):
        with type(self).gospl_init_lock:
            if type(self).gospl_manager is None:
                type(self).gospl_manager = GosplManager(ROOT/'output'/'gospl')
            return type(self).gospl_manager

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def log_message(self, fmt, *args):
        if not any(route in str(args[0] if args else "") for route in ('/api/status', '/api/terrain/status')):
            super().log_message(fmt, *args)

    def terrain_jobs(self):
        with self.terrain_init_lock:
            if type(self).terrain_manager is None:
                type(self).terrain_manager = TerrainManager(ROOT / 'output' / 'terrain')
            return type(self).terrain_manager

    def send_path(self, path, mime='application/octet-stream', name=None):
        with path.open('rb') as source:
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if name:
                self.send_header('Content-Disposition', f'attachment; filename="{name}"')
            self.end_headers()
            shutil.copyfileobj(source, self.wfile, length=256*1024)

    def send_display(self, sampled):
        """Binary reply: 4-byte header length, JSON header, then the raw field arrays.

        Fields are sent as float32/int32/uint8 rather than JSON numbers because a 768x384
        grid is 294,912 values per field; as text that would be megabytes of parsing per
        frame for no gain in precision that a display can show.
        """
        blocks, header = [], {'width': sampled['width'], 'height': sampled['height'], 'fields': []}
        offset = 0
        for name, values in sampled['fields'].items():
            if values.dtype == np.uint8:
                kind, data = 'uint8', np.ascontiguousarray(values, np.uint8)
            elif np.issubdtype(values.dtype, np.integer):
                kind, data = 'int32', np.ascontiguousarray(values, np.int32)
            else:
                kind, data = 'float32', np.ascontiguousarray(values, np.float32)
            payload = data.tobytes()
            header['fields'].append({'name': name, 'type': kind, 'offset': offset, 'count': int(data.size)})
            # Pad each block to 4 bytes: the reader builds Int32Array/Float32Array views over
            # this buffer, and those throw on an unaligned byte offset.
            padding = -len(payload) % 4
            blocks.append(payload + b'\0' * padding)
            offset += len(payload) + padding
        encoded = json.dumps(header).encode('utf-8')
        # Pad the header too: the blocks start at 4 + len(header), so an odd header length
        # would misalign every field offset and the reader's typed-array views would throw.
        encoded += b' ' * (-len(encoded) % 4)
        body = len(encoded).to_bytes(4, 'little') + encoded + b''.join(blocks)
        return self.send_data(body, 'application/octet-stream')

    def send_data(self, data, mime="application/json", name=None, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if name:
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        self.wfile.write(data)

    def trusted_request(self):
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        if origin:
            expected = f"http://{self.headers.get('Host')}"
            if origin != expected:
                return False
        return True

    def send_archive(self, index, orientation=None):
        with self.manager.export_file(index, orientation) as output:
            output.seek(0, os.SEEK_END)
            length = output.tell()
            output.seek(0)
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(length))
            self.send_header('Content-Disposition', 'attachment; filename="deep-time-history.zip"')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            shutil.copyfileobj(output, self.wfile, length=256 * 1024)

    def do_GET(self):
        if not self.trusted_request():
            return self.send_data(json_bytes({"error": "Local requests only."}), status=403)
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path
        if not route.startswith("/api/"):
            return super().do_GET()
        try:
            if route == "/api/health":
                return self.send_data(json_bytes({"app": "deep-time", "version": 4, "root": str(ROOT),
                                                 "shutting_down": self.manager.shutting_down}))
            if route == "/api/config":
                return self.send_data(json_bytes({"defaults": fresh_world_defaults()}))
            if route == "/api/status":
                return self.send_data(json_bytes(self.manager.status()))
            if route == '/api/workers':
                return self.send_data(json_bytes(self.manager.workers_status()))
            if route == "/api/native-frame":
                return self.send_data(json_bytes(self.manager.native_frame(int(query.get('index', [-1])[0]))))
            if route == "/api/initial":
                if 'preset' in query:
                    config = validate_config({key: query[key][0] for key in DEFAULT_CONFIG if key in query})
                    return self.send_data(json_bytes(make_initial(config, preset=query['preset'][0])))
                with self.manager.lock:
                    initial_path = self.manager.path() / "initial.json"
                    return self.send_data(initial_path.read_bytes())
            if route == "/api/runs":
                return self.send_data(json_bytes({"runs": self.manager.list_runs()}))
            if route.startswith('/api/gospl/'):
                jobs=self.gospl_jobs()
                job_id=query.get('job_id',[None])[0]
                if route=='/api/gospl/status':return self.send_data(json_bytes(jobs.status(job_id)))
                if route=='/api/gospl/list':return self.send_data(json_bytes({'jobs':jobs.list_jobs()}))
                if route=='/api/gospl/download':return self.send_path(jobs.download(job_id),'application/zip','gospl-history.zip')
            if route.startswith('/api/terrain/'):
                jobs = self.terrain_jobs()
                job_id = query.get('job_id', [None])[0]
                if route == '/api/terrain/status':
                    return self.send_data(json_bytes(jobs.status(job_id)))
                if route == '/api/terrain/list':
                    return self.send_data(json_bytes({'jobs': jobs.list_jobs()}))
                if route == '/api/terrain/preview':
                    path, _ = jobs.completed_path(job_id, 'preview.png')
                    return self.send_path(path, 'image/png')
                if route == '/api/terrain/download':
                    fmt = query.get('format', ['zip'])[0]
                    path = jobs.download(job_id, fmt)
                    mime = {'png': 'image/png', 'zip': 'application/zip', 'npy': 'application/octet-stream'}[fmt]
                    return self.send_path(path, mime, path.name)
                if route == '/api/terrain/tile':
                    return self.send_data(jobs.tile(job_id, query.get('x', [-1])[0], query.get('y', [-1])[0]), 'image/png')
                if route == '/api/terrain/sample':
                    return self.send_data(json_bytes(jobs.sample(job_id, query.get('x', [-1])[0], query.get('y', [-1])[0])))
            index = int(query.get("frame", query.get("index", ["-1"]))[0])
            orientation = normalize_orientation(json.loads(query.get('orientation', ['null'])[0]))
            if route == "/api/record":
                return self.send_data(json_bytes(self.manager.record(query.get("run_id", [None])[0], orientation)))
            if route == "/api/history":
                cell = int(query.get("cell", ["-1"])[0])
                return self.send_data(json_bytes(self.manager.history(index, cell, query.get("run_id", [None])[0], orientation)))
            if route == "/api/frame":
                return self.send_data(json_bytes(self.manager.frame(index, orientation)))
            if route == "/api/display":
                # Display-only: the saved native mesh resampled onto a finer grid. Separate from
                # /api/frame because only some fields exist per mesh face -- mixing resolutions
                # in one payload would make the viewer index the 192x96 fields with the wrong
                # stride. The caller uses these for the map image and keeps /api/frame for
                # inspection, painting and every layer stored only on the display grid.
                names = [n for n in query.get('fields', ['elevation'])[0].split(',') if n]
                width = int(query.get('width', [str(mesh_display.DEFAULT_WIDTH)])[0])
                return self.send_display(self.manager.display_fields(index, names, width, orientation))
            if route in ("/api/heightmap", "/api/preview"):
                colored = route.endswith("preview")
                return self.send_data(png_bytes(self.manager.frame(index, orientation), colored), "image/png", "terrain_preview.png" if colored else "heightmap_16bit.png")
            if route == "/api/export":
                return self.send_archive(index, orientation)
            self.send_data(json_bytes({"error": "Unknown endpoint."}), status=404)
        except (ValueError, OSError, KeyError) as exc:
            self.send_data(json_bytes({"error": str(exc)}), status=400)

    def do_POST(self):
        if not self.trusted_request():
            return self.send_data(json_bytes({"error": "Local requests only."}), status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 16 * 1024 * 1024:
                raise ValueError("Request exceeds the 16 MB limit.")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object.")
            route = urlparse(self.path).path
            if route == '/api/workers':
                return self.send_data(json_bytes(self.manager.set_workers(body)))
            if route == '/api/orient-initial':
                initial = body.get('initial')
                if not isinstance(initial, dict):
                    raise ValueError('Choose a starting world to rotate.')
                config = validate_config({key: initial.get(key) for key in ('width', 'height')})
                initial = validate_initial(initial, config)
                return self.send_data(json_bytes(orient_initial(initial, body.get('orientation'))))
            if route == "/api/initial":
                config = validate_config(body.get("config", {}))
                return self.send_data(json_bytes(make_initial(config, preset=body.get('preset', 'pangaea'))))
            if route == "/api/run":
                return self.send_data(json_bytes(self.manager.start(body.get("config", {}), body.get("initial"))))
            if route=='/api/gospl/start':
                if self.manager.shutting_down:raise ValueError('The app is shutting down. Reopen it before exporting.')
                return self.send_data(json_bytes(self.gospl_jobs().start(self.manager,**{key:body[key] for key in
                    ('run_id','start_index','end_index','subdivisions','dt_years','rainfall_m_yr','orientation') if key in body})))
            if route=='/api/gospl/cancel':return self.send_data(json_bytes(self.gospl_jobs().cancel(body.get('job_id'))))
            if route == '/api/gospl/results/inspect':
                return self.send_data(json_bytes(inspect_result(body.get('path'))))
            if route == '/api/gospl/results/browse':
                return self.send_data(json_bytes({'path': choose_gospl_result()}))
            if route == '/api/terrain/gospl':
                if self.manager.shutting_down:
                    raise ValueError('The app is shutting down. Reopen it before importing.')
                return self.send_data(json_bytes(self.terrain_jobs().start_gospl_result(
                    body.get('path'), epoch_index=body.get('epoch_index'),
                    width=body.get('width', 8192), detail=body.get('detail', 0.))))
            if route == '/api/terrain':
                return self.send_data(json_bytes(self.terrain_jobs().start(self.manager, body.get('frame', -1),
                    run_id=body.get('run_id'), width=body.get('width', 8192), detail=body.get('detail', 1.), orientation=body.get('orientation'),
                    reconstruct_margins=body.get('reconstruct_margins', False))))
            if route == '/api/terrain/cancel':
                return self.send_data(json_bytes(self.terrain_jobs().cancel(body.get('job_id'))))
            if route == "/api/cancel":
                return self.send_data(json_bytes(self.manager.cancel(body.get('run_id'))))
            if route == "/api/pause":
                return self.send_data(json_bytes(self.manager.pause(body.get('run_id'))))
            if route == "/api/resume":
                return self.send_data(json_bytes(self.manager.resume(body.get('run_id'))))
            if route == '/api/branch':
                return self.send_data(json_bytes(self.manager.branch(body.get('run_id'), body.get('world_design'))))
            if route == "/api/shutdown":
                self.manager.prepare_shutdown()
                if self.terrain_manager is not None:
                    self.terrain_manager.cancel()
                if self.gospl_manager is not None:
                    self.gospl_manager.cancel()
                self.send_data(json_bytes({"ok": True, "shutting_down": True}))
                def finish_shutdown():
                    worker = self.manager.worker
                    if worker and worker.is_alive():
                        worker.join()
                    self.server.shutdown()
                threading.Thread(target=finish_shutdown, daemon=True).start()
                return
            if route == "/api/load":
                return self.send_data(json_bytes(self.manager.load(body.get("run_id"))))
            self.send_data(json_bytes({"error": "Unknown endpoint."}), status=404)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.send_data(json_bytes({"error": str(exc)}), status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument('--workers', type=int, help='Concurrent workers, from 1 through this host\'s physical core count')
    parser.add_argument("--open", action="store_true", help="Open the local user interface in your browser")
    args = parser.parse_args()
    Handler.manager = SimulationManager(workers=args.workers)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    Handler.terrain_manager = TerrainManager(ROOT / 'output' / 'terrain')
    Handler.gospl_manager = GosplManager(ROOT / 'output' / 'gospl')
    print(f"Deep Time is running at http://127.0.0.1:{args.port}", flush=True)
    print(f"Saved histories: {RUNS}", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        Handler.manager.prepare_shutdown()
    finally:
        Handler.manager.prepare_shutdown()
        if Handler.manager.worker and Handler.manager.worker.is_alive():
            Handler.manager.worker.join()
        Handler.manager.parallel.close()
        Handler.terrain_manager.cancel()
        if Handler.terrain_manager.worker and Handler.terrain_manager.worker.is_alive():
            Handler.terrain_manager.worker.join(timeout=30)
        if Handler.gospl_manager is not None:
            Handler.gospl_manager.cancel()
            if Handler.gospl_manager.worker and Handler.gospl_manager.worker.is_alive():
                Handler.gospl_manager.worker.join(timeout=30)
        server.server_close()


if __name__ == "__main__":
    main()
