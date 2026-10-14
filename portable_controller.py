"""Local fresh-run controller; independent of the archived production guardian.

Launch with --port 8768. A separate parent process enforces the same absolute
1800 seconds/Myr deadline used cooperatively by the simulation and its outputs.
Windows Job Object containment prevents orphan workers. A timeout recovers only
the saved-run viewer; it never resumes the simulation automatically. --workers
is a performance control forwarded to the server (whose default is up to two workers).
"""
import argparse
import copy
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARKER = ROOT / 'output' / 'portable-controller-active.json'
FAILURE = ROOT / 'output' / 'portable-controller-failure.json'


def write_control_json(path, value):
    """Publish a receipt atomically without importing the numerical server."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class WindowsOwnedServer:
    """Contain exactly this child and its descendants in a Windows Job Object.

    Assignment happens while the primary thread is suspended, before the child
    can spawn. No PID-based enumeration or unrelated-process termination is used.
    A job remains owned until every member has exited, including after a partial
    taskkill result. The parent intentionally fails closed on query/API errors.
    """
    def __init__(self, command):
        import _winapi
        from ctypes import wintypes
        self.api = _winapi
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
            ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
            ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.job = self.kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = None
        thread = None
        inherited = []
        try:
            class Limits(ctypes.Structure):
                _fields_ = [('user_times', ctypes.c_int64 * 2), ('flags', wintypes.DWORD),
                            ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t),
                            ('active_limit', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                            ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]
            class ExtendedLimits(ctypes.Structure):
                _fields_ = [('basic', Limits), ('io', ctypes.c_uint64 * 6),
                            ('memory', ctypes.c_size_t * 4)]
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits),
                                                       ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            # Preserve the launcher's redirected logs and inherit only these
            # duplicates, never unrelated open handles from the controller.
            import msvcrt
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESTDHANDLES
            process = _winapi.GetCurrentProcess()
            for index, stream in enumerate((sys.stdin, sys.stdout, sys.stderr)):
                try:
                    handle = msvcrt.get_osfhandle(stream.fileno())
                except (AttributeError, OSError, ValueError):
                    handle = -1
                missing = open(os.devnull, 'rb' if index == 0 else 'ab') if handle < 0 else None
                try:
                    if missing is not None:
                        handle = msvcrt.get_osfhandle(missing.fileno())
                    inherited.append(_winapi.DuplicateHandle(process, handle, process, 0, True,
                                                              _winapi.DUPLICATE_SAME_ACCESS))
                finally:
                    if missing is not None:
                        missing.close()
            startup.hStdInput, startup.hStdOutput, startup.hStdError = inherited
            startup.lpAttributeList = {'handle_list': inherited}
            self.handle, thread, self.pid, _ = _winapi.CreateProcess(
                None, subprocess.list2cmdline(command), None, None, True,
                0x00000004 | 0x08000000, None, str(ROOT), startup)
            if not self.kernel.AssignProcessToJobObject(self.job, self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            if self.kernel.ResumeThread(thread) == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            if self.handle is not None:
                _winapi.TerminateProcess(self.handle, 1)
                _winapi.WaitForSingleObject(self.handle, 5000)
                _winapi.CloseHandle(self.handle)
                self.handle = None
            _winapi.CloseHandle(self.job)
            self.job = None
            raise
        finally:
            if thread is not None:
                _winapi.CloseHandle(thread)
            for handle in inherited:
                _winapi.CloseHandle(handle)

    def poll(self):
        waited = self.api.WaitForSingleObject(self.handle, 0)
        if waited == self.api.WAIT_TIMEOUT:
            return None
        if waited != self.api.WAIT_OBJECT_0:
            raise OSError('Cannot verify owned child exit.')
        return self.api.GetExitCodeProcess(self.handle)

    def active_processes(self):
        class Accounting(ctypes.Structure):
            _fields_ = [('times', ctypes.c_int64 * 4), ('faults', ctypes.c_uint32),
                        ('total', ctypes.c_uint32), ('active', ctypes.c_uint32),
                        ('terminated', ctypes.c_uint32)]
        data = Accounting()
        if not self.kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(data),
                                                     ctypes.sizeof(data), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(data.active)

    def terminate_tree(self):
        if not self.kernel.TerminateJobObject(self.job, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        # Never detach a job with a possibly surviving process tree.
        if self.active_processes() != 0:
            raise RuntimeError('Owned process tree is still alive; recovery is blocked.')
        self.api.CloseHandle(self.handle)
        self.api.CloseHandle(self.job)
        self.handle = self.job = None


def launch_server(port, workers=None):
    if os.name != 'nt':
        raise RuntimeError('This local controller requires Windows process-tree containment.')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--port', str(port)]
    if workers is not None:
        command += ['--workers', str(workers)]
    return WindowsOwnedServer(command)


def stop_owned_tree(child, *, run_command=subprocess.run, clock=time.monotonic,
                    sleep=time.sleep, timeout=10.):
    """Record taskkill diagnostics; only process handles/job state prove exit."""
    result = dict(child_pid=child.pid, termination_verified=False)
    try:
        if child.poll() is None:
            killed = run_command(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                 check=False, capture_output=True, text=True, timeout=5.)
            result.update(taskkill_returncode=killed.returncode,
                          taskkill_stdout=killed.stdout, taskkill_stderr=killed.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        result['taskkill_error'] = str(error)
    try:
        # A nonzero taskkill result can accompany successful root termination.
        # Job membership, unlike PID lists, also covers surviving descendants.
        if child.active_processes():
            child.terminate_tree()
        deadline = clock() + timeout
        while child.poll() is None or child.active_processes():
            if clock() >= deadline:
                raise RuntimeError('Owned process tree did not exit within the bounded wait.')
            sleep(.05)
        result['termination_verified'] = True
    except (OSError, RuntimeError) as error:
        result['blocker'] = str(error)
    return result


def recover_view(child, marker, port, workers=None, *, launch=launch_server,
                 stop=stop_owned_tree, publish=write_control_json):
    """Recover the viewer only; simulation resume always remains an API action."""
    receipt = dict(kind='portable_controller_deadline_failure', version=1,
                   observed_at_utc=datetime.now(timezone.utc).isoformat(),
                   run_id=marker.get('run_id'), interval=dict(marker),
                   error='Interval deadline exceeded; simulation recovery is required.',
                   automatic_resume=False,
                   **stop(child))
    archive = FAILURE.parent / 'controller-failures' / (uuid.uuid4().hex + '.json')
    publish(archive, receipt)
    publish(FAILURE, receipt)
    if not receipt['termination_verified']:
        raise RuntimeError('Controller recovery blocked: ' + receipt.get('blocker', 'termination unverified'))
    child.close()
    publish(MARKER, dict(active=False, error=receipt['error'], run_id=receipt['run_id']))
    # serve() loads and validates the saved checkpoint, but never resumes it.
    try:
        return launch(port, workers)
    except Exception as error:
        receipt['blocker'] = 'Saved-run viewer failed to launch: ' + str(error)
        publish(archive, receipt)
        publish(FAILURE, receipt)
        raise RuntimeError(receipt['blocker']) from error


def serve(port, workers=None):
    import budget
    import coarse_history
    import server

    class PortableManager(server.SimulationManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                failure = json.loads(FAILURE.read_text(encoding='utf-8'))
                source = failure['interval'].get('source_time_myr', -1.)
                if (failure.get('run_id') == self.current.get('run_id')
                        and self.current.get('state') in ('paused', 'interrupted')
                        and self.current.get('time_myr', 0.) <= source):
                    self.current['error'] = failure['error']
                    self.current['controller_failure'] = failure
            except (OSError, ValueError, KeyError, TypeError):
                pass

        def _guarded_checkpoint(self, result, run_id):
            if result.get('controller') == 'portable_fresh_v1':
                return None
            return super()._guarded_checkpoint(result, run_id)

        def commit(self, simulation, next_output):
            budget.require_finalization_time('portable_checkpoint')
            saved = copy.deepcopy(self.current)
            saved.update(controller='portable_fresh_v1', state='paused', can_resume=True,
                         checkpoint_version=server.CHECKPOINT_VERSION,
                         checkpoint_time_myr=float(simulation.t), time_myr=float(simulation.t),
                         integration_time_myr=float(simulation.t),
                         progress=min(1., simulation.t/simulation.config['duration_myr']),
                         next_output_myr=next_output)
            server.write_checkpoint(self.path()/'checkpoint.npz', simulation, saved, self.compatibility())
            budget.require_finalization_time('portable_checkpoint_written')
            self.current.update(checkpoint_version=server.CHECKPOINT_VERSION,
                                checkpoint_time_myr=float(simulation.t), next_output_myr=next_output,
                                time_myr=float(simulation.t), integration_time_myr=float(simulation.t),
                                progress=min(1., simulation.t/simulation.config['duration_myr']))

        def _run_owned(self, config, initial, resuming=False):
            self.current['controller'] = 'portable_fresh_v1'
            durable = copy.deepcopy(self.current)
            try:
                server.write_json(MARKER, dict(active=True, deadline=time.perf_counter()+1800,
                                              run_id=self.current['run_id'], phase='initialization'))
                if resuming:
                    simulation, saved = server.read_checkpoint(self.path()/self._resume_checkpoint_name,
                                                               self.compatibility(), server.Simulation)
                    next_output = saved['next_output_myr']
                    with self.lock:
                        self.current['state'] = 'running'
                else:
                    base = dict(config, history_mode=None)
                    simulation = server.Simulation(base, initial)
                    precursor = self.path()/'fresh-source.npz'
                    server.write_checkpoint(precursor, simulation, self.current, self.compatibility())
                    coarse_history.activate(simulation, source_receipt=dict(kind='portable_fresh_initial_state',
                        run_id=self.current['run_id'], time_myr=0,
                        checkpoint_file=str(precursor.resolve()),
                        checkpoint_sha256=hashlib.sha256(precursor.read_bytes()).hexdigest()))
                    next_output = min(config['duration_myr'], config['snapshot_myr'])
                    self._append_snapshot(simulation)
                    self.commit(simulation, next_output)
                durable = copy.deepcopy(self.current)
                self.persist()
                server.write_json(MARKER, dict(active=False))
                stop = float(config['duration_myr'])
                while simulation.t < stop-1e-8 and not self.stop.is_set() and not self.pause_requested.is_set():
                    dt = min(1., config['dt_myr'], next_output-simulation.t, stop-simulation.t)
                    with budget.step_budget(dt) as clock:
                        server.write_json(MARKER, dict(active=True, deadline=clock.deadline,
                            run_id=self.current['run_id'], source_time_myr=float(simulation.t)))
                        source = float(simulation.t)
                        simulation.step(dt)
                        if abs(simulation.t-source-dt) > 1e-8:
                            raise ValueError('Physical interval did not finish')
                        if simulation.t >= next_output-1e-8:
                            self._append_snapshot(simulation)
                            next_output = min(stop, next_output+config['snapshot_myr'])
                        if simulation.t >= stop-1e-8:
                            self.save_final(self.frame(self.current['frame_count']-1))
                        self.commit(simulation, next_output)
                        self.current.update(self.timing.update(simulation.t))
                        self.persist()
                        durable = copy.deepcopy(self.current)
                        server.write_json(MARKER, dict(active=False))
                self.current.update(state='complete' if simulation.t >= stop-1e-8 else 'paused',
                                    can_resume=simulation.t < stop-1e-8)
                self.current.update(self.timing.finish() if simulation.t >= stop-1e-8 else self.timing.pause())
                self.persist()
            except (Exception, budget.BudgetExhausted) as error:
                import traceback
                traceback.print_exc()
                self.current = durable
                self.current.update(state='error', error=str(error), can_resume=False)
                self.persist()
                server.write_json(MARKER, dict(active=False, error=str(error)))

    server.SimulationManager = PortableManager
    sys.argv = [sys.argv[0], '--port', str(port)]
    if workers is not None:
        sys.argv += ['--workers', str(workers)]
    server.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--workers', type=int, help='Concurrent workers; defaults to the server default.')
    args = parser.parse_args()
    if args.child:
        serve(args.port, args.workers)
        return
    MARKER.parent.mkdir(exist_ok=True)
    write_control_json(MARKER, dict(active=False))
    child = launch_server(args.port, args.workers)
    try:
        while child.poll() is None:
            try:
                marker = json.loads(MARKER.read_text(encoding='utf-8'))
                if marker.get('active') and time.perf_counter() >= marker['deadline']:
                    child = recover_view(child, marker, args.port, args.workers)
                    print('Interval deadline exceeded; saved-run viewer recovered without resuming.', flush=True)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(.25)
    finally:
        if child.handle is not None:
            stopped = stop_owned_tree(child)
            if not stopped['termination_verified']:
                raise RuntimeError('Owned process tree cleanup blocked: ' + stopped.get('blocker', 'unverified'))
            child.close()


if __name__ == '__main__':
    main()
