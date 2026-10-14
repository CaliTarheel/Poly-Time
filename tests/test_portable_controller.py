"""Controller recovery tests use synthetic processes, never a simulation run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portable_controller as controller


class FakeTree:
    def __init__(self, *, refuse=False, root_dead=False):
        self.pid = 1234
        self.alive = not root_dead
        self.members = 1
        self.refuse = refuse
        self.closed = False

    def poll(self):
        return None if self.alive else 1

    def active_processes(self):
        return self.members

    def terminate_tree(self):
        if self.refuse:
            raise OSError('Access denied: owned process is still alive')
        self.alive = False
        self.members = 0

    def close(self):
        if self.alive or self.members:
            raise RuntimeError('Cannot detach a live process tree')
        self.closed = True


class ControllerTests(unittest.TestCase):
    def kill_result(self, code=0):
        return Mock(return_value=subprocess.CompletedProcess(['taskkill'], code, 'partial success', ''))

    def test_original_called_process_error_does_not_hide_successful_job_cleanup(self):
        tree = FakeTree()
        command = Mock(side_effect=subprocess.CalledProcessError(255, ['taskkill']))
        result = controller.stop_owned_tree(tree, run_command=command)
        self.assertTrue(result['termination_verified'])
        self.assertIn('255', result['taskkill_error'])
        self.assertEqual(tree.members, 0)

    def test_nonzero_taskkill_is_diagnostic_and_descendants_are_still_terminated(self):
        tree = FakeTree()
        result = controller.stop_owned_tree(tree, run_command=self.kill_result(255))
        self.assertTrue(result['termination_verified'])
        self.assertEqual(result['taskkill_returncode'], 255)
        self.assertEqual(tree.members, 0)

    def test_dead_root_never_sends_pid_based_kill_and_still_cleans_owned_descendants(self):
        tree = FakeTree(root_dead=True)
        command = self.kill_result()
        result = controller.stop_owned_tree(tree, run_command=command)
        command.assert_not_called()
        self.assertTrue(result['termination_verified'])
        self.assertEqual(tree.members, 0)

    def test_zero_taskkill_is_not_evidence_that_the_tree_exited(self):
        tree = FakeTree(refuse=True)
        result = controller.stop_owned_tree(tree, run_command=self.kill_result())
        self.assertFalse(result['termination_verified'])
        self.assertIn('Access denied', result['blocker'])

    def test_wait_is_bounded_when_termination_has_not_completed(self):
        tree = FakeTree()
        tree.terminate_tree = lambda: None
        clock = iter((0., 0., 11.))
        result = controller.stop_owned_tree(tree, run_command=self.kill_result(),
                                             clock=lambda: next(clock), sleep=lambda _: None)
        self.assertFalse(result['termination_verified'])
        self.assertIn('bounded wait', result['blocker'])

    def test_query_failure_blocks_recovery_even_when_root_is_dead(self):
        tree = FakeTree(root_dead=True)
        tree.active_processes = Mock(side_effect=OSError('Cannot verify job membership'))
        result = controller.stop_owned_tree(tree, run_command=self.kill_result())
        self.assertFalse(result['termination_verified'])
        self.assertIn('Cannot verify', result['blocker'])

    def test_recovery_publishes_receipt_before_launch_and_keeps_workers(self):
        tree = FakeTree()
        published = []
        recovered = type('Recovered', (), {'pid': 5678})()
        def launch(port, workers):
            self.assertTrue(tree.closed)
            self.assertTrue(published[0][1]['termination_verified'])
            self.assertEqual((port, workers), (8768, 2))
            return recovered
        marker = dict(active=True, run_id='fixture', source_time_myr=183., deadline=1900.)
        result = controller.recover_view(tree, marker, 8768, 2, launch=launch,
            stop=lambda value: controller.stop_owned_tree(value, run_command=self.kill_result(255)),
            publish=lambda path, value: published.append((path, dict(value))))
        self.assertIs(result, recovered)
        receipt = next(value for path, value in reversed(published) if path == controller.FAILURE)
        self.assertFalse(receipt['automatic_resume'])
        self.assertEqual(receipt['interval'], marker)
        self.assertEqual(marker['deadline'], 1900.)

    def test_refused_termination_never_starts_a_second_server(self):
        tree = FakeTree(refuse=True)
        launch, publish = Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, 'Controller recovery blocked'):
            controller.recover_view(tree, {'run_id': 'fixture'}, 8768, launch=launch,
                stop=lambda value: controller.stop_owned_tree(value, run_command=self.kill_result(255)),
                publish=publish)
        launch.assert_not_called()
        self.assertFalse(tree.closed)
        self.assertFalse(publish.call_args.args[1]['termination_verified'])

    def test_atomic_receipt_is_complete_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'receipt.json'
            controller.write_control_json(target, {'event': 'deadline'})
            self.assertEqual(json.loads(target.read_text()), {'event': 'deadline'})
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['receipt.json'])

    def test_failed_view_launch_is_recorded_without_automatic_resume(self):
        tree = FakeTree()
        publish = Mock()
        with self.assertRaisesRegex(RuntimeError, 'viewer failed to launch'):
            controller.recover_view(tree, {'run_id': 'fixture'}, 8768,
                launch=Mock(side_effect=OSError('fixture spawn refused')),
                stop=lambda value: controller.stop_owned_tree(value, run_command=self.kill_result()),
                publish=publish)
        self.assertTrue(tree.closed)
        receipt = publish.call_args.args[1]
        self.assertTrue(receipt['termination_verified'])
        self.assertIn('fixture spawn refused', receipt['blocker'])
        self.assertFalse(receipt['automatic_resume'])

    def test_fsync_failure_preserves_previous_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'receipt.json'
            controller.write_control_json(target, {'event': 'previous'})
            with patch.object(controller.os, 'fsync', side_effect=OSError('fixture flush failed')):
                with self.assertRaisesRegex(OSError, 'fixture flush failed'):
                    controller.write_control_json(target, {'event': 'new'})
            self.assertEqual(json.loads(target.read_text()), {'event': 'previous'})
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['receipt.json'])

    @unittest.skipUnless(os.name == 'nt', 'Windows Job Object containment')
    def test_real_windows_owned_child_and_grandchild_exit_before_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / 'ready.json'
            script = ('import json,subprocess,sys,time; '
                      'from pathlib import Path; '
                      'child=subprocess.Popen([sys.executable,"-B","-c","import time; time.sleep(120)"]); '
                      'Path(sys.argv[1]).write_text(json.dumps({"pid":child.pid})); time.sleep(120)')
            # A hidden Start-Process launch may have no console stdin.
            with patch.object(sys, 'stdin', None):
                tree = controller.WindowsOwnedServer([sys.executable, '-B', '-c', script, str(ready)])
            try:
                deadline = time.monotonic() + 10.
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(ready.exists(), 'Synthetic child did not start')
                self.assertGreaterEqual(tree.active_processes(), 2)
                # Reproduce a partial taskkill failure without killing any
                # arbitrary PID: the ownership job must do the actual cleanup.
                result = controller.stop_owned_tree(tree, run_command=self.kill_result(255))
                self.assertTrue(result['termination_verified'], result)
                self.assertEqual(tree.active_processes(), 0)
                self.assertIsNotNone(tree.poll())
            finally:
                if tree.active_processes():
                    tree.terminate_tree()
                    deadline = time.monotonic() + 10.
                    while tree.active_processes() and time.monotonic() < deadline:
                        time.sleep(.05)
                tree.close()


if __name__ == '__main__':
    unittest.main()
