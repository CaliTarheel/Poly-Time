"""Run the unittest suite one module per process and compare the outcome with the known failures.

    python -B ci/run_tests.py [--shard I/N] [--timeout SECONDS] [--results FILE] [--logs DIR] [--survey]

Every module of tests/ runs in its own interpreter, so a module that hangs, crashes or cannot be
imported costs only itself. The run fails when a test fails that ci/known_failures.json does not
list; a listed test that fails is reported and tolerated, and one that passes is reported so the
list can be shortened. --survey records outcomes and never fails.

Standard library only: this file must run before any dependency of the engine is known to work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN = ROOT / 'ci' / 'known_failures.json'
DURATIONS = ROOT / 'ci' / 'test_durations.json'
FAILING = ('failed', 'error', 'timeout', 'crash')
RANK = dict(passed=0, skipped=0, expected_failure=0, failed=2, error=3)


def identity(test):
    """A stable name for a test, a failed import, or a failed class or module fixture."""
    name = test.id()
    if name.startswith('unittest.loader._FailedTest.'):
        return 'tests.' + name.rsplit('.', 1)[1] + '::import'
    if ' (' in name and name.endswith(')'):          # "setUpClass (tests.test_x.Class)"
        fixture, owner = name[:-1].split(' (', 1)
        return owner + '::' + fixture
    return name


def brief(error):
    lines = [line for line in ''.join(traceback.format_exception_only(error[0], error[1])).splitlines() if line.strip()]
    return lines[-1].strip()[:300] if lines else error[0].__name__


def detail(error):
    """The traceback, for the results file: enough to diagnose a failure from the artifact alone."""
    return ''.join(traceback.format_exception(*error))[-6000:]


class Recorder(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.outcomes = {}
        self.clock = {}

    def record(self, test, outcome, message='', error=None):
        key = identity(test)
        previous = self.outcomes.get(key)
        # A subtest that failed is not undone by its parent finishing.
        if previous is None or RANK[outcome] > RANK[previous['outcome']]:
            self.outcomes[key] = dict(outcome=outcome, message=message, seconds=(previous or {}).get('seconds', 0.))
            if error is not None:
                self.outcomes[key]['traceback'] = detail(error)

    def startTest(self, test):
        super().startTest(test)
        self.clock[identity(test)] = time.perf_counter()

    def stopTest(self, test):
        super().stopTest(test)
        key = identity(test)
        if key in self.outcomes and key in self.clock:
            self.outcomes[key]['seconds'] = round(time.perf_counter() - self.clock[key], 3)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'passed')

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.record(test, 'failed', brief(error), error)

    def addError(self, test, error):
        super().addError(test, error)
        self.record(test, 'error', brief(error), error)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'skipped', str(reason)[:300])

    def addExpectedFailure(self, test, error):
        super().addExpectedFailure(test, error)
        self.record(test, 'expected_failure')

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, 'failed', 'unexpected success of a test marked as an expected failure')

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error is not None:
            self.record(test, 'failed' if issubclass(error[0], test.failureException) else 'error', brief(error), error)


def run_module(module, out):
    """Child process: run one module and write its outcomes."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    started = time.perf_counter()
    recorder = Recorder()
    unittest.defaultTestLoader.loadTestsFromName('tests.' + module).run(recorder)
    Path(out).write_text(json.dumps(dict(module=module, seconds=round(time.perf_counter() - started, 2),
                                         tests=recorder.outcomes)), encoding='utf-8')


def shard(modules, index, count):
    """Deterministic shards, balanced by recorded duration where one is known."""
    durations = json.loads(DURATIONS.read_text(encoding='utf-8')) if DURATIONS.exists() else {}
    typical = sorted(durations.values())[len(durations) // 2] if durations else 1.
    bins = [[0., []] for _ in range(count)]
    for module in sorted(modules, key=lambda name: (-durations.get(name, typical), name)):
        lightest = min(bins, key=lambda entry: entry[0])
        lightest[0] += durations.get(module, typical)
        lightest[1].append(module)
    return sorted(bins[index - 1][1])


def stop(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(process.pid)], capture_output=True)
    else:
        process.kill()
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shard', default='1/1', help='I/N: run the I-th of N shards')
    parser.add_argument('--timeout', type=float, default=1200., help='seconds allowed to one module')
    parser.add_argument('--results', type=Path, default=ROOT / 'tmp' / 'ci-results.json')
    parser.add_argument('--logs', type=Path, default=ROOT / 'tmp' / 'ci-logs')
    parser.add_argument('--survey', action='store_true', help='record outcomes; never fail')
    parser.add_argument('--module', help=argparse.SUPPRESS)
    parser.add_argument('--out', help=argparse.SUPPRESS)
    parser.add_argument('modules', nargs='*', help='module names such as test_checkpoint; default all')
    args = parser.parse_args()
    if args.module:
        run_module(args.module, args.out)
        return 0
    index, count = (int(value) for value in args.shard.split('/'))
    if not 1 <= index <= count:
        parser.error('--shard needs 1 <= I <= N')
    modules = args.modules or sorted(path.stem for path in (ROOT / 'tests').glob('test_*.py'))
    modules = shard(modules, index, count)
    args.logs.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment.setdefault(name, '1')
    tests, timing = {}, {}
    began = time.perf_counter()
    for position, module in enumerate(modules, 1):
        out, log = args.logs / (module + '.json'), args.logs / (module + '.log')
        out.unlink(missing_ok=True)
        started = time.perf_counter()
        with log.open('w', encoding='utf-8', errors='replace') as stream:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--module', module, '--out', str(out)],
                                       cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=args.timeout)
                problem = None if out.exists() else 'crash'
            except subprocess.TimeoutExpired:
                stop(process)
                problem = 'timeout'
        seconds = round(time.perf_counter() - started, 2)
        timing[module] = seconds
        if problem is None:
            found = json.loads(out.read_text(encoding='utf-8'))['tests']
        else:
            tail = log.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-1:] or ['']
            found = {'tests.%s::%s' % (module, problem): dict(
                outcome=problem, seconds=seconds,
                message=('no result after %.0f s' % args.timeout) if problem == 'timeout' else
                        ('interpreter exited %s: %s' % (process.returncode, tail[0][:240])))}
        for result in found.values():
            result['module'] = module
        tests.update(found)
        bad = sum(result['outcome'] in FAILING for result in found.values())
        print('[%3d/%d] %-58s %4d tests %3d failing %7.1f s' % (position, len(modules), module, len(found), bad, seconds), flush=True)

    known = json.loads(KNOWN.read_text(encoding='utf-8')) if KNOWN.exists() else dict(reasons={}, tests={})
    listed, reasons = known.get('tests', {}), known.get('reasons', {})
    failing = {name: result for name, result in tests.items() if result['outcome'] in FAILING}
    new = {name: result for name, result in failing.items() if name not in listed}
    ran = {result['module'] for result in tests.values()}
    # A listed entry is judged only if its module ran in this shard.
    mine = {name for name in listed if name.split('::')[0].split('.')[1] in ran}
    recovered = sorted(name for name in mine if name in tests and tests[name]['outcome'] not in FAILING)
    gone = sorted(name for name in mine if name not in tests and '::' not in name)
    summary = dict(shard=args.shard, modules=len(modules), tests=len(tests), seconds=round(time.perf_counter() - began, 1),
                   outcomes={outcome: sum(result['outcome'] == outcome for result in tests.values())
                             for outcome in ('passed', 'skipped', 'expected_failure', *FAILING)},
                   failing=len(failing), new_failures=sorted(new), known_failures=len(failing) - len(new),
                   listed_but_passing=recovered, listed_but_absent=gone)
    args.results.write_text(json.dumps(dict(summary=summary, module_seconds=timing, tests=tests), indent=1, sort_keys=True),
                            encoding='utf-8')
    lines = ['### Tests, shard %s' % args.shard, '',
             '%d modules, %d tests in %.0f s: %d passed, %d skipped, %d failing (%d listed as known, %d new).' % (
                 len(modules), len(tests), summary['seconds'], summary['outcomes']['passed'], summary['outcomes']['skipped'],
                 len(failing), summary['known_failures'], len(new)), '']
    if new:
        lines += ['**New failures**', ''] + ['- `%s` (%s): %s' % (name, result['outcome'], result['message'])
                                             for name, result in sorted(new.items())] + ['']
    if recovered:
        lines += ['Listed as known failures but passing here (remove from `ci/known_failures.json`):', '']
        lines += ['- `%s`' % name for name in recovered] + ['']
    if summary['known_failures']:
        counts = {}
        for name in failing:
            if name in listed:
                counts[listed[name]] = counts.get(listed[name], 0) + 1
        lines += ['Known failures by reason:', ''] + ['- %d: %s' % (number, reasons.get(reason, reason))
                                                      for reason, number in sorted(counts.items(), key=lambda item: -item[1])] + ['']
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as stream:
            stream.write(text + '\n')
    return 0 if args.survey or not new else 1


if __name__ == '__main__':
    raise SystemExit(main())
