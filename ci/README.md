# Tests on every pull request

`.github/workflows/tests.yml` runs the whole `tests/` suite on a hosted Windows runner for every
pull request, in eight parallel shards of about ten minutes each. It runs on Windows with the pinned
`requirements-portable.txt` because the native backends are embedded Windows binaries and many
expected numbers are pinned to that platform.

`ci/run_tests.py` runs each test module in its own interpreter, so a module that hangs, crashes or
cannot be imported costs only itself. The same command works locally:

    python -B ci/run_tests.py                      # everything
    python -B ci/run_tests.py test_checkpoint      # chosen modules
    python -B ci/run_tests.py --shard 3/8          # one shard, as the workflow runs it

## What a green tick means

No test failed that is not listed in `ci/known_failures.json`. That file lists the tests that fail
on a clean machine for reasons that predate the change under test, each with one of a few stated
reasons (a package that is not in this repository, a stale fixture, and so on). It does not mean
those tests pass.

- A listed test that fails is reported and tolerated.
- A listed test that passes is reported, so its entry can be removed. Repairing a test and
  deleting its line is how the list shrinks.
- Any other failure, a timeout, or a crashed interpreter fails the run. The job summary names it
  and the uploaded `test-results-N` artifact holds every outcome and each module's output.

## Keeping it current

`ci/test_durations.json` only balances the shards; a module missing from it is given a typical
duration. To rebuild both files from a run, download its artifacts and collect the `tests` and
`module_seconds` of the eight `ci-results-N.json` files.
