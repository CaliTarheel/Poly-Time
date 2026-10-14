# Super-Light-Time

The guarded production source of an exploratory tectonic and terrain evolution simulator, derived from Asein Lite. Its model covers plate motion, ocean and continental material, collision stacks, subduction, arcs, and relief, with a local browser interface for inspecting history.

The initial publication, commit `6f2b2ccd`, preserves the simulation code used by the `20261007-coarse-burial-speed-history` continuation in **`coarse_rigid_sheet_history_v1`** mode. This development branch adds the diagnostics and validation described in [Coarse handoff](docs/COARSE_HANDOFF.md); those source changes do not establish adoption by the sealed active run. The mode's unresolved processes and observed problems are listed in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

**This is a source snapshot, not a standalone runnable simulation release.** The exact production server requires an owned guardian, an initial budget and token, and a valid resumed-state acceptance chain. Those private run bindings and the historical state are not published. A portable launcher remains future work; the production guards have been preserved.

## Inspect the interface in a separate local copy

Use Python 3.12. In a new clone, preferably with a virtual environment activated:

```sh
python -m pip install -r requirements.txt
python -B server.py --port 8766 --open
```

This command opens the local browser interface only. **It does not establish a valid evolving run.** The production worker rejects unbound or fresh evolution: its `_run_owned` path requires the initial budget/token and guardian bindings, and requires `resuming=True`. NumPy and Pillow are the core dependencies.

Inherited Lite configuration defaults and the included historical launchers do not make this production overlay portable. `run_simulation.py` is not a way to launch it as a fresh production run, and writing a configuration JSON cannot supply the missing owned process identities or receipt chain. See [Current mode](docs/CURRENT_MODE.md) for the preparation and acceptance requirements.

## What is included

| Location | Contents |
| --- | --- |
| Root Python modules and `web/` | The recorded runtime baseline, with this branch's source changes described in [Coarse handoff](docs/COARSE_HANDOFF.md). |
| `tests/` | Regression tests and their included fixtures; presence in this repository is not a claim that every test has been run for this publication. |
| `benchmarks/`, `examples/`, `capture/` | Supporting source and examples retained from the project. |
| `reference_operations/` | Archived preparation, supervision, budget, and delivery code. Private bindings have been removed or redacted; these are references, not a ready-to-run production controller. |
| `docs/inherited/` | Selected earlier design and scientific reference documents. They include historical results and optional or proposed features; they do not define the active mode. |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | Source provenance and file hashes for the original publication at `6f2b2ccd`; it is not a hash manifest for the modified development branch. |

The source run's checkpoint data, output history, credentials, account state, and private operational receipts are excluded. The code snapshot is neither a self-contained reproduction of that world nor a ready-to-evolve fresh-world package.

## Interpreting the model

The goal is plausible tectonic worldbuilding with inspectable motion and material accounting. Results depend on initial conditions, process choices, resolution, and the saved state's history. This is not a calibrated reconstruction of Earth or a prediction of a unique geological future.

In the recorded coarse mode, material sheets move rigidly while overlapping columns remain represented. Internal sheet shortening, thinning, and gravitational spreading are unresolved. A convincing map does not establish that these omitted processes have been solved.

The production continuation used an independent Windows Job controller with an **1800 wall-second limit per simulated 1 Myr**, including mandatory output and acceptance work. The interface command above does not install that guard or satisfy the worker's bindings. The limit is a control policy, not a measured speed guarantee for this code on another machine.

A completed, validated 1000 Myr history is not included in this release. Read [Current mode](docs/CURRENT_MODE.md) and [Known limitations](KNOWN_LIMITATIONS.md) alongside the inherited scientific documents when extending the model.

## Attribution and licensing

Super-Light-Time retains code and documentation from the Asein Lite / Deep Time project. Existing author and reference attributions remain applicable. No new license is assigned by this snapshot; public availability alone should not be read as an additional license grant.
