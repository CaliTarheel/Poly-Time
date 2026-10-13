# Asein Lite starting conditions

These are the original **0 Myr inputs for the Asein Lite history** that later
continued as `20261012-coarse-winding-recovery-history`. The starting run was
`20261002-122914-072bbc`, created on 2 October 2026 with the selected effective
subduction force of **3e13 N/m**. Earlier 5e12 and 5e13 trials are different runs.

## Retrieve the files

The dataset is ordinary Git content, about 520 KiB; Git LFS is not needed.
While the dataset pull request is open, clone its published branch:

```sh
git clone --branch codex/asein-starting-conditions --single-branch https://github.com/CaliTarheel/Super-Light-Time.git
cd Super-Light-Time
python -B datasets/asein-lite-start/verify.py
```

To retrieve the directory into an existing clone without switching branches:

```sh
git fetch origin codex/asein-starting-conditions
git restore --source=FETCH_HEAD -- datasets/asein-lite-start
```

The second command places the dataset in the working tree and replaces any local
files at that path. After the PR is merged, a normal pull of `main` includes it.
For an immutable copy, use the dataset commit linked by the PR instead of the
branch name. GitHub's **Download ZIP** option also includes these files when that
branch or commit is selected.

## Files

| File | Meaning |
| --- | --- |
| `initial.json` | The normalized initial map actually saved by the starting run. Use this with `config.json` as the primary input pair. |
| `config.json` | The normalized configuration actually saved by that run. |
| `original-artwork.json` | Byte-for-byte original artwork from `20261001-153852-27d072`, before Lite input normalization. It retains historical metadata that Lite ignored. |
| `requested-config.json` | The original 3e13 configuration request. Its parsed JSON equals `config.json`; formatting differs. |
| `manifest.json` | File sizes, SHA-256 hashes, source identity, provenance, and verification scope. |
| `SHA256SUMS` | Checksums of the four input files. |
| `verify.py` | Standard-library-only verification of the archived bytes and input relationships. It does not load or start a simulation. |

All four input files preserve their original bytes. The artwork and normalized
map have the same 131,072 crust entries, with unchanged ordering and orientation.
The artwork's `preset` label is `established50`, but regenerating that preset is
not a substitute for using the saved map. Crust codes are 0 for ocean, 1 for
continental crust, and 2 for craton.

## Recorded starting settings

The JSON files contain the full settings, including trench selection, ocean ages,
and explicit continental attachments. A few useful identifiers are:

| Setting | Value at the original start |
| --- | --- |
| Seed | 37 |
| Map dimensions | 512 x 256 |
| Effective subduction force | 30,000,000,000,000 N/m |
| Mesh level / coastline geometry level | 4 / 4 |
| Mechanics nodes | 1,024 |
| Deforming regions / adaptive refinement | 1 / 1 |
| Requested duration / timestep / saved-frame cadence | 1,000 / 1 / 2 Myr |
| Physics profile | `reviewed_v1` |
| Subduction response / retained phases | `fixed_trench` / `disabled` |
| Source commit | `a645f277fbc79197df189279bae98003d5a6afa6` |
| Recorded NumPy version | 2.3.5 |

The source commit identifies the historical Deep-Time/Lite checkout used for
that fresh start. It is not the current Super-Light-Time source revision.

## Reusing these conditions

A compatible fresh-world initializer takes `config.json` as its configuration
and `initial.json` as its initial-map object. Keep the full files together;
the seed alone does not describe this world. Configuration settings take
precedence over some retained initial-map metadata. For example, the Lite
configuration disables continental lifecycle and rift traction even though the
original map retains their earlier declarations.

The present Super-Light-Time server is a guarded continuation runtime. Its
inherited `run_simulation.py` does **not** supply a supported fresh startup or the
required controller bindings. These files archive the inputs; they do not make
that server a portable launcher. See [Current mode](../../docs/CURRENT_MODE.md).

The ongoing history passed through later source repairs and explicit mode
changes. In particular, its later `coarse_rigid_sheet_history_v1` mode disables
automatic fine refinement; that is not the original configuration above.
Starting these inputs under another source revision may produce a different
history. Reproducing the evolved history exactly would additionally require its
historical runtime versions, numerical environment, state, and accepted
transitions. No evolved checkpoint or operational control binding is included.

Publication checks confirm file hashes, dimensions, crust-array equality, and
configuration equality. No fresh simulation or long-term replay was run to
publish this dataset.
