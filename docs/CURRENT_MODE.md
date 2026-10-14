# Current recorded mode

## Source and scope

The published runtime comes from run **`20261007-coarse-burial-speed-history`**, using **`coarse_rigid_sheet_history_v1`**. The continuation starts at an explicitly accepted **121 Myr** source boundary. Its prior saved history remains historical evidence rather than being recomputed or relabeled by this source snapshot.

The root [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) records the copied source in the original publication, commit `6f2b2ccd`. This development branch adds the diagnostic and validation changes described in [Coarse handoff](COARSE_HANDOFF.md); modified files must not be described as byte-identical to the sealed runtime, and the original manifest is not a current-branch checksum list. Supporting tests and inherited documents have their own provenance. Run-specific checkpoints, output frames, private receipts, and machine bindings are excluded.

## Mechanical choices

| Process | Behavior in the recorded coarse continuation |
| --- | --- |
| Plate motion | Existing plate driving and resistance laws determine actual motion. |
| Material transport | Finite rigid-sheet advection remains active. |
| Overlapping crust | Retained stacked columns and their existing support/accounting remain active. |
| Ocean and subduction transport | Existing native processes remain active. |
| Arcs, baseline erosion, and ordinary foundering | Existing process paths remain active according to the saved configuration. |
| Internal P1 sheet strain | Unresolved: the mode does not solve internal sheet shortening or thinning. |
| Gravitational spreading | Unresolved: the fine deformation/gravity solution is not being claimed. |
| Automatic fine refinement | Disabled. Arc birth can still add faces, so this does not fix the total geometry size. |

The mode introduces no new plate-driving constitutive law, imposed motion, or frozen-motion completion fallback. Its reduction concerns internal mechanical resolution. Mountains can still grow through retained crustal stacks and arc addition, with the limitations described in [KNOWN_LIMITATIONS.md](../KNOWN_LIMITATIONS.md).

The burial-speed successor adopts a focused geometry implementation and ordered, read-only parallel work, while retaining the named coarse mode. The broader upstream feature set was not adopted wholesale. Deep-stack eclogite experiments, new remeshing policies, or fine contact/gravity behavior described elsewhere must not be inferred to be enabled here.

## Portability and starting another experiment

This repository preserves the exact guarded production entry point. `server.py` can open its browser interface, but its `_run_owned` worker unconditionally requires the initial budget/token and owned guardian bindings, and requires `resuming=True`. A clone without the original state and acceptance chain cannot evolve a fresh world through that path.

Lite defaults remain in inherited configuration code. They are not a supported unbound startup mode for this production overlay. A new `Simulation` constructor does not establish the named coarse-mode source boundary, and `run_simulation.py` does not supply the production controller requirements. A configuration JSON alone cannot create the missing owned process identities and receipt chain.

The coarse continuation was prepared through an explicit paused-state migration using `coarse_history.activate` and a valid immutable parent-checkpoint receipt. This operation records the changed scope and advances no simulated time. Reproducing the recorded world requires that source state and its configuration in addition to the code; they are not bundled with this public repository.

The scripts under `reference_operations/` document the production preparation and supervision approach. Their original private bindings have been omitted or redacted. They are not an executable recipe for accessing or restarting the original run. A portable startup and supervision path for a separate experiment remains future work; this publication has not stripped the production guards or introduced a replacement launcher.

## Interval timing and acceptance

The recorded production controller supplies one absolute **1800-second deadline for each simulated 1 Myr interval**. Physical integration, nested retries, column/topology work, required output, and acceptance writes share that interval. An independent owned Windows Job guard enforces the cutoff; a browser status check or periodic monitor is not that guard.

The root checkpoint remains the 121 Myr source boundary. Later durable checkpoints belong to immutable per-token interval directories. Acceptance requires the controller token and clock chain, guardian acknowledgment, server acceptance closure, and final guardian release, all matching the owned identities and original deadline. Tentative integration time, a checkpoint filename, or a saved frame alone does not establish acceptance. A partial, late, or cut-off interval remains partial.

This contract describes the original production setup. Opening the server interface in a clone does not install its independent guard or reproduce its acceptance chain, and therefore does not establish a runnable production worker. The configured 1800 seconds is a limit, not a performance measurement or a promise of completion speed.

## Interpreting inherited documents

`docs/inherited/` preserves useful scientific foundations, reference material, and development history. Some sections describe earlier defaults, optional experiments, or proposed work. Use this document for the recorded mode's scope and the source manifest for its code provenance. The inherited Scotese rules remain a guiding reference; computational simplifications and the documented scientific qualifications should not be presented as universal geological laws.
