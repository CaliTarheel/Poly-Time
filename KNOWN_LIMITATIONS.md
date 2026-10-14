# Known limitations

These notes describe the recorded `coarse_rigid_sheet_history_v1` continuation and the source published with it. They distinguish implemented behavior from work still needed. Historical documents under `docs/inherited/` cover a broader set of experiments and defaults.

## The production snapshot is not yet portable

The exact published `server.py` requires an initial budget/token, an owned guardian and its bindings, and a valid resumed state. Its production worker requires `resuming=True`; opening the interface does not enable fresh or unbound evolution. The necessary historical state and private acceptance receipts are excluded from this repository.

Neither inherited Lite configuration defaults, a plain configuration JSON, nor `run_simulation.py` supplies those requirements. Archived operation scripts are references with private bindings removed, not a portable launcher. A separately designed and reviewed startup path is future work. The production guards have deliberately been retained in this source snapshot.

## Internal deformation and gravitational spreading

The coarse mode advects material sheets rigidly. It does not solve their internal P1 shortening, thinning, or gravitational spreading. Overlapping crust is retained in stacked columns, so collisions and arc addition can still build relief, but that is not a replacement for deformable crust or a fully relaxed mountain belt.

This is a computational simplification, not a claim that natural continental interiors remain rigid. Restoring the omitted mechanics can change later motion, collision geometry, and terrain. Refinement may therefore require a separate continuation or rerun; it cannot be assumed to be cosmetic post-processing of an unchanged history.

## Boundary geometry and ownership can disagree

A review of the saved 130 Myr state found two concrete examples:

- A short boundary segment near **57.3° S, 80.5° W** lacked a connection to the surrounding reconstructed boundary.
- A small triangle near **29.2° S, 90.4° W**, close to a triple junction, came from a categorical plate label whose fractional material support did not dominate the continuous ownership field.

The boundary reconstruction can retain a raw mesh contact when a continuous boundary is unresolved. These examples expose a mismatch between categorical labels and the piecewise-linear ownership representation; they are not simply a map seam or camera artifact. Their effect on forces and later topology has not yet been quantified. Removing the line or deleting the small plate's material would not by itself establish a correct repair.

Names also need interpretation. In this state, **Ocean Basin 17** was assigned to an already disconnected oceanic region. Its recorded isolation event did not demonstrate newly opened seafloor or establish that it would evolve like Earth's Pacific Plate.

## Collision columns and relief need further validation

The retained stack supports overlapping material and associated buoyancy accounting. Large combined stack thicknesses should not be read as the thickness of one intact crustal sheet. The current approximation does not resolve the full vertical deformation, pressure-temperature evolution, or gravitational relaxation of a deep collision stack.

The separate upstream deep-stack eclogite prototype was not adopted into this continuation. Ordinary retained column and foundering processes must not be confused with that additional proposed treatment. High relief and deeply stacked regions remain priorities for checking material conservation, force consistency, and physical interpretation.

The retained ordinary law uses a 50 km stack-depth threshold, a 10 Myr residence gate, and a 20 Myr exponential removal timescale. Its version-2 removable inventory prevents renewed compositional credit. However, local burial integration returns face-mean eligibility and the sink changes one thickness per face: on partial cover, accounting closure does not demonstrate that thinning is confined to buried material or that exposed coastlines are preserved. This remains a spatial-resolution concern; the absent 75 km experiment's measured losses are not measurements of the ordinary law. See [the mode-specific handoff assessment](docs/COARSE_HANDOFF.md) for source status and adoption requirements.

A synthetic repeated half-cover regression reaches the ordinary law's 30 km compositional cap even though its initial integrated buried inventory is smaller. The branch's new diagnostics expose the uniform face allocation and its limiting inventories; they do not localize removal or establish how much never-deep material was removed in the active world. Ledger/frame validation also preserves historical attribution gaps rather than certifying them as closed. These source changes are not evidence of deployment to the sealed run.

## Workload can still grow

Automatic fine refinement is disabled in the recorded coarse mode, but physical arc birth and topology changes can add geometry. This is not a fixed bound on mesh size or interval cost. Conservative merging or history compaction needs its own checks for ownership, sheet identity, material inventories, and provenance.

The 1800-second production cutoff is distinct from measured completed-step time. Small geometry tests, a fast individual phase, and a healthy browser interface do not prove that a whole interval, including required output, finishes within that limit.

## Validation and reproducibility remain limited

The current history includes explicit source transitions. The code at the latest transition was not used unchanged for every earlier saved epoch. Inherited default settings neither reproduce that experiment nor establish a runnable fresh world through this guarded production entry point.

Numerical regression tests establish particular program properties. They do not alone establish resolution convergence, robust behavior across seeds, calibrated Earth dynamics, or geologically accurate mountain heights. Useful next checks include boundary connectivity and force consistency, material budgets, checkpoint reproducibility, sensitivity to time step and resolution, and sustained behavior across long histories.

The published source includes optional and historical paths that the coarse continuation does not activate. In particular, the presence of fine contact/gravity code, display options, or a design document is not evidence that the corresponding process was solved in a saved coarse-mode interval. A completed, validated 1000 Myr history is not part of this code release.
