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

## Juvenile magma capacity and deferred point edifices

Whole-island lateral growth can be blocked by material on just one side of the island. The finite overlap test must still protect that material, including neighboring arcs on the same plate. Small new foundations can also lack a profile that satisfies both the retained 8 km column minimum and the 20-degree constructive-slope capacity. Rejected supplied magma remains in the pending source ledger; rejection does not mean the subduction source stopped producing magma.

A focused local alternative is to add magma to the existing juvenile face containing its source, without expanding the footprint or changing its reference area. Actual retained volume, column inventories, trace columns, and source provenance must all agree. Deposition uniformly thickens that containing face; preserving the source coordinates does not resolve an individual volcanic cone or transport within the face. The complete connected arc still supplies the slope stencil, and any already steeper face must not become steeper. This has finite capacity: it does not join neighboring islands or cure the lack of lateral spreading in the coarse mode.

At 176 Myr, a complete saved pending-source transaction retained 18,272.867 cubic kilometres across 67 of 169 existing-arc groups with this fallback, versus zero additional volume under the unchanged policy. All 310 new-island birth groups still admitted zero volume. This used only about 0.35% of the 5.184 million cubic kilometres pending: the eight largest blocked islands had effectively no local capacity. The trial passed physical/source/placement-receipt volume closure, frame validation, and typed checkpoint reload; geometry, reference arrays, configuration, and historical placement receipts stayed unchanged. It covered one emplacement transaction, not tectonic time integration or live deployment. An earlier direct column/trace trial on 168 archived failed-growth groups retained 18,023.406 cubic kilometres across 66 components. Equal-thickness deposition over the immediate shared-edge neighborhood admitted less volume in preliminary probes; these tests do not establish a general geological transport law.

The retained backup design is **seeded point volcanic centers** along the overriding subduction arc. Stable centers could accumulate explicitly budgeted source volume before receiving a resolved polygonal edifice. Seeded pseudo-random placement may break up regular spacing, but must remain tied to valid overriding sources, persist across checkpoints, and advect with the receiving plate. An unresolved center still needs volume, owner, provenance, age, and basal-location records; it cannot become free crust when displayed or resolved. Promotion should depend on accumulated volume, spacing and the resolution needed to represent a funded edifice, rather than point count alone. A center must explicitly be either pending magma without a retained-crust/mechanical contribution, or a retained point edifice that has already debited its magma budget and participates in mass, composition/history, and relevant force/removal accounting. Resolving a retained edifice into polygons transfers its existing inventory and history; it creates no new magma and must not increment supply or crust-addition counters again. The funded finite edifice must still satisfy basal-datum, ownership/exposure, material-exclusion, column and slope constraints. Neighboring edifices would need conservative overlap or coalescence rules. This point representation is a deferred option, not active physics or a promise that dense points imply emerged islands.

## Workload can still grow

Automatic fine refinement is disabled in the recorded coarse mode, but physical arc birth and topology changes can add geometry. This is not a fixed bound on mesh size or interval cost. Conservative merging or history compaction needs its own checks for ownership, sheet identity, material inventories, and provenance.

The 1800-second production cutoff is distinct from measured completed-step time. Small geometry tests, a fast individual phase, and a healthy browser interface do not prove that a whole interval, including required output, finishes within that limit.

## Validation and reproducibility remain limited

The current history includes explicit source transitions. The code at the latest transition was not used unchanged for every earlier saved epoch. Inherited default settings neither reproduce that experiment nor establish a runnable fresh world through this guarded production entry point.

Numerical regression tests establish particular program properties. They do not alone establish resolution convergence, robust behavior across seeds, calibrated Earth dynamics, or geologically accurate mountain heights. Useful next checks include boundary connectivity and force consistency, material budgets, checkpoint reproducibility, sensitivity to time step and resolution, and sustained behavior across long histories.

The published source includes optional and historical paths that the coarse continuation does not activate. In particular, the presence of fine contact/gravity code, display options, or a design document is not evidence that the corresponding process was solved in a saved coarse-mode interval. A completed, validated 1000 Myr history is not part of this code release.
