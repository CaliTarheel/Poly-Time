# Deep-stack handoff: scope for the coarse continuation

## Which source this note describes

The reviewed `HANDOFF-DEEP-STACK-ECLOGITE.md` describes a separate upstream experiment. Its statement that the code is merged does **not** mean that the experiment is present in this repository or active in `20261007-coarse-burial-speed-history`.

The reviewed handoff has SHA-256 `222fb42395a26ed28db29b9ca5eaaff31d4016bca8e953078fa8aa45408d80cf`. It describes an experimental 75 km deep-stack stage, an adoption pulse, and a 0.5 Myr removal timescale. This coarse source has **no such stage or adoption API**. In particular, the handoff's `foundering_deep_stack_version`, `upgrade_deep_stack`, and proposed `deep_stack_returned_km` field must not be treated as available controls here.

The handoff's measured 112 Myr volumes, coastline changes, test counts, and relief improvements belong to that upstream experiment. They are not results for the current coarse continuation or acceptance evidence for this branch.

[SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) describes the original public snapshot at commit `6f2b2ccd`. The diagnostic and validation changes below modify that baseline in a development branch. The manifest remains historical provenance; it does not certify that the modified branch is byte-identical to the sealed running source.

## The ordinary law retained here

The recorded coarse continuation retains the ordinary version-2 foundering inventory and the existing local burial integration. The relevant source is [eclogite_sink.py](../eclogite_sink.py), [burial_depth.py](../burial_depth.py), and [crust_inventory.py](../crust_inventory.py).

| Quantity | Retained behavior |
| --- | --- |
| Depth threshold | `ECLOGITE_DEPTH_KM = 50`: eligibility is measured below the top of the represented stack, including the face's own thickness. |
| Residence gate | `HEATING_DELAY_MYR = 10`: the existing face-level burial or root clock gates eligibility. It is not a solved temperature field. |
| Removal timescale | `FOUNDERING_TAU_MYR = 20`: the step fraction is `1 - exp(-dt / 20)`. This is an exponential timescale, not a promise that all eligible material disappears after 20 Myr. |
| Composition limit | Version 2 spends the remaining conservative removable-crust inventory. It does not grant a fresh fraction of the remaining column at each step or source transition. |
| Column floor | The existing residual floor also limits removal. That numerical reserve is not a physical proof that a deep residue should persist. |
| Bookkeeping | Removal decreases column thickness and spends the removable inventory, while recording cumulative foundering and mantle return. |

These are reduced process closures. The ordinary sink does not resolve temperature-dependent phase evolution or a retained dense eclogite layer. The separate retained-dense-crust policy in the source must not be confused with this ordinary sink or the absent 75 km experiment.

## What the handoff reveals that still matters here

Local burial integration thresholds the disjoint covered regions before returning a **face-mean eligible thickness**. The subsequent ordinary sink still changes one thickness value for the whole face. Accurate integration of the eligible volume therefore does not establish spatially localized thinning.

For a face that is only partly covered, removal attributed to its buried part can lower the exposed part as well. Repeated evolution can change the amount of represented material that meets the depth criterion. A synthetic half-covered ordinary-law regression reaches its 30 km compositional cap even though the initial integrated buried inventory is smaller. This exposes a spatial limitation of the whole-face representation; it is not a repair or a measurement of the production world's affected volume. The upstream experiment's quantified over-removal cannot simply be assigned to the present law. Version-2 inventory conservation prevents renewed compositional credit, but does not prove that each removed parcel was physically below the threshold or that coastlines remain unchanged.

Diagnostics and ledger/frame validation can make this distinction visible and reject inconsistent saved data. They do not split faces, localize the sink, restore internal deformation, or repair an existing coastline. The code's diagnostic fields and their regression tests are the authority for any new source checks; the upstream handoff is not evidence that those checks have already passed here.

The coarse mode also leaves internal P1 sheet shortening, thinning, and gravitational spreading unresolved. Retained stacked columns can support mountains without providing a fully relaxed collision belt. These omissions remain explicit even if accounting and validation improve.

## Diagnostic and validation changes in this branch

### Observe the ordinary transaction without changing its law

`ordinary_loss_diagnostics` is a pure observer called before `apply` in the non-trace material-column pass. Its version-1 report explicitly records `column_assignment = "uniform whole-face thickness decrement"` and `spatial_depletion_resolved = False`. It introduces no physical state, new geometry operation, migration pulse, or changed loss coefficient.

The report expresses eligible, requested, and admitted loss volumes in physical km³. It also separates `inventory_blocked_requested_loss_km3` from `residual_floor_blocked_requested_loss_km3`. These are **disjoint allocations in inventory-then-floor order**: the floor sees only the inventory-admitted request. They must not be added to independent hypothetical estimates of what each limiter would block by itself.

When the existing local burial pass supplies its already computed `covered_union_area_km2`, the observer sets `exact_partial_cover_available = True` and reports the number of partly covered faces and their eligible and admitted-removal volumes. It reuses the union footprint; summing overlapping pair areas would not provide the same classification at a multi-sheet stack. If that footprint is unavailable, the flag is false and these partial-cover metrics are omitted.

`uniform_removal_assigned_to_uncovered_area_km3` measures the portion of the uniform face decrement assigned to the currently uncovered area of partly covered faces. It is an allocation consequence, **not proof that never-deep material was removed**: uncovered columns can have their own deep roots, and the face representation does not resolve each parcel's vertical history.

### Validate bookkeeping without inventing historical attribution

The foundering record path stages the mantle-return ledger and process totals and validates the current step's plate, sheet, and contact partitions before committing those bookkeeping records. Saved-frame checks reject nonfinite or inconsistent foundering/inventory data and contradictory migration-record/version combinations.

New frames assert `mantle_return_ledger_version = 1` only when each complete cumulative plate, sheet, and contact partition closes to the ledger total. An inherited historical attribution gap remains version 0; writing a new frame does not fill that gap or certify missing old attribution. The marker describes accounting closure, not spatially resolved depletion or geological validation.

Focused regressions cover observer outputs and bounded ordinary-law inventory accounting, including repeated partial cover. Their physical lesson is that a closing composition ledger can coexist with the unresolved spatial limitation described above. Full test results belong to the change's validation report; this note does not claim that the entire inherited suite passes or that a live adoption trial has been completed.

## Changes that require a separate physical implementation

The handoff's proposed irreversible `deep_stack_returned_km` field belongs to its 75 km law. Adding that field alone to this source would neither implement a conservative spatial sink nor justify adopting the experimental law. A new field needs defined transport, strain, split, merge, arc-birth, channel, and remesh behavior, together with checkpoint compatibility and a versioned source transition.

Likewise, splitting partly covered faces to localize removal changes represented geometry and workload. It needs independent conservation and coastline checks, including exposed children and depleted residues, before adoption. Disabling automatic fine refinement does not by itself supply or forbid a correctly designed local split; the design must state its cost and mechanical consequences.

Any future return of felsic crust to the mantle must be identified as a model closure and accounted for separately. It must not be justified merely by a mafic eclogite timescale or described as solved natural felsic foundering. The current handoff work does not introduce that return stage, an elevation ceiling, or an unmeasured claim of faster evolution.

## Source review and active-run adoption are separate

1. Develop and review changes in the isolated source branch. Keep the sealed running source and historical checkpoints unchanged.
2. Demonstrate each claimed change with focused fixtures and saved-state validation. For accounting work, distinguish inventory closure from spatially correct removal. For a future localized sink, include partial cover, repeated removal, later thickening, remapping, and coastline response.
3. Describe compatibility explicitly. Preserve historical fields and receipts; missing new diagnostics in an old frame do not establish that the corresponding process was solved. Any physical-law change needs its own versioned migration and reviewed source boundary.
4. Before adopting into a production continuation, prepare the source pins, accepted predecessor, controller bindings, and required output/acceptance checks. The existing production worker still requires its owned guardian and valid resumed-state chain; a configuration JSON is insufficient.
5. Preserve the original interval deadline and all partial evidence. A source fix does not renew an exhausted attempt, authorize another successor, or allow a late interval to become accepted. Keep the current independent guard behavior distinct from operator notification preferences.

This note and changes in a development branch are not proof of deployment to the active run. No live adoption, migration pulse, restart, or deadline change follows from the handoff document itself. See [Current mode](CURRENT_MODE.md) and [Known limitations](../KNOWN_LIMITATIONS.md) for the continuing scope limits.
