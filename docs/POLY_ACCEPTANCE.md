# Region-world acceptance and failure behavior

`RegionWorld.step()` computes a candidate state and publishes it only after geometry
and material budgets pass. Any exception leaves the call-entry world, clock,
meshes, boundaries, sutures, and ledger unchanged. Existing references to that
world's fields are unchanged on failure. This guarantee covers a single `step`
call; separately requested operations such as `rift()` precede that transaction.
A successful step may replace mutable child objects, as topology changes already
do. Historical ledger entries are append-only during a step and share their
objects with the candidate; the ledger list itself is copied.

Validation does not repair or rescale the candidate. Its geometry checks measure:

- Uncovered sphere area and excess plate ownership independently. Excess ownership
  is sum of plate areas minus their union, counting multiplicity at triple overlap.
- Per-plate uncovered area, crust lying outside the plate, and overlapping crust
  pieces. The first two compare spatial differences; overlap is measured in the
  common plate frame before any rotation/reconstruction.
- Orphan crust, nonfinite/negative material thickness and fluxes, and disagreement
  between a continental mesh and its stored mean thickness.

Thus equal holes and overlaps cannot cancel. The measurement rotation itself is
restricted to the acceptance allowance so the geometry helper's ordinary
near-full-sphere shortcut cannot erase a large hole before it is measured.

## Budgets

The cap, collision, island, arc, erosion, force-driven and rifting fixtures on
`poly/erosion` at `5cfd633` produced 81 region steps with these observed maxima:
59.8 km² uncovered area, 2.04 km² overlap, 52.2 km² unbooked reconstruction drift,
and about 1,707 km³ unexplained continental-volume change (3.5e-5 of the affected
world's continental volume). These are a calibration sample, not qualification
of arbitrary seeds or million-year intervals.

The defaults in `poly.acceptance` provide margin above that behavior:

| Quantity | Ceiling |
| --- | --- |
| Each aggregate geometry defect; per-step area balance or rebuild drift | 1 ppm of sphere area, about 510 km² on Earth |
| Signed cumulative rebuild drift | 10 ppm, about 5,101 km² |
| Absolute cumulative rebuild drift | 100 ppm, about 51,006 km² |
| Explicit numerical area additions plus removals per step | 100 ppm |
| Explicit numerical area additions plus removals over the run | 1,000 ppm |
| Per-step and signed cumulative continental-volume residual | 1e-4 of reference volume, at least 1 km³ |
| Absolute cumulative continental-volume residual | 1e-3 of reference volume, at least 1 km³ |

Numerical area additions/removals record discarded ocean debris, polygon cleaning,
and accepted simplification area changes. They are an additional classification
of existing ledger terms, not additional physical gains/losses. Unbooked effects
of geometric reconstruction still appear in `rebuild_drift_km2`. These diagnostics
measure area changes, not the geographic distance by which a coastline moves.

Cumulative limits do not grow with step count. Both signed and absolute budgets
are enforced: opposite errors cannot keep cancelling forever. Exhaustion is a
reason to inspect or improve numerical behavior, not silently reset the budget.
Long-run qualification may motivate an explicit policy change with evidence.

## Continental material contract

The residual is:

`volume_after - volume_before - arc_km3 + sediment_to_sea_km3`.

`arc_km3` means magma actually emplaced into continental material. Pending arc-root
and island banks are separate reservoirs and are not part of this check. Sediment
redeposited on the same continent and collision thickening are internal transfers;
only sediment exported to the unrepresented sea reservoir is a sink. Future LIPs,
continental destruction, or a distinct sediment reservoir must explicitly extend
this source/sink contract rather than bypass the check.

The ledger carries the initial volume, cumulative sources and sinks, signed and
absolute residuals, and budget. A checkpoint retaining the ledger retains these
budgets. When loading an older ledger without these fields, accounting begins at
the first newly checked step; no claim is made about the unmeasured older history.
The current checker deliberately tolerates the documented small volume error from
geometric reconstruction. It neither hides that error nor claims exact physical
mass conservation.

## Regression coverage

`python -m unittest discover -s tests -p 'test_poly*.py' -q`

The acceptance tests include equal-sized gaps and overlaps, overlapping crust,
orphaned ownership, source/sink closure, repeated small area/volume errors,
cumulative recorded repair, invalid fluxes, and a failed late step that must
preserve byte-for-byte state and remain usable afterward. Existing geological
fixtures exercise legitimate collision, accretion, magmatism, erosion and rifting.
