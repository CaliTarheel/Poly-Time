# Poly runs, restart, and qualification

`python -m poly.run` runs the polygon/Euler-frame `RegionWorld`. It is independent
of the inherited root worker and its private production-run bindings.

## Start and resume

Install `requirements.txt`, then use an empty output directory:

```sh
python -m poly.run start --output runs/example --steps 10 --dt 0.5 --seed 7
python -m poly.run resume --output runs/example --steps 10
```

The default initial world has eight Voronoi plates and requests 30% continental
crust, with 250 km initial boundary spacing and 5,000 global force samples.
Continental crust fraction is not emergent land fraction. Motion is force-driven,
including the existing rift decision at the start of each output interval. These
are modest development defaults, not a scientifically qualified resolution.

`--steps` counts output intervals, each `--dt` Myr; force/kinematic motion is split
further when required by the existing displacement limit. Resume counts an
interrupted partial interval as the first requested interval. It uses the saved
configuration and does not accept replacement physical settings. `--steps 0`
creates a checkpoint of the initial world.

Useful alternatives:

```sh
# Controlled prescribed cap motion; no force solve or rift decision.
python -m poly.run start --output runs/cap --fixture cap --motion prescribed --steps 20 --dt 0.5

# A larger initial tessellation, explicitly denser force quadrature, no rift tests.
python -m poly.run start --output runs/seed3 --plates 12 --seed 3 --samples 20000 --no-rifting
```

`--spacing` sets initial boundary spacing, `--speed` the prescribed/initial angular
speed multiplied by planetary radius in km/Myr, and `--continent-fraction` the
requested global continental crust fraction. The cap fixture has ocean crust only;
plate count, seed, and continent fraction apply to the Voronoi fixture. Initialization
uses the current builder's coverage guarantees. These commands do not change the
physical rules in `regions`, `dynamics`, or `mesh`.

## Output and acceptance

Each run directory contains:

| File | Purpose |
| --- | --- |
| `manifest.json` | Configuration, exact source hashes, Python/platform/dependency versions, completion/failure status. |
| `checkpoint.json` | Full last committed accepted state and continuation phase. Authoritative for restart. |
| `diagnostics.json` | Current plate/crust areas, continental volume, boundary lengths, ocean-age area bins, pending/retired material, cumulative gain/loss terms, numerical drift. |

After each successfully accepted `RegionWorld.step`, the checkpoint is written to
a temporary file, flushed, and atomically replaced. A failed step or serialization
does not replace the preceding checkpoint. A crash can leave the diagnostic or
manifest file behind the checkpoint; resume uses the checkpoint and rewrites them.
Before the first accepted step, the initial checkpoint is the recoverable state.

The runner preserves the remaining duration of an interrupted output interval, so
resume does not perform another rift decision halfway through that interval. It
records rift events on the same first accepted substep as `dynamics.run`. Failed
attempts and rejected topology changes are retried from the last committed state.
There is no automatic adaptive retry or forced acceptance after a failure.

## Checkpoint format and scope

Schema version 1 is data-only JSON with explicit array/tuple/dictionary tags. It
stores frames, full spherical regions, crust geometry and age/thickness, continental
meshes, boundaries and identities, polarity, accumulated convergence, pending arc
and escape budgets, retired arc reservoir and plate policy where supported,
sutures, parents/names, time, complete ledger, and simplification scheduling.
The continent lookup cache is recomputed. Unknown persistent attributes are an
error; adding model state requires a corresponding schema change or deliberate
migration. Missing fields, unsupported versions, malformed arrays, invalid frame
rotations, broken references, negative budgets, and corrupted polygon payloads fail.

Polygon coordinates are stored as spherely's native S2 bytes, with per-payload
SHA-256 and area checks. This deliberately uses a fixed native `Geography` decoder
and never evaluates pickle or imports a class named by the file. The codec is
bound to the exact spherely version. WKB was rejected for restart because converting
S2 coordinates to degrees can collapse almost-coincident vertices into a degenerate
edge. The native form preserves the exact accepted geometry, including large
polygons, holes, and disconnected components.

The CLI also requires matching source hashes and the recorded numerical environment
on resume. An upgraded model/codec needs an explicit migration, not default values
for newly introduced budgets. Same-runtime tests check exact uninterrupted/resumed
state equality, including force-driven restarts inside an interval; bit-identical
results across different hardware/BLAS implementations are not promised.

For application use:

```python
from poly.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(world, "checkpoint.json", metadata={"experiment": "trial-1"})
world, metadata = load_checkpoint("checkpoint.json")
```

The low-level API checks schema and geometry-codec compatibility. The caller owns
its motion/configuration metadata and run-level provenance; use the CLI for the
stricter source/environment check and mid-interval continuation protocol.

## Timestep and initial-resolution comparison

```sh
python -m poly.run qualify --output runs/qualification --duration 2 --dt 0.5
python -m poly.run qualify --output runs/cap-check --fixture cap --motion prescribed --duration 10 --dt 0.5
```

This runs the same fixture/seed for the same duration three times: baseline,
half timestep, and half initial boundary spacing. Each trial has ordinary run
files; `qualification.json` collects measurements, endpoint differences, and any
failures. Default duration is two Myr, and longer trials are explicit. A failed
trial is retained in the report and makes the command exit nonzero.

Boundary spacing also affects the discretized starting coastline, so that third
comparison measures both starting-geometry and evolution sensitivity. It is not
a pure solver-order measurement. Mesh spacing and force quadrature do not change
between the three trials. Ocean-age bins are areas in Myr ranges 0–20, 20–100,
100–200 and 200+, plus unknown ages.

These are reproducible qualification inputs and diagnostics, not a claim that the
model has passed long-duration geological validation. No threshold is invented for
acceptable geological convergence. Use the saved states and ledger to investigate
differences before setting scientific acceptance criteria. Export to GPlates,
rendered history movies, checkpoint compaction, and geological calibration remain
separate work.
