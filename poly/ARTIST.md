# Artist evolution prototype

This branch adds an opt-in artist scenario to the spherical polygon core. It is a working
implementation and qualification harness, not a claim that a convincing 1,000 Myr world
has been achieved. It depends on the reviewed stack through PR #16.

## Run and inspect

Use the repository's numerical dependencies, including the native geometry codec's
recorded spherely version. The implementation was exercised with Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0, spherely 0.1.1 and Shapely 2.2.0. Install the standalone
dependencies with `python -m pip install -r poly/requirements.txt`. Shapely 2.1 or later
supplies the constrained triangulation; its version and GEOS version enter the run receipt.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m poly.artist_run start \
  --output runs/artist-01 --seed 1 --plates 8 \
  --land .55 --continents .65 --duration 1000 \
  --dt .5 --snapshot 5 --width 1600 --spacing 150 --samples 20000

python -m poly.artist_run resume --output runs/artist-01 --duration 100

python -m poly.artist_run render --checkpoint runs/artist-01/states/state-000050.000.json \
  --output map-50Myr.png --width 8192
```

Open `runs/artist-01/replay.html` locally. It embeds the actual saved maps, with a draggable
globe, flat projection, time slider, playback, boundary overlay and inherited belt overlay.
The maps and timeline do not require a server or external JavaScript libraries. A target of
1,000 Myr is an instruction to the runner; inspect `manifest.json` for the duration actually
accepted. A failed or interrupted integration preserves its last accepted checkpoint and
records the error. A partially attempted step is never presented as an accepted snapshot.
Every displayed epoch also keeps a native checkpoint under `states/`, so its mechanical
fields can be inspected or rendered again at a higher resolution. Rendering does not advance
the simulation or increase its physical resolution.

`checkpoint.json` contains the accepted native spherical geometries, moving frames,
continental fields, boundary state, material ledger and artist history. `history.json` records
the measurements and geometry for each displayed epoch. `diagnostics.json` contains core
area and volume budgets. Source and package hashes accompany the checkpoint. Resuming
with changed source or a changed numerical environment is deliberately rejected.

The duration argument counts complete outer integration intervals. On resuming a checkpoint
inside an interval, its unfinished part counts as the first interval. The manifest records the
resulting exact target time. Force and rift decisions retain the original timestep cadence;
changing the snapshot interval does not change integration decisions.
The artist integrator can halve a rejected timestep up to six times when finite spatial
coverage or per-step area closure exceeds its existing acceptance budget. Rejections are
recorded on the accepted ledger entry. Budgets stay unchanged; volume errors, cumulative
budget exhaustion and other failures stop the run. This is numerical refinement, not
evidence of timestep convergence.

## What the implementation does

The default `--layout varied` applies invertible spherical twists to the initial continent
outlines and shared plate network. Both sides of each plate boundary use the same curved
line, and actual native polygon areas calibrate continental coverage. `--layout simple`
retains the earlier unwarped geometry for comparisons. The twists choose a starting situation;
they never reshape evolved material or manufacture simulated history.
Continents are spherical polygons independent of the starting plates. Initial contacts are
classified again after placing the actual crust: converging continental sides start as
collisions, while a continent/ocean trench starts with the continental side overriding.
A chosen starting state supplies shelves and inherited mountain belts as a thickness field. Those belts
are initial conditions, not fabricated simulated prehistory. The default continent coverage is
65% of the sphere. An equal-area quadrature calibrates water to 55% emerged land, and the
equivalent global water depth is then fixed. Sea level is solved after each accepted mechanical
step; erosion uses the previous accepted water level. Every accepted ledger entry records
the sampled land fraction, sea level and water residual. Output does not update the model.
If an equal-area water quadrature point falls on a tiny numerical slit, the actual native
geometry is audited against the existing coverage budget before seeking nearby valid
surface samples. The displaced point count and maximum offset are recorded. The underlying
crust and rendered gap mask are not filled or hidden. Large coverage defects still fail.

Each continental province has an identity, formation time and a rotation into its material
reference frame. Ownership transfers reframe those coordinates, so fine relief moves with
the same rock rather than swimming over it in world coordinates. Reference rotations and
contact belts survive checkpoints. Collision belts remain as inherited weaknesses after
activity ceases. In this profile, closing 100 km does not automatically fuse two entire plates.
Disconnected continental masses are separate material bodies even when a plate owns several
of them. Their local fields and total volume are retained, so a small terrane is not tested
against the area of every continent on its plate. Partition remapping and any geometry cleanup
are recorded in the ledger.

Resolved relief comes from the continental thickness mesh and oceanic age-depth relation.
Domain-warped procedural detail amplifies that relief in material coordinates. It is an output
surface, not a second inventory of crust. Mechanics still use moving polygons and Euler
rotations; increasing map resolution does not create additional resolved tectonic physics.
The continental field's ordinary interior spacing is approximately 200 km. Boundary spacing
and force quadrature are separate settings.

Remeshing preserves native polygon coverage, including holes and disconnected components.
Boundary fragments use constrained triangulation in a gnomonic plane, where great-circle
edges are straight. This avoids fragile hand-written ear clipping. Native spherical polygons
remain authoritative, and the final coverage check still compares the mesh with their area.
Eight spherical charts cover continents that exceed one tangent plane, with shared seam
nodes stitched back into one field. Degenerate roundoff fragments cannot destroy the main
field or acquire invented area. Backward Euler transport conserves true node-area volume
within each disconnected continent, and erosion uses that same area measure. Captured
failed-run geometries exercise these cases.
The artist profile uses a yielding approximation for spreading: conductance is proportional
to `0.001 + (max(mean_edge_thickness - 35 km, 0) / 10 km)^3`. Thick belts relax quickly, while
ordinary cratons and thin margins move slowly. Volume is retained and relief is not clamped
to a chosen maximum. These effective mobility parameters require calibration. The lateral-escape coast probe uses the same local frame as the ocean it tests.
Artist sediment deposition fills available onshore lowland accommodation up to 200 m above
the current water level, within 500 km of an erosion source. Submarine transport and surplus go to the existing explicit
sea-export ledger. This prevents a tiny surviving receiver from becoming an arbitrarily high
sediment spike. Offshore export is accounted material, but an evolving offshore sediment
surface is still absent.

## What remains before accepting a billion-year world

- Continental opening stretches a band about 100 km wide on each side, conserving the
  existing rock volume. New ocean crust forms after the sampled edge thins to about 12 km;
  the previously stretched continental margins remain. This is a reduced extension rule,
  not a resolved fault network or lithospheric thermal model.
- Basement and deposited sediment are not separate evolving fields. The existing erosion
  and sediment export accounting remains in use.
- Material rotations track ownership changes. Continuous internal deformation and mixtures
  of buried source provinces are not resolved by those rotations.
- Collision history does not yet provide a calibrated mechanical model of every long-lived
  orogen. Whole-plate fusion is disabled in this profile; sustained belts still need long-run
  tests and comparison with plausible continental thicknesses and plate motions.
- No resolved mantle convection, global stress solution or predictive reconstruction is claimed.
  The reduced force balance and rift rules need sensitivity tests across seeds and timesteps.
- Fine relief is procedural. Drainage networks, sedimentary basins, climate-based colouring
  and a finished cartographic treatment need further work and visual review.

The next acceptance gate is a continuous, ledger-checked 100 Myr history, followed by 500
and 1,000 Myr runs, multiple starting seeds, timestep comparisons, and inspection of actual
landforms. Passing short tests alone is insufficient.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover \
  -s tests -p 'test_poly_*.py' -q
```

Tests distinguish continental coverage from emergence, conserve water after uplift, compare
material-attached terrain before and after actual plate transfer, verify exact restart surfaces,
exercise a force-driven scenario, check submerged erosion, and cover captured degenerate
geometries and wide connected continental meshes. Starting-state checks exercise invertible
twists, gap-free curved shared boundaries, local crust at every initial trench and collision,
and preservation of the last accepted checkpoint during interruption. Historical worlds without artist state
retain their existing checkpoint fields and whole-plate fusion policy.
