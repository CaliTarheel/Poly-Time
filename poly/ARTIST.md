# Artist evolution prototype

This branch adds an opt-in artist scenario to the spherical polygon core. It is a working
implementation and qualification harness, not a claim that a convincing 1,000 Myr world
has been achieved. It depends on the reviewed stack through PR #16.

## Run and inspect

Use the repository's numerical dependencies, including the native geometry codec's
recorded spherely version. The implementation was exercised with Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0 and spherely 0.1.1.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m poly.artist_run start \
  --output runs/artist-01 --seed 1 --plates 8 \
  --land .55 --continents .65 --duration 1000 \
  --dt .5 --snapshot 5 --width 1600 --spacing 150 --samples 20000

python -m poly.artist_run resume --output runs/artist-01 --duration 100
```

Open `runs/artist-01/replay.html` locally. It embeds the actual saved maps, with a draggable
globe, flat projection, time slider, playback, boundary overlay and inherited belt overlay.
The maps and timeline do not require a server or external JavaScript libraries. A target of
1,000 Myr is an instruction to the runner; inspect `manifest.json` for the duration actually
accepted. A failed integration preserves its checkpoint and records the error.

`checkpoint.json` contains the accepted native spherical geometries, moving frames,
continental fields, boundary state, material ledger and artist history. `history.json` records
the measurements and geometry for each displayed epoch. `diagnostics.json` contains core
area and volume budgets. Source and package hashes accompany the checkpoint. Resuming
with changed source or a changed numerical environment is deliberately rejected.

The duration argument counts complete outer integration intervals. On resuming a checkpoint
inside an interval, its unfinished part counts as the first interval. The manifest records the
resulting exact target time. Force and rift decisions retain the original timestep cadence;
changing the snapshot interval does not change integration decisions.

## What the implementation does

Continents are lobed spherical polygons independent of the starting Voronoi plates. A chosen
starting state supplies shelves and inherited mountain belts as a thickness field. Those belts
are initial conditions, not fabricated simulated prehistory. The default continent coverage is
65% of the sphere. An equal-area quadrature calibrates water to 55% emerged land, and the
equivalent global water depth is then fixed. Sea level is solved after each accepted mechanical
step; erosion uses the previous accepted water level. Output does not update the model.

Each continental province has an identity, formation time and a rotation into its material
reference frame. Ownership transfers reframe those coordinates, so fine relief moves with
the same rock rather than swimming over it in world coordinates. Reference rotations and
contact belts survive checkpoints. Collision belts remain as inherited weaknesses after
activity ceases. In this profile, closing 100 km does not automatically fuse two entire plates.

Resolved relief comes from the continental thickness mesh and oceanic age-depth relation.
Domain-warped procedural detail amplifies that relief in material coordinates. It is an output
surface, not a second inventory of crust. Mechanics still use moving polygons and Euler
rotations; increasing map resolution does not create additional resolved tectonic physics.
The continental field's ordinary interior spacing is approximately 200 km. Boundary spacing
and force quadrature are separate settings.

Remeshing preserves native polygon coverage, including holes and disconnected components.
Eight spherical charts cover continents that exceed one tangent plane, with shared seam
nodes stitched back into one field. Degenerate roundoff fragments cannot destroy the main
field or acquire invented area. Backward Euler transport conserves true node-area volume
within each disconnected continent, and erosion uses that same area measure. Captured
failed-run geometries exercise these cases.

## What remains before accepting a billion-year world

- Finite-width rifting and passive-margin growth are not implemented. Rift history is recorded,
  while new opening gaps still follow the existing ocean-creation rule.
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
geometries and wide connected continental meshes. Historical worlds without artist state
retain their existing checkpoint fields and whole-plate fusion policy.
