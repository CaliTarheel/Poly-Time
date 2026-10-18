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
Resolved disconnected continental masses are separate material bodies even when a plate owns
several of them. Their local fields and total volume are retained, so a terrane is not tested
against the area of every continent on its plate. Partition remapping and any geometry cleanup
are recorded in the ledger. The artist profile uses the existing 20,000 km2 continental mesh
threshold (a disc about 160 km across) as the minimum independent continental body. This is
an artist-model resolution choice, separate from the unchanged 250,000 km2 minimum plate area.
It does not remove meaningful microplates.
Clipped mesh vertices can also have microscopic control areas inside a large continent. The
artist profile ties those cells to regional field patches with at least 10,000 km2 support
(one quarter of the ordinary 200 km spacing squared). Nearest resolved nodes are selected by
mesh-edge distance, so disconnected coasts cannot exchange rock. A component with no resolved
node keeps one volume-weighted value. Each patch retains the sum of its real node-area volumes;
no artificial nodal mass or thickness ceiling is introduced. Grouping is deterministic and
idempotent on unchanged geometry, and the internal redistribution and smallest resulting patch
are recorded. This prevents a newly clipped 1.9 km2 cell from supporting its own mountain spike
just before a snapshot, after the timed spreading solve has already run.

Resolved relief comes from the continental thickness mesh and oceanic age-depth relation.
Domain-warped procedural detail amplifies that relief in material coordinates. It is an output
surface, not a second inventory of crust. Mechanics still use moving polygons and Euler
rotations; increasing map resolution does not create additional resolved tectonic physics.
The continental field's ordinary interior spacing is approximately 200 km. Boundary spacing
and force quadrature are separate settings.

Artist fracture candidates start with the existing great-circle loading screen. Up to three
supercritical candidates are routed within a 350 km corridor using resolved crustal thickness
as an effective resistance proxy and inherited collision/suture belts as weaknesses. A temporary
100 km by 50 km search lattice and 200 km smoothing scale choose a regional path; that lattice
is not the planet's material representation. A homogeneous resistance field receives no
decorative wiggles. This is a bounded path search, not a global stress or fracture solution.
The actual curved spherical polygon partitions the parent. Both children share that same
boundary, and existing segment state is assigned using its actual sides. Loading is fitted
again using the curved partition, segment lengths and local normals; a path cannot inherit a
supercritical straight cut's old score. The source field, effective resistance, search corridor
and smoothing scales need calibration.

Every real opening gap beside continental rock can stretch that material, including the first
opening while a new boundary still has a provisional transform label. Previously that first
gap became an age-zero ocean slit before the boundary was retyped as a ridge; later continental
stretching could not undo the already-emplaced slit. The artist profile now stretches a finite
200 km margin band first. A compact smooth thinning kernel leaves remote cratons unchanged and
funds the new footprint from that band's existing volume. Each coastal section stops extending
when its resolved edge thickness reaches 12 km, allowing local ocean birth; a thick section of
the same coast no longer keeps a thin section extending through a whole-coast average.

Remeshing preserves native polygon coverage, including holes and disconnected components.
Boundary fragments use constrained triangulation in a gnomonic plane, where great-circle
edges are straight. This avoids fragile hand-written ear clipping. Native spherical polygons
remain authoritative, and the final coverage check still compares the mesh with their area.
Eight spherical charts cover continents that exceed one tangent plane, with shared seam
nodes stitched back into one field. Degenerate roundoff fragments cannot destroy the main
field or acquire invented area. Backward Euler transport conserves true node-area volume
within each disconnected continent, and erosion uses that same area measure. Captured
failed-run geometries exercise these cases.
Chart selection uses the same stable duplicate-cleaned rings as triangulation. Using a raw
ring's point-weighted centre before removing duplicates could select a chart that its cleaned
ring could not use, dropping the main component. A captured 27.8 million km2 continent tests
full coverage across this transition; the full-sphere case also uses native octant charts.
The artist profile uses a yielding approximation for spreading: conductance is proportional
to `0.001 + (max(mean_edge_thickness - 35 km, 0) / 10 km)^3`. Thick belts relax quickly, while
ordinary cratons and thin margins move slowly. Volume is retained and relief is not clamped
to a chosen maximum. These effective mobility parameters require calibration. The lateral-escape coast probe uses the same local frame as the ocean it tests.
Artist sediment deposition fills available onshore lowland accommodation up to 200 m above
the current water level, within 500 km of an erosion source. Submarine transport and surplus go to the existing explicit
sea-export ledger. This prevents a tiny surviving receiver from becoming an arbitrarily high
sediment spike. Offshore export is accounted material, but an evolving offshore sediment
surface is still absent.
After every plate has resolved its losses, sub-resolution continental bodies are coarsened.
Their rock joins the nearest resolved continental field within the existing 300 km material
reach, distributed through that field's local area weights. This is numerical homogenization,
not physical transport across a small ocean gap. Source province, material reference, transferred
volume and receiver are retained in the step ledger; individual sub-resolution texture is lost.
An isolated body's rock remains in a passive inventory with its source province and a location
in its moving plate frame. It is emplaced when a resolved receiver comes within reach. The inventory
is checkpointed, follows splits and handovers, and is included in continental-volume conservation.
It has no independent relief, force, erosion or deformation solve while unresolved.
The old surface footprint remains on the same plate as age-unknown marine cover, merged into
one mask per plate. No gap, ridge, new magma or fabricated seafloor age is introduced. Coarsened
area and volume, deposition and outstanding inventory are reported explicitly. Coarsening runs
again after cleanup, plate regrouping and arc emplacement, before solving water level. Water is
calibrated after initial coarsening, so the starting emerged fraction remains 55%.
This rule prevents rock from being squeezed into shrinking numerical specks. It deliberately
omits small islands as mechanical bodies; it does not establish that resolved collision relief
or the 300 km homogenization reach is calibrated.
Whole-terrane accretion also waits until every original owner has resolved its border losses.
A terrane arriving at a trench can simultaneously overlap a third plate. Transferring its
uncut outline early carried that third overlap into the receiver. The artist profile now
cuts those original overlaps first, retaining the rock volume, then transfers the surviving
terrane. A captured 18.59 Myr contact tests the original coverage and volume budgets. Its
fixture retains native geometry and frames with simplified mean thickness fields; it is
a single-step regression, not a continued qualification history.

The diagnostics and every accepted ledger entry include the maximum continental thickness
over all resolved mesh nodes and any historical uniform bodies. Passive unresolved material
has no defined surface height. Snapshot metrics also report the corresponding maximum
resolved height. The older `maximum_height_km` remains a maximum of equal-area samples of the
amplified surface; those samples can miss a narrow high island. Inspect the native-field maximum
when qualifying relief. No thickness or height is clipped to make a map pass.

## What remains before accepting a billion-year world

- Continental opening stretches a band about 200 km wide on each side, conserving the
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
The captured 95.176 Myr rotation failure is a regression fixture: reconstruction of a duplicate
vertex must preserve the signed area of the separated loops, rather than accepting the removal
of a finite fold under a plate-sized relative allowance. The original spatial and cumulative
acceptance budgets remain unchanged.
Planetary coarsening checks conserve total rock, preserve surface coverage, retain isolated
material instead of sending it to a distant continent, and carry source reference coordinates
through plate handover and exact checkpoint restart. Diagnostics separate resolved continental
volume from passive unresolved volume while the acceptance ledger closes their sum.
Fracture checks follow a curved inherited weakness, rescore its actual geometry, preserve
coverage and source references through the curved split, and keep the first provisional rift
gap continental. A mixed-thickness coast tests local breakup, and a homogeneous resistance
field tests that routing does not manufacture noise. Controlled prescribed-opening runs
compare the previous and new rules with identical initial rock and motions; these are process
calibrations, separate from a force-driven planetary qualification.
A captured coastal field tests regional support, volume conservation and repeated projection.
Separate component volumes stay separate, and a uniform 250 km thickness remains 250 km,
demonstrating that the projection is a resolution choice rather than a hidden relief ceiling.
