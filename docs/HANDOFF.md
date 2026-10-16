# Poly-Time handoff

Written 2026-10-16 at the end of a long session. It says where the polygon model stands, how to
run and test it, what was learned the hard way, and what to do next, in order. `POLY-TIME-DESIGN.md`
beside this file is the design; this is the state.

## Where the code is

- Repo: https://github.com/CaliTarheel/Poly-Time (public). Local clone `C:\Users\thoma\Downloads\Poly-Time`
  (checked out on `poly/erosion`) and a worktree `C:\Users\thoma\Downloads\Poly-Time-arcs` on `poly/arcs`.
  The worktree exists so arcs-branch commits never switch the main checkout under a running job.
- Branches and PRs:
  - `main` = PRs #1-#6 merged (regions-first core, suture and rift, continental mesh).
  - `poly/arcs` (PR #7 -> main): arcs, ocean depth from age, escape fixes, islands, terranes, plate
    speeds, slab rollback, trench suction, hotspots and plume heads, the dust fixes. Head 5bff9b9.
  - `poly/erosion` (PR #8 -> poly/arcs): erosion on top of all of that. Head 39801c9.
  - Both green locally (41 and 42 tests) and on CI at those commits. Merging #7 then #8 is the
    owner's call; nothing is waiting on code.
- Python: the repo's own `.venv` (`.venv/Scripts/python.exe`, numpy 2.3, scipy, spherely, Pillow).
  From the worktree use `../Poly-Time/.venv/Scripts/python.exe` with `PYTHONPATH=.`.
- Tests: `python -m unittest tests.test_poly_core tests.test_poly_regions tests.test_poly_evolve
  tests.test_poly_dynamics` (about 3 minutes). The old grid engine's tests are also in `tests/`
  and the CI shards run them with a known-failures list; they are not this model.

## Run scripts (beside the repo, not in it)

All take `<myr>` and an optional resume pickle, print one line per Myr, and pickle the world
every Myr so a run can be resumed or inspected.

| script | what |
|---|---|
| `../Poly-Time-calib.py 40` | the Earth-like world (`voronoi_world(count=8, seed=3, speed_km_myr=30., continent_fraction=.3)`) with hotspots on, vertex and piece counts per Myr, no maps; pickle `../Poly-Time-calib-last.pkl` |
| `../Poly-Time-islands.py 30` | same world with maps every 5 Myr to `../Poly-Time-maps/islands-NNNmyr.png` |
| `../Poly-Time-earth6.py 60 [resume.pkl]` | the 60 Myr evidence run of PR #7 (pre-island arcs) |
| `../Poly-Time-speed.py <myr> <drag multiple of 1e15>` | speed calibration, rms and max per Myr |

Maps: `poly.render.draw(world, path, width=1440)`; `mesh=True` outlines every continental mesh
triangle; 4096 px takes about 12 minutes (per-pixel containment against every piece).

States worth keeping: `../Poly-Time-calib-040.pkl` (final code, 40 Myr, 25 plates, 539 pieces),
`../Poly-Time-earth6-last.pkl` (60 Myr, pre-island arcs). Old pickles load: `Boundary.arcs()`
upgrades the one-column arc state, and a pickled world has no plumes until `enable_hotspots()`.

## What the model does now (one line each; the design doc has the rest)

Regions-first plates carried by rotation, boundaries re-derived each step with per-segment
identity; crust in plate frames with a closed ledger (every km2 created or destroyed is booked by
process, every km3 of continent accounted for); continental mesh with thickening, spreading,
lateral escape, erosion; subduction initiation, collision, suture and thicken, tension rifting
(sutures and fresh plume heads are weak); arcs that thicken a continent or build spaced islands
after a ridge-first incubation and a ribbon when mature; terranes that dock whole; forearcs that
shed from retreating trenches, open back-arcs for 25 Myr and weld back; plumes with heads
(plateaus or flood basalts) and tails (island chains); ocean depth from age on the maps.

## Knobs (defaults; all in `poly/regions.py` unless said)

| knob | value | meaning |
|---|---|---|
| `DETAIL_KM2` (sphere.py) | 4 | level of detail; smaller parts are debris |
| `SIMPLIFY_KM2`, `SIMPLIFY_NET` (sphere.py) | 100, 1e-4 | vertex removal threshold; net area a ring may lose per pass |
| `MIN_RUN_KM` | 50 | a boundary kind runs at least this far |
| `ARC_KM3_PER_KM_MYR`, `ARC_REFERENCE_KM_MYR` | 30, 50 | arc magma per km of trench at the reference convergence |
| `ISLAND_SHARE`, `ISLAND_KM2`, `ISLAND_SPACING_KM`, `ARC_EMERGE_KM3_PER_KM` | .05, 50, 70, 600 | emergent share, chunk, site spacing, ridge-first incubation |
| `ARC_WIDTH_KM x ARC_THICK_KM` | 60 x 30 | the mature ribbon (60 Myr at reference convergence) |
| `TERRANE_KM2` | 1e6 | below this a continent piece docks, above it collides |
| `ROLLBACK_KM_MYR`, `ROLLBACK_MARGIN_KM_MYR`, `FOREARC_KM`, `FOREARC_MAX_RUN_KM`, `BACKARC_LIFE_MYR` | 20, 10, 250, 3000, 25 | hinge retreat at 80 Myr slab age, opening margin, forearc width, stretch, lifetime |
| `HOTSPOT_COUNT`, `HOTSPOT_KM3_PER_MYR..STRONG`, `LIP_KM3`, `LIP_THICK_KM` | 12, 1e4..1.5e5, 5e6, 30 | plumes |
| `Parameters.basal_drag_pa_s_per_m` (dynamics.py) | 3e15 | set so the Earth-like world moves like Earth |
| `Parameters.trench_suction` (dynamics.py) | .2 | share of slab pull drawing the overriding plate |
| `Rifting.continent_strength_n_per_m` (dynamics.py) | 3e12 | rift rate: 1e12 gave 8 rifts in 40 Myr, 3e12 gave 6 |

## Evidence from the final 40 Myr run (code 5bff9b9, 2,209 s)

25 plates at the end (10 forearcs; 25 shed, 15 welded back); 598 arc islands, 76 arc ribbons,
44 terranes docked, 5 plume heads (4 ocean plateaus, 1 flood basalt), 7 rifts, 5 collisions.
Books: 76.159 Mkm2 created, 76.158 destroyed, drift 732 km2. Continental volume +5.95e7 km3
against 5.99e7 km3 of arc and plume magma booked. 539 crust pieces, 102k vertices, 5.7% of edges
under 100 m. 86-138 s per Myr late in the run. Non-forearc plates 20-40 km/Myr, fastest 60-130.

## What to do next, in order

1. **Merge PRs #7 and #8** (owner). Nothing is waiting on code.
2. **The island population is the cost.** By 35 Myr the model adds 40-50 islands a Myr, each a
   24-vertex polygon plus its hole in the ocean piece; 500 pieces cost 130 s per Myr, so 1,000 Myr
   is about 30 hours. Options, cheapest first: keep an island as a point feature (plate, position,
   volume) until it reaches a few hundred km2 and only then cut a polygon; bigger chunks; or hold
   all of a plate's islands in one multipolygon piece so the per-piece overhead (clean, contains,
   rotation) is paid once. Measure with `../Poly-Time-calib.py` and the per-Myr vertex line.
3. **Trace the volume gap with plumes.** Continental volume closes to 0.2% of its change without
   plumes and 0.6% with them (3.8e5 km3 over 40 Myr, 0.007% of the total). Instrument
   `_continent_km3()` before and after each stage of `step` (the pattern used for terranes) on a
   state with active plumes; suspects are the plateau remainder returned to the bank in
   `_hotspots` and plume chunks landing on islands.
4. **Look at the rollback dynamics once more.** Suction at .2 barely moves a 60-100 Mkm2
   overriding plate, so nearly every old-slab margin sheds and the 25 Myr lifetime is what caps
   the count. Candidates: a trench-normal "advance" term that includes the overriding plate's
   own rollback-driven drift, shedding only when a back-arc would open faster than some floor,
   or a per-margin cooldown after a weld. Also remnant arcs: today the whole arc rides with the
   forearc and nothing is left behind.
5. **Ribbons come early on fast trenches** (from 13 Myr at 200+ km/Myr convergence; 76 in 40
   Myr). Decide whether the ribbon should exist at all now that islands grow together, or raise
   its maturity.
6. **GPlates export** (deferred by the owner): rotation file plus per-step polygons, the stated
   route to GPlates compatibility.
7. **Housekeeping**: prune the old grid engine and its tests from this repo; normalise line
   endings (sphere.py and regions.py are CRLF, the tests LF, so edit with newline-preserving I/O);
   the standing crust-vs-plate offset of about 100 km2 after some sutures.

## Lessons that cost time (do not relearn them)

- When a run stalls with modest state, dump the stack
  (`faulthandler.dump_traceback_later(240, exit=True)` on the resume pickle); do not profile
  first. A test timed while another run hogs the CPU reads 10x slow.
- Watch the vertex count every Myr whenever simplification, zones or boundary typing change. The
  two blow-ups this session were a per-ring area budget that cancelled removals instead of
  trimming them, and boundary kinds flipping every few metres on a slow boundary (each flip a
  zone strip, each strip cutting more dust). `edges under 100 m` and `simplify_shared(stats=)`
  per-round candidates/go are the diagnostics.
- Any per-piece loop in the step is now multiplied by hundreds of islands (`dynamics.cuts`
  rotated every continent for every candidate cut: 630k rotations per plate per sub-step).
- A prescribed plate must be imposed inside the force balance, not overridden after it; a free
  sliver between a slab and its trench lets the slab-pulled plate race.
- Dock terranes as bodies: slice by slice, the rest of an island keeps arriving at the same spot
  and stacks onto its own first slice.
- Small convex shapes lose area to simplification on the same side every pass; a hemisphere loop
  defeats the vertex-fan area formula (use `left_area_km2`, Gauss-Bonnet).
- Processes that create continent (islands, plumes) must be opt-in or seeded explicitly in test
  worlds, or every accounting test breaks.
- S2 `contains` is False for points on an outline; never decide loss or ownership on containment
  alone for outline nodes.
- The 0.1% accuracy standard is the bar; do not chase sub-4 km2 geometry.

## Memory for the next agent session

The user's auto-memory (`~/.claude/projects/C--Users-thoma-Downloads-Lite-Time/memory/poly-time.md`)
holds the same facts in session order, with commit hashes.
