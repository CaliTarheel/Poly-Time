# Poly-Time design

Poly-Time replaces Super-Light-Time's fixed ownership grid with explicit polygons on the sphere, in the
spirit of the GPlates world-building process: plates bounded by typed boundary lines, ocean floor laid
down as isochron strips at ridges, ocean destroyed only where it crosses a trench, continents that
collide and suture. The physics (force balance, crust processes) chooses the motions and the events
instead of a person drawing them.

## Why

In Super-Light-Time, plate ownership and the ocean floor are fractions on a fixed grid of 5,120 cells.
Each step moves those fractions and then rescales every cell to exactly 100% ("normalization"). Where
plates converge without a trench, normalization deletes the overlapping ocean with no record, and it
also takes area from the overriding plate. At 343 Myr in the Asein35 run, about a fifth of the ocean
closed each step was lost that way at converging boundaries with no trench, while trench sinks recorded
only a third of what went down at real trenches. The same grid draws boundaries a cell at a time,
lets plates of one or two cells survive and run away, and cuts boundary lines through continental crust
in collision zones. Poly-Time removes the grid so that none of this can happen silently.

## Principles

1. **Closed crust ledger.** The sphere is tiled exactly by plate polygons, and each plate exactly by its
   crust polygons. Area enters only at ridges (new ocean) and leaves only across trenches (subducted
   ocean). Continental overlap in a collision becomes thickness, not lost area. Every term is recorded
   each step and the books must balance to round-off.
2. **Boundaries are first-class.** Every plate boundary is a stored line with a type: ridge, trench
   (with its overriding plate), transform, collision, or converging. Overlap or gap along a boundary is
   resolved by that boundary's rule, never by a generic cleanup.
3. **Explicit crust polygons.** Continental crust and ocean crust are polygons carried by their plate.
   Ocean polygons are isochron strips with a birth time; continental polygons carry an adaptive TIN for
   elevation, thickness and history where detail is needed.
4. **Geometry on the sphere.** All clipping, union and area use S2 through `spherely` (BSD). GPlates
   compatibility comes from exporting GPlates formats (rotations, plate polygons, isochrons), not from
   linking GPlates.

## One step

1. **Motions.** Solve each plate's rotation (Euler pole and rate) from the force balance. In the
   kinematic core the rotations are prescribed.
2. **Carry.** Rotate every plate polygon and its crust polygons by the plate's rotation.
3. **Resolve each boundary.** The carried plates now overlap where they converge and leave gaps where
   they diverge. Each overlap or gap is attributed to the boundary that swept it (the area between the
   boundary carried by one plate and by the other) and resolved by that boundary's type:
   - *ridge*: the gap is split along the ridge carried by the half-stage rotation; each side gains a new
     ocean strip of age zero;
   - *trench*: the boundary moves with the overriding plate; the downgoing plate's overlap is removed
     and booked as subducted ocean (area, age, length of trench); continental crust reaching the trench
     cannot subduct and turns the trench into a collision;
   - *collision*: overlapping continental crust is kept as thickening (area times thickness conserved);
   - *transform*: ideally nothing; any sliver from a boundary that is not a small circle about the
     relative pole is split along the half-stage line and booked as transform sliver;
   - *converging, no trench*: the overlap is booked as compression of the plate edge, so it is visible
     and bounded, and drives subduction initiation; it is never deleted.
4. **Topology events.** Subduction initiation (a converging boundary becomes a trench), collision and
   suture (a boundary closes and two plates may merge), rifting (a new ridge or rift line splits a plate
   and its crust polygons), ridge jumps and plate death. Every event is a change of boundary lines,
   followed by re-resolving the affected plate polygons.
5. **Crust processes.** Continental TIN processes (relief, erosion, arcs, collision stacking) run on the
   crust polygons; ocean properties come from isochron ages.
6. **Check the ledger.** Plate areas sum to the sphere; crust areas sum to each plate; the change of each
   is exactly created minus subducted plus recorded slivers. A step that does not balance is rejected.

## How boundaries evolve (phase 2, started)

- Every step, ridges, transforms and converging boundaries are re-typed from their plates' current
  relative motion: opening makes ridge, closing makes converging, mostly sliding (more than about 3.7
  times the opening or closing, i.e. within 15 degrees) makes transform. A 20% margin keeps a boundary
  near the threshold from flickering. A boundary whose type changes along it is split there, and
  neighbours of one plate pair and type are joined again.
- Closing never makes a trench directly. A converging boundary records how far it has closed (its
  overlap is booked as compression); at 100 km it becomes a trench. Ocean against continent: the ocean
  goes down. Ocean against ocean: the older goes down. Continent against continent: collision.
- A trench where more than 500 km2 of continental crust arrives in one step becomes a collision.

## Plates as regions (regions first)

Keeping a network of boundary lines and junctions consistent as plates move broke again and again
(junctions crossing, pinches, slivers, plates turning inside out). So the plates are the primary
objects (`poly/regions.py`): each plate is its crust, stored in the plate's own frame as GPlates stores
features, and its region is where that crust is. Every step:

1. each region is carried by its rotation;
2. the carried regions overlap where plates converge and leave gaps where they diverge. Both are
   resolved inside a strip around each run of the old boundary, by that run's type: at a trench the
   downgoing plate loses the overlap; anywhere else the overlap or gap is split along the pair's
   half-stage line. Pairs are resolved one after another, so ground where three plates overlap ends
   with exactly one owner. What is left of a gap goes to the nearest plate (booked as junction);
3. each plate's crust is updated in its own frame, age bands are merged, and the crust is cleaned;
4. shared boundaries are tied together (a point one plate has on its neighbour's edge becomes a
   vertex of both) and simplified on both sides at once, so the plates still tile the sphere;
5. the regions are rebuilt from the crust; a plate that came apart becomes separate plates, and
   crumbs of one plate left beside or inside another go to that plate;
6. the boundaries are derived again from where the regions touch. Every segment has a persistent
   identity: the old boundaries are indexed one by one with their midpoints carried by the pair's
   mean rotation, and a new segment takes the state (type, trench polarity, closing-so-far) and
   identity of the nearest carried old segment within 300 km, with one continuity rule: a segment
   whose nearest old boundary differs from both its neighbours' is beside another contact of the
   same plates, not part of it, and takes its neighbours' boundary when that is in reach. A segment
   with no old boundary in reach is a new contact, typed from the motion.

**Level of detail.** Nothing smaller than 4 km2 is tracked (`DETAIL_KM2`): a hole, a loop or a
leftover crust piece under that is debris, dropped or merged and still counted in the ledger's drift;
an outline vertex that cuts off less than that is not kept. Islands and arcs at that scale will come
from a separate process, not from the crust geometry.

**Geometry repair.** S2's outputs are valid, but a ring read back as longitude and latitude and
rotated moves by about 1e-16 and can touch or cross itself. A loop S2 says crosses itself is split at
the crossing into the loops that meet there, which keeps the signed area exactly; loops of the ring's
orientation with real area are built, folds and slits are debris. Snapping every vertex to a lattice
was tried and made things worse: the lattice is far coarser than the native error, and shared edges
pass through different plate frames, so it widened every slit.

The checks are the same: the plates tile the sphere, each plate's crust tiles it, and the area created
equals the area destroyed, term by term.

## Collisions, sutures and rifts (phase 2, continued)

- **Continent keeps its volume.** Continental crust that loses area anywhere but down a trench
  (collision, compression, slivers) thickens instead: each piece carries a mean thickness (35 km to
  start), and the volume taken is added back to the piece, or to its plate's largest other continental
  piece if it was taken whole. The ledger records the volume thickened and the total continental volume
  every step. (One thickness per piece until continents carry a mesh.)
- **Suture.** A collision records how far it has closed; at 100 km the two plates weld. The smaller
  plate's crust passes into the larger's frame unchanged, the boundaries between them go, and its other
  boundaries become the larger plate's with their state. The suture line is kept, in the welded plate's
  frame like its crust, and moves with it (and passes to a new plate when a rift takes its side).
- **Rifting by tension.** At the start of each step of a force-driven run, straight cuts (great circles)
  through each large plate's centre and through each of its continents are tested, twelve directions
  each. The plate moves rigidly, so each side of a cut needs the other to carry it: the external torque
  on one side (drivers, neighbours' resistance, basal drag) fitted as an even pull across the cut is the
  cut's tension. The strength along the cut is the mean of ocean (1.6e13 N/m) and continent (3e12 N/m)
  by length, so continents are where plates break; a stretch of cut within 150 km of an old suture is
  weaker still (3e11 N/m), and the great circle fitted to each suture is tried as a cut of its own, so
  plates break again where they welded. The cut whose tension most exceeds its strength splits the
  plate, at most one a step; the new plate starts in its parent's frame, takes the crust and boundary
  state on its side, and the cut is typed from the motion at the next step. Both parts must be at
  least 3 Mkm2.
- **Slab pull by age.** Pull per metre of trench scales with the square root of the age of the crust
  going down (3e13 N/m at 80 Myr, capped at 1.25x), sampled 150 km down the downgoing side; young
  ocean at a new trench pulls little.
- The three-continent test world is continent-poor (9.4 Mkm2, under 2% of the sphere, against about
  30% on Earth), so most of its cuts are nearly all ocean and most of its rifts are oceanic; on an
  Earth-like world the continental weakness dominates.

## What carries over from Super-Light-Time

- The force balance physics (slab pull, basal drag, ridge push, resistances), with inputs taken from
  polygons and boundary lines instead of grid fractions and contour pieces.
- The continental crust mesh and its processes (relief, erosion, arcs, collision stacking) as the TIN
  inside continental polygons.
- Frames, viewer and checkpoints, re-plumbed.

Rewritten: ownership, ocean transport, spreading, subduction capture, boundary reconstruction, trench
matching, initiation and rifting (about 70 of the 182 modules a step uses touch the grid).

## Phases

1. **Kinematic core**: geometry layer, plate and crust polygons, typed boundaries, ridge strips and trench
   consumption under prescribed rotations, closed ledger, tests, map rendering.
2. **Topology events**: initiation, collision and suture, rifting and splitting, triple junctions.
3. **Dynamics**: the force balance on polygons and boundary lines.
4. **Crust**: continental TIN and its processes; isochron ocean properties.
5. **Runs**: frames, viewer, GPlates export, comparison with Super-Light-Time runs.
