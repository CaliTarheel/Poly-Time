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
6. the boundaries are derived again from where the regions touch, each segment keeping the type,
   trench polarity and closing-so-far of the nearest old segment of the same plates.

The checks are the same: the plates tile the sphere, each plate's crust tiles it, and the area created
equals the area destroyed, term by term.

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
