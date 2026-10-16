# Geological state after the current feature work

Status: implementation plan, not implemented physics. Reviewed against main
`a5a3d2b` and the arc/erosion stack `5cfd633` on 2026-10-16.

The polygon/Euler-frame model remains the foundation. Terranes, rollback,
plate-speed changes and LIPs are already being developed. This plan describes
the interfaces and geological behaviour to agree on around that work, without
requiring a mantle-convection solver or a wholesale engine replacement.

## Order of work

1. Land conservative state transfer, domain-respecting continental meshes,
   independent gap/overlap and material checks, and reliable restart support.
2. Agree on the event and surface-state contracts below with the active feature
   work. Add small prescribed-motion fixtures before tuning force-driven runs.
3. Implement sustained continental collision and finite-width rifted margins.
4. Separate basement and sediment, add post-rift subsidence, then derive land,
   shallow-water connections and deeper ocean basins from one height datum.
5. Qualify the combined model over longer histories and multiple initial worlds.

Useful observational targets are persistent mountain belts, basin development,
changing continental flooding, seaway connectivity and ocean-age distributions.
These are not claims that a fictional planet must match Earth's exact rates.

## Contract: geological state survives ownership changes

Plate ID denotes a moving frame, not the identity of a geological feature.
Natural disconnection, absorption, explicit rift, accretion and plate fusion
must all publish the same kind of spatial transfer information: source frame,
destination frame, transferred domain, and source-feature lineage.

| State | Required transfer rule |
| --- | --- |
| Boundary type and polarity | Inherit by spatial correspondence; relabel the participating plates. |
| Accumulated strain or convergence | Remap an intensive field with a declared weighting. |
| Magma, sediment and other inventories | Partition or combine conserved amounts; never copy a whole inventory into every new segment. |
| Sutures and other material lines | Clip between daughters, rotate to the receiving frames, retain source lineage. |
| Retired boundary inventory | Keep an explicit reservoir or named physical sink; do not silently discard it. |
| Thermal and deformation history | Travel with the affected crust, not with a currently nearest boundary. |

Additions to this state must be included in checkpoints and replay tests in the
same change. A topology edit at zero velocity must not create magmatism, erase
an external trench, or move a material line to a different place.

## Sustained continental collision

Current behaviour: a collision run reaching 100 km of accumulated convergence
can merge its entire two plates. This conflates local crustal welding and plate
retirement. The new terrane treatment should not inherit this as a requirement
for every larger collision.

Represent these as separate decisions:

- **Accretion:** a particular crustal body changes host plate.
- **Collision:** the boundary remains active, with shortening, thickening and
  any chosen lateral escape acting locally.
- **Mechanical coupling:** resistance changes with collision state while both
  plate rotations can remain independently solved.
- **Plate fusion:** an explicit event retires one rotation only when the chosen
  model's criteria hold; an isolated short collision run is insufficient by itself.

Before implementation, choose and document the fusion criterion. A possible
minimal model uses sustained low relative motion together with the extent and
continuity of the coupled contact, checks remaining contacts, and exposes its
thresholds as parameters. These are proposed modelling choices, not calibrated
laws. Avoid a globally acting fixed-distance trigger as the only criterion.

Acceptance fixtures:

1. A local continent-continent collision permits continued convergence and
   thickening without immediately deleting remote boundaries between the plates.
2. A small terrane accretes without fusing its former host oceanic plate to the
   overriding plate.
3. A deliberately fully coupled pair can fuse, retaining external boundary state,
   material geometry and volume.
4. A later rift crossing the old suture leaves the proper suture fragment on
   each daughter plate.

## Finite-width rifts and passive margins

Current behaviour partitions crust along a great circle, then fills opening with
new ocean. It has no pre-breakup continental stretching history. A minimal next
model needs a finite deforming strip within otherwise rigid plates.

Use an explicit progression: intact continent -> stretching margin -> breakup
-> oceanic spreading. Store stretching and breakup/cooling age on the margin.
Conserve basement volume during thinning, and create new oceanic crust only in
the portion that has actually broken up. An abandoned rift must be able to remain
a continental basin rather than becoming an ocean automatically.

Do not commit to a particular thinning or cooling coefficient until the active
rollback and LIP interfaces are settled. Rollback can supply extension; LIPs may
supply uplift, heating or magmatic additions. Neither should independently and
silently overwrite the other process's height or crust-volume field.

Acceptance fixtures:

- Prescribed extension thins a continental strip while closing its volume budget.
- Breakup creates age-zero ocean between two preserved continental margins.
- An abandoned rift retains its thinning and cooling history.
- A quiet passive margin subsides after extension stops, without requiring a
  currently active boundary beside it.
- Halving the integration step yields converging event timing and volume totals.

## One surface datum; separate crust, sediments and water

The present Airy formula is a useful provisional baseline. It must not make
`kind == 'continent'` synonymous with dry land. For example, the existing 20 km
continental island crust evaluates to -2.25 km under the 35 km reference law.
Rendering it as land cannot resolve that model inconsistency.

Start with a shared, read-only surface sampler used by maps and exports. Return
crust kind, height relative to a declared datum, water-relative height and masks.
An optional prescribed sea level is a scenario input, not a water-volume solution.

The subsequent physical state should distinguish:

| Quantity | Meaning |
| --- | --- |
| Basement thickness/composition | Crystalline crust and its density assumptions. |
| Sediment column | Deposited material, with its own mass/volume and optional compaction. |
| Stretching and thermal state | Inherited deformation and cooling affecting subsidence. |
| Surface height | Derived from the above and explicitly represented loading/uplift terms. |
| Sea level | A stated reference or a water-inventory solution using basin capacity. |

Keep the initial implementation small: separate basement from a sediment column,
and define a single documented isostatic height function. Full flexural loading,
compaction and dynamic topography can follow. Every added process must state
whether it transfers existing material, adds mantle-derived material, changes
density/temperature, or changes water inventory.

Sediment exported to the sea must remain in an explicit reservoir until an ocean
deposition model consumes it. Merely adding sediment to basement thickness loses
the distinction needed for later subsidence and stratigraphy.

Acceptance fixtures include a submerged continent, an emerged volcanic body with
an explicit height law, a flooded continental interior, a rift basin filled by
sediment, and a seaway that opens or closes when sea level changes. Crust area
must remain unchanged in a pure sea-level scenario.

## Event timing must not depend on the output interval

Separate output cadence from integration substeps and from event accumulation.
At the reviewed base, `_collide` uses 500 km2 arriving in one call: a single
600 km2 arrival triggers collision, while two 300 km2 arrivals do not. The active
terrane/collision work should choose a cumulative or geometric contact rule and
provide 1, 0.5 and 0.25 Myr comparisons for the same prescribed history.

The rift search also currently runs once per outer `dynamics.run` step. A change
to movie-frame cadence must not silently change the number of physical rift
opportunities. Use a defined physical event schedule or convergence-controlled
event checks, with that schedule included in restart state.

## Qualification and release gate

Use a small, reproducible suite covering low and high continental fractions,
microplates, ocean-ocean subduction, sustained continental collision, rifting and
reassembly. Start with prescribed motions to isolate geometry and material
behaviour, then repeat representative worlds with solved forces.

Record cumulative material sources/sinks and numerical repair, independent gaps
and overlaps, pending inventories, plate-size distribution, ocean-age
distribution, boundary lengths, emerged land and shallow-water area when those
surfaces are available, and runtime/geometry growth. Check timestep and spatial
resolution convergence over short controlled episodes; do not require identical
chaotic plate histories after hundreds of Myr.

Save the configuration, code revision, dependency versions, seed and final
accepted state. A short passing test suite is a mechanism check, not qualification
of a 1000 Myr history. Report the actual duration and tested configurations with
each longer-run result.

## References informing the design

- Scotese and Wright (2018), [PALEOMAP PaleoDEMs](https://www.earthbyte.org/webdav/ftp/Data_Collections/Scotese_Wright_2018_PaleoDEM/Scotese_Wright2018_PALEOMAP_PaleoDEMs.pdf):
  distinguishes continental positions, topography, flooding and the persistence
  of geographic features through successive time slices.
- [pyGPlates crustal thickness and tectonic subsidence example](https://www.gplates.org/docs/pygplates/sample-code/pygplates_reconstruct_crustal_thickness_and_tectonic_subsidence):
  demonstrates the connection between deforming regions, thickness and subsidence.

These references motivate the represented quantities; they do not validate the
present empirical coefficients or make the proposed simplified model a calibrated
Earth reconstruction.
