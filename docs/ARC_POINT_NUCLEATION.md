# Volcanic points and compact arc foundations

This branch extends the local arc deposition change with version-1 volcanic points and compact foundation proposals. It is an exploratory representation of unresolved pending magma, not an eruption, magma-plumbing or mantle-flow solver.

## From pending source to resolved crust

1. Existing subduction supply creates or replenishes geographically located source origins. Each origin already records its volume, persistent overriding-plate host, trench episode, connected source component and transport history.
2. A positive pending origin appears as an amber volcanic-center point. The feature is derived from the canonical pending ledger; it adds no separate crust, topography, mechanical force or second magma inventory. It moves exactly once with that ledger's host and keeps its original host identity if a plate slot is retired.
3. Compatible, connected local origins can fund a compact foundation centered on a real source. A stateless hash of the world seed and anchor origin controls its small shape/orientation variation. Source positions and magma supply are not randomized or relocated.
4. The proposed foundation must contain every contributing source, preserve the induced source connectivity, fit uncovered intended-owner water and pass the existing exact geographic exclusion checks. Physical columns remain between 8 and 75 km and constructive slope remains at most 20 degrees.
5. An accepted proposal creates a 24-face polygonal crust patch and consumes only the accepted per-origin pending volume. Failed or unused supply remains pending. The existing arc growth and local deposition paths subsequently handle the patch.

Density here means funded magma volume per physical footprint, not point count. The existing conversion is 25 km of source-column equivalent per source-area unit. The 8 km minimum is crust-column thickness, not a required volcano height. A patch becomes an island only if its modeled surface reaches sea level.

The compact proposal is an additional shape choice alongside the existing elongated footprint. Its roughly circular footprint can distribute the same volume with gentler slopes. The older shape is tried if compact admission fails for a local group. Local group partitions can differ between the two shapes, so this is not a guarantee that every previously admissible placement remains available in the same transaction.

## Saved state and interface

Fresh native simulations declare `native_arc_point_version = 1`. An older checkpoint without that flag retains its previous behavior. Explicit legacy profile/footprint versions also disable the extension. Publishing this source does not upgrade a running simulation or rewrite historical frames.

Versioned snapshots include `arc_point_policy`, `volcanic_point_features` and promotion receipts. Validators check that every point corresponds exactly to one positive pending origin and that promotion receipts agree with actual source partitions, lineage, physical area and volume. The existing source placement ledger remains the enduring material record. Typed checkpoint reload preserves the policy and canonical source inventory.

The Volcanic centers toggle shows screen-sized amber rings on flat maps and globes. Hover gives the pending volume, recorded host and source time. Symbols never paint land into the elevation raster or claim island size. Old unversioned frames do not acquire invented point histories.

## Focused validation

`tests/test_arc_point_nucleation.py` covers seeded reproducibility, pole/seam rotation, single advection, retired host identity, conservation, failed promotion without mutation, lineage isolation, source subdivision, checkpoint continuation and invalid snapshot/receipt rejection.

One physical fixture at a -6,000 m basement uses 100 km² of source equivalent (2,500 km³). The old elongated proposal cannot meet the 20-degree slope limit; a compact 24-face foundation can. Two compatible 50 km² supplies accumulate and promote through the actual material transaction, whereas the first supply alone remains pending. This is a controlled geometry result, not a claim that every source in a saved world will form an island.

`tests/test_arc_points_ui.cjs` checks map/globe projection, longitude seam and pole behavior, screen sizing, hover and legacy-frame clearing. Browser screenshot review was unavailable in this environment; visual clipping and contrast have not been independently checked.

## Saved-world validation

An isolated pending-source transaction on the preserved 176 Myr checkpoint created 18 compact foundations (432 faces), placing 64,128.835 km³ into 7,019.218 km² of new crust. Existing-arc deposition remained 18,272.867 km³, for 82,401.702 km³ placed in total. Pending point features decreased from 1,324 to 1,298; 5,101,614.252 km³ remained pending.

All 18 new foundations were submerged: their reconstructed surfaces remained between approximately -4,579 and -2,146 m. These are crust foundations, not 18 emerged islands. Later accumulation must build their surface to sea level.

The source-volume residual was 1.05 × 10⁻⁹ km³ and the physical-volume residual was -1.35 × 10⁻⁶ km³; existing conservation tolerances passed. Strict frame checks, normal zero-time raster/domain refresh, typed checkpoint reload and snapshot revalidation passed. The source checkpoint, production source files, configuration, tectonic time, existing vertices and shared random state remained unchanged. The full transaction output was byte-identical across the repeated trials.

This validates the pending-material transaction and its saved-state representation. It does not constitute a multi-step tectonic integration or live-run adoption. Separate focused tests exercise later ordinary growth and local deposition into a newly promoted compact patch.

The selected regression checks include 198 passing Python tests and 33 passing UI tests. Four inherited failing cases and four cases requiring unavailable private fixtures remain documented in the pull request; the inherited suite is not wholly green.
