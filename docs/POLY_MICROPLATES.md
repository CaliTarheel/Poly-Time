# Microplate retention and force integration

Every `World` and `RegionWorld` has its own immutable `plate_policy`:

```python
from poly.policy import PlatePolicy

world.plate_policy = PlatePolicy(
    minimum_plate_km2=250_000.,
    numerical_crumb_km2=100.,
)
```

The default minimum has changed from 1,000,000 to 250,000 km². An existing
plate or detached component at least this large can remain an independently
moving plate. A smaller component is transferred to a neighbour. This is a
model resolution policy, not a claim that smaller geological plates cannot
exist. Lower `minimum_plate_km2` to retain smaller plates, keeping it greater
than `numerical_crumb_km2`.

`PlatePolicy.legacy()` restores the previous 1,000,000 km² minimum. The network
to region-world conversion preserves the selected policy.

Detached numerical crumbs below `numerical_crumb_km2` are handled in batches,
separately from resolved plate fragments. This does not change the geometry
engine's `DETAIL_KM2` limit or its outline simplification tolerances. Setting a
very small retention minimum does not recover geometry those operations have
already discarded.

Absorption selects the neighbour sharing the longest spherical boundary arc
with the transferred region. Shared arc overlap is measured with a 1 cm
positional tolerance so frame roundoff cannot erase a contact. Projected arc
intervals are unioned, so subdividing an adjacent outline does not double the
length. The tolerance does not bridge metre-scale gaps. Equal lengths within
that tolerance use the lowest plate id. Only when
no neighbour shares a line (for example a separated numerical crumb after
cleanup) does the selector use nearest distance, again with a stable id tie.
Thus dictionary insertion order does not decide a plate's geological parent.

## Basal drag on a small plate

The global Fibonacci samples remain the large-plate integration method. When
a plate contains fewer than `Parameters.minimum_drag_samples` (default 128),
its basal drag uses an exact spherical polygon integral of the same force law.
This avoids an absent or rank-deficient basal drag tensor when the global
sampling grid puts zero or one points inside a microplate. It does not change
slab-pull, ridge-push, boundary-resistance, or drag coefficients.

`poly.quadrature.basal_moment_m2` uses the surface divergence theorem to
integrate `I - r rᵀ` from each oriented great-circle boundary edge and the
polygon's spherical area. Holes, disconnected parts, and greater-than-
hemisphere complements retain their correct contributions. Accuracy is with
respect to the polygon supplied; outline accuracy is still a separate issue.
The selected method and sample count are reported in
`balance.last['drag_quadrature']`.

The fallback resolves basal drag, not every possible resolution issue. Rift
cut searches continue to use the global samples and their existing minimum
plate/part cutoffs; those cutoffs are much larger than retained microplates.
Changing the rift cutoffs to small scales requires a separate cut-resolution
study. The transition at the sampling threshold also retains the existing
large-plate quadrature error; setting `minimum_drag_samples` above the global
sample count requests polygon integration for every plate.

## Checks

`tests/test_poly_microplates.py` checks persistent small plates under zero
motion, per-world and legacy policies, detached plates versus numerical
crumbs, insertion-order-independent recipients, spherical-cap integral
convergence, rotation covariance, complements and disconnected components, and
a nonzero force solve with no global sample inside the microplate.
