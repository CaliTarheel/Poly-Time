# Surface height and emergence

`poly.surface` samples the existing physical height laws independently of map
colours. Continental crust can be underwater. Oceanic crust is identified by its
material type even in an extreme prescribed sea-level scenario where it emerges.

```python
from poly import sphere, surface, render

points = sphere.from_lonlat([0., 45.], [0., 10.])
values = surface.sample(world, points, sea_level_km=0.)
values.elevation_km             # height relative to the model's existing zero datum
values.height_above_water_km    # height minus the prescribed sea level
values.continental_crust        # material classification, independent of emergence
values.land                    # water-relative height >= 0
values.water                   # water-relative height < 0

render.draw(world, 'map.png', sea_level_km=.1)  # 100 m higher water surface
```

The sampler does not modify crust, sea level, or frames. It uses the current
continental Airy law (35 km reference thickness, 0.15 elevation factor) and GDH1
ocean cooling depth. The sea-level argument is a display/export scenario, not a
water-inventory solver, and does not alter erosion or any other process.

Missing crust is returned as NaN height and an empty crust-kind string. These
points are neither land nor water. The plate ID remains available if a containing
plate exists; plate zero means no containing plate was found. Map pixels without
a known height are black instead of being silently filled with water.

Rendering uses water-depth colours for submerged material of either crust type.
Small continental pieces receive enlarged visibility dots only if emerged under
the same height law; these dots are presentation markers, not a land-area measure.
Use the sampler and suitable area weights for quantitative masks. Equirectangular
pixel counts alone are not area-weighted measurements.

This exposes an existing physical approximation: 20 km continental arc crust has
height -2.25 km under the present reference law. The renderer no longer overrides
that result with a land colour. A separate emergence or loading model must be an
explicit physical change, shared by rendering and future exports.
