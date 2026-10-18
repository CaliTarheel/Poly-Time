"""A continent's thickness as a field on a triangle mesh that moves with its plate.

The mesh lives in the plate's frame like the crust polygon it covers: its nodes are the polygon's
outline vertices plus interior points about SPACING_KM apart. Each connected polygon is
triangulated in its own gnomonic plane, then triangles are clipped to its actual outline and holes.
Great circles are straight in that plane. Each node carries a thickness; the piece's volume is
the thickness integrated over the mesh, scaled to the polygon's area for numerical roundoff.

When the polygon changes (shortening at a collision, a rift, debris let go), the mesh is rebuilt on
the new polygon with the old field carried over, and the volume the new mesh is short of the
target is put back onto the nodes nearest the ground that was lost: shortening thickens the crust
where it happened, not everywhere.
"""
from __future__ import annotations

import numpy as np
import spherely
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from poly import sphere

SPACING_KM = 200.   # interior node spacing
REFERENCE_KM = 35.  # thickness of continent at sea level
ISOSTASY = .15      # elevation per km of extra thickness: (rho_mantle - rho_crust) / rho_mantle
# Thick crust flows outward under its own weight: thickness diffuses with this diffusivity. A
# plateau 1,000 km across takes about 50 Myr to spread (Tibet).
SPREADING_KM2_PER_MYR = 3e3
# Erosion: crust above sea level is worn down at this fraction of its elevation per Myr, as
# thickness (0.2 km/Myr of rock at 1 km, since isostasy lifts the crust as it is unloaded, so the
# land loses only 0.15 km of height for each km of rock removed). The sediment settles on the
# piece's own lowlands within DEPOSIT_REACH_KM; what finds none goes to the sea.
ERODE_PER_MYR = .2
LOWLAND_KM = .2
DEPOSIT_REACH_KM = 500.
REACH_KM = 300.     # shortening thickens the crust within about this of where it happened
MINIMUM_KM2 = 2e4   # a piece smaller than this (a 140 km disc) keeps one thickness, no mesh
LOST_KM = 1.        # a node this far outside the piece's new outline was on the ground lost
# S2 clipping can retain disconnected fragments below a square metre. Their native geometry
# remains in the crust polygon and its volume; they cannot support a useful continental mesh.
MINIMUM_COMPONENT_KM2 = 1e-6


class ContinentMesh:
    def __init__(self, nodes, triangles, thickness):
        self.nodes = np.asarray(nodes, float)
        self.triangles = np.asarray(triangles, int).reshape(-1, 3)
        self.thickness = np.asarray(thickness, float)

    # ------------------------------------------------------------ building
    @classmethod
    def build(cls, shape, thickness_km, field=None, spacing_km=SPACING_KM):
        """A mesh over `shape` (in the plate's frame), `thickness_km` everywhere or from `field`.

        `field(points)` gives the thickness at points (an older mesh's `sample`). Returns None
        when the shape is too small or too thin to mesh.
        """
        polygon_area = sphere.area_km2(shape)
        if polygon_area < MINIMUM_KM2:
            return None
        nodes, triangles = [], []
        for rings in sphere.polygons(shape):
            # Test a bounding spherical cap, rather than a cancellation-prone signed loop area.
            # A small shell could also bound the complement of a cap; that is impossible when
            # the complete geography occupies less than a hemisphere. Keep unsupported large
            # geometries on the ordinary path rather than guessing their orientation.
            if polygon_area < sphere.SPHERE_KM2 / 2 and _microscopic(rings[0]):
                continue
            rings = [sphere.tidy_ring(r, tolerance=1e-13) for r in rings]
            if len(rings[0]) < 3 and polygon_area < sphere.SPHERE_KM2 / 2:
                continue  # a native zero-width component collapsed in the degree conversion
            boundary = np.vstack(rings)
            centre = sphere.unit(boundary.mean(axis=0))
            if (polygon_area < sphere.SPHERE_KM2 / 2 and np.isfinite(centre).all()
                    and np.min(boundary @ centre) > .2
                    and abs(sum(sphere.loop_area_km2(r) for r in rings)) < MINIMUM_COMPONENT_KM2):
                continue  # projected minor loops: an elongated roundoff fold, not a cap
            part = _triangulate(rings, spacing_km)
            parts = [part] if part is not None else _chart_meshes(rings, spacing_km)
            for part in parts:
                if part is None:
                    continue  # unresolved zero-width support; bounded by full native-area check
                xyz, faces = part
                triangles.extend(faces + len(nodes))
                nodes.extend(xyz)
        if not triangles:
            return None
        nodes, triangles = _weld_nodes(np.asarray(nodes, float), np.asarray(triangles, int))
        thickness = np.full(len(nodes), float(thickness_km)) if field is None else np.asarray(field(nodes), float)
        mesh = cls(nodes, triangles, thickness)
        error = abs(mesh.area_km2() - polygon_area)
        if error > max(1e-5, polygon_area * 1e-9):
            raise ValueError('Continental mesh coverage differs from its native polygon by %.6g km2.' % error)
        return mesh if mesh.area_km2() > 0. else None

    # ------------------------------------------------------------ measures
    def triangle_areas_km2(self):
        a, b, c = (self.nodes[self.triangles[:, k]] for k in range(3))
        numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
        denominator = 1. + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
        return np.abs(2. * np.arctan2(numerator, denominator)) * sphere.RADIUS_KM ** 2

    def node_areas_km2(self):
        """A third of each incident triangle's area, per node."""
        areas = np.zeros(len(self.nodes))
        third = self.triangle_areas_km2() / 3.
        for k in range(3):
            np.add.at(areas, self.triangles[:, k], third)
        return areas

    def area_km2(self):
        return float(self.triangle_areas_km2().sum())

    def volume_km3(self, polygon_km2):
        """Thickness integrated over the mesh, scaled to the polygon's area."""
        covered = self.area_km2()
        if covered <= 0.:
            return 0.
        return float((self.node_areas_km2() * self.thickness).sum()) * polygon_km2 / covered

    def elevation_km(self, thickness=None):
        """Airy isostatic elevation (km above sea level) of crust `thickness` km thick."""
        thickness = self.thickness if thickness is None else np.asarray(thickness, float)
        return ISOSTASY * (thickness - REFERENCE_KM)

    def sample(self, points):
        """Interpolate within a containing face, without blending across coastlines or holes.

        New ground outside the old mesh (during recutting) takes the nearest boundary edge's
        value. It does not mix nodes from opposite shores or disconnected components.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.full(len(points), np.nan)
        if not len(points):
            return result
        triangles = _usable_triangles(self.nodes, self.triangles)
        if not len(triangles):
            raise ValueError('Continental mesh has no numerically resolved sampling faces.')
        inverse = np.linalg.inv(self.nodes[triangles])
        centres = sphere.unit(self.nodes[triangles].sum(axis=1))
        nearest = cKDTree(centres)
        # The nearby faces normally suffice on a regular mesh. Fall back to an exhaustive
        # containment search for long, thin boundary faces; nearest-centre is never the rule
        # deciding which continent or side of an inlet supplies a value.
        count = min(8, len(triangles))
        for start in range(0, len(points), 4096):
            batch = points[start:start + 4096]
            candidates = np.asarray(nearest.query(batch, k=count)[1]).reshape(len(batch), count)
            weights = np.einsum('pi,pkij->pkj', batch, inverse[candidates])
            inside = np.all(weights >= -1e-10, axis=2) & (weights.sum(axis=2) > 0.)
            rows = np.flatnonzero(inside.any(axis=1))
            columns = np.argmax(inside[rows], axis=1)
            w = np.maximum(weights[rows, columns], 0.)
            tri = triangles[candidates[rows, columns]]
            result[start + rows] = (w * self.thickness[tri]).sum(axis=1) / w.sum(axis=1)
        for tri, transform in zip(triangles, inverse):
            missing = np.flatnonzero(np.isnan(result))
            if not len(missing):
                break
            # Positive cone coordinates locate a point in a minor spherical triangle. Their
            # normalized values are barycentric coordinates in the face's gnomonic plane.
            weights = points[missing] @ transform
            hit = np.all(weights >= -1e-10, axis=1) & (weights.sum(axis=1) > 0.)
            w = np.maximum(weights[hit], 0.)
            result[missing[hit]] = (w @ self.thickness[tri]) / w.sum(axis=1)
        missing = np.flatnonzero(np.isnan(result))
        if len(missing):
            edges = _edges(triangles)
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            boundary = unique[counts == 1]
            best = np.full(len(missing), np.inf)
            for i, j in boundary:
                a, b = self.nodes[i], self.nodes[j]
                direction = b - a
                fraction = np.clip((points[missing] - a) @ direction / (direction @ direction), 0., 1.)
                nearest = sphere.unit(a + fraction[:, None] * direction)
                distance = np.sum((points[missing] - nearest) ** 2, axis=1)
                hit = distance < best
                result[missing[hit]] = ((1. - fraction[hit]) * self.thickness[i]
                                        + fraction[hit] * self.thickness[j])
                best[hit] = distance[hit]
        return result

    # ------------------------------------------------------------- changes
    def add_volume(self, volume_km3, polygon_km2, near=None, reach_km=REACH_KM):
        """Thicken by `volume_km3` in all, on the nodes nearest `near` (points) or everywhere."""
        areas = self.node_areas_km2() * polygon_km2 / max(self.area_km2(), 1e-300)
        if near is None or not len(near):
            weight = np.ones(len(self.nodes))
        else:
            d = cKDTree(np.asarray(near, float).reshape(-1, 3)).query(self.nodes, k=1)[0] * sphere.RADIUS_KM
            weight = np.exp(-.5 * (d / (reach_km / 2.)) ** 2)
            if weight.sum() < 1e-12:
                weight = np.ones(len(self.nodes))
        total = float((weight * areas).sum())
        if total > 0.:
            self.thickness = self.thickness + volume_km3 * weight / total

    def recut(self, shape, polygon_km2, target_km3):
        """The mesh rebuilt on the piece's new `shape`, holding `target_km3` of crust.

        The old field is carried onto the new nodes; what the new mesh is short of the target (the
        crust that was on the ground lost) is put back near the old nodes that are gone, so
        shortening thickens the crust where it happened. Returns None when the shape can no longer
        be meshed (the caller keeps one thickness).
        """
        new = ContinentMesh.build(shape, 0., field=self.sample)
        if new is None:
            return None
        # Nodes clearly outside the new shape were on the ground lost. (A point on the outline
        # counts as outside to S2, so containment alone would name every surviving boundary node.)
        lon, lat = sphere.to_lonlat(self.nodes)
        points = spherely.points(lon, lat)
        spherely.prepare(shape)
        gone = ~np.asarray(spherely.contains(shape, points))
        if gone.any():
            km = np.asarray(spherely.distance(shape, points[gone]), float) / 1e3
            gone[np.flatnonzero(gone)[km < LOST_KM]] = False
        deficit = target_km3 - new.volume_km3(polygon_km2)
        new.add_volume(deficit, polygon_km2, near=self.nodes[gone] if gone.any() else None)
        new.lost_nodes = self.nodes[gone]  # where the ground went, for what follows a collision
        return new

    def spread(self, dt_myr, diffusivity=SPREADING_KM2_PER_MYR):
        """Let the thickness flow down its gradient for `dt_myr`: crust is conserved exactly.

        Backward-Euler finite volumes on mesh edges. Actual node areas supply the mass matrix;
        microscopic clipped cells therefore neither impose millions of explicit substeps nor
        acquire the artificial one-square-kilometre mass of the former stability floor.
        Every disconnected component conserves its own volume.
        """
        if len(self.triangles) == 0 or dt_myr <= 0.:
            return
        edges = np.vstack((self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        i, j = edges[:, 0], edges[:, 1]
        areas = self.node_areas_km2()
        if not np.isfinite(diffusivity) or diffusivity < 0. or np.any(areas <= 0.):
            raise ValueError('Diffusion requires positive finite node areas and nonnegative diffusivity.')
        if diffusivity == 0.:
            return
        n = len(areas)
        graph = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        count, labels = connected_components(graph, directed=False)
        component_area = np.bincount(labels, weights=areas, minlength=count)
        component_volume = np.bincount(labels, weights=areas * self.thickness, minlength=count)
        mean = (component_volume / component_area)[labels]
        degree = np.asarray(graph.sum(axis=1)).ravel()
        operator = diags(areas) + dt_myr * diffusivity * (diags(degree) - graph)
        # Subtract each component's mean so a constant field remains exact even when a very
        # small mass diagonal is added to much larger diffusion coefficients.
        result = mean + spsolve(operator.tocsc(), areas * (self.thickness - mean))
        correction = np.bincount(labels, weights=areas * (self.thickness - result), minlength=count)
        result += (correction / component_area)[labels]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite continental diffusion result.')
        self.thickness = result

    def erode(self, dt_myr, polygon_km2, sea_level_km=0.):
        """Wear the highlands down for `dt_myr`, settling the sediment on nearby lowlands.

        Returns (km3 eroded, km3 settled on this piece, km3 gone to the sea).
        """
        elevation = self.elevation_km() - sea_level_km
        high = elevation > 0.
        if not high.any() or dt_myr <= 0.:
            return 0., 0., 0.
        areas = self.node_areas_km2() * polygon_km2 / max(self.area_km2(), 1e-300)
        cut = np.where(high, ERODE_PER_MYR * elevation * dt_myr, 0.)
        cut = np.minimum(cut, np.maximum(self.thickness - REFERENCE_KM - sea_level_km / ISOSTASY, 0.))
        removed = float((cut * areas).sum())
        self.thickness = self.thickness - cut
        if removed <= 0.:
            return 0., 0., 0.
        low = elevation <= LOWLAND_KM
        if not low.any():
            return removed, 0., removed
        d = cKDTree(self.nodes[high]).query(self.nodes[low], k=1)[0] * sphere.RADIUS_KM
        weight = np.exp(-.5 * (d / (DEPOSIT_REACH_KM / 2.)) ** 2) * areas[low]
        if weight.sum() <= 1e-12:
            return removed, 0., removed
        self.thickness[low] += removed * (weight / weight.sum()) / areas[low]
        return removed, removed, 0.

    def rotated(self, matrix):
        return ContinentMesh(sphere.rotate(self.nodes, matrix), self.triangles, self.thickness)


def _edges(triangles):
    return np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)


def _usable_triangles(nodes, triangles):
    """Faces whose spherical interpolation can be resolved in binary64.

    Clipping at a coincident edge can create zero-width faces. They retain no useful field
    support. New meshes check the omitted area against the native domain; old checkpoint
    sampling ignores these faces while the full saved volume remains the remapping target.
    """
    triangles = np.asarray(triangles, int).reshape(-1, 3)
    if not len(triangles):
        return triangles
    condition = np.linalg.cond(np.asarray(nodes)[triangles])
    return triangles[np.isfinite(condition) & (condition < 1e12)]


def _microscopic(ring):
    if not len(ring):
        return True
    angle = np.max(sphere.arc_km(ring[0], ring)) / sphere.RADIUS_KM
    cap_area = 4. * np.pi * sphere.RADIUS_KM ** 2 * np.sin(angle / 2.) ** 2
    return cap_area < MINIMUM_COMPONENT_KM2


def _weld_nodes(nodes, triangles):
    """Shared chart seams are ordinary interior mesh edges, not separate continents."""
    parent = np.arange(len(nodes))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in sorted(cKDTree(nodes).query_pairs(1e-12)):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(nodes))])
    unique, inverse = np.unique(roots, return_inverse=True)
    faces = _usable_triangles(nodes[unique], inverse[triangles])
    used, remap = np.unique(faces, return_inverse=True)
    return nodes[unique][used], remap.reshape(-1, 3)


def _mesh_geography(rings):
    """Repair each loop independently so the output geometry cleaner cannot erase slit holes.

    The simulation has its own explicit, ledgered detail policy. Meshing retains even
    sub-detail holes until that policy acts on the authoritative polygon itself.
    """
    shape = sphere.polygon_robust(rings[0])
    for hole in rings[1:]:
        shape = sphere.difference(shape, sphere.polygon_robust(hole[::-1]))
    return shape


def _chart_meshes(rings, spacing_km):
    """Cover wide domains with eight gnomonic charts, each a spherical octant.

    The native-domain coverage check still applies after stitching. This removes the old
    silent collapse to one mean thickness when a continent outgrows one tangent plane.
    """
    shape = _mesh_geography(rings)
    parts = []
    axes = np.eye(3)
    for x in (-1., 1.):
        for y in (-1., 1.):
            for z in (-1., 1.):
                ring = axes * np.array([x, y, z])[:, None]
                if sphere.loop_area_km2(ring) < 0.:
                    ring = ring[::-1]
                chart = sphere.intersection(shape, sphere.polygon(ring))
                for domain in sphere.polygons(chart):
                    parts.append(_triangulate(domain, spacing_km))
    return parts


def _triangulate(rings, spacing_km):
    """Cover one connected polygon, including its holes, with domain-clipped faces.

    All outline vertices participate in the initial Delaunay triangulation, so an initial face
    cannot contain an entire hole. Clipping may split a face, but leaves only simple polygons.
    Triangulating those fragments retains the coastline without introducing a new dependency.
    """
    # Native S2 vertices that differ by an ulp can coincide after WKB's degree conversion.
    # Remove only sub-micrometre repeats/spikes before rebuilding the meshing domain. The
    # authoritative native polygon remains untouched; build checks total mesh coverage.
    rings = [sphere.tidy_ring(r, tolerance=1e-13) for r in rings]
    if len(rings[0]) < 3:
        return None
    rings = [rings[0]] + [r for r in rings[1:] if len(r) >= 3]
    boundary = np.vstack(rings)
    centre, e1, e2 = _frame(boundary)
    if not np.isfinite(centre).all() or np.min(boundary @ centre) < .2:
        return None  # one connected component is too wide for a single gnomonic plane
    # A valid native S2 component may touch itself at vertices after degree conversion.
    # Reuse the spherical loop repair; the full mesh/native-area comparison remains mandatory.
    shape = _mesh_geography(rings)
    xy = _project(boundary, centre, e1, e2)
    step = spacing_km / sphere.RADIUS_KM
    xs = np.arange(xy[:, 0].min(), xy[:, 0].max() + step, step)
    ys = np.arange(xy[:, 1].min(), xy[:, 1].max() + step * .866, step * .866)
    gx, gy = np.meshgrid(xs, ys)
    gx[1::2] += step / 2.
    candidates = _unproject(np.column_stack((gx.ravel(), gy.ravel())), centre, e1, e2)
    lon, lat = sphere.to_lonlat(candidates)
    spherely.prepare(shape)
    inside = np.asarray(spherely.contains(shape, spherely.points(lon, lat)))
    near_edge = cKDTree(boundary).query(candidates, k=1)[0] < .5 * step
    nodes = np.vstack((boundary, candidates[inside & ~near_edge]))
    try:
        initial = Delaunay(_project(nodes, centre, e1, e2)).simplices
    except (ValueError, QhullError):
        return None
    tree = cKDTree(nodes)
    extra, faces = [], []

    def index(point):
        distance, ident = tree.query(point)
        if distance < 1e-12:
            return int(ident)
        if extra:
            distances = np.linalg.norm(np.asarray(extra) - point, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < 1e-12:
                return len(nodes) + nearest
        extra.append(point)
        return len(nodes) + len(extra) - 1

    for tri in initial:
        ring = nodes[tri]
        # scipy orders the planar simplex counter-clockwise; keep the minor spherical face.
        face = sphere.polygon(ring)
        if sphere.area_km2(face) > .5 * sphere.SPHERE_KM2:
            face = sphere.polygon(ring[::-1])
        if spherely.covers(shape, face):
            faces.append(tri)
            continue
        clipped = sphere.intersection(shape, face)
        for part in sphere.polygons(clipped):
            if len(part) != 1:
                raise ValueError('A clipped mesh face contains a hole; cannot triangulate safely.')
            fragment = part[0]
            local = _ear_clip(_project(fragment, centre, e1, e2))
            ids = np.array([index(point) for point in fragment], int)
            faces.extend(ids[local])
    if not faces:
        return None
    if extra:
        nodes = np.vstack((nodes, extra))
    faces = _usable_triangles(nodes, faces)
    if not len(faces):
        return None
    faces = np.asarray(faces, int)
    # Clipping can leave original outline/interior vertices unused. They must not remain as
    # zero-area nodes participating in remapping or the explicit transport stability limit.
    used, inverse = np.unique(faces, return_inverse=True)
    return nodes[used], inverse.reshape(-1, 3)


def _ear_clip(points):
    """Triangulate a simple planar clipping fragment without crossing its outline."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return np.zeros((0, 3), int)
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    signed = sum(cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))
    indices = list(range(len(points))) if signed >= 0. else list(reversed(range(len(points))))
    tolerance = max(np.ptp(points, axis=0).max() ** 2 * 1e-13, 1e-24)
    faces = []
    while len(indices) > 3:
        for k, b in enumerate(indices):
            a, c = indices[k - 1], indices[(k + 1) % len(indices)]
            pa, pb, pc = points[[a, b, c]]
            turn = cross(pb - pa, pc - pb)
            if abs(turn) <= tolerance:
                # S2 can retain collinear vertices on a clipped edge.
                indices.pop(k)
                break
            if turn < 0.:
                continue
            other = [i for i in indices if i not in (a, b, c)]
            if any(cross(pb - pa, points[i] - pa) >= -tolerance
                   and cross(pc - pb, points[i] - pb) >= -tolerance
                   and cross(pa - pc, points[i] - pc) >= -tolerance for i in other):
                continue
            faces.append((a, b, c))
            indices.pop(k)
            break
        else:
            raise ValueError('Cannot triangulate a clipped mesh face without crossing its outline.')
    if len(indices) == 3:
        a, b, c = points[indices]
        if abs(cross(b - a, c - a)) > tolerance:
            faces.append(tuple(indices))
    return np.asarray(faces, int).reshape(-1, 3)


def _frame(points):
    centre = sphere.unit(np.asarray(points, float).mean(axis=0))
    up = np.array([0., 0., 1.]) if abs(centre[2]) < .9 else np.array([1., 0., 0.])
    e1 = sphere.unit(np.cross(centre, up))
    e2 = np.cross(centre, e1)
    return centre, e1, e2


def _project(points, centre, e1, e2):
    """Gnomonic projection about `centre`: great circles become straight lines."""
    points = np.asarray(points, float)
    depth = points @ centre
    return np.column_stack((points @ e1 / depth, points @ e2 / depth))


def _unproject(xy, centre, e1, e2):
    return sphere.unit(centre[None] + xy[:, :1] * e1[None] + xy[:, 1:] * e2[None])
