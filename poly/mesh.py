"""A continent's thickness as a field on a triangle mesh that moves with its plate.

The mesh lives in the plate's frame like the crust polygon it covers: its nodes are the polygon's
outline vertices plus interior points about SPACING_KM apart, triangulated in the gnomonic
projection about the piece's centre (great circles are straight there, so a plane Delaunay is a
spherical one). Each node carries a thickness; the piece's volume is the thickness integrated over
the mesh, scaled by the little of the polygon the triangles do not cover.

When the polygon changes (shortening at a collision, a rift, debris let go), the mesh is rebuilt on
the new polygon with the old field carried over, and the volume the new mesh is short of the
target is put back onto the nodes nearest the ground that was lost: shortening thickens the crust
where it happened, not everywhere.
"""
from __future__ import annotations

import numpy as np
import spherely
from scipy.spatial import Delaunay, cKDTree

from poly import sphere

SPACING_KM = 200.   # interior node spacing
REFERENCE_KM = 35.  # thickness of continent at sea level
ISOSTASY = .15      # elevation per km of extra thickness: (rho_mantle - rho_crust) / rho_mantle
# Thick crust flows outward under its own weight: thickness diffuses with this diffusivity. A
# plateau 1,000 km across takes about 50 Myr to spread (Tibet).
SPREADING_KM2_PER_MYR = 3e3
REACH_KM = 300.     # shortening thickens the crust within about this of where it happened
MINIMUM_KM2 = 2e4   # a piece smaller than this (a 140 km disc) keeps one thickness, no mesh
LOST_KM = 1.        # a node this far outside the piece's new outline was on the ground lost


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
        if sphere.area_km2(shape) < MINIMUM_KM2:
            return None
        rings = [r for poly in sphere.polygons(shape) for r in poly]
        if not rings:
            return None
        boundary = np.vstack(rings)
        centre, e1, e2 = _frame(boundary)
        if np.min(boundary @ centre) < .2:
            return None  # wider than about 150 degrees: no single gnomonic plane
        xy = _project(boundary, centre, e1, e2)
        step = spacing_km / sphere.RADIUS_KM
        xs = np.arange(xy[:, 0].min(), xy[:, 0].max() + step, step)
        ys = np.arange(xy[:, 1].min(), xy[:, 1].max() + step * .866, step * .866)
        gx, gy = np.meshgrid(xs, ys)
        gx[1::2] += step / 2.  # hexagonal packing
        grid = np.column_stack((gx.ravel(), gy.ravel()))
        candidates = _unproject(grid, centre, e1, e2)
        lon, lat = sphere.to_lonlat(candidates)
        spherely.prepare(shape)
        inside = np.asarray(spherely.contains(shape, spherely.points(lon, lat)))
        near_edge = cKDTree(boundary).query(candidates, k=1)[0] < .5 * step
        interior = candidates[inside & ~near_edge]
        nodes = np.vstack((boundary, interior))
        try:
            triangles = Delaunay(_project(nodes, centre, e1, e2)).simplices
        except Exception:
            return None
        centroids = sphere.unit(nodes[triangles].sum(axis=1))
        lon, lat = sphere.to_lonlat(centroids)
        keep = np.asarray(spherely.contains(shape, spherely.points(lon, lat)))
        triangles = triangles[keep]
        if len(triangles) < 1:
            return None
        thickness = np.full(len(nodes), float(thickness_km)) if field is None else np.asarray(field(nodes), float)
        mesh = cls(nodes, triangles, thickness)
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
        """Thickness at `points` (plate frame): inverse-distance over the three nearest nodes."""
        points = np.asarray(points, float).reshape(-1, 3)
        k = min(3, len(self.nodes))
        d, idx = cKDTree(self.nodes).query(points, k=k)
        d, idx = np.asarray(d).reshape(len(points), k), np.asarray(idx).reshape(len(points), k)
        w = 1. / np.maximum(d, 1e-12)
        return (w * self.thickness[idx]).sum(axis=1) / w.sum(axis=1)

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

        Finite volumes on the mesh edges: the crust moving along an edge per Myr is the
        diffusivity times the thickness difference (an edge's width and length cancel to first
        order), sub-cycled to keep the explicit scheme stable.
        """
        if len(self.triangles) == 0 or dt_myr <= 0.:
            return
        edges = np.vstack((self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        i, j = edges[:, 0], edges[:, 1]
        areas = np.maximum(self.node_areas_km2(), 1.)
        degree = np.bincount(np.concatenate((i, j)), minlength=len(self.nodes))
        limit = .25 * areas / np.maximum(degree, 1) / diffusivity  # stable sub-step, per node
        cycles = int(np.ceil(dt_myr / max(float(limit.min()), 1e-9)))
        step = dt_myr / cycles
        t = self.thickness.copy()
        for _ in range(cycles):
            flow = diffusivity * step * (t[j] - t[i])  # km3 along each edge, from j to i
            change = np.zeros(len(t))
            np.add.at(change, i, flow)
            np.add.at(change, j, -flow)
            t += change / areas
        self.thickness = t

    def rotated(self, matrix):
        return ContinentMesh(sphere.rotate(self.nodes, matrix), self.triangles, self.thickness)


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
