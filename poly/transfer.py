"""Geometry-only helpers for carrying state through changes of plate ownership.

All clipping and distance calculations use minor great-circle arcs on the sphere.
"""
from __future__ import annotations

import struct

import numpy as np
import spherely

from poly import sphere


def segment_distances(points, edges):
    """Distance in km from every point to every minor great-circle segment."""
    points, edges = np.asarray(points), np.asarray(edges)
    if len(points) > 128:  # bound the temporary point-by-edge-by-coordinate arrays
        return np.concatenate([segment_distances(points[k:k + 128], edges)
                               for k in range(0, len(points), 128)])
    a, b = edges[:, 0], edges[:, 1]
    cross = np.cross(a, b)
    norm = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(norm[:, None], 1e-30)
    dots = points @ normals.T
    projected = points[:, None, :] - dots[:, :, None] * normals[None, :, :]
    projected /= np.maximum(np.linalg.norm(projected, axis=2, keepdims=True), 1e-30)
    da = sphere.arc_km(projected, a[None])
    db = sphere.arc_km(projected, b[None])
    on = (np.abs(da + db - sphere.arc_km(a, b)[None]) < 1e-5) & (norm[None] > 1e-15)
    ends = np.minimum(sphere.arc_km(points[:, None], a[None]), sphere.arc_km(points[:, None], b[None]))
    foot = sphere.arc_km(points[:, None], projected)
    return np.where(on, np.minimum(ends, foot), ends)


def lines(geography):
    """Read line members of an S2 result, ignoring isolated points at clipping tangencies."""
    data = spherely.to_wkb(geography)
    result = []

    def read(pos):
        endian = '<' if data[pos] == 1 else '>'
        kind = struct.unpack_from(endian + 'I', data, pos + 1)[0]
        pos += 5
        dims = 2 + (kind // 1000 in (1, 2)) + 2 * (kind // 1000 == 3)
        kind %= 1000
        if kind == 1:
            return pos + 8 * dims
        count = struct.unpack_from(endian + 'I', data, pos)[0]
        pos += 4
        if kind == 2:
            values = np.frombuffer(data, endian + 'f8', count * dims, pos).reshape(count, dims)
            if count >= 2:
                result.append(sphere.from_lonlat(values[:, 0], values[:, 1]))
            return pos + 8 * dims * count
        if kind == 3:
            for _ in range(count):
                n = struct.unpack_from(endian + 'I', data, pos)[0]
                pos += 4 + 8 * dims * n
            return pos
        for _ in range(count):
            pos = read(pos)
        return pos

    read(0)
    return result


def transfer_sutures(sutures, plate, to, piece, source_frame, target_frame, whole=False):
    """Clip each transferred suture into its two owners; preserve metadata and world position."""
    result = []
    for s in sutures:
        if s['plate'] != plate:
            result.append(s)
            continue
        world = sphere.rotate(s['points'], source_frame)
        if whole:
            result.append(dict(s, plate=to, points=sphere.rotate(world, target_frame.T)))
            continue
        lon, lat = sphere.to_lonlat(world)
        line = spherely.create_linestring(list(zip(lon, lat)))
        for owner, frame, geography in (
                (plate, source_frame, spherely.difference(line, piece)),
                (to, target_frame, spherely.intersection(line, piece))):
            for part in lines(geography):
                result.append(dict(s, plate=owner, points=sphere.rotate(part, frame.T)))
    return result
