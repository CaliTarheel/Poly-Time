"""Bounded-memory, tectonically conditioned terrain refinement.

The simulator supplies the geography and its history.  This module supplies a
deterministic, finer *procedural surface* for a selected recorded epoch; it does
not rerun tectonics or reconstruct individual fine features through time.
Only NumPy, Pillow and the standard library are required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import time
import zlib

import numpy as np
from PIL import Image

from orientation import normalize_orientation, ORIENTATION_CONVENTION, reproject_raster, rotation_matrix
import native_frame_sampling


TERRAIN_VERSION = 2
SUPPORTED_WIDTHS = (2048, 4096, 8192, 16384)
RADIUS_KM = 6371.0
SAMPLE_CHUNK = 32_768


class TerrainCancelled(Exception):
    """Raised at a row-band boundary when the caller cancels a terrain job."""


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise TerrainCancelled("Terrain generation was cancelled.")


def _blur(field, passes=1):
    """Compact spherical smoothing, including the half-turn across either pole."""
    result = np.asarray(field, dtype=np.float32).copy()
    for _ in range(passes):
        north = np.concatenate((np.roll(result[:1], result.shape[1] // 2, axis=1), result[:-1]))
        south = np.concatenate((result[1:], np.roll(result[-1:], result.shape[1] // 2, axis=1)))
        result = (4 * result + north + south + np.roll(result, 1, axis=1)
                  + np.roll(result, -1, axis=1)) * np.float32(.125)
    return result


def _sampling(lon, lat, width, height):
    """Bilinear cell-centred sampling indices, with correct polar continuation."""
    x = (lon + np.float32(np.pi)) * np.float32(width / (2 * np.pi)) - np.float32(.5)
    y = (np.float32(np.pi / 2) - lat) * np.float32(height / np.pi) - np.float32(.5)
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    fx, fy = x - ix, y - iy
    rows = []
    for row in (iy, iy + 1):
        cross = (row < 0) | (row >= height)
        ry = np.where(row < 0, -row - 1, np.where(row >= height, 2 * height - row - 1, row))
        rx = ix + cross.astype(np.int32) * (width // 2)
        rows.append((ry.clip(0, height - 1), rx % width, (rx + 1) % width))
    # Equirectangular cell centres stop short of either pole.  Opposite-cell
    # reflection alone leaves a longitude-dependent value at that single point;
    # blend the final half-cell to the polar ring's mean instead.
    north_cap = np.clip(-2 * y, 0, 1).astype(np.float32)
    south_cap = np.clip(2 * (y - (height - 1)), 0, 1).astype(np.float32)
    return rows, fx.astype(np.float32), fy.astype(np.float32), north_cap, south_cap


def _sample(field, sampling):
    rows, fx, fy, north_cap, south_cap = sampling
    y0, x00, x01 = rows[0]
    y1, x10, x11 = rows[1]
    a = field[y0, x00] * (1 - fx) + field[y0, x01] * fx
    b = field[y1, x10] * (1 - fx) + field[y1, x11] * fx
    result = a * (1 - fy) + b * fy
    return (result * (1 - north_cap - south_cap)
            + field[0].mean() * north_cap + field[-1].mean() * south_cap)


def _hash3(x, y, z, seed):
    """Integer-only spatial hash: reproducible and independent of tiling/order."""
    value = (x.astype(np.uint32) * np.uint32(73856093)
             ^ y.astype(np.uint32) * np.uint32(19349663)
             ^ z.astype(np.uint32) * np.uint32(83492791)
             ^ np.uint32(seed & 0xffffffff))
    value ^= value >> np.uint32(16)
    value *= np.uint32(2246822519)
    value ^= value >> np.uint32(13)
    value *= np.uint32(3266489917)
    value ^= value >> np.uint32(16)
    return (value >> np.uint32(8)).astype(np.float32) * np.float32(2.0 / 16777215.0) - np.float32(1)


def _noise3(x, y, z, frequency, seed):
    """Smooth 3-D value noise evaluated on a sphere, with no map-edge seam."""
    xx, yy, zz = x * frequency, y * frequency, z * frequency
    ix, iy, iz = np.floor(xx).astype(np.int32), np.floor(yy).astype(np.int32), np.floor(zz).astype(np.int32)
    fx, fy, fz = xx - ix, yy - iy, zz - iz
    # Quintic interpolation gives continuous first and second derivatives.
    fx = (fx * fx * fx * (fx * (fx * 6 - 15) + 10)).astype(np.float32)
    fy = (fy * fy * fy * (fy * (fy * 6 - 15) + 10)).astype(np.float32)
    fz = (fz * fz * fz * (fz * (fz * 6 - 15) + 10)).astype(np.float32)
    result = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                result += (_hash3(ix + dx, iy + dy, iz + dz, seed)
                           * (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                           * (fz if dz else 1 - fz))
    return result


def _prepare(frame):
    try:
        width, height = int(frame["width"]), int(frame["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The source frame must include its width and height.") from exc
    if width < 4 or height < 2 or width != 2 * height:
        raise ValueError("Terrain refinement needs a global 2:1 source grid.")
    fields = {}
    for key in ("elevation", "crust", "boundary"):
        try:
            data = np.asarray(frame[key])
            if data.size != width * height or not np.isfinite(data).all():
                raise ValueError
            fields[key] = data.astype(np.float32).reshape(height, width)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"The source frame has an invalid {key} field.") from exc
    z, crust, boundary = fields["elevation"], fields["crust"], fields["boundary"]
    if not np.isin(crust, [0, 1, 2, 3]).all() or not np.isin(boundary, range(7)).all():
        raise ValueError("Source crust or boundary codes are invalid.")
    smooth = _blur(z, 1)
    gx = (np.roll(smooth, -1, axis=1) - np.roll(smooth, 1, axis=1)) * .5
    gy = np.gradient(smooth, axis=0)
    fields["slope"] = np.clip(np.hypot(gx, gy) / 1400, 0, 1.5).astype(np.float32)
    fields["craton"] = _blur(crust == 2, 1)
    fields["arc"] = _blur(crust == 3, 1)
    fields["continent"] = _blur(crust > 0, 1)
    for name, code, passes in (("collision", 4, 4), ("subduction", 2, 3),
                                ("rift", 5, 2), ("ridge", 1, 2)):
        fields[name] = np.clip(_blur(boundary == code, passes) * 2.5, 0, 1)
    age = np.asarray(frame.get("age", np.full(width * height, 100)), dtype=np.float32)
    if age.size != width * height or not np.isfinite(age).all():
        raise ValueError("The source frame has an invalid age field.")
    fields["young_ocean"] = np.exp(-np.maximum(age.reshape(height, width), 0) / 45).astype(np.float32)
    fields["suture"] = np.zeros_like(z)
    fields["deformation"] = np.zeros_like(z)
    trace_count = 0
    if "trace_xyz" in frame:
        xyz = np.asarray(frame["trace_xyz"], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("The source frame has invalid material-marker positions.")
        trace_count = len(xyz)
        if trace_count:
            lon = np.arctan2(xyz[:, 1], xyz[:, 0])
            lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
            cells = (np.clip(((np.pi / 2 - lat) * height / np.pi).astype(int), 0, height - 1) * width
                     + (((lon + np.pi) * width / (2 * np.pi)).astype(int) % width))
            weights = np.bincount(cells, minlength=width * height).reshape(height, width).astype(np.float32)
            den = _blur(weights, 4)
            for source, target in (("trace_suture", "suture"), ("trace_uplift_m", "deformation")):
                values = np.asarray(frame.get(source, np.zeros(trace_count)), dtype=np.float32)
                if values.shape != (trace_count,) or not np.isfinite(values).all():
                    raise ValueError(f"The source frame has an invalid {source} field.")
                values = np.clip(values if target == "suture" else values / 6000, 0, 1)
                numerator = np.bincount(cells, weights=values, minlength=width * height).reshape(height, width)
                field = np.divide(_blur(numerator, 4), den, out=np.zeros_like(z), where=den > 1e-5)
                # Trace history is a broad conditioning envelope, not a new rock map.
                fields[target] = field * np.clip(den * 8, 0, 1) * fields["continent"]
    return fields, trace_count


def _coordinates(width, height, y0, y1):
    lon = ((np.arange(width, dtype=np.float64) + .5) * (2 * np.pi / width) - np.pi).astype(np.float32)[None, :]
    lat = (np.pi / 2 - (np.arange(y0, y1, dtype=np.float64) + .5) * (np.pi / height)).astype(np.float32)[:, None]
    coslat = np.cos(lat)
    return lon, lat, coslat * np.cos(lon), coslat * np.sin(lon), np.sin(lat)


def _synthesize(fields, width, height, y0, y1, seed, detail):
    source_h, source_w = fields["elevation"].shape
    lon, lat, x, y, z = _coordinates(width, height, y0, y1)
    sampling = _sampling(lon, lat, source_w, source_h)
    original = _sample(fields["elevation"], sampling)
    if detail == 0:
        return original, original
    # Stable world-space noise has no frame-index seed.  Fine detail remains a
    # world-space synthesis, however, and must not be called material tracking.
    noises = [_noise3(x, y, z, np.float32(frequency), seed + octave * 104729)
              for octave, frequency in enumerate((32, 64, 128, 256, 512, 1024, 2048))]
    # Displace coast/interior sampling by a fraction of a source cell.  This
    # breaks straight interpolated shore segments while retaining broad basins.
    shore = np.exp(-np.abs(original) / np.float32(800))
    polar_guard = np.minimum(1, np.cos(lat) * np.float32(source_h / np.pi))
    angular = np.float32(2 * np.pi / source_w) * detail * (.10 + .18 * shore) * polar_guard
    shifted_lat = lat + angular * noises[1]
    shifted_lon = lon + angular * noises[0] / np.maximum(np.cos(lat), .12)
    north, south = shifted_lat > np.pi / 2, shifted_lat < -np.pi / 2
    shifted_lon += np.where(north | south, np.pi, 0).astype(np.float32)
    shifted_lat = np.where(north, np.pi - shifted_lat,
                           np.where(south, -np.pi - shifted_lat, shifted_lat))
    sampling = _sampling(shifted_lon, shifted_lat, source_w, source_h)
    base = _sample(fields["elevation"], sampling)
    craton = _sample(fields["craton"], sampling)
    arc = _sample(fields["arc"], sampling)
    continent = _sample(fields["continent"], sampling)
    slope = _sample(fields["slope"], sampling)
    convergence = np.maximum(_sample(fields["collision"], sampling),
                             _sample(fields["subduction"], sampling) * continent)
    suture = _sample(fields["suture"], sampling)
    deformation = _sample(fields["deformation"], sampling)
    mountain = np.clip(np.maximum(base - 550, 0) / 2400 + .40 * slope + .55 * convergence, 0, 1.5)
    # Low plains stay quiet; rugged relief follows high terrain, active margins,
    # arcs, and surviving deformation memory.  Cratonic interiors are subdued.
    ruggedness = 1 - .78 * craton
    ridges = ((1 - np.abs(noises[2])) ** 3 - .44) * .68
    ridges += ((1 - np.abs(noises[3])) ** 3 - .44) * .24
    ridges += ((1 - np.abs(noises[4])) ** 3 - .44) * .08
    fine = .44 * noises[3] + .29 * noises[4] + .18 * noises[5] + .09 * noises[6]
    broad = .6 * noises[1] + .4 * noises[2]
    ridge_amplitude = (90 + 1050 * mountain + 350 * arc + 140 * suture + 80 * deformation) * ruggedness
    rough_amplitude = (55 + 180 * mountain + 130 * slope + 130 * arc + 60 * suture) * ruggedness
    land_detail = ridge_amplitude * ridges + rough_amplitude * fine + (35 + 55 * suture) * broad * ruggedness
    land_detail += 85 * _sample(fields["rift"], sampling) * (fine - .35 * ridges)
    ocean_amplitude = (45 + 95 * _sample(fields["young_ocean"], sampling)
                       + 180 * _sample(fields["ridge"], sampling)
                       + 70 * _sample(fields["subduction"], sampling))
    ocean_detail = ocean_amplitude * (.75 * fine + .35 * ridges)
    blend = np.clip(.5 + base / 400, 0, 1)
    added = detail * (land_detail * blend + ocean_detail * (1 - blend))
    # Added abyssal texture cannot invent land in a deep basin, and texture on
    # established land cannot excavate a new ocean.  Narrow shore zones can vary.
    allowance = np.maximum(np.abs(base) * .38, 95)
    added = np.clip(added, -allowance, allowance)
    return np.clip(base + added, -11999, 20000).astype(np.float32), original


def _png_chunk(handle, kind, data):
    handle.write(struct.pack(">I", len(data)))
    handle.write(kind)
    handle.write(data)
    handle.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))


def _write_height_png(path, elevation, band_rows, cancel, progress):
    height, width = elevation.shape
    with Path(path).open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        _png_chunk(handle, b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0))
        compressor = zlib.compressobj(level=4)
        for y0 in range(0, height, band_rows):
            _check_cancel(cancel)
            y1 = min(height, y0 + band_rows)
            pixels = np.rint(np.clip(np.asarray(elevation[y0:y1], dtype=np.float64) + 12000, 0, 65535)).astype(">u2")
            raw = b"".join(b"\0" + row.tobytes() for row in pixels)
            compressed = compressor.compress(raw)
            if compressed:
                _png_chunk(handle, b"IDAT", compressed)
            if progress:
                progress(.83 + .15 * y1 / height)
        compressed = compressor.flush()
        if compressed:
            _png_chunk(handle, b"IDAT", compressed)
        _png_chunk(handle, b"IEND", b"")


def _color_array(z, world_width, world_height, y0=0):
    stops = np.array([-11000, -6500, -4000, -1500, -1, 0, 400, 1200, 2400, 4200, 6500, 9000])
    colors = np.array([[5, 18, 34], [10, 38, 65], [19, 70, 96], [37, 115, 135],
                       [94, 162, 166], [121, 154, 117], [132, 157, 111],
                       [165, 160, 114], [164, 139, 103], [154, 126, 107],
                       [207, 201, 185], [247, 245, 234]])
    rgb = np.stack([np.interp(z, stops, colors[:, k]) for k in range(3)], axis=-1)
    gy, gx = np.gradient(z)
    latitude = np.pi / 2 - (np.arange(y0, y0 + z.shape[0]) + .5) * np.pi / world_height
    dx = 2 * np.pi * RADIUS_KM * 1000 / world_width * np.maximum(np.cos(latitude), .08)
    dy = np.pi * RADIUS_KM * 1000 / world_height
    gx = gx * 5 / dx[:, None]
    gy = gy * 5 / dy
    light = (.42 * gx + .56 * gy + .71) / np.sqrt(1 + gx * gx + gy * gy)
    shade = np.clip(.55 + .62 * light, .43, 1.16)
    return np.uint8(np.clip(rgb * shade[..., None], 0, 255))


def color_tile(elevation, x, y, size=512):
    """Colour an actual-resolution pixel crop, reading only a one-cell halo.

    ``x`` and ``y`` are pixel offsets, not tile numbers.  ``elevation`` may be a
    two-dimensional array/memmap or the path of an elevation_m.npy file.
    """
    owned = isinstance(elevation, (str, Path))
    data = np.load(elevation, mmap_mode="r", allow_pickle=False) if owned else np.asarray(elevation)
    try:
        if data.ndim != 2:
            raise ValueError("Elevation must be a two-dimensional array.")
        height, width = data.shape
        x, y, size = int(x), int(y), int(size)
        if size < 1 or size > 2048 or x < 0 or y < 0 or x >= width or y >= height:
            raise ValueError("The requested terrain tile is outside the map or too large.")
        tile_w, tile_h = min(size, width - x), min(size, height - y)
        xs = np.arange(x - 1, x + tile_w + 1) % width
        ys = np.arange(y - 1, y + tile_h + 1).clip(0, height - 1)
        halo = np.asarray(data[np.ix_(ys, xs)], dtype=np.float32)
        rgb = _color_array(halo, width, height, y - 1)[1:-1, 1:-1]
        return Image.fromarray(rgb)
    finally:
        if owned:
            del data


def _write_preview(path, elevation, cancel):
    height, width = elevation.shape
    preview_w = min(2048, width)
    preview_h = preview_w // 2
    preview = np.empty((preview_h, preview_w), dtype=np.float32)
    # Supported production sizes divide exactly.  The general path also keeps
    # small development/test dimensions useful without a full-sized copy.
    edges_x = np.linspace(0, width, preview_w + 1).astype(int)
    edges_y = np.linspace(0, height, preview_h + 1).astype(int)
    for row in range(preview_h):
        if row % 32 == 0:
            _check_cancel(cancel)
        strip = np.asarray(elevation[edges_y[row]:edges_y[row + 1]], dtype=np.float32).mean(axis=0)
        if width % preview_w == 0:
            preview[row] = strip.reshape(preview_w, width // preview_w).mean(axis=1)
        else:
            preview[row] = np.add.reduceat(strip, edges_x[:-1]) / np.diff(edges_x)
    Image.fromarray(_color_array(preview, preview_w, preview_h)).save(path)


def output_points(width, first, last, orientation=None):
    """Exact float64 pixel-centre directions, expressed in the source world."""
    index = np.arange(first, last, dtype=np.int64)
    longitude = (index % width+.5)*(2*np.pi/width)-np.pi
    latitude = np.pi/2-(index//width+.5)*(2*np.pi/(width))
    cosine = np.cos(latitude)
    points = np.column_stack((cosine*np.cos(longitude), cosine*np.sin(longitude), np.sin(latitude)))
    if orientation is not None and any(normalize_orientation(orientation).values()):
        points = points@rotation_matrix(orientation).T
    return points


def _sampled_detail(points, baseline, crust, seed, detail):
    """Optional short-wavelength texture; never displace the source coastline."""
    if detail == 0:
        return baseline
    x, y, z = np.asarray(points, np.float32).T
    texture = np.zeros(len(points), np.float32)
    for octave, (frequency, weight) in enumerate(((256, .45), (512, .30), (1024, .17), (2048, .08))):
        texture += weight*_noise3(x, y, z, frequency, seed+104729*octave)
    amplitude = 40.+220.*np.clip(np.abs(baseline)/4000., 0., 1.)
    amplitude *= np.where(np.asarray(crust) == 2, .25, 1.)
    # This bound vanishes at the actual zero contour. Even maximum user detail
    # cannot flip land to sea, move an island, or fill a material gap.
    allowance = np.minimum(detail*amplitude, .15*np.abs(baseline))
    return baseline+allowance*texture


def build_sampled_terrain(sample_height, output_dir, *, source, width=8192, seed=12,
                          detail=0., progress=None, cancel=None, band_rows=32,
                          orientation=None, sample_chunk=SAMPLE_CHUNK):
    """Regrid a continuous spherical source directly, without an interim raster.

    ``sample_height`` accepts source-world unit XYZ and returns either heights
    or a dictionary with elevation and optional crust. It is called in bounded
    batches. The caller prepares its locator once, and owns source validation.
    No regional bias subtraction, coastline warp or second rotation resample is
    applied. NPY retains float32 source heights when procedural detail is zero.
    """
    if isinstance(width, bool) or int(width) != width:
        raise ValueError('Terrain width must be an integer.')
    width = int(width)
    if width not in SUPPORTED_WIDTHS and not (32 <= width <= 1024 and width % 2 == 0):
        raise ValueError('Choose a terrain width of 2048, 4096, 8192 or 16384.')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError('Terrain seed must be an integer.')
    if not np.isfinite(detail) or not 0 <= float(detail) <= 2:
        raise ValueError('Terrain detail must be between zero and two.')
    if isinstance(band_rows, bool) or int(band_rows) != band_rows or not 1 <= int(band_rows) <= 128:
        raise ValueError('Band height must be an integer between 1 and 128.')
    if isinstance(sample_chunk, bool) or int(sample_chunk) != sample_chunk or not 1 <= sample_chunk <= SAMPLE_CHUNK:
        raise ValueError(f'Sampling batches must contain 1..{SAMPLE_CHUNK} points.')
    orientation = normalize_orientation(orientation)
    source = dict(source)
    radius_km = float(source.get('radius_m', RADIUS_KM*1000.))/1000.
    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError('The spherical source radius must be positive and finite.')
    # Reject nonportable source metadata before creating output products.
    json.dumps(source, allow_nan=False)
    height, band_rows, seed, detail = width//2, int(band_rows), int(seed), float(detail)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir/name for name in ('elevation_m.npy', 'heightmap_16bit.png', 'preview.png', 'terrain_metadata.json')]
    if any(path.exists() for path in targets):
        raise ValueError('The terrain output directory already contains terrain products.')
    _check_cancel(cancel)
    started = time.monotonic()
    elevation = np.lib.format.open_memmap(targets[0], mode='w+', dtype=np.float32, shape=(height, width))
    minimum, maximum, area_sum, weighted_sum, land_sum = np.inf, -np.inf, 0., 0., 0.
    clipping_count, maximum_detail = 0, 0.
    try:
        for y0 in range(0, height, band_rows):
            y1 = min(height, y0+band_rows)
            block = np.empty((y1-y0)*width, np.float32)
            for start in range(0, len(block), int(sample_chunk)):
                _check_cancel(cancel)
                end = min(len(block), start+int(sample_chunk))
                points = output_points(width, y0*width+start, y0*width+end, orientation)
                sampled = sample_height(points)
                baseline = np.asarray(sampled['elevation'] if isinstance(sampled, dict) else sampled, float)
                if baseline.shape != (len(points),) or not np.isfinite(baseline).all():
                    raise ValueError('Spherical terrain sampler returned invalid elevations.')
                crust = sampled.get('crust', np.zeros(len(points), np.uint8)) if isinstance(sampled, dict) else np.zeros(len(points), np.uint8)
                values = _sampled_detail(points, baseline, crust, seed, detail)
                if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
                    raise ValueError('Spherical terrain elevation cannot be represented in float32.')
                maximum_detail = max(maximum_detail, float(np.max(np.abs(values-baseline), initial=0)))
                block[start:end] = values.astype(np.float32)
                if progress:
                    progress(.80*(y0*width+end)/(height*width))
            block = block.reshape(y1-y0, width)
            elevation[y0:y1] = block
            minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
            clipping_count += int(np.count_nonzero((block < -12000.) | (block > 53535.)))
            for row, values in enumerate(block, y0):
                weight = math.cos(math.pi/2-(row+.5)*math.pi/height)
                area_sum += weight*width
                weighted_sum += float(values.astype(np.float64).sum())*weight
                land_sum += int(np.count_nonzero(values >= 0))*weight
        elevation.flush()
        _write_height_png(targets[1], elevation, band_rows, cancel, progress)
        _write_preview(targets[2], elevation, cancel)
        _check_cancel(cancel)
        metadata = dict(terrain_version=TERRAIN_VERSION, state='complete', width=width, height=height,
            seed=seed, detail=detail, source_type=source.get('source_type', 'spherical_surface'),
            source=source, source_width=source.get('width'), source_height=source.get('height'),
            time_myr=float(source.get('time_myr', 0.)), orientation=orientation,
            orientation_convention=ORIENTATION_CONVENTION,
            orientation_application='Output pixel centres are inverse-rotated into the original spherical source before sampling; no intermediate raster or second resampling.',
            resolution=dict(terrain_grid=[width, height],
                equatorial_pixel_km=2*math.pi*radius_km/width,
                meaning='Direct sampling of the authoritative spherical source; finer pixels retain available geometry but do not create new resolved physics.'),
            coordinates=dict(projection='equirectangular', radius_km=radius_km, row_order='north to south',
                longitude_range=[-180, 180], latitude_range=[90, -90], registration='pixel centres',
                longitude_degrees='-180 + (column + 0.5) * 360 / width', latitude_degrees='90 - (row + 0.5) * 180 / height'),
            files={'elevation_m.npy': dict(dtype='float32', shape=[height, width], units='metres relative to source sea-level datum'),
                   'heightmap_16bit.png': dict(encoding='unsigned 16-bit grayscale', decode='elevation_metres = pixel_value - 12000',
                       rounding_error_max_m=.5, representable_range_m=[-12000, 53535], clipped_pixels=clipping_count,
                       note='NPY preserves elevations outside the PNG range; the PNG clamps those pixels.'),
                   'preview.png': dict(purpose='colour and exaggerated relief shading; display only', width=min(2048, width), height=min(2048, width)//2)},
            method='Direct spherical source sampling'+('; optional bounded 3D short-wavelength texture' if detail else ''),
            baseline=dict(review_raster_used=False, coastline_warp=False, regional_bias_correction=False,
                          maximum_added_detail_m=maximum_detail, detail_zero='Exact sampler output rounded once to float32.'),
            history=dict(seed_changes_with_epoch=False, fine_features_material_tracked=False),
            downstream=dict(role='final/base topography for landscape or climate preparation', gospl_ready_forcing=False),
            stats=dict(minimum_m=minimum, maximum_m=maximum, area_weighted_mean_m=weighted_sum/area_sum,
                       land_area_fraction=land_sum/area_sum),
            elapsed_seconds=round(time.monotonic()-started, 3),
            working_memory=dict(output='disk-backed float32 NPY', sample_batch_points=int(sample_chunk),
                                row_band_limit=band_rows, full_output_copy=False))
        targets[3].write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        if progress:
            progress(1.)
        return metadata
    finally:
        elevation.flush()
        elevation._mmap.close()


def build_terrain(frame, output_dir, width=8192, seed=12, detail=1.0,
                  progress=None, cancel=None, band_rows=32, orientation=None, reconstruct_margins=False):
    """Write a selected epoch's refined global surface with bounded work arrays.

    Production widths are 2048, 4096, 8192 and 16384.  Even widths 32..1024 are
    accepted for tests/development.  The caller owns ``output_dir`` and handles
    removal of incomplete products after cancellation; this function never
    recursively deletes caller data.  Success is marked by terrain_metadata.json.
    Orientation is applied to the finished surface, after procedural synthesis
    and bias correction, so repositioning moves the same terrain and fine detail.
    """
    if isinstance(width, bool) or int(width) != width:
        raise ValueError("Terrain width must be an integer.")
    width = int(width)
    if width not in SUPPORTED_WIDTHS and not (32 <= width <= 1024 and width % 2 == 0):
        raise ValueError("Choose a terrain width of 2048, 4096, 8192 or 16384.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("Terrain seed must be an integer.")
    if not np.isfinite(detail) or not 0 <= float(detail) <= 2:
        raise ValueError("Terrain detail must be between 0 and 2.")
    if isinstance(band_rows, bool) or int(band_rows) != band_rows or not 1 <= int(band_rows) <= 128:
        raise ValueError("Band height must be an integer between 1 and 128.")
    if not isinstance(reconstruct_margins, (bool, np.bool_)):
        raise ValueError('Margin reconstruction must be enabled or disabled explicitly.')
    orientation = normalize_orientation(orientation)
    rotated = any(orientation.values())
    height, band_rows, detail, seed = width // 2, int(band_rows), float(detail), int(seed)
    if native_frame_sampling.is_native(frame):
        _check_cancel(cancel)
        if reconstruct_margins:
            import continental_margin
            frame = continental_margin.upgrade_frame(frame)
        prepared = native_frame_sampling.prepare(frame)
        source = {key: frame[key] for key in ('run_id', 'index', 'time_myr', 'width', 'height',
                    'engine_sha256', 'source_engine_sha256', 'source_frame_sha256', 'mesh_version',
                    'surface_reconstruction_version', 'arc_surface_version', 'continental_margin_version',
                    'continental_margin_parameters', 'continental_margin_revision') if key in frame}
        source.update(source_type='native_history', control_faces=len(frame['mesh_faces']),
                      material_faces=len(frame['material_faces']), material_vertices=len(frame['material_vertices']),
                      reconstruct_margins=bool(reconstruct_margins))
        return build_sampled_terrain(lambda points: native_frame_sampling.sample_frame(frame, points, prepared),
            output_dir, source=source, width=width, seed=seed, detail=detail, progress=progress, cancel=cancel,
            band_rows=band_rows, orientation=orientation)
    if reconstruct_margins:
        raise ValueError('Reconstructing physical continental margins requires a native mesh history.')
    fields, trace_count = _prepare(frame)
    source_h, source_w = fields["elevation"].shape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in ("elevation_m.npy", "heightmap_16bit.png", "preview.png", "terrain_metadata.json")]
    rotation_path = output_dir / '.orientation_work.npy'
    if any(path.exists() for path in targets) or rotation_path.exists():
        raise ValueError("The terrain output directory already contains terrain products.")
    _check_cancel(cancel)
    started = time.monotonic()
    elevation = np.lib.format.open_memmap(targets[0], mode="w+", dtype=np.float32, shape=(height, width))
    rotation_array = None
    bias_sum = np.zeros((source_h, source_w), dtype=np.float64)
    bias_weight = np.zeros_like(bias_sum)
    coarse_x = np.minimum(((np.arange(width) + .5) * source_w / width).astype(int), source_w - 1)
    x_count = np.bincount(coarse_x, minlength=source_w)
    try:
        for y0 in range(0, height, band_rows):
            _check_cancel(cancel)
            y1 = min(height, y0 + band_rows)
            terrain, original = _synthesize(fields, width, height, y0, y1, seed, detail)
            elevation[y0:y1] = terrain
            # Accumulate row by row in a fixed order, so band partition cannot
            # affect rounding.  A smooth coarse residual is removed below to
            # avoid adding a new long-wavelength continental/basin elevation.
            for row in range(y0, y1):
                coarse_y = min(int((row + .5) * source_h / height), source_h - 1)
                area = math.cos(math.pi / 2 - (row + .5) * math.pi / height)
                residual = (terrain[row - y0].astype(np.float64) - original[row - y0]) * area
                bias_sum[coarse_y] += np.bincount(coarse_x, weights=residual, minlength=source_w)
                bias_weight[coarse_y] += x_count * area
            if progress:
                progress((.60 if rotated else .75) * y1 / height)
        bias = np.divide(bias_sum, bias_weight, out=np.zeros_like(bias_sum), where=bias_weight > 0)
        bias = _blur(bias, 3)
        minimum, maximum, area_sum, weighted_sum, land_sum = np.inf, -np.inf, 0., 0., 0.
        for y0 in range(0, height, band_rows):
            _check_cancel(cancel)
            y1 = min(height, y0 + band_rows)
            lon, lat, _, _, _ = _coordinates(width, height, y0, y1)
            correction = _sample(bias, _sampling(lon, lat, source_w, source_h))
            corrected = np.clip(elevation[y0:y1] - correction, -11999, 20000).astype(np.float32)
            elevation[y0:y1] = corrected
            minimum, maximum = min(minimum, float(corrected.min())), max(maximum, float(corrected.max()))
            weights = np.cos(lat).astype(np.float64)
            area_sum += float(weights.sum()) * width
            weighted_sum += float((corrected * weights).sum())
            land_sum += float(((corrected >= 0) * weights).sum())
            if progress:
                progress(.60 + .05 * y1 / height if rotated else .75 + .08 * y1 / height)
        elevation.flush()
        if rotated:
            _check_cancel(cancel)
            rotation_array = np.lib.format.open_memmap(rotation_path, mode='w+', dtype=np.float32, shape=(height, width))
            try:
                reproject_raster(elevation, rotation_array, orientation,
                                 progress=(lambda fraction: progress(.65+.15*fraction)) if progress else None,
                                 cancel=cancel)
            except InterruptedError as exc:
                raise TerrainCancelled('Terrain rotation was cancelled.') from exc
            rotation_array.flush()
            # Windows forbids replacing files while their mappings are open.
            # The original here is this job's newly synthesized output, never a
            # saved world or another completed terrain product.
            rotation_array._mmap.close()
            rotation_array = None
            elevation._mmap.close()
            elevation = None
            rotation_path.replace(targets[0])
            elevation = np.load(targets[0], mmap_mode='r+', allow_pickle=False)
            minimum, maximum, area_sum, weighted_sum, land_sum = np.inf, -np.inf, 0., 0., 0.
            for y0 in range(0, height, band_rows):
                _check_cancel(cancel)
                y1 = min(height, y0+band_rows)
                values = np.asarray(elevation[y0:y1])
                _, lat, _, _, _ = _coordinates(width, height, y0, y1)
                weights = np.cos(lat).astype(np.float64)
                minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
                area_sum += float(weights.sum())*width
                weighted_sum += float((values*weights).sum())
                land_sum += float(((values >= 0)*weights).sum())
                if progress:
                    progress(.80+.03*y1/height)
        _write_height_png(targets[1], elevation, band_rows, cancel, progress)
        _write_preview(targets[2], elevation, cancel)
        _check_cancel(cancel)
        metadata = {
            "terrain_version": TERRAIN_VERSION, "state": "complete",
            "width": width, "height": height, "seed": seed, "detail": detail,
            "orientation": orientation, "orientation_convention": ORIENTATION_CONVENTION,
            "orientation_application": "The completed terrain is rigidly rotated after synthesis and coarse-bias correction. Bilinear resampling can change pixel-edge details; it does not regenerate the noise in its new position.",
            "source_width": source_w, "source_height": source_h,
            "time_myr": float(frame.get("time_myr", 0)),
            "source": {"width": source_w, "height": source_h,
                       "time_myr": float(frame.get("time_myr", 0)), "material_marker_count": trace_count,
                       **{key: frame[key] for key in ("run_id", "index", "engine_sha256", "source_engine_sha256") if key in frame}},
            "resolution": {
                "tectonic_grid": [source_w, source_h], "terrain_grid": [width, height],
                "equatorial_pixel_km": 2 * math.pi * RADIUS_KM / width,
                "meaning": "Fine terrain pixels are a procedural refinement of the recorded tectonic frame, not additional resolved tectonic history."},
            "coordinates": {"projection": "equirectangular", "radius_km": RADIUS_KM,
                            "row_order": "north to south", "longitude_range": [-180, 180],
                            "latitude_range": [90, -90], "registration": "pixel centres",
                            "longitude_degrees": "-180 + (column + 0.5) * 360 / width",
                            "latitude_degrees": "90 - (row + 0.5) * 180 / height"},
            "files": {"elevation_m.npy": {"dtype": "float32", "shape": [height, width], "units": "metres relative to model sea level"},
                      "heightmap_16bit.png": {"encoding": "unsigned 16-bit grayscale", "decode": "elevation_metres = pixel_value - 12000", "rounding_error_max_m": .5},
                      "preview.png": {"purpose": "colour and exaggerated relief shading; display only", "width": min(2048, width), "height": min(2048, width) // 2}},
            "conditioning": ["source relief and relief gradients", "cratonic stability", "island-arc crust",
                             "collision and subduction proximity", "rifts and young ocean floor",
                             "available material-marker suture and cumulative uplift envelopes"],
            "method": "Spherical 3-D multiscale noise, ridged relief, bounded coast displacement and smooth removal of coarse-scale added elevation bias.",
            "history": {"seed_changes_with_epoch": False, "fine_features_material_tracked": False,
                        "description": "The same world-coordinate noise is used at every epoch. Tectonic conditioning follows each recorded frame; individual fine ridges/coast details do not follow material trajectories."},
            "downstream": {"role": "initial/base topography for subsequent landscape and climate preparation",
                           "gospl_ready_forcing": False,
                           "note": "Elevation contains procedural relief and the tectonic model's existing relief relaxation. It is not a separate uplift/subsidence or horizontal-advection forcing field, a drainage solution, or a ROCKE-3D input package."},
            "stats": {"minimum_m": minimum, "maximum_m": maximum,
                      "area_weighted_mean_m": weighted_sum / area_sum,
                      "land_area_fraction": land_sum / area_sum},
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "working_memory": "Float32 output is written through a disk-backed array; synthesis and PNG encoding use bounded row bands. Nonidentity orientation uses a second temporary disk array and bounded inverse-warp chunks."}
        targets[3].write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        if progress:
            progress(1.)
        return metadata
    finally:
        if rotation_array is not None:
            rotation_array._mmap.close()
        if elevation is not None:
            elevation.flush()
            elevation._mmap.close()
        # Only this function's temporary file is removed. Incomplete public
        # products remain the caller's responsibility, as for synthesis errors.
        rotation_path.unlink(missing_ok=True)
