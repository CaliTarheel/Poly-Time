"""Rigid orientation of a saved globe, without changing its tectonic history.

Angles are degrees about fixed world axes, applied yaw (Z), pitch (Y), then
roll (X). Cartesian points and angular velocities are row vectors: ``xyz @ R``.
Raster cell centres use north-to-south rows and longitude [-180, 180). A raster
rotation always samples the original frame, never pushes cells into a new grid.
It therefore has no unfilled pixels; finite resolution can still change a
feature's edge by one source cell. Repeated UI adjustments should use the
original frame to avoid accumulating interpolation loss.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np


CONTINUOUS_FIELDS = frozenset(('elevation', 'age', 'crustal_thickness_km',
    'crustal_root_km', 'rift_thermal_support_m', 'rift_cooling_age_myr',
    'foreland_deflection_m', 'erosion_rate_m_myr', 'rift_damage', 'rift_strength_relative',
    'deformation_weight', 'geometric_strain_percent'))
CATEGORICAL_FIELDS = frozenset(('plate', 'crust', 'boundary', 'domain', 'craton', 'trench', 'refinement_level'))
GRID_FIELDS = CONTINUOUS_FIELDS | CATEGORICAL_FIELDS
ORIENTATION_CONVENTION = 'Degrees about fixed world axes: yaw Z, then pitch Y, then roll X; row-vector Cartesian positions xyz @ R.'
_CHUNK = 65_536
_VECTOR_KEYS = frozenset(('angular_velocity', 'center', 'normal', 'tangent', 'xyz',
                         'angular_motion_proxy_before', 'angular_motion_proxy_after'))


def normalize_orientation(value=None):
    """Return finite yaw/pitch/roll in [-180, 180); None means identity.

    All three angles permit a complete turn, including pitch. Unknown keys,
    booleans, strings and nonfinite values are rejected rather than silently
    changing a requested placement. Missing angles default to zero.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {'yaw', 'pitch', 'roll'}:
        raise ValueError('Orientation must contain only yaw, pitch and roll in degrees.')
    result = {}
    for key in ('yaw', 'pitch', 'roll'):
        angle = value.get(key, 0.)
        if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, Real):
            raise ValueError(f'Orientation {key} must be a finite number.')
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError(f'Orientation {key} must be a finite number.')
        angle = (angle + 180.) % 360. - 180.
        result[key] = 0. if angle == 0 else angle
    return result


def rotation_matrix(value=None):
    """Return the proper orthogonal 3x3 matrix for row-vector ``xyz @ R``."""
    angles = normalize_orientation(value)

    def trig(angle):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        # Cardinal rotations also retain exact zeros/ones for exact cell shifts.
        c = round(c) if abs(c-round(c)) < 1e-15 else c
        s = round(s) if abs(s-round(s)) < 1e-15 else s
        return c, s

    c, s = trig(angles['yaw'])
    rz = np.array(((c, s, 0.), (-s, c, 0.), (0., 0., 1.)))
    c, s = trig(angles['pitch'])
    ry = np.array(((c, 0., -s), (0., 1., 0.), (s, 0., c)))
    c, s = trig(angles['roll'])
    rx = np.array(((1., 0., 0.), (0., c, s), (0., -s, c)))
    return rz @ ry @ rx


def _metadata(value, matrix):
    """Copy the recorded geometry in event/plate metadata, leaving IDs intact."""
    if isinstance(value, list):
        return [_metadata(item, matrix) for item in value]
    if isinstance(value, tuple):
        return tuple(_metadata(item, matrix) for item in value)
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {key: _metadata(item, matrix) for key, item in value.items()}
    if 'lon_deg' in value and 'lat_deg' in value:
        import world_design
        result.update(world_design.rotate_region(result, matrix))
    if 'geometry_xyz' in value:
        points = np.asarray(value['geometry_xyz'], dtype=float)
        if points.size == 0:
            result['geometry_xyz'] = []
        elif points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Recorded system geometry must contain finite Cartesian 3-vectors.')
        else:
            result['geometry_xyz'] = (points @ matrix).tolist()
    # Volcanic origin traces and the latest finite source corridors are live
    # world-coordinate evidence, even when no displayed material is present.
    for key,corners in (('trace_xyz',2),('corridors',3)):
        if key in value:
            points=np.asarray(value[key],float)
            if points.size==0:result[key]=[]
            elif points.ndim!=3 or points.shape[1:]!=(corners,3) or not np.isfinite(points).all():
                raise ValueError('Recorded volcanic source geometry is malformed: '+key)
            else:result[key]=(points@matrix).tolist()
    for key in _VECTOR_KEYS.intersection(value):
        vector = np.asarray(value[key], dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f'Recorded {key} must be a finite Cartesian 3-vector.')
        rotated = vector @ matrix
        result[key] = rotated if isinstance(value[key], np.ndarray) else rotated.tolist()
    if 'lon' in value and 'lat' in value:
        lon, lat = float(value['lon']), float(value['lat'])
        if not math.isfinite(lon) or not math.isfinite(lat) or not -90 <= lat <= 90:
            raise ValueError('Recorded event longitude/latitude is invalid.')
        lon, lat = math.radians(lon), math.radians(lat)
        point = np.array((math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))) @ matrix
        # Longitude at a pole is arbitrary; use a deterministic zero.
        result['lon'] = (math.degrees(math.atan2(point[1], point[0])) + 180.) % 360. - 180. if math.hypot(point[0], point[1]) > 1e-14 else 0.
        result['lat'] = math.degrees(math.asin(float(np.clip(point[2], -1., 1.))))
    return result


def _dimensions(frame):
    try:
        w, h = frame['width'], frame['height']
        if isinstance(w, (bool, np.bool_)) or isinstance(h, (bool, np.bool_)):
            raise ValueError
        if int(w) != w or int(h) != h or h < 1 or w != 2*h:
            raise ValueError
        return int(w), int(h)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError('Orientation requires a global 2:1 raster with positive integer width and height.') from None


def orient_metadata(value, orientation=None):
    """Copy recursive lon/lat records and named Cartesian vectors coherently."""
    angles = normalize_orientation(orientation)
    return _metadata(value, rotation_matrix(angles)) if any(angles.values()) else deepcopy(value)


def _inverse_coordinates(cells, width, height, matrix):
    lon = ((cells % width) + .5) * (2*np.pi/width) - np.pi
    lat = np.pi/2 - ((cells // width) + .5) * (np.pi/height)
    coslat = np.cos(lat)
    points = np.stack((coslat*np.cos(lon), coslat*np.sin(lon), np.sin(lat)), axis=-1) @ matrix.T
    x = (np.arctan2(points[..., 1], points[..., 0]) + np.pi) * (width/(2*np.pi)) - .5
    y = (np.pi/2 - np.arcsin(np.clip(points[..., 2], -1., 1.))) * (height/np.pi) - .5
    return x, y


def inverse_cells(cells, width, height, orientation=None):
    """Return original flat cell IDs sampled at destination flat cell centres.

    Accepts a scalar or integer array and retains its shape. This is the same
    nearest-cell mapping used for categorical fields and map inspection.
    """
    w, h = _dimensions(dict(width=width, height=height))
    cells = np.asarray(cells)
    if not np.issubdtype(cells.dtype, np.integer) or np.any(cells < 0) or np.any(cells >= w*h):
        raise ValueError('Destination cells must be integer indices inside the globe.')
    angles = normalize_orientation(orientation)
    if not any(angles.values()):
        return cells.copy()
    x, y = _inverse_coordinates(cells, w, h, rotation_matrix(angles))
    return np.clip(np.floor(y+.5).astype(np.int64), 0, h-1)*w + np.floor(x+.5).astype(np.int64) % w


def _bilinear(field, x, y, pole_means):
    h, w = field.shape
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x-ix, y-iy
    values = []
    for row in (iy, iy+1):
        crossed = (row < 0) | (row >= h)
        yy = np.where(row < 0, -row-1, np.where(row >= h, 2*h-row-1, row)).clip(0, h-1)
        xx = ix + crossed*(w//2)
        values.append(field[yy, xx % w]*(1-fx) + field[yy, (xx+1) % w]*fx)
    north, south = np.clip(-2*y, 0, 1), np.clip(2*(y-h+1), 0, 1)
    return ((values[0]*(1-fy)+values[1]*fy)*(1-north-south)
            + pole_means[0]*north + pole_means[1]*south)


def reproject_raster(source, destination, orientation=None, categorical=False, progress=None, cancel=None):
    """Inverse-warp a 2D source into a separate 2D ndarray or disk-backed array.

    Both arrays must be C-contiguous, same-shaped global 2:1 grids. Source and
    destination must not alias or use the same memmap file. Coordinates and
    samples use at most 65,536 pixels per chunk. The caller owns and cleans up
    its destination. Progress is 0..1; cancellation raises InterruptedError.
    Identity and column-aligned yaw copy pixels exactly. Bilinear output needs
    a floating destination; categorical output always retains source values.
    """
    original_source, original_destination = source, destination
    source, destination = np.asarray(source), np.asarray(destination)
    if source.ndim != 2 or destination.shape != source.shape:
        raise ValueError('Source and destination must be same-shaped 2D global rasters.')
    h, w = source.shape
    _dimensions(dict(width=w, height=h))
    if not source.flags.c_contiguous or not destination.flags.c_contiguous or not destination.flags.writeable:
        raise ValueError('Raster arrays must be C-contiguous and the destination writable.')
    if (np.shares_memory(source, destination) or
            (getattr(original_source, 'filename', None) is not None and
             getattr(original_source, 'filename', None) == getattr(original_destination, 'filename', None))):
        raise ValueError('Source and destination must be separate arrays/files.')
    if not categorical and not np.issubdtype(destination.dtype, np.floating):
        raise ValueError('Continuous interpolation requires a floating destination.')
    angles = normalize_orientation(orientation)
    matrix = rotation_matrix(angles)
    shift = angles['yaw']*w/360.
    exact = angles['pitch'] == 0 and angles['roll'] == 0 and abs(shift-round(shift)) < 1e-10
    poles = (float(source[0].mean(dtype=np.float64)), float(source[-1].mean(dtype=np.float64))) if not categorical else None
    flat_source, flat_destination = source.reshape(-1), destination.reshape(-1)
    for start in range(0, w*h, _CHUNK):
        if cancel is not None and cancel():
            raise InterruptedError('Globe rotation was cancelled.')
        stop = min(start+_CHUNK, w*h)
        cells = np.arange(start, stop)
        if exact:
            indices = (cells//w)*w + ((cells % w)-round(shift)) % w
            flat_destination[start:stop] = flat_source[indices]
        else:
            x, y = _inverse_coordinates(cells, w, h, matrix)
            if categorical:
                indices = np.clip(np.floor(y+.5).astype(np.int64), 0, h-1)*w + np.floor(x+.5).astype(np.int64) % w
                flat_destination[start:stop] = flat_source[indices]
            else:
                flat_destination[start:stop] = _bilinear(source, x, y, poles)
        if progress:
            progress(stop/(w*h))
    if cancel is not None and cancel():
        raise InterruptedError('Globe rotation was cancelled.')
    return destination


def orient_initial(initial, orientation=None):
    """Orient an initial-map dictionary, sharing the frame's categorical warp."""
    return orient_frame(initial, orientation)


def orient_frame(frame, orientation=None):
    """Copy and orient one saved frame or an initial ``{width,height,crust}`` map.

    Identity is an exact deep copy, including array bytes and container types.
    Otherwise category rasters share nearest-cell sampling; elevation and age
    are bilinear with longitude wrap and same-pole continuation. Floating raster
    dtypes are retained; continuous integer inputs become float64 so interpolation
    is not truncated. Flat and 2D raster shapes are retained. Only the documented
    GRID_FIELDS are warped; scalar statistics, IDs and timestamps remain original.

    All trace positions, plate angular velocities and recorded event locations/
    fracture frames receive the same rigid rotation. Domains currently contain
    only identity, area and time metadata, which are copied unchanged. Working
    coordinate memory is bounded independently of map resolution.
    """
    angles = normalize_orientation(orientation)
    w, h = _dimensions(frame)
    sources = {}
    for key in GRID_FIELDS.intersection(frame):
        field = np.asarray(frame[key])
        if field.shape not in ((w*h,), (h, w)) or not np.issubdtype(field.dtype, np.number):
            raise ValueError(f'Recorded {key} must be a flat or height-by-width numeric raster.')
        if np.iscomplexobj(field) or not np.isfinite(field).all():
            raise ValueError(f'Recorded {key} must contain finite real values.')
        sources[key] = field
    if not any(angles.values()):
        return deepcopy(frame)
    matrix = rotation_matrix(angles)
    result = {}
    for key, value in frame.items():
        if key in sources:
            continue
        if key in ('events', 'plates', 'initial_fractures', 'ridge_episodes', 'rift_records', 'boundary_segments',
                   'backarc_basins', 'trench_systems', 'rift_systems', 'local_accretion_contacts',
                   'collision_contacts', 'collision_resistance_diagnostics', 'native_accretion_diagnostics',
                   'arc_material_diagnostics', 'arc_pending_source', 'arc_source_placements',
                   'arc_source_cohort_state', 'volcanic_point_features', 'world_design'):
            result[key] = _metadata(value, matrix)
        elif key in ('trace_xyz', 'mesh_vertices', 'material_vertices'):
            points = np.asarray(value, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f'Recorded {key} must contain finite N-by-3 Cartesian positions.')
            result[key] = points @ matrix
        else:
            result[key] = deepcopy(value)
    column_shift = angles['yaw'] * w / 360.
    if angles['pitch'] == 0 and angles['roll'] == 0 and abs(column_shift-round(column_shift)) < 1e-10:
        for key, field in sources.items():
            result[key] = np.roll(field.reshape(h, w), round(column_shift), axis=1).reshape(field.shape)
        return result
    for key, field in sources.items():
        dtype = np.float64 if key in CONTINUOUS_FIELDS and not np.issubdtype(field.dtype, np.floating) else field.dtype
        result[key] = np.empty(field.shape, dtype=dtype)
    if not sources:
        return result
    continuous = {key: field.reshape(h, w) for key, field in sources.items() if key in CONTINUOUS_FIELDS}
    rift_valid = None
    if 'rift_cooling_age_myr' in continuous:
        rift_valid = (continuous['rift_cooling_age_myr'] >= 0).astype(float)
        continuous['rift_cooling_age_myr'] = np.maximum(continuous['rift_cooling_age_myr'], 0.)
    pole_means = {key: (float(field[0].mean(dtype=np.float64)), float(field[-1].mean(dtype=np.float64)))
                  for key, field in continuous.items()}
    for start in range(0, w*h, _CHUNK):
        stop = min(start+_CHUNK, w*h)
        cells = np.arange(start, stop)
        x, y = _inverse_coordinates(cells, w, h, matrix)
        nearest = np.clip(np.floor(y+.5).astype(np.int64), 0, h-1)*w + np.floor(x+.5).astype(np.int64) % w
        for key in CATEGORICAL_FIELDS.intersection(sources):
            result[key].reshape(-1)[start:stop] = sources[key].reshape(-1)[nearest]
        if not continuous:
            continue
        for key, field in continuous.items():
            # All meridians meet at each pole. Fade the half polar cell to its
            # ring mean so its scalar limit cannot depend on arbitrary longitude.
            values = _bilinear(field, x, y, pole_means[key])
            if key == 'rift_cooling_age_myr':
                coverage = _bilinear(rift_valid, x, y, (rift_valid[0].mean(), rift_valid[-1].mean()))
                values = np.divide(values, coverage, out=np.full(len(values), -1.), where=coverage >= .5)
            result[key].reshape(-1)[start:stop] = values
    return result
