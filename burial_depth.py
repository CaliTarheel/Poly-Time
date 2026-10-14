"""Integrate a burial threshold on disjoint spherical stack regions.

Each lower face is partitioned by its actual covering triangles. A region has
one constant overburden (sum of the local upper columns); the depth threshold
is applied there before integration. Region areas form a union, while columns
within a region form a stack. Neither pair-area summation nor mean depth can
represent both of these operations.
"""
from fractions import Fraction
from math import gcd
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import getcontext, localcontext
import hashlib
from threading import RLock
from time import perf_counter

import numpy as np

import mesh_coverage as geometry
from parallel_runtime import active_runtime, share_inputs, read_inputs
from spherical_predicates import edge_distances, plane_distances
from deforming_regions import IncompleteContactStepError

VERSION = 1
# Below this many uncached faces the process dispatch costs more than it saves.
MIN_PARALLEL_PARTITION_FACES = 16

# Process-local pure geometry only. No Simulation/native-locator field is added.
# These limits bound retained represented payload; entry/root counts also bound
# Python container overhead. Active call snapshots are separate temporary inputs.
_REUSE_ENABLED = ContextVar('burial_geometry_reuse_enabled', default=True)
_REUSE_FINGERPRINTS = ContextVar('burial_geometry_reuse_fingerprints', default=False)


def _array_geometry_key(value):
    if (type(value) is not np.ndarray or value.dtype.hasobject or value.dtype.fields is not None
            or value.dtype.metadata is not None or not value.flags.c_contiguous):
        raise TypeError('Exact geometry reuse requires explicit contiguous nonobject arrays.')
    return value.dtype.str, value.shape, value.strides, value.tobytes(order='C')


def _scalar_geometry_key(value):
    a = np.asarray(value)
    if a.ndim or a.dtype.hasobject:
        raise TypeError('Exact geometry reuse requires numerical scalars.')
    return type(value).__module__, type(value).__qualname__, a.dtype.str, a.tobytes()


def _geometry_arithmetic_key():
    c = getcontext()
    return (c.prec, c.rounding, c.Emin, c.Emax, c.capitals, c.clamp,
            tuple(sorted((signal.__name__, value) for signal, value in c.traps.items())),
            tuple(sorted((signal.__name__, value) for signal, value in c.flags.items())),
            tuple(sorted(np.geterr().items())))


def _geometry_clone(value):
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, list): return [_geometry_clone(item) for item in value]
    if isinstance(value, tuple): return tuple(_geometry_clone(item) for item in value)
    return value


def _geometry_payload(value):
    if isinstance(value, np.ndarray): return value.nbytes
    if isinstance(value, (bytes, str)): return len(value)
    if isinstance(value, (tuple, list)): return sum(_geometry_payload(x) for x in value)
    return 16


class _ExactGeometryReuse:
    def __init__(self, max_entries=16384, max_bytes=64*1024*1024, max_roots=8):
        if min(max_entries, max_bytes, max_roots) <= 0: raise ValueError('Positive geometry reuse bounds required.')
        self.max_entries, self.max_bytes, self.max_roots = max_entries, max_bytes, max_roots
        self.lock = RLock()
        self.entries, self.roots = OrderedDict(), OrderedDict()
        self.generations = {}
        self.next_root = 0
        self.entry_bytes = self.root_bytes = 0
        self.counts = dict(hits=0, misses=0, evictions=0, bypasses=0, warning_probe_bypasses=0,
                           source_invalidations=0, root_snapshots=0, kernel_calls=0,
                           uncached_kernel_seconds=0., partition_hits=0, partition_misses=0,
                           metric_hits=0, metric_misses=0)
        self.fingerprints = OrderedDict()

    def _clear(self):
        self.counts['evictions'] += len(self.entries)
        self.entries.clear(); self.roots.clear()
        self.entry_bytes = self.root_bytes = 0

    def reset(self):
        with self.lock:
            self._clear(); self.generations.clear(); self.fingerprints.clear()
            for name in self.counts: self.counts[name] = 0

    def _intern(self, key):
        # Intern whole triangle and upper-index vectors once per consumer call.
        # Dict lookup checks exact key bytes after hash matching.
        with self.lock:
            if key in self.roots:
                record = self.roots[key]; self.roots.move_to_end(key)
                return record
            cost = _geometry_payload(key)
            if cost > self.max_bytes: return None
            if len(self.roots) >= self.max_roots or self.root_bytes+cost > self.max_bytes:
                self._clear()
            self.next_root += 1
            record = (self.next_root, key)
            self.roots[key] = record; self.root_bytes += cost
            self.counts['root_snapshots'] += 1
            # Enforce the bound before a kernel can throw; exceptions are uncached.
            while self.entries and self.root_bytes+self.entry_bytes > self.max_bytes:
                _, (_, old_cost) = self.entries.popitem(last=False)
                self.entry_bytes -= old_cost; self.counts['evictions'] += 1
            return record

    def _generation(self, role, kernel):
        g = kernel.__globals__
        names = ('_split', '_precise_partition', '_positive_binary64_winding', '_uncertain_winding',
                 'edge_distances', 'plane_distances', '_precise_rotation_integral')
        signature = (kernel,)+tuple(g.get(name) for name in names)+(geometry._unit,
                     geometry._triangle_planes, geometry._polygon_area, np.__version__)
        with self.lock:
            old = self.generations.get(role)
            if old is not None and old != signature:
                self._clear(); self.counts['source_invalidations'] += 1
            self.generations[role] = signature
        return signature

    def _intern_many(self, keys):
        with self.lock:
            missing = [key for key in keys if key not in self.roots]
            if (len(keys) > self.max_roots or
                    sum(_geometry_payload(key) for key in keys) > self.max_bytes): return None
            if (len(self.roots)+len(missing) > self.max_roots or
                    self.root_bytes+sum(_geometry_payload(key) for key in missing) > self.max_bytes):
                self._clear()
            return tuple(self._intern(key) for key in keys)

    def bypass(self):
        with self.lock: self.counts['bypasses'] += 1

    def run(self, compute):
        start = perf_counter()
        try: return compute()
        finally:
            with self.lock:
                self.counts['kernel_calls'] += 1
                self.counts['uncached_kernel_seconds'] += perf_counter()-start

    def _fingerprint(self, role, represented):
        if not _REUSE_FINGERPRINTS.get(): return
        h = hashlib.sha256()
        def add(x):
            if isinstance(x, bytes): h.update(str(len(x)).encode()); h.update(b':'); h.update(x)
            elif isinstance(x, (tuple, list)):
                h.update(b'[')
                for y in x: add(y)
                h.update(b']')
            else: h.update(repr(x).encode()); h.update(b';')
        add((role, represented)); fingerprint = h.hexdigest()
        with self.lock:
            if fingerprint in self.fingerprints:
                row = self.fingerprints.pop(fingerprint); row['calls'] += 1
            else: row = dict(role=role, fingerprint_sha256=fingerprint, calls=1)
            self.fingerprints[fingerprint] = row
            if len(self.fingerprints) > 4096: self.fingerprints.popitem(last=False)

    def get(self, role, key, represented, kernel, compute):
        signature = self._generation(role, kernel)
        self._fingerprint(role, represented)
        with self.lock:
            if key in self.entries:
                value, cost = self.entries.pop(key); self.entries[key] = value, cost
                self.counts['hits'] += 1
                self.counts[role+'_hits'] = self.counts.get(role+'_hits', 0)+1
                return _geometry_clone(value)
            self.counts['misses'] += 1
            self.counts[role+'_misses'] = self.counts.get(role+'_misses', 0)+1
        before = _geometry_arithmetic_key()
        decimal_before = getcontext().copy()
        modes = np.geterr()
        if any(mode not in ('ignore', 'raise', 'warn') for mode in modes.values()):
            self.bypass(); return _geometry_clone(self.run(compute))
        # A warn->raise probe leaves the numerical operations unchanged. A clean
        # probe proves that skipping this call cannot skip an FP warning. On any
        # FP event, rerun in the original restored caller mode: warning filters,
        # registries, source attribution and exceptions retain normal semantics.
        probe_modes = {name: ('raise' if mode == 'warn' else mode) for name, mode in modes.items()}
        try:
            with np.errstate(**probe_modes): result = self.run(compute)
        except FloatingPointError:
            with self.lock: self.counts['warning_probe_bypasses'] += 1
            # The approved kernels use Decimal localcontext. Restore even an
            # outgoing partial caller context before a failed probe is rerun;
            # the real original-mode call alone determines observable effects.
            c = getcontext()
            for name in ('prec', 'rounding', 'Emin', 'Emax', 'capitals', 'clamp'):
                setattr(c, name, getattr(decimal_before, name))
            c.traps.update(decimal_before.traps); c.flags.update(decimal_before.flags)
            self.bypass(); return _geometry_clone(self.run(compute))
        if _geometry_arithmetic_key() != before:
            self.bypass(); return _geometry_clone(result)  # Preserve real outgoing Decimal effects.
        return self._retain(role, key, signature, result)

    def _retain(self, role, key, signature, result):
        retained = _geometry_clone(result)
        cost = _geometry_payload(key)+_geometry_payload(retained)
        with self.lock:
            if self.generations.get(role) != signature or cost+self.root_bytes > self.max_bytes:
                self.counts['bypasses'] += 1; return _geometry_clone(result)
            while self.entries and (len(self.entries) >= self.max_entries or
                                      self.entry_bytes+self.root_bytes+cost > self.max_bytes):
                _, (_, removed) = self.entries.popitem(last=False)
                self.entry_bytes -= removed; self.counts['evictions'] += 1
            old = self.entries.pop(key, None)
            if old is not None: self.entry_bytes -= old[1]
            self.entries[key] = retained, cost; self.entry_bytes += cost
        return _geometry_clone(retained)

    def contains(self, key):
        with self.lock:
            return key in self.entries

    def store(self, role, key, represented, kernel, result, seconds):
        """Record a miss whose kernel ran elsewhere, exactly as get() would retain it."""
        signature = self._generation(role, kernel)
        self._fingerprint(role, represented)
        with self.lock:
            self.counts['misses'] += 1
            self.counts[role+'_misses'] = self.counts.get(role+'_misses', 0)+1
            self.counts['kernel_calls'] += 1
            self.counts['uncached_kernel_seconds'] += seconds
        return self._retain(role, key, signature, result)

    def statistics(self, include_fingerprints=False):
        with self.lock:
            return dict(self.counts, entries=len(self.entries), interned_roots=len(self.roots),
                        accounted_payload_bytes=self.entry_bytes+self.root_bytes,
                        max_entries=self.max_entries, max_payload_bytes=self.max_bytes,
                        max_interned_roots=self.max_roots,
                        recorded_fingerprints=len(self.fingerprints),
                        repeated_fingerprints=sum(row['calls']>1 for row in self.fingerprints.values()),
                        fingerprint_calls=sum(row['calls'] for row in self.fingerprints.values()),
                        fingerprints=[dict(row) for row in self.fingerprints.values()] if include_fingerprints else [])


_GEOMETRY_REUSE = _ExactGeometryReuse()


@contextmanager
def geometry_reuse(*, enabled=True, collect_fingerprints=False):
    """Private validation controls only; never serialized as physical state."""
    one = _REUSE_ENABLED.set(bool(enabled)); two = _REUSE_FINGERPRINTS.set(bool(collect_fingerprints))
    try: yield
    finally: _REUSE_FINGERPRINTS.reset(two); _REUSE_ENABLED.reset(one)


def geometry_cache_reset():
    _GEOMETRY_REUSE.reset()


def geometry_cache_statistics(*, include_fingerprints=False):
    return _GEOMETRY_REUSE.statistics(include_fingerprints)


class _PartitionGeometrySession:
    def __init__(self, triangles, upper):
        self.original_triangles, self.original_upper = triangles, upper
        self.keys = None
        if _REUSE_ENABLED.get():
            try:
                keys = (_array_geometry_key(triangles), _array_geometry_key(upper))
                if sum(_geometry_payload(key) for key in keys) <= _GEOMETRY_REUSE.max_bytes:
                    self.keys = keys
                    # Immutable snapshots preserve the represented input exactly
                    # throughout this consumer loop; no caller array is frozen.
                    self.triangles = np.frombuffer(keys[0][3], dtype=triangles.dtype).reshape(triangles.shape)
                    self.upper = np.frombuffer(keys[1][3], dtype=upper.dtype).reshape(upper.shape)
            except TypeError: pass

    def _cache_key(self, face, selected, reference_area, radius):
        """(key, represented) for reusable inputs, else None after counting a bypass."""
        if self.keys is None or not _REUSE_ENABLED.get():
            _GEOMETRY_REUSE.bypass()
            return None
        try:
            arguments = (_scalar_geometry_key(face), _array_geometry_key(selected),
                         _scalar_geometry_key(reference_area), _scalar_geometry_key(radius), _geometry_arithmetic_key())
        except TypeError:
            _GEOMETRY_REUSE.bypass()
            return None
        # A source-generation change clears the old roots before interning.
        _GEOMETRY_REUSE._generation('partition', _partition_face_uncached)
        records = _GEOMETRY_REUSE._intern_many(self.keys)
        if records is None:
            _GEOMETRY_REUSE.bypass()
            return None
        return ('partition', tuple(record[0] for record in records), arguments), (self.keys, arguments)

    def partition_face(self, face, selected, reference_area, radius):
        cached = self._cache_key(face, selected, reference_area, radius)
        if cached is None:
            return _GEOMETRY_REUSE.run(lambda: _partition_face_uncached(self.original_triangles, face, selected, self.original_upper, reference_area, radius))
        key, represented = cached
        return _GEOMETRY_REUSE.get('partition', key, represented, _partition_face_uncached,
            lambda: _partition_face_uncached(self.triangles, face, selected, self.upper, reference_area, radius))

    def partition_faces(self, jobs):
        """Yield partition_face(*job) for each job, in order.

        With an active multi-worker runtime, uncached faces run the same kernel
        in worker processes on byte-identical shared inputs, under the caller's
        Decimal context and with NumPy warnings raised. A face whose worker met
        any floating-point event or exception is recomputed here at its own
        position, so warnings and errors keep their serial meaning. Worker
        results enter the same bounded cache before they are returned.
        """
        jobs = list(jobs)
        runtime = active_runtime()
        modes = np.geterr()
        if (runtime is None or runtime.policy.status()['requested_workers'] == 1
                or self.keys is None or not _REUSE_ENABLED.get()
                or any(mode not in ('ignore', 'raise', 'warn') for mode in modes.values())):
            for job in jobs:
                yield self.partition_face(*job)
            return
        # The same key partition_face builds; faces that cannot be keyed stay serial.
        keys, missing = [None]*len(jobs), []
        _GEOMETRY_REUSE._generation('partition', _partition_face_uncached)
        records = _GEOMETRY_REUSE._intern_many(self.keys)
        if records is not None:
            roots = tuple(record[0] for record in records)
            for index, (face, selected, reference_area, radius) in enumerate(jobs):
                try:
                    arguments = (_scalar_geometry_key(face), _array_geometry_key(selected),
                                 _scalar_geometry_key(reference_area), _scalar_geometry_key(radius),
                                 _geometry_arithmetic_key())
                except TypeError:
                    continue
                keys[index] = ('partition', roots, arguments), (self.keys, arguments)
                if not _GEOMETRY_REUSE.contains(keys[index][0]):
                    missing.append(index)
        computed = {}
        if len(missing) >= MIN_PARALLEL_PARTITION_FACES:
            computed = self._parallel(runtime, [jobs[index] for index in missing], missing, modes)
        for index, job in enumerate(jobs):
            outcome = computed.pop(index, None)
            if outcome is None:
                yield self.partition_face(*job)
            else:
                result, seconds = outcome
                key, represented = keys[index]
                yield _GEOMETRY_REUSE.store('partition', key, represented, _partition_face_uncached,
                                            result, seconds)

    def _parallel(self, runtime, jobs, indices, modes):
        # Cost grows steeply with the number of covering faces (the exact
        # fallback most of all), and thick stacks are adjacent in face order.
        # Deal faces round-robin from most to fewest covers so every batch
        # gets a similar share; results are keyed by index, not batch order.
        count = min(len(jobs), runtime.policy.status()['requested_workers']*8)
        ranked = sorted(range(len(jobs)), key=lambda position: -len(jobs[position][1]))
        probe = {name: ('raise' if mode == 'warn' else mode) for name, mode in modes.items()}
        context = getcontext().copy()
        with share_inputs(dict(triangles=self.triangles, upper=self.upper)) as shared:
            rows = runtime.map(_partition_batch, ((shared, context, probe,
                [(indices[position],)+tuple(jobs[position]) for position in ranked[start::count]])
                for start in range(count)))
        return {index: outcome for row in rows for index, outcome in row if outcome is not None}


def partition_geometry_session(triangles, upper):
    return _PartitionGeometrySession(triangles, upper)


def reuse_rotation_metric(polygon, radius, kernel):
    if not _REUSE_ENABLED.get():
        _GEOMETRY_REUSE.bypass(); return _GEOMETRY_REUSE.run(lambda: kernel(polygon, radius))
    try:
        arguments = (_array_geometry_key(polygon), _scalar_geometry_key(radius), _geometry_arithmetic_key())
    except TypeError:
        _GEOMETRY_REUSE.bypass(); return _GEOMETRY_REUSE.run(lambda: kernel(polygon, radius))
    return _GEOMETRY_REUSE.get('metric', ('metric', arguments), arguments, kernel,
                             lambda: kernel(polygon, radius))


def _positive_binary64_winding(polygon):
    """Whether a cast polygon still has a positive convex fan, exactly.

    Guard-digit clipping can resolve a positive region whose distinct Decimal
    vertices round onto opposite sides of the same binary64 line.  Evaluate
    the floats as exact rationals at that representation boundary: zero fan
    triangles are harmless, but a negative fan or no positive fan means there
    is no represented two-dimensional convex region to return.
    """
    polygon = np.asarray(polygon, float)
    if len(polygon) < 3:
        return False
    points = [[Fraction.from_float(float(value)) for value in point] for point in polygon]
    a = points[0]
    signs = []
    for b, c in zip(points[1:-1], points[2:]):
        u, v = ([right-left for left, right in zip(a, point)] for point in (b, c))
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        signs.append(sum(x*y for x, y in zip(a, cross)))
    return any(value > 0 for value in signs) and not any(value < 0 for value in signs)


def _uncertain_winding(polygon):
    """Detect negative or cancellation-dominated fan determinants, not tiny area."""
    if len(polygon)<3:return False
    a=polygon[0];u=polygon[1:-1]-a;v=polygon[2:]-a
    products=np.cross(u,v)
    determinant=products@a
    # Bound the products used in the translated determinant. A thin but
    # well-resolved triangle remains on the ordinary double-precision path.
    terms=(np.abs(u[:,[1,2,0]]*v[:,[2,0,1]])
           +np.abs(u[:,[2,0,1]]*v[:,[1,2,0]]))
    roundoff=32*np.finfo(float).eps*(terms@np.abs(a))
    return bool(np.any(determinant<=roundoff))


def _homogeneous(point):
    """Binary64 point as exact integers [X, Y, Z, W], W > 0, value (X/W, Y/W, Z/W)."""
    ratios = [float(value).as_integer_ratio() for value in point]
    w = max(d for _, d in ratios)  # power-of-two denominators divide the largest
    return [n*(w//d) for n, d in ratios]+[w]


def _homogeneous_det(a, b, c):
    """det of the spatial parts: the represented determinant times WaWbWc > 0."""
    return (a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])
            +a[2]*(b[0]*c[1]-b[1]*c[0]))


def _homogeneous_equal(p, q):
    return p[0]*q[3] == q[0]*p[3] and p[1]*q[3] == q[1]*p[3] and p[2]*q[3] == q[2]*p[3]


def _precise_partition(triangles,face,selected,upper):
    """Reclip original represented triangles when repeated cuts lose winding.

    This never reverses a polygon, changes material, or relaxes a conservation
    or interface-work check. Exact homogeneous coordinates retain incidence
    during repeated cuts; 80-digit normalization is needed only on return.
    Both paths use the physical original-edge halfspaces without a tolerance.

    Points are integer vectors [X, Y, Z, W] with W > 0 holding the exact
    rational point the Fraction formulation computed. Every decision is the
    sign, or zero, of a determinant or of a distance to a plane through the
    origin; both are unchanged by positive scaling, so no division is needed.
    The crossing of edge A->B, with plane distances Da and Db, is B*Da-A*Db:
    the homogeneous form of the Fraction value (b*Da-a*Db)/(Da-Db).
    Decimal division is correctly rounded, so the returned 80-digit unit
    vectors depend only on those rational values.
    """
    from decimal import Decimal,localcontext

    def unit(point):
        w=Decimal(point[3])
        a=[Decimal(point[0])/w,Decimal(point[1])/w,Decimal(point[2])/w]
        length=sum(x*x for x in a).sqrt()
        return [x/length for x in a]

    def rational(point):
        return tuple(Fraction(value,point[3]) for value in point[:3])

    with localcontext() as context:
        context.prec=80
        source={int(index):[_homogeneous(point) for point in triangles[index]]
                for index in np.r_[face,upper[selected]]}
        # Values via the exact rationals, as before: a stored -0.0 returns as 0.0.
        original_points={key:np.asarray(key,float) for key in
                         (rational(point) for triangle in source.values() for point in triangle)}
        for a,b,c in source.values():
            if _homogeneous_det(a,b,c)<=0:
                raise ValueError('Burial partition requires positively wound source triangles.')
            # 1+a.b+b.c+c.a, scaled by the positive WaWbWc.
            if (a[3]*b[3]*c[3]+(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])*c[3]
                    +(b[0]*c[0]+b[1]*c[1]+b[2]*c[2])*a[3]+(c[0]*a[0]+c[1]*a[1]+c[2]*a[2])*b[3])<=0:
                raise ValueError('Burial partition requires minor convex source triangles.')

        def positive(polygon):
            if len(polygon)<3:return False
            signs=[_homogeneous_det(polygon[0],b,c) for b,c in zip(polygon[1:-1],polygon[2:])]
            if any(value<0 for value in signs):
                raise ValueError('Exact burial clipping did not preserve convex winding.')
            return any(value>0 for value in signs)

        def clip(polygon,plane):
            if len(polygon)<3:return []
            x,y,z=plane
            distance=[point[0]*x+point[1]*y+point[2]*z for point in polygon]
            inside=[value>=0 for value in distance]
            output=[]
            for index,point in enumerate(polygon):
                previous=(index-1)%len(polygon)
                if inside[index]!=inside[previous]:
                    before,after=distance[previous],distance[index]
                    if before==0:crossing=polygon[previous]
                    elif after==0:crossing=point
                    else:
                        crossing=[p*before-q*after for p,q in zip(point,polygon[previous])]
                        if crossing[3]<0:crossing=[-value for value in crossing]
                        common=gcd(*crossing)
                        if common>1:crossing=[value//common for value in crossing]
                    output.append(crossing)
                if inside[index]:output.append(point)
            output=[point for index,point in enumerate(output) if not _homogeneous_equal(point,output[index-1])]
            # Exact redundant edge vertices have no area. Remove them before
            # binary64 conversion can round a collinear point to the wrong
            # side and invalidate an otherwise large convex region. There is
            # no distance threshold: every nonzero thin region survives.
            while len(output)>=3:
                redundant=next((index for index in range(len(output))
                                if _homogeneous_det(output[index-1],output[index],output[(index+1)%len(output)])==0),None)
                if redundant is None:break
                del output[redundant]
            return output

        def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

        regions=[(source[int(face)],())]
        for pair in selected:
            triangle=source[int(upper[pair])]
            planes=[cross(a,b) for a,b in zip(triangle,triangle[1:]+triangle[:1])]
            next_regions=[]
            for polygon,cover in regions:
                inside=polygon
                for plane in planes:
                    outside=clip(inside,tuple(-v for v in plane))
                    if positive(outside):next_regions.append((outside,cover))
                    inside=clip(inside,plane)
                    if len(inside)<3:break
                if positive(inside):next_regions.append((inside,cover+(int(pair),)))
            regions=next_regions
        result = []
        for polygon, cover in regions:
            # Original binary64 unit vectors define the source geometry and
            # cached area. Do not renormalize them a second time on return.
            keys = [rational(point) for point in polygon]
            represented = np.asarray([original_points[key] if key in original_points else unit(point)
                                      for key, point in zip(keys, polygon)], float)
            if _positive_binary64_winding(represented):
                result.append((represented, cover))
            else:
                # A tiny positive fan can become inverted at binary64 return
                # while the rest of this exact polygon represents a large
                # valid region. Keep its individually representable fan
                # triangles with the same stack, rather than discard the
                # entire region. Only exact-positive fans are eligible and
                # each still passes the unchanged represented-winding guard.
                # The unchanged whole-footprint gate bounds any lost material
                # below the coordinate representation; no area is repaired.
                for index in range(1,len(polygon)-1):
                    if _homogeneous_det(polygon[0],polygon[index],polygon[index+1])<=0:
                        continue
                    triangle=represented[[0,index,index+1]]
                    if _positive_binary64_winding(triangle):
                        result.append((triangle,cover))
        return result


def _split(polygon, planes, radius, clipper=None):
    """Disjoint intersection/complement of a convex spherical clipping triangle.

    A complement piece violates the first clipping plane it fails. The
    remainder satisfies all earlier planes, so later pieces cannot overlap it.
    Every retained polygon is convex and can use the same great-circle kernel.
    """
    inside = polygon
    outside = []
    # Most regions in a refined cover are wholly outside the next triangle.
    # Avoid allocating clipped polygons when a separating plane proves that.
    distances=(plane_distances(polygon[:,None,:],np.asarray(planes)[None,:,:])
               if clipper is None else edge_distances(
                   polygon[:,None,:],clipper[None,:,:],np.roll(clipper,-1,axis=0)[None,:,:]))
    if np.any(np.max(distances, axis=0) < 0.):
        return np.empty((0, 3)), [polygon]
    if np.all(distances >= 0.):
        return polygon, []
    for index,plane in enumerate(planes):
        d=(plane_distances(inside,plane) if clipper is None else
           edge_distances(inside,clipper[index],clipper[(index+1)%len(clipper)]))
        positive=[];negative=[]
        for slot,point in enumerate(inside):
            previous=(slot-1)%len(inside)
            if (d[slot]>0. and d[previous]<0.) or (d[slot]<0. and d[previous]>0.):
                fraction=abs(d[previous])/(abs(d[previous])+abs(d[slot]))
                crossing=geometry._unit((1-fraction)*inside[previous]+fraction*point)
                positive.append(crossing);negative.append(crossing)
            if d[slot]>=0.:positive.append(point)
            if d[slot]<=0.:negative.append(point)
        # Both halves use one crossing and one sign decision. Shared boundaries
        # have zero area; they must not become separately rounded sliver strips.
        def clean(points):
            points=np.asarray(points,float).reshape(-1,3)
            return points[np.any(points!=np.roll(points,1,axis=0),axis=1)] if len(points)>1 else points
        inside,piece=clean(positive),clean(negative)
        if geometry._polygon_area(piece, radius) > 0.:
            outside.append(piece)
        if len(inside) < 3:
            break
    return inside, outside


def _partition_face_uncached(triangles, face, selected, upper, reference_area, radius):
    """Disjoint local stack geometry shared by depth and interface mechanics."""
    if len(np.unique(upper[selected])) != len(selected):
        raise ValueError('Duplicate covering pair in burial geometry.')
    regions = [(triangles[face], ())]
    for pair in selected:
        triangle = triangles[upper[pair]]
        planes = geometry._triangle_planes(triangle)
        next_regions = []
        for polygon, covering in regions:
            inside, outside = _split(polygon, planes, radius,triangle)
            next_regions.extend((piece, covering) for piece in outside)
            if geometry._polygon_area(inside, radius) > 0.:
                next_regions.append((inside, covering+(int(pair),)))
        regions = next_regions
    if any(_uncertain_winding(polygon) for polygon,_ in regions):
        regions=_precise_partition(triangles,face,selected,upper)
    areas = np.array([geometry._polygon_area(polygon, radius) for polygon, _ in regions])
    error = abs(float(areas.sum())-reference_area)/reference_area
    if error > 2e-10:
        raise ValueError('Local burial regions do not conserve their lower footprint.')
    return regions, areas, error


def partition_face(triangles, face, selected, upper, reference_area, radius):
    """Exact disjoint geometry with detached, bounded process-local reuse."""
    return partition_geometry_session(triangles, upper).partition_face(face, selected, reference_area, radius)


def _partition_batch(job):
    """Worker: [(index, (result, seconds) or None)]; None defers that face to the caller."""
    shared, context, modes, items = job
    rows = []
    with read_inputs(shared) as data:
        with localcontext(context):
            for index, face, selected, reference_area, radius in items:
                start = perf_counter()
                try:
                    with np.errstate(**modes):
                        result = _partition_face_uncached(data['triangles'], face, selected, data['upper'],
                                                          reference_area, radius)
                except Exception:
                    rows.append((index, None))
                else:
                    # Pieces can be views of the shared triangles; copy before unmapping.
                    rows.append((index, (_geometry_clone(result), perf_counter()-start)))
    return rows


def integrate(s, thickness, pairs, *, depth_km, heating_delay_myr):
    """Return face-mean eligible thickness plus scope/contact attribution.

    Heating remains the existing face-level residence closure. Contact ledger
    allocation weights each region's eligible volume by its contributing upper
    thicknesses. This allocation is bookkeeping, not a traction or heat law.
    """
    upper, lower, pair_area, contact = pairs
    surface = s.material_surface
    triangles = np.asarray(surface['vertices'])[np.asarray(surface['faces'])]
    area = np.asarray(surface['area_km2'], float)
    thickness = np.asarray(thickness, float)
    n = len(area)
    if (thickness.shape != (n,) or not np.isfinite(thickness).all()
            or np.any(thickness <= 0) or not np.isfinite(area).all() or np.any(area <= 0)):
        raise ValueError('Burial integration needs positive finite aligned columns and areas.')
    radius = float(surface.get('radius_km', 6371.))
    heated = np.asarray(s.parcel_burial_myr, float) >= heating_delay_myr
    root_heated = np.asarray(getattr(s, 'parcel_root_age_myr', np.zeros(n)), float) >= heating_delay_myr
    covered_volume = np.zeros(n)
    own_volume = area*np.clip(thickness-depth_km, 0., thickness)
    union_area = np.zeros(n)
    eligible_pair_volume = np.zeros(len(upper))
    order = np.lexsort((upper, lower))
    selected_lower, starts = np.unique(lower[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    regions_count = 0
    maximum_regions = 0
    maximum_area_error = 0.
    sheets = np.asarray(s.parcel_collision_sheet)
    geometry_session = partition_geometry_session(triangles, upper)
    spans = list(zip(selected_lower, starts, ends))
    partitions = geometry_session.partition_faces(
        (face, order[start:end], area[face], radius) for face, start, end in spans)
    for (face, start, end), (regions, areas, error) in zip(spans, partitions):
        selected = order[start:end]
        maximum_area_error = max(maximum_area_error, error)
        own_volume[face] = 0.
        represented = np.zeros(len(selected))
        pair_position = {int(pair): i for i, pair in enumerate(selected)}
        for region_area, (_, covering) in zip(areas, regions):
            if not covering:
                own_volume[face] += region_area*np.clip(thickness[face]-depth_km, 0., thickness[face])
                continue
            indices = np.asarray(covering, int)
            # Same-sheet triangulation edges carry no stack depth. A genuine
            # positive-area self overlap is ambiguous rather than another
            # physical layer. Reject the complete transaction so adaptive
            # stepping can recompute the geometry over a finer interval.
            if (len(np.unique(sheets[upper[indices]])) != len(indices)
                    and region_area > area[face]*2e-10):
                raise IncompleteContactStepError(
                    'Overlapping faces of one sheet do not define a burial stack; '
                    'the complete coupled timestep must be retried.')
            load = thickness[upper[indices]]
            overburden = float(load.sum())
            eligible = region_area*np.clip(overburden+thickness[face]-depth_km, 0., thickness[face])
            union_area[face] += region_area
            covered_volume[face] += eligible
            eligible_pair_volume[indices] += eligible*heated[face]*load/overburden
            for pair in indices:
                represented[pair_position[int(pair)]] += region_area
        if not np.allclose(represented, pair_area[selected], rtol=2e-9, atol=area[face]*2e-11):
            raise ValueError('Burial polygons disagree with the cached overlap geometry.')
        regions_count += len(regions)
        maximum_regions = max(maximum_regions, len(regions))
    covered = covered_volume*heated
    own = own_volume*root_heated
    eligible_volume = covered+own
    eligible = np.minimum(eligible_volume/area, thickness)
    share = np.divide(covered, eligible_volume, out=np.zeros(n), where=eligible_volume > 0)
    covered_area = np.bincount(lower, weights=pair_area, minlength=n)
    scope = dict(covered_km3=float(covered_volume.sum()), own_root_km3=float(own_volume.sum()),
                 root_inventory_km3=float((covered_volume+own_volume).sum()),
                 eligible_km3=float(area@eligible),
                 heated_covered_fraction=float(covered.sum()/max(float(covered_volume.sum()), 1e-300)),
                 heated_root_fraction=float(own.sum()/max(float(own_volume.sum()), 1e-300)),
                 overlapping_face_pairs=int(len(upper)),
                 faces_with_cover_beyond_own_area=int(np.count_nonzero(covered_area > area)),
                 depth_integration_version=VERSION, local_stack_regions=regions_count,
                 maximum_regions_per_face=maximum_regions,
                 maximum_region_area_relative_error=maximum_area_error,
                 covered_union_area_km2=float(union_area.sum()))
    attribution = dict(lower=lower, weight=pair_area, contact=contact,
                       covered_area=covered_area, covered_share=share,
                       covered_union_area_km2=union_area.copy(),
                       eligible_pair_volume_km3=eligible_pair_volume)
    return eligible, scope, attribution
