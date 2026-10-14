"""Conserved juvenile columns, water-loaded freeboard and emplacement capacity.

Version one adds retained juvenile material above the stored ocean datum. Its
freeboard follows the existing density contrast; it has no prescribed summit.
The 20-degree constructive-slope capacity is an explicit, uncalibrated model
choice, informed by 10--20 degree seamount flanks in Chaytor et al. (2007),
doi:10.1029/2007GC001712. It is not a universal volcanic stability threshold.
Source that cannot fit is pending magma, never a rescaled displayed mountain.
"""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
import crustal_structure as columns
import material_surface
import material_reconstruction
import arc_surface

VERSION = 1
SOURCE_COLUMN_KM = 25.
MECHANICAL_REFERENCE_KM = 8.
MAX_CONSTRUCTIVE_SLOPE_DEG = 20.
MAX_GRADE = float(np.tan(np.deg2rad(MAX_CONSTRUCTIVE_SLOPE_DEG)))


def version(s):
    value=getattr(s,'native_arc_birth_profile_version',0)
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value not in (0,VERSION):
        raise ValueError('Unsupported juvenile birth profile version.')
    if value and getattr(s,'native_arc_emplacement_version',0)!=1:
        raise ValueError('Column-derived juvenile profiles require finite footprint admission.')
    return int(value)


def freeboard(thickness_km, basement_m):
    """Water-loaded buoyancy of actual added columns over a declared ocean datum.

    The ocean/mantle background is represented by the sampled datum. No ocean
    column is invented or removed by this function. These are reference birth
    heights; subsequent thickness changes retain normal structure semantics.
    """
    thickness,basement=np.broadcast_arrays(np.asarray(thickness_km,float),np.asarray(basement_m,float))
    if not np.isfinite(thickness).all() or not np.isfinite(basement).all() or np.any(thickness<0):
        raise ValueError('Juvenile freeboard needs finite nonnegative columns and datum.')
    air=np.where(basement<0,basement/columns.WATER_FACTOR,basement)+columns.AIRY_M_PER_KM*thickness
    return np.where(air<0,air*columns.WATER_FACTOR,air)


def face_gradient_bounds(vertices, faces, nodal_height_m):
    """Exact maximum radial-P1 gradient on every minor spherical triangle.

    h(q)=(a.q)/(b.q), with b.v_i=1 at each unit vertex. Inside a minor
    triangle b.q>=1. The numerator a-h*b is a convex combination of its
    vertex values; its norm is therefore bounded by their maximum. At each
    vertex it is tangent and b.q=1, so the bound is attained there. Subtract
    a face-local datum before solving to avoid cancellation from sea depth.
    """
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    heights=np.asarray(nodal_height_m,float)[faces]
    triangles=vertices[faces]
    relative=heights-heights[:,:1]
    a=np.linalg.solve(triangles,relative[:,:,None])[...,0]
    b=np.linalg.solve(triangles,np.ones_like(relative)[:,:,None])[...,0]
    gradient=a[:,None,:]-relative[:,:,None]*b[:,None,:]
    return np.linalg.norm(gradient,axis=2).max(axis=1)/(6371.*1000.)


def measure(vertices, faces, area_km2, height_m, basement_m):
    """Measure constructive relief on the actual shared-node arc reconstruction.

    True free-boundary nodes meet the ocean. This measures the added profile
    relative to its local stored datum, not the gradient of changing regional
    bathymetry, and does not alter either heights or physical face columns.
    """
    faces=np.asarray(faces,int)
    stencil=material_reconstruction.prepare(vertices,faces,np.zeros(len(faces),int),area_km2=area_km2)
    prepared=arc_surface.prepare(stencil,np.ones(len(faces),np.int64))
    nodal=material_reconstruction.vertex_values(stencil,np.asarray(height_m)-np.asarray(basement_m))
    nodal[prepared['boundary_nodes']]=0.
    nodes=np.asarray(vertices)[stencil['vertex_index']]
    gradients=face_gradient_bounds(nodes,stencil['face_vertices'],nodal)
    return dict(face_grade=gradients,nodal_uplift_m=nodal,
                maximum_grade=float(gradients.max(initial=0.)),
                maximum_slope_deg=float(np.rad2deg(np.arctan(gradients.max(initial=0.)))))


def birth(patch, source_area_km2, basement_m, *, area_km2=None):
    """Construct a smooth radial column profile with an exact 25-km source mean.

    Geometry is normalized exactly as append_surface will normalize it. The
    retained 8-km minimum remains explicit; all volume above it uses a smooth
    radial taper. Capacity can reject this profile but never renormalizes its
    freeboard independently of its conserved physical column.
    """
    vertices=np.asarray(patch['vertices'],float)
    vertices=vertices/np.linalg.norm(vertices,axis=1)[:,None]
    faces=np.asarray(patch['faces'],int)
    area=(material_surface.spherical_face_areas(vertices,faces) if area_km2 is None
          else np.asarray(area_km2,float))
    radius=np.asarray(patch['profile_radius'],float)
    shape=np.square(np.clip(1.-radius*radius,0.,1.))
    volume=float(source_area_km2)*SOURCE_COLUMN_KM
    amplitude=(volume-MECHANICAL_REFERENCE_KM*float(area.sum()))/float(area@shape)
    thickness=MECHANICAL_REFERENCE_KM+amplitude*shape
    height=freeboard(thickness,basement_m)
    measured=measure(vertices,faces,area,height,basement_m)
    bounded=bool(np.all(thickness>=columns.MIN_THICKNESS_KM) and np.all(thickness<=columns.MAX_THICKNESS_KM))
    admissible=bounded and measured['maximum_grade']<=MAX_GRADE*(1.+1e-10)
    diagnostic=dict(version=VERSION,admissible=admissible,
        reason='accepted' if admissible else 'constructive_slope_capacity' if bounded else 'physical_column_capacity',
        maximum_grade=measured['maximum_grade'],maximum_slope_deg=measured['maximum_slope_deg'],
        slope_limit_deg=MAX_CONSTRUCTIVE_SLOPE_DEG,maximum_allowed_grade=MAX_GRADE,
        source_area_km2=float(source_area_km2),source_volume_km3=volume,
        actual_area_km2=float(area.sum()),actual_volume_km3=float(area@thickness),
        minimum_column_km=float(thickness.min()),maximum_column_km=float(thickness.max()),
        minimum_reference_height_m=float(height.min()),maximum_reference_height_m=float(height.max()),
        basement_m=float(basement_m),mechanical_reference_column_km=MECHANICAL_REFERENCE_KM,
        reference_self_GPE_created_km4=float(.5*np.sum(area*np.square(thickness-MECHANICAL_REFERENCE_KM))))
    return dict(thickness_km=thickness,height_m=height,area_km2=area,diagnostics=diagnostic)


def growth(s, selected, plan):
    """Prospective mixed columns may not worsen any already steeper face.

    Tectonic relief already present is retained. This is a source-admission
    check, not erosion or a post-step smoothing operation.
    """
    surface=s.material_surface
    fields={key:np.asarray(value)[selected].copy() for key,value in s.structure.items()}
    old_height=columns.elevation(fields)
    old_area=np.asarray(surface['area_km2'])[selected]
    fields['thickness_km']=(old_area*fields['thickness_km']+plan['added_area_km2']*SOURCE_COLUMN_KM)/plan['area_km2']
    import crust_inventory
    if crust_inventory.present(fields):
        old_reference=np.asarray(s.mass)[selected]
        new_reference=old_reference+plan['added_area_km2']
        fields['area_factor']=plan['area_km2']/new_reference
        crust_inventory.grow_reference(fields,np.arange(len(selected)),old_reference,new_reference,
                                        plan['added_area_km2']*SOURCE_COLUMN_KM)
    try:columns._state(fields)
    except ValueError:return dict(version=VERSION,admissible=False,reason='physical_column_capacity')
    ids=plan['vertex_ids'];faces=np.searchsorted(ids,surface['faces'][selected])
    basal=s.parcel_arc_basal_m[selected]
    before=measure(surface['vertices'][ids],faces,old_area,old_height,basal)
    after=measure(plan['vertices'],faces,plan['area_km2'],columns.elevation(fields),basal)
    limits=np.maximum(MAX_GRADE,before['face_grade'])
    excess=float(np.max(after['face_grade']-limits,initial=0.))
    accepted=excess<=max(1e-10,float(limits.max(initial=0.))*1e-10)
    return dict(version=VERSION,admissible=accepted,
        reason='accepted' if accepted else 'constructive_slope_capacity',
        maximum_grade=after['maximum_grade'],maximum_slope_deg=after['maximum_slope_deg'],
        slope_limit_deg=MAX_CONSTRUCTIVE_SLOPE_DEG,maximum_allowed_grade=float(limits.max(initial=0.)),
        maximum_per_face_grade_excess=excess,previous_maximum_grade=before['maximum_grade'])


def snapshot_fields(s):
    if not version(s):return {}
    from copy import deepcopy
    import arc_source_cohorts
    arc_source_cohorts.normalize(s,None,0)
    return dict(arc_birth_profile_version=VERSION,arc_source_cohort_version=1,
        arc_source_placements=deepcopy(getattr(s,'native_arc_source_placements',[])),
        arc_source_cohort_state=deepcopy(s.arc_source_cohort_state),
        arc_birth_profile=dict(version=VERSION,
        source_column_km=SOURCE_COLUMN_KM,mechanical_reference_column_km=MECHANICAL_REFERENCE_KM,
        maximum_constructive_slope_deg=MAX_CONSTRUCTIVE_SLOPE_DEG,
        policy='column-derived Airy/water-load birth freeboard; finite constructive slope capacity; unplaced source remains pending',
        calibration='Uncalibrated model parameter; not a universal volcanic stability angle'))


def validate_frame(frame):
    import arc_point_nucleation
    arc_point_nucleation.validate_frame(frame)
    import arc_birth_footprint
    arc_birth_footprint.validate_frame(frame)
    from arc_emplacement_geometry import validate_deposition_policy
    deposition_version=validate_deposition_policy(frame)
    modes=('birth','growth','deposition') if deposition_version else ('birth','growth')
    tag=frame.get('arc_birth_profile_version',0)
    if isinstance(tag,(bool,np.bool_)) or not isinstance(tag,Integral) or tag not in (0,VERSION):
        raise ValueError('Unsupported saved juvenile birth profile version.')
    if not tag:
        if (any(key in frame for key in ('arc_birth_profile','arc_source_cohort_version','arc_source_placements','arc_source_cohort_state'))
                or 'birth_profile_capacity' in frame.get('arc_material_diagnostics',{})):
            raise ValueError('Juvenile profile diagnostics require their version.')
        return
    metadata=frame.get('arc_birth_profile')
    expected=dict(version=VERSION,source_column_km=25.,mechanical_reference_column_km=8.,maximum_constructive_slope_deg=20.)
    if frame.get('arc_emplacement_version')!=1 or not isinstance(metadata,dict):
        raise ValueError('Column-derived profiles require their metadata and finite-footprint policy.')
    for key,value in expected.items():
        if isinstance(metadata.get(key),bool) or metadata.get(key)!=value:
            raise ValueError('Invalid juvenile birth profile parameter: '+key)
    if not isinstance(metadata['version'],Integral):
        raise ValueError('Juvenile profile version must be an integer.')
    import arc_source_cohorts
    if type(frame.get('arc_source_cohort_version')) is not int or frame['arc_source_cohort_version']!=1:
        raise ValueError('Column-derived juvenile births require their source cohort version.')
    state=frame.get('arc_source_cohort_state')
    arc_source_cohorts.validate_state(state,frame.get('time_myr'))
    inventory=frame['arc_pending_source']
    pending=inventory.get('source_provenance',[])
    arc_source_cohorts.validate(pending,len(inventory['area_km2']))
    ledger=frame.get('arc_source_placements')
    if not isinstance(ledger,list):raise ValueError('Juvenile placements require their enduring origin ledger.')
    spent_by_origin={};original_by_origin={}
    def nonnegative(value):
        return not isinstance(value,(bool,np.bool_)) and isinstance(value,Real) and np.isfinite(value) and value>=0.
    def balance(actual,expected):return abs(actual-expected)<=max(1e-7,abs(expected)*1e-11)
    for index,entry in enumerate(ledger):
        if (not isinstance(entry,dict) or type(entry.get('placement_id')) is not int or entry['placement_id']!=index+1
                or type(entry.get('arc_id')) is not int or entry['arc_id']<1
                or entry.get('mode') not in modes):
            raise ValueError('Invalid enduring juvenile placement identity.')
        if entry['mode']=='deposition' and (type(entry.get('receiving_face_id')) is not int
                                             or entry['receiving_face_id']<1):
            raise ValueError('Enduring juvenile deposition requires its actual receiving face identity.')
        area=entry.get('area_km2');volume=entry.get('volume_km3');time=entry.get('placement_time_myr')
        point=np.asarray(entry.get('geometry_xyz'),float)
        if (not nonnegative(area) or area==0 or not nonnegative(volume) or not balance(volume,25.*area)
                or not nonnegative(time) or time>float(frame['time_myr'])+1e-10
                or point.shape!=(1,3) or not np.isfinite(point).all()
                or abs(np.linalg.norm(point[0])-1.)>2e-10):
            raise ValueError('Invalid juvenile placement position, time or source volume.')
        provenance=entry.get('source_provenance');arc_source_cohorts.validate([provenance],1)
        origin=provenance['origin_id'];original=provenance['original_area_km2']
        if origin in original_by_origin and original_by_origin[origin]!=original:
            raise ValueError('An enduring magma origin changed its originally supplied volume.')
        if provenance['source_time_myr']>time+1e-10:
            raise ValueError('Magma cannot be emplaced before its recorded source time.')
        original_by_origin[origin]=original
        spent_by_origin[origin]=spent_by_origin.get(origin,0.)+area
    for provenance,amount in zip(pending,inventory['area_km2']):
        origin=provenance['origin_id'];original=provenance['original_area_km2']
        if origin in original_by_origin and original_by_origin[origin]!=original:
            raise ValueError('Pending and placed magma disagree about their common origin.')
        original_by_origin[origin]=original
        spent_by_origin[origin]=spent_by_origin.get(origin,0.)+float(amount)
    if any(not balance(amount,original_by_origin[origin]) for origin,amount in spent_by_origin.items()):
        raise ValueError('Each entered magma origin must remain fully placed or pending.')
    stats=frame.get('stats',{})
    if not isinstance(stats,dict):
        raise ValueError('Juvenile cumulative statistics must be a mapping.')
    if 'arc_added_km2' in stats and not balance(sum(row['area_km2'] for row in ledger),stats['arc_added_km2']):
        raise ValueError('Enduring juvenile source placements disagree with cumulative emitted area.')
    deposited=sum(row['area_km2'] for row in ledger if row['mode']=='deposition')
    cumulative=('arc_deposited_source_area_km2','arc_deposited_volume_km3')
    # Old epochs and future-only activations have no deposited history. Their
    # missing counters mean zero; a real deposition history must be explicit.
    if 'stats' in frame and (deposited>0. or any(key in stats for key in cumulative)):
        values=[stats.get(key) for key in cumulative]
        if (not all(nonnegative(value) for value in values)
                or not balance(values[0],deposited) or not balance(values[1],25.*deposited)):
            raise ValueError('Cumulative juvenile deposition disagrees with enduring source placements.')
    report=frame.get('arc_material_diagnostics',{}).get('birth_profile_capacity')
    if report is None and frame.get('time_myr')==0. and not frame.get('arc_material_diagnostics'):return
    if not isinstance(report,dict) or type(report.get('version')) is not int or report['version']!=VERSION:
        raise ValueError('Marked juvenile profiles require their capacity ledger.')
    for key in ('requested_birth_area_km2','placed_birth_area_km2','profile_pending_area_km2',
                'maximum_accepted_birth_grade','reference_self_GPE_created_km4'):
        value=report.get(key)
        if isinstance(value,bool) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
            raise ValueError('Invalid juvenile profile capacity scalar: '+key)
    if report['maximum_accepted_birth_grade']>MAX_GRADE*(1.+1e-10):
        raise ValueError('Juvenile birth exceeded its constructive slope capacity.')
    for row in frame['arc_material_diagnostics']['emplacement_geometry']['sources']:
        if row.get('mode') not in modes:
            raise ValueError('Juvenile capacity source mode requires its explicit policy.')
        profile=row.get('profile_capacity')
        if (not isinstance(profile,dict) or profile.get('version')!=VERSION
                or isinstance(profile.get('version'),bool) or not isinstance(profile.get('version'),Integral)):
            raise ValueError('Each marked juvenile source requires its capacity decision.')
        count=len(row['source_indices'])
        origin_ids=row.get('source_origin_ids')
        if (not isinstance(origin_ids,list) or len(origin_ids)!=count or len(set(origin_ids))!=count
                or any(type(origin) is not int or origin<1 for origin in origin_ids)):
            raise ValueError('Juvenile capacity decisions require distinct source origin IDs.')
        parts=[]
        for key in ('source_available_area_km2','source_placed_area_km2','source_pending_area_km2'):
            values=np.asarray(row.get(key),float)
            if values.shape!=(count,) or not np.isfinite(values).all() or np.any(values<0):
                raise ValueError('Invalid per-origin juvenile amount partition.')
            parts.append(values)
        available,placed,pending_amount=parts
        if (not np.allclose(available,placed+pending_amount,rtol=1e-11,atol=1e-7)
                or not balance(float(placed.sum()),row['accepted_area_km2'])
                or not balance(float(pending_amount.sum()),row['pending_area_km2'])
                or not balance(float(available.sum()),row['requested_area_km2'])
                or row.get('consumed_origins_contained') is not True):
            raise ValueError('Juvenile capacity decisions do not close each actual source origin.')
        if row['accepted_area_km2']>0:
            grade=profile.get('maximum_grade');limit=profile.get('maximum_allowed_grade')
            if (profile.get('admissible') is not True or isinstance(grade,bool) or isinstance(limit,bool)
                    or not isinstance(grade,Real) or not isinstance(limit,Real)
                    or not np.isfinite([grade,limit]).all() or grade<0 or limit<MAX_GRADE
                    or grade>limit+max(1e-10,limit*1e-10)):
                raise ValueError('Placed juvenile source lacks a valid capacity decision.')
            if row['mode']=='birth' and limit!=MAX_GRADE:
                raise ValueError('A juvenile birth cannot inherit tectonic oversteepening.')
            if row['mode']=='deposition':
                previous=profile.get('previous_maximum_grade')
                if (isinstance(previous,(bool,np.bool_)) or not isinstance(previous,Real)
                        or not np.isfinite(previous) or previous<0. or limit!=max(MAX_GRADE,previous)):
                    raise ValueError('Juvenile deposition cannot relax its original constructive slope limit.')
            if row['mode'] in ('growth','deposition'):
                excess=profile.get('maximum_per_face_grade_excess')
                if (isinstance(excess,bool) or not isinstance(excess,Real) or not np.isfinite(excess)
                        or excess<0 or excess>max(1e-10,limit*1e-10)):
                    raise ValueError('Juvenile growth exceeds its per-face constructive capacity.')
