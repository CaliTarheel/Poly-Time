"""Opt-in induced initiation along finite, weakening ocean-bearing faults.

This Lite closure records actual accepted shortening and along-strike slip;
it is not a resolved brittle/thermal failure calculation. Old ocean age is an
incoming control-volume cooling-age approximation. No historical damage,
elapsed time or missing slab inventory is reconstructed on activation.
"""
from copy import copy, deepcopy
import math
import numpy as np
from ridge_geometry import rotate

VERSION=1
POLICY='induced oceanic transform/oblique fault initiation from accepted local shortening and shear'
DEFAULTS=dict(version=VERSION,enabled=False,minimum_convergence_km_myr=2.,
    minimum_consecutive_myr=10.,minimum_shortening_km=100.,minimum_shear_slip_km=100.,
    minimum_ocean_age_myr=20.,minimum_ocean_fraction=.8,minimum_length_km=300.)


def normalize(raw=None):
    if raw is None:return deepcopy(DEFAULTS)
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS):raise ValueError('Unknown Lite initiation controls.')
    value=dict(DEFAULTS,**raw)
    if type(value['version']) is not int or value['version']!=VERSION or type(value['enabled']) is not bool:
        raise ValueError('Lite initiation needs original version1 and boolean enabled controls.')
    for key in set(DEFAULTS)-{'version','enabled'}:
        number=value[key]
        if (isinstance(number,(bool,np.bool_)) or not isinstance(number,(int,float,np.number))
                or not np.isfinite(number) or number<=0.):raise ValueError('Lite initiation thresholds must be finite and positive.')
        value[key]=float(number)
    if value['minimum_ocean_fraction']>1.:raise ValueError('Lite incoming ocean fraction must be in (0,1].')
    return value


def enabled(s):
    raw=getattr(s,'config',{}).get('effective_subduction',{}).get('initiation')
    state=getattr(s,'effective_subduction_initiation',None)
    if raw is None or not normalize(raw)['enabled']:
        if state is not None:raise ValueError('Lite initiation state has no matching enabled policy.')
        return False
    import effective_subduction
    if not effective_subduction.enabled(s):raise ValueError('Lite initiation requires effective subduction.')
    if (not isinstance(state,dict) or state.get('version')!=VERSION or type(state.get('version')) is not int
            or state.get('policy')!=POLICY or state.get('parameters')!=normalize(raw)
            or not isinstance(state.get('candidates'),list) or not isinstance(state.get('births'),list)
            or not isinstance(state.get('events'),list) or type(state.get('next_candidate_id')) is not int
            or state['next_candidate_id']<1):raise ValueError('Invalid Lite initiation checkpoint metadata.')
    activation,epoch=state.get('activation_myr'),state.get('epoch_myr')
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v)
            for v in (activation,epoch)) or not 0.<=activation<=epoch<=float(s.t)):
        raise ValueError('Invalid Lite initiation activation or accepted clock.')
    ids=[]
    for row in state['candidates']:
        if (not isinstance(row,dict) or type(row.get('id')) is not int or row['id']<1
                or row.get('phase') not in ('weakening','pending','born','retired')
                or any(type(row.get(k)) is not int or row[k]<=0 for k in ('incoming_plate_uid','overriding_plate_uid'))
                or row.get('incoming_plate_uid')==row.get('overriding_plate_uid')):
            raise ValueError('Invalid Lite initiation candidate identity.')
        ids.append(row['id'])
        for key in ('created_myr','last_seen_myr','consecutive_myr','shortening_km','shear_slip_km','length_km'):
            value=row.get(key)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<0.:
                raise ValueError('Invalid Lite initiation candidate history.')
        for key,total in (('shortening_lower_bound_km','shortening_km'),('shear_lower_bound_km','shear_slip_km'),
                          ('consecutive_lower_bound_myr','consecutive_myr')):
            value=row.get(key,0.)
            if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                    or not np.isfinite(value) or value<0. or value>row[total]+1e-10*max(1.,row[total])):
                raise ValueError('Invalid Lite initiation spatial history bound.')
        if type(row.get('bounded_history',False)) is not bool:
            raise ValueError('Invalid Lite initiation spatial history policy.')
        if not activation<=row['created_myr']<=row['last_seen_myr']<=epoch:raise ValueError('Lite initiation invents candidate history outside accepted time.')
        for name in ('segments_start','segments_end'):
            xyz=np.asarray(row.get(name),float)
            if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all() or np.any(abs(np.linalg.norm(xyz,axis=1)-1.)>1e-10):
                raise ValueError('Lite initiation requires finite unit candidate fault geometry.')
        if len(row['segments_start'])!=len(row['segments_end']):raise ValueError('Lite initiation candidate endpoints must align.')
        if np.any(np.linalg.norm(np.cross(row['segments_start'],row['segments_end']),axis=1)<=1e-12):
            raise ValueError('Lite initiation candidate finite arcs must be nondegenerate.')
    if ids!=sorted(set(ids)) or any(ident>=state['next_candidate_id'] for ident in ids):
        raise ValueError('Lite initiation candidate IDs must remain unique and ordered.')
    return True


def upgrade(s,parameters=None):
    """Add only configuration and zero-history policy metadata at this epoch."""
    if enabled(s):return deepcopy(s.effective_subduction_initiation)
    import effective_subduction
    effective_subduction.validate_compatibility(s)
    if not effective_subduction.enabled(s):raise ValueError('Lite initiation activation requires existing effective subduction.')
    controls=normalize(dict(enabled=True) if parameters is None else parameters)
    if not controls['enabled']:raise ValueError('Explicit Lite initiation activation must be enabled.')
    config=deepcopy(s.config);config['effective_subduction']=dict(config['effective_subduction'],initiation=controls)
    state=dict(version=VERSION,policy=POLICY,parameters=deepcopy(controls),activation_myr=float(s.t),
        epoch_myr=float(s.t),candidates=[],next_candidate_id=1,births=[],events=[],
        retrospective_time_myr=0.,retrospective_shortening_km=0.,retrospective_shear_slip_km=0.,
        physical_state_changed=False,
        provenance='Only subsequent accepted finite-fault observations contribute; no inherited ocean damage or slab history.',
        ocean_age_scope='Existing incoming control-volume cooling age; wet length uses actual native material exclusion.')
    subject=copy(s);subject.config=config;subject.effective_subduction_initiation=state
    enabled(subject)
    s.config=config;s.effective_subduction_initiation=state
    return deepcopy(state)


def initialize(s):
    """Fresh opt-in worlds use the same zero-history activation contract."""
    controls=normalize(s.config['effective_subduction'].get('initiation'))
    if not controls['enabled']:return False
    # A newly authored configuration is not active metadata until this call.
    raw=s.config['effective_subduction']['initiation']
    subject=copy(s);subject.config=deepcopy(s.config);subject.config['effective_subduction'].pop('initiation')
    upgrade(subject,raw)
    s.config=subject.config;s.effective_subduction_initiation=subject.effective_subduction_initiation
    return True


def _pieces(s):
    """Current finite fault pieces; never use a parent's averaged convergence."""
    contour=getattr(s,'native_boundary_geometry',None)
    if not isinstance(contour,dict):raise ValueError('Lite initiation requires resolved native finite fault geometry.')
    a,b,n=(np.asarray(contour.get(key),float) for key in ('segments_start','segments_end','segment_normals'))
    parent=np.asarray(contour.get('contact_index'))
    if (a.ndim!=2 or a.shape[1]!=3 or b.shape!=a.shape or n.shape!=a.shape
            or parent.shape!=(len(a),) or parent.dtype.kind not in 'iu'
            or np.any(parent<0) or np.any(parent>=len(s.ba))
            or not all(np.isfinite(v).all() for v in (a,b,n))
            or any(np.any(abs(np.linalg.norm(v,axis=1)-1.)>1e-10) for v in (a,b,n))):
        raise ValueError('Lite initiation finite native pieces are malformed or unaligned.')
    import effective_subduction
    import trench_history
    ids=trench_history._match(s,include_shutdown=False)
    reserved=np.zeros(len(ids),bool)
    for trace in s.trench_systems:
        if not trace.get('effective_subduction') or trace['phase'] in ('shutdown','joined'):continue
        reserved|=supported_parents(s,trace,ids==trace['id'])
    ages=np.asarray(s.age,float)
    if not np.isfinite(ages).all() or np.any(ages<0.):raise ValueError('Lite initiation requires finite nonnegative control ocean ages.')
    middle=(a+b);middle/=np.linalg.norm(middle,axis=1)[:,None]
    length=6371.*np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    n=n-middle*np.sum(n*middle,axis=1)[:,None]
    magnitude=np.linalg.norm(n,axis=1)
    if np.any(magnitude<=1e-12):raise ValueError('Lite fault normals need a resolved midpoint tangent.')
    n/=magnitude[:,None]
    import native_subduction
    if native_subduction.enabled(s):
        contexts={int(owner):native_subduction._material(s,int(owner)) for owner in np.unique(np.r_[s.bp,s.bq])}
        water=[]
        for first,last,edge,extent in zip(a,b,parent,length):
            fractions=[]
            for owner in (s.bp[edge],s.bq[edge]):
                wet=0.
                for low,high in native_subduction.exact.ocean_intervals(first,last,contexts[int(owner)]):
                    left=(1-low)*first+low*last;left/=np.linalg.norm(left)
                    right=(1-high)*first+high*last;right/=np.linalg.norm(right)
                    wet+=6371.*math.atan2(float(np.linalg.norm(np.cross(left,right))),float(left@right))
                fractions.append(float(np.clip(wet/max(extent,1e-30),0.,1.)))
            water.append(tuple(fractions))
    else:
        water=[(float(s.crust[s.ba[edge]]==0),float(s.crust[s.bb[edge]]==0)) for edge in parent]
    relative=np.cross(np.asarray(s.omega)[np.asarray(s.bq)[parent]]-np.asarray(s.omega)[np.asarray(s.bp)[parent]],middle)*6371.
    normal=np.sum(relative*n,axis=1)
    shear=np.linalg.norm(relative-normal[:,None]*n,axis=1)
    records=[]
    for piece,edge in enumerate(parent):
        edge=int(edge);p,q=int(s.bp[edge]),int(s.bq[edge])
        if (reserved[edge] or not s.active[p] or not s.active[q] or p==q or length[piece]<=1e-10
                or int(s.bcode[edge]) not in (2,3,4)):
            continue
        records.append(dict(index=piece,parent=edge,start=a[piece],end=b[piece],mid=middle[piece],
            length=float(length[piece]),normal_speed=float(normal[piece]),shear=float(shear[piece]),
            slots=(p,q),uids=(int(s.plate_uid[p]),int(s.plate_uid[q])),
            water=water[piece],
            normal=n[piece],relative_omega=np.asarray(s.omega[q])-np.asarray(s.omega[p]),
            age=(float(ages[s.ba[edge]]),float(ages[s.bb[edge]]))))
    return records


def _groups(records):
    parents=list(range(len(records)));seen={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,row in enumerate(records):
        pair=tuple(sorted(row['uids']))
        for xyz in (row['start'],row['end']):
            key=(pair,*tuple(np.round(xyz,12)))
            if key in seen:
                first,second=find(i),find(seen[key]);parents[max(first,second)]=min(first,second)
            else:seen[key]=i
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    return list(groups.values())


def _measure(group,controls,incoming=None):
    length=math.fsum(row['length'] for row in group)
    if length<controls['minimum_length_km']:return None
    pair=tuple(sorted(group[0]['uids']))
    options=[]
    for uid in pair:
        # Parent-scale force/intake cannot qualify dry or hot sibling support.
        if any(row['water'][row['uids'].index(uid)]<controls['minimum_ocean_fraction']
                or row['age'][row['uids'].index(uid)]<controls['minimum_ocean_age_myr'] for row in group):continue
        water=math.fsum(row['length']*row['water'][row['uids'].index(uid)] for row in group)/length
        age=math.fsum(row['length']*row['water'][row['uids'].index(uid)]*row['age'][row['uids'].index(uid)] for row in group)
        age/=max(water*length,1e-30)
        if water>=controls['minimum_ocean_fraction'] and age>=controls['minimum_ocean_age_myr']:
            options.append((age,-uid,uid,water))
    if incoming is None:
        if not options:return None
        age,_,incoming,water=max(options)
    else:
        options=[row for row in options if row[2]==incoming]
        if not options:return None
        age,_,_,water=options[0]
    over=pair[1] if pair[0]==incoming else pair[0]
    bounds=[_rate_bounds(row) for row in group]
    return dict(incoming_plate_uid=incoming,overriding_plate_uid=over,length_km=length,
        mean_ocean_age_myr=age,mean_ocean_fraction=water,
        minimum_convergence_km_myr=min(-row['normal_speed'] for row in group),
        mean_convergence_km_myr=math.fsum(row['length']*max(-row['normal_speed'],0.) for row in group)/length,
        mean_shear_km_myr=math.fsum(row['length']*row['shear'] for row in group)/length,
        minimum_shear_km_myr=min(row['shear'] for row in group),
        guaranteed_convergence_km_myr=min(row[0] for row in bounds),
        guaranteed_shear_km_myr=min(row[1] for row in bounds))


def _rate_bounds(row):
    """Spatial lower bounds, not the midpoint observations used by the law.

    Every point on an arc is at most 2 sin(angle/4) from its midpoint. Both
    Euler velocity and the projected unit normal vary by at most this chord
    times |omega| and 1 respectively. The same bound applies to the norm of
    the shear projection. Thus a later subarc cannot inherit a faster old
    midpoint's slip. Synthetic records without omega are piecewise constant.
    """
    margin=0.
    if 'relative_omega' in row:
        a,b=np.asarray(row['start']),np.asarray(row['end'])
        angle=math.atan2(float(np.linalg.norm(np.cross(a,b-a))),float(a@b))
        speed=6371.*float(np.linalg.norm(row['relative_omega']))
        margin=speed*(4.*math.sin(angle/4.)+256*np.finfo(float).eps)
    return max(0.,-row['normal_speed']-margin),max(0.,row['shear']-margin)


def _distance(group,row):
    """Distance to finite carried arcs, including their real endpoints."""
    a,b=np.asarray(row['segments_start']),np.asarray(row['segments_end'])
    normals=np.cross(a,b);normals/=np.linalg.norm(normals,axis=1)[:,None]
    arc=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    worst=0.
    for record in group:
        point=record['mid'];dot=normals@point
        foot=point-dot[:,None]*normals;foot/=np.linalg.norm(foot,axis=1)[:,None]
        left=np.arccos(np.clip(np.sum(foot*a,axis=1),-1.,1.));right=np.arccos(np.clip(np.sum(foot*b,axis=1),-1.,1.))
        on=left+right<=arc+128*np.finfo(float).eps
        distance=np.minimum(np.arccos(np.clip(a@point,-1.,1.)),np.arccos(np.clip(b@point,-1.,1.)))
        distance[on]=np.arcsin(np.minimum(abs(dot[on]),1.))
        worst=max(worst,float(distance.min())*6371.)
    return worst


def _covered(points,a,b,radius):
    """Each point projects inside real finite support, with a lateral allowance."""
    # Translated cross products avoid cancellation on short arcs. Signed
    # endpoint tangents avoid subtracting three nearly equal arc lengths:
    # that old test could reject an endpoint of its own represented arc.
    normals=np.cross(a,b-a);normals/=np.linalg.norm(normals,axis=1)[:,None]
    first=np.cross(normals,a);last=np.cross(b,normals)
    for point in points:
        dot=normals@point
        on=(first@point>=-128*np.finfo(float).eps)&(last@point>=-128*np.finfo(float).eps)
        lateral=6371.*np.arcsin(np.minimum(abs(dot),1.))
        if not np.any(on & (lateral<=radius)):return False
    return True


def _arc_intervals(first,last,a,b,radius):
    """Intervals on one current arc inside the union of carried finite arcs.

    The existing lateral matching allowance does not extend endpoints. Each
    source arc contributes its two endpoint halfspaces and a spherical strip.
    Degree-two bends have bounded joins; genuine ends have no caps. Analytic
    roots clip the entire target arc, not just sampled points.
    """
    cross=np.cross(first,last-first);angle=math.atan2(float(np.linalg.norm(cross)),float(first@last))
    tangent=np.cross(cross/np.linalg.norm(cross),first)
    tolerance=128*np.finfo(float).eps
    normal=np.cross(a,b-a);normal/=np.linalg.norm(normal,axis=1)[:,None]
    starts=np.cross(normal,a);ends=np.cross(b,normal)
    strip=math.sin(min(math.pi/2.,radius/6371.));covered=[]
    regions=[((start,0.),(end,0.),(n,-strip),(-n,-strip))
             for n,start,end in zip(normal,starts,ends)]
    regions.extend(_bend_regions(a,b,radius))
    for planes in regions:
        intervals=[(0.,angle)]
        for plane,level in planes:
            cosine,sine=float(first@plane),float(tangent@plane)
            amplitude=math.hypot(cosine,sine);roots=[]
            if amplitude>abs(level):
                phase=math.atan2(sine,cosine);offset=math.acos(level/amplitude)
                for base in (phase-offset,phase+offset):
                    roots.extend(base+2*math.pi*k for k in (-1,0,1) if 0.<base+2*math.pi*k<angle)
            result=[]
            for low,high in intervals:
                cuts=[low]+sorted(x for x in roots if low<x<high)+[high]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    middle=(left+right)/2.
                    if cosine*math.cos(middle)+sine*math.sin(middle)>=level-tolerance:
                        result.append((left,right))
            intervals=result
            if not intervals:break
        covered.extend(intervals)
    merged=[]
    for low,high in sorted(covered):
        if low<=tolerance:low=0.
        if high>=angle-tolerance:high=angle
        if high-low<=tolerance:continue
        if merged and low<=merged[-1][1]+tolerance:merged[-1]=(merged[-1][0],max(high,merged[-1][1]))
        else:merged.append((low,high))
    return angle,tangent,merged


def _bend_regions(a,b,radius):
    """Fill only the missing wedge at a genuine degree-two polyline bend.

    A full vertex disk would extend a nearby real endpoint when its adjacent
    segment is short. Each join is restricted to the outward joint wedge and
    both incident arcs' remote-end halfspaces. Reversed/duplicate copies of
    one arc never turn its endpoint into a bend.
    """
    vertices={}
    for first,last in zip(a,b):
        for point,other in ((first,last),(last,first)):
            key=tuple(np.round(point,12));neighbor=tuple(np.round(other,12))
            if key==neighbor:continue
            vertex=vertices.setdefault(key,[point,{},0.]);center,adjacent,_=vertex
            vertex[2]=max(vertex[2],float(np.linalg.norm(point-center)))
            if neighbor in adjacent:
                vertex[2]=max(vertex[2],float(np.linalg.norm(other-adjacent[neighbor])))
            adjacent[neighbor]=other
    result=[]
    for center,adjacent,merged_error in vertices.values():
        if len(adjacent)!=2:continue
        planes=[(center,math.cos(min(math.pi/2.,radius/6371.)))]
        directions=[];magnitudes=[]
        coordinate_error=128*np.finfo(float).eps+4.*merged_error
        for other in adjacent.values():
            normal=np.cross(center,other-center);magnitude=float(np.linalg.norm(normal))
            if magnitude<=coordinate_error:break
            magnitudes.append(magnitude);normal/=magnitude
            direction=np.cross(normal,center);directions.append(direction)
            planes.extend(((-direction,0.),(np.cross(other,normal),0.)))
        if len(directions)!=2:continue
        # Overlapping copies with different far endpoints are still one
        # branch; they must not turn a real endpoint into a capped joint.
        # Normalizing a short arc amplifies endpoint uncertainty by 1/sin L.
        # Reject unresolved joins conservatively, including representatives
        # merged by the existing rounded endpoint-identity key.
        uncertainty=coordinate_error*sum(1./value for value in magnitudes)
        if (uncertainty>=1. or (directions[0]@directions[1]>0.
                and np.linalg.norm(np.cross(*directions))<=uncertainty)):continue
        result.append(tuple(planes))
    return result


def _retained_core(group,row,radius):
    """Only previously observed finite support may inherit a fault's clock."""
    a,b=np.asarray(row['segments_start']),np.asarray(row['segments_end']);core=[]
    for record in group:
        first,last=np.asarray(record['start']),np.asarray(record['end'])
        angle,tangent,intervals=_arc_intervals(first,last,a,b,radius)
        for low,high in intervals:
            # Keep the checkpoint's existing nondegenerate-arc contract.
            # Unresolvable endpoint slivers receive no inherited history.
            if math.sin(high-low)<=1e-12:continue
            item=dict(record)
            item['start']=first if low==0. else first*math.cos(low)+tangent*math.sin(low)
            item['end']=last if high==angle else first*math.cos(high)+tangent*math.sin(high)
            if np.linalg.norm(np.cross(item['start'],item['end']))<=1e-12:continue
            item['mid']=item['start']+item['end'];item['mid']/=np.linalg.norm(item['mid'])
            item['length']=6371.*(high-low)
            item['complete_piece']=low==0. and high==angle
            # The discarded part could contain all of the observed water.
            # A clipped piece must not inherit its parent's wet fraction.
            # This lower bound uses the already exact finite-arc wet length;
            # it is deliberately conservative when the piece was mixed.
            excluded=max(0.,record['length']-item['length'])
            if 'water' in item:
                item['water']=tuple(float(np.clip((fraction*record['length']-excluded)/item['length'],0.,1.))
                                    for fraction in record['water'])
            if 'relative_omega' in item:
                n=np.asarray(item['normal']);n=n-item['mid']*(n@item['mid']);n/=np.linalg.norm(n)
                item['normal']=n
                relative=np.cross(item['relative_omega'],item['mid'])*6371.
                item['normal_speed']=float(relative@n)
                item['shear']=float(np.linalg.norm(relative-item['normal_speed']*n))
            core.append(item)
    return core


def _same_support(group,row,radius):
    """Bidirectional finite coverage prevents halo growth and borrowed history."""
    a,b=np.asarray(row['segments_start']),np.asarray(row['segments_end'])
    c,d=np.array([r['start'] for r in group]),np.array([r['end'] for r in group])
    return _arcs_covered(c,d,a,b,radius) and _arcs_covered(a,b,c,d,radius)


def _arcs_covered(c,d,a,b,radius):
    for first,last in zip(c,d):
        angle,_,intervals=_arc_intervals(first,last,a,b,radius)
        if intervals!=[(0.,angle)]:return False
    return True


def _complete_parents(s,group):
    parent=np.asarray(s.native_boundary_geometry['contact_index'])
    selected={row['index'] for row in group if row.get('complete_piece',True)}
    return all(set(np.flatnonzero(parent==edge))<=selected for edge in {row['parent'] for row in group})


def _birth_interior(s,group,controls,incoming):
    """No parent-scale force may include a clipped, untrained sibling piece."""
    parent=np.asarray(s.native_boundary_geometry['contact_index'])
    selected={row['index'] for row in group if row.get('complete_piece',True)}
    complete={edge for edge in {row['parent'] for row in group}
              if set(np.flatnonzero(parent==edge))<=selected}
    interiors=[part for part in _groups([row for row in group if row['parent'] in complete])
               if _measure(part,controls,incoming) is not None]
    # One candidate creates one connected trace; other support receives no
    # inherited force and can begin fresh observations after this birth.
    return max(interiors,key=lambda part:math.fsum(row['length'] for row in part),default=[])


def supported_parents(s,row,select):
    """New force sources cannot recruit untrained parents through a match halo."""
    law=row.get('effective_subduction',{})
    if not law.get('initiation_finite_support',False):return select
    if row['id']!=law.get('support_origin_trench_id'):
        raise ValueError('Initiated finite support needs an explicit owner/topology transfer before loading a descendant trace.')
    geometry=np.asarray(row['geometry_xyz'],float)
    if geometry.shape!=(law.get('support_endpoint_count'),3) or len(geometry)%2:
        raise ValueError('Initiated finite force support lost its carried endpoint representation.')
    a,b=geometry[::2],geometry[1::2]
    if np.any(np.linalg.norm(np.cross(a,b),axis=1)<=1e-12):
        raise ValueError('Initiated finite force support needs nondegenerate arcs.')
    contour=s.native_boundary_geometry;parent=np.asarray(contour['contact_index'])
    starts,ends=np.asarray(contour['segments_start']),np.asarray(contour['segments_end'])
    radius=max(120.,1.8*__import__('trench_history')._spacing(s));result=np.asarray(select,bool).copy()
    for edge in np.flatnonzero(result):
        pieces=parent==edge
        if not np.any(pieces):result[edge]=False;continue
        c,d=starts[pieces],ends[pieces]
        if not _arcs_covered(c,d,a,b,radius):result[edge]=False
    return result


def prepare(s):
    """Pure current observations; never age a candidate or activate a load."""
    if not enabled(s):return []
    controls=s.effective_subduction_initiation['parameters']
    return [dict(_measure(group,controls),parent_indices=sorted(set(row['parent'] for row in group)))
        for group in _groups(_pieces(s)) if _measure(group,controls) is not None]


def update(s,dt):
    """Commit exactly one accepted clock interval and any pending births."""
    if not enabled(s):return
    state=s.effective_subduction_initiation;epoch=float(state['epoch_myr']);now=float(s.t)
    if isinstance(dt,(bool,np.bool_)) or not np.isscalar(dt) or not np.isfinite(dt) or dt<0.:
        raise ValueError('Lite initiation needs finite nonnegative accepted elapsed time.')
    if now==epoch:
        return
    elapsed=now-epoch
    if elapsed<0. or not math.isclose(elapsed,float(dt),rel_tol=0.,abs_tol=64*np.finfo(float).eps*max(now,float(dt),1.)):
        raise ValueError('Lite initiation clock does not match the accepted motion interval.')
    import trench_history
    proposed=deepcopy(state);controls=state['parameters'];slots=trench_history._slots(s)
    live=[row for row in proposed['candidates'] if row['phase']=='weakening']
    for row in live:
        owner=slots.get(row['overriding_plate_uid'])
        if owner is None or row['incoming_plate_uid'] not in slots:
            row.update(phase='retired',reason='owner_changed');continue
        rotation=np.asarray(s.omega[owner])*elapsed
        row['segments_start']=rotate(np.asarray(row['segments_start']),rotation).tolist()
        row['segments_end']=rotate(np.asarray(row['segments_end']),rotation).tolist()
    groups=_groups(_pieces(s))
    radius=max(120.,1.8*trench_history._spacing(s))
    matches={i:[] for i in range(len(groups))};cores={}
    old_matches={row['id']:[] for row in live if row['phase']=='weakening'}
    for i,group in enumerate(groups):
        pair=set(group[0]['uids'])
        for row in live:
            if row['phase']!='weakening' or pair!={row['incoming_plate_uid'],row['overriding_plate_uid']}:continue
            core=_retained_core(group,row,radius)
            # Clipping can leave several trained intervals on the same
            # connected observed fault. They share a conservative history;
            # a real split into distinct observed groups still fails the
            # one-to-one match below, and a birth uses connected interiors.
            if core and _measure(core,controls,row['incoming_plate_uid']) is not None:
                matches[i].append(row);old_matches[row['id']].append(i);cores[i,row['id']]=core
    accepted=set();births=[]
    for i,group in enumerate(groups):
        choices=matches[i]
        reuse=len(choices)==1 and len(old_matches[choices[0]['id']])==1
        if reuse:
            row=choices[0];full=_same_support(group,row,radius);group=cores[i,row['id']]
            measurement=_measure(group,controls,row['incoming_plate_uid']);accepted.add(row['id'])
            # A removed fast end can make the old whole-fault mean larger
            # than the surviving core ever experienced. Only accumulated
            # spatial minima are transferable to a restricted footprint.
            row.setdefault('shortening_lower_bound_km',0.);row.setdefault('shear_lower_bound_km',0.)
            row.setdefault('consecutive_lower_bound_myr',0.)
            row['bounded_history']=row.get('bounded_history',False) or not full
            if row['bounded_history']:
                row['shortening_km']=row['shortening_lower_bound_km']
                row['shear_slip_km']=row['shear_lower_bound_km']
                row['consecutive_myr']=row['consecutive_lower_bound_myr']
            row['shear_lower_bound_km']+=measurement['guaranteed_shear_km_myr']*elapsed
            row['shear_slip_km']+=(measurement['guaranteed_shear_km_myr'] if row['bounded_history']
                                     else measurement['mean_shear_km_myr'])*elapsed
            convergence=(measurement['guaranteed_convergence_km_myr'] if row['bounded_history']
                         else measurement['minimum_convergence_km_myr'])
            if convergence>=controls['minimum_convergence_km_myr']:
                row['consecutive_myr']+=elapsed
                row['shortening_km']+=(measurement['guaranteed_convergence_km_myr'] if row['bounded_history']
                                      else measurement['mean_convergence_km_myr'])*elapsed
            else:row['consecutive_myr']=row['shortening_km']=0.
            if measurement['guaranteed_convergence_km_myr']>=controls['minimum_convergence_km_myr']:
                row['consecutive_lower_bound_myr']+=elapsed
                row['shortening_lower_bound_km']+=measurement['guaranteed_convergence_km_myr']*elapsed
            else:row['consecutive_lower_bound_myr']=row['shortening_lower_bound_km']=0.
        else:
            measurement=_measure(group,controls)
            if measurement is None:continue
            row=dict(id=proposed['next_candidate_id'],phase='weakening',created_myr=now,
                consecutive_myr=0.,shortening_km=0.,shear_slip_km=0.,
                shortening_lower_bound_km=0.,shear_lower_bound_km=0.,consecutive_lower_bound_myr=0.,bounded_history=False)
            proposed['next_candidate_id']+=1;proposed['candidates'].append(row)
        row.update(measurement,last_seen_myr=now,
            segments_start=[r['start'].tolist() for r in group],segments_end=[r['end'].tolist() for r in group])
        if (row['consecutive_myr']>=controls['minimum_consecutive_myr']
                and row['shortening_km']>=controls['minimum_shortening_km']
                and row['shear_slip_km']>=controls['minimum_shear_slip_km']):
            interior=_birth_interior(s,group,controls,row['incoming_plate_uid'])
            if interior:
                if len(interior)!=len(group):
                    if (row['consecutive_lower_bound_myr']<controls['minimum_consecutive_myr']
                            or row['shortening_lower_bound_km']<controls['minimum_shortening_km']
                            or row['shear_lower_bound_km']<controls['minimum_shear_slip_km']):continue
                    row.update(bounded_history=True,consecutive_myr=row['consecutive_lower_bound_myr'],
                        shortening_km=row['shortening_lower_bound_km'],shear_slip_km=row['shear_lower_bound_km'])
                row['phase']='pending';births.append((row,interior))
    for row in live:
        if row['phase']=='weakening' and row['id'] not in accepted:
            row.update(phase='retired',reason='finite_fault_unmatched_or_ambiguous_topology')
    proposed['epoch_myr']=now
    # Publish only after all measurements and clock checks. Native callers
    # own the adaptive deep-copy/restore transaction including trace births.
    s.effective_subduction_initiation=proposed
    for candidate,group in births:
        edges=np.array(sorted(set(row['parent'] for row in group)),int)
        down=np.full(len(s.ba),slots[candidate['incoming_plate_uid']],int)
        row=trench_history._birth(s,edges,down,inherited=False)
        points=np.array([r['mid'] for r in group]);weights=np.array([r['length'] for r in group])
        center=np.sum(points*weights[:,None],axis=0);center/=np.linalg.norm(center)
        endpoints=np.array([[r['start'],r['end']] for r in group]).reshape(-1,3)
        row.update(geometry_xyz=endpoints.tolist(),center=center.tolist(),length_km=float(weights.sum()),anchor_gap_km=0.)
        force=float(s.effective_subduction_settings['force_n_per_m'])
        row['effective_subduction']=dict(version=1,force_n_per_m=force,activation_pending=True,
            initiation_candidate_id=candidate['id'],eligible_myr=now,
            initiation_finite_support=True,support_origin_trench_id=row['id'],support_endpoint_count=len(endpoints),
            provenance='Accepted induced finite-fault gates; no slab inventory, retrospective intake or startup clock.')
        row.update(phase='initiating',maturity=0.)
        candidate['trench_id']=row['id']
        event=dict(time_myr=now,candidate_id=candidate['id'],trench_id=row['id'],kind='eligible_pending',
            incoming_plate_uid=candidate['incoming_plate_uid'],overriding_plate_uid=candidate['overriding_plate_uid'],
            force_n_per_m=force,consecutive_myr=candidate['consecutive_myr'],
            shortening_km=candidate['shortening_km'],shear_slip_km=candidate['shear_slip_km'],
            force_activation='before next instantaneous solve; no force or intake at eligibility snapshot')
        proposed['births'].append(deepcopy(event));proposed['events'].append(event)
        if hasattr(s,'_record'):
            s._record('effective_subduction_initiation',
                'Accepted local oceanic shortening and shear made a finite induced trench eligible; traction starts at the next solved source.',
                ('effective_subduction_initiation',VERSION,candidate['id']),
                plates=(slots[candidate['incoming_plate_uid']],slots[candidate['overriding_plate_uid']]),
                xyz=row['center'],details=dict(event,mechanism=POLICY,parameters=deepcopy(controls)))


def activate_pending(s):
    """Source boundary only: activate before the next solve under rollback."""
    if not enabled(s):return False
    import effective_subduction
    import trench_history
    slots=trench_history._slots(s);changed=False
    for row in s.trench_systems:
        law=row.get('effective_subduction',{})
        if not law.get('activation_pending',False):continue
        if row['downgoing_plate_uid'] not in slots or row['overriding_plate_uid'] not in slots:
            trench_history._shutdown(s,row,'plate_owner_lost');law['activation_pending']=False;continue
        law.update(activation_pending=False,activated_myr=float(s.t))
        row.update(phase='mature',maturity=1.)
        candidate=next(r for r in s.effective_subduction_initiation['candidates'] if r['id']==law['initiation_candidate_id'])
        candidate['phase']='born'
        s.effective_subduction_initiation['events'].append(dict(kind='source_activated',time_myr=float(s.t),
            candidate_id=candidate['id'],trench_id=row['id'],force_n_per_m=law['force_n_per_m']))
        changed=True
    if changed:effective_subduction.prepare(s)
    return changed


def snapshot(s):
    if not enabled(s):return {}
    state=s.effective_subduction_initiation
    return dict(effective_subduction_initiation_version=VERSION,effective_subduction_initiation=deepcopy(state),
        effective_subduction_initiation_diagnostics=dict(parameters=deepcopy(state['parameters']),
            activation_myr=state['activation_myr'],accepted_epoch_myr=state['epoch_myr'],
            weakening_candidates=sum(r['phase']=='weakening' for r in state['candidates']),
            pending_trenches=sum(r['phase']=='pending' for r in state['candidates']),
            activated_trenches=sum(r['phase']=='born' for r in state['candidates']),
            birth_count=len(state['births']),
            activation_scope='Pending eligible traces carry no force or intake until the next solved source interval.'))
