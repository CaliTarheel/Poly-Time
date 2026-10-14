"""Explicit local entry-depth pressure/thermal coupling for reduced columns.

The thin-sheet displacement adds ambient mantle hydrostatic load and geotherm
depth. It does not supply resolved lithosphere pressure, slab conduction,
exposure, or a shared law for active continental-stack/entry overlap. Existing phase and heat
inventories evolve locally and are then conservatively projected to each face.
"""
from copy import deepcopy
import numpy as np
import entry_regions
import entry_depth
import finite_entry_arc
import mesh_coverage
import column_density
import dense_crust as phase
import crustal_structure as columns
from exact_polygon import ExactPolygon


def enabled(s):
    state=getattr(s,'entry_phase_depth',None)
    if state is None:return False
    if not isinstance(state,dict) or type(state.get('version')) is not int or state['version']!=1:
        raise ValueError('Unsupported entry-depth phase policy.')
    for key in ('relative_tolerance','absolute_tolerance'):
        value=state.get(key)
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<=0.:
            raise ValueError('Entry-depth quadrature tolerances must be finite and positive.')
    budget=state.get('max_panels')
    if type(budget) is not int or budget<1:raise ValueError('Entry-depth quadrature requires a positive panel budget.')
    return True


def upgrade(s,*,relative_tolerance=1e-6,absolute_tolerance=1e-10,max_panels=1024):
    """Explicitly select the uniform-mantle depth closure on a detached world."""
    from copy import copy
    if enabled(s):
        prepare(s)
        return deepcopy(s.entry_phase_depth)
    staged=copy(s)
    staged.continental_entry_regions=deepcopy(getattr(s,'continental_entry_regions',None))
    if hasattr(s,'parcel_entry_region'):staged.parcel_entry_region=np.asarray(s.parcel_entry_region).copy()
    staged.entry_phase_depth=dict(version=1,relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,max_panels=max_panels,
        pressure='ambient uniform-mantle hydrostatic increment',
        temperature='existing geotherm at translated local column depth')
    enabled(staged);prepare(staged)
    s.entry_phase_depth=staged.entry_phase_depth
    return deepcopy(s.entry_phase_depth)


def prepare(s,*,coordinate_mode='along_slab_conveyor'):
    """Snapshot source depths; horizontal mode is read-only until a new policy is versioned."""
    if (not isinstance(coordinate_mode,str) or coordinate_mode not in
            ('along_slab_conveyor','horizontal_surface')):
        raise ValueError('Entry source-depth coordinate mode is unsupported.')
    if not enabled(s) or getattr(s,'retained_dense_crust_version',0)!=1 or not entry_regions.enabled(s):
        raise ValueError('Local entry-depth sources require explicit retained phases and persistent entry regions.')
    specification=entry_regions.specification(s)
    potential=entry_regions.frozen(s,coordinate_mode=coordinate_mode)
    mesh=s.material_surface
    if potential is None:return dict(corners={},triangles={},epoch_myr=float(s.t),
        face_ids=np.asarray(s.parcel_patch).copy(),specification=None,coordinate_mode=coordinate_mode)
    result=potential.evaluate(mesh['vertices'],mesh['faces'],potential.volumes,potential.sheets,radius=potential.radius)
    finite_pieces={};finite_planes={}
    lengths=specification['finite_half_lengths_km']
    for local,face in enumerate(potential.indices):
        if not np.isfinite(lengths[local]):continue
        triangle=np.asarray(mesh['vertices'])[mesh['faces'][face]]
        left,right,_,_=finite_entry_arc.endpoint_planes(specification['hinge_normals'][local],
            specification['finite_midpoints'][local],lengths[local],potential.radius)
        _,_,_,_,fraction,pieces=finite_entry_arc.integrate(
            result['corner_entry_coordinate_m'][local],triangle@left,triangle@right)
        depths=result['corner_entry_depth_m'][local]/1000.
        finite_pieces[int(face)]=dict(zero_fraction=max(0.,1.-fraction),
            triangles=[(float(area),bary@depths) for area,bary in pieces],
            barycentric_triangles=[bary for _,bary in pieces])
        finite_planes[int(face)]=np.stack((left,right))
    return dict(corners={int(i):depth.copy()/1000. for i,depth in zip(potential.indices,result['corner_entry_depth_m'])},
        triangles={int(i):np.asarray(mesh['vertices'])[mesh['faces'][i]].copy() for i in potential.indices},
        finite_pieces=finite_pieces,finite_planes=finite_planes,
        epoch_myr=float(s.t),face_ids=np.asarray(s.parcel_patch).copy(),
        specification=deepcopy(specification),coordinate_mode=coordinate_mode)


def _entered_region_triangles(depth_state,face,polygon,radius):
    """Clip the entered reference pieces to one spherical stack region.

    This source-stage partition uses the same disjoint stack polygons as
    phase_evolution. The mechanical entry potential retains its own fixed
    reference measure; no source-area approximation is fed back into forces.
    """
    if isinstance(polygon, ExactPolygon):
        raise ValueError('Entry-phase clipping does not support exact-ray polygon precision; '
                         'the stack region cannot be rounded to binary64.')
    triangle=depth_state['triangles'][face]
    finite=depth_state['finite_pieces'].get(face)
    if finite is None:
        _,_,_,_,_,pieces=finite_entry_arc.integrate(
            depth_state['corners'][face],np.ones(3),np.ones(3))
        barycentric=[bary for _,bary in pieces]
    else:
        barycentric=finite['barycentric_triangles']
    result=[]
    for bary in barycentric:
        entered=bary@triangle
        entered/=np.linalg.norm(entered,axis=1,keepdims=True)
        clipped=mesh_coverage.clip_triangle(polygon,entered)
        for second in range(1,len(clipped)-1):
            piece=np.array([clipped[0],clipped[second],clipped[second+1]])
            area=mesh_coverage._polygon_area(piece,radius)
            if area>0.:
                depths=np.array([entry_depth.point_depth(depth_state['corners'][face],triangle,point)
                                 for point in piece])
                result.append((area,depths))
    return result


def expand(s,state,dt,rows,index,weight,prepared,controls,*,trace=False):
    """Local source histories precede integration; markers use their own depth."""
    import phase_evolution
    if not enabled(s):raise ValueError('Prepared entry sources require their explicitly selected policy.')
    depth_state=prepared['entry_depth']
    if depth_state.get('coordinate_mode','along_slab_conveyor')!='along_slab_conveyor':
        raise ValueError('Horizontal entry-depth candidates cannot enter the native source policy.')
    if (depth_state['epoch_myr']!=s.t or not np.array_equal(depth_state['face_ids'],s.parcel_patch)):
        raise ValueError('Entry source geometry is stale after a time or material-identity change.')
    specification=entry_regions.specification(s)
    saved=depth_state['specification']
    if ((specification is None)!=(saved is None) or
            (saved is not None and any(not np.array_equal(saved[key],specification[key]) for key in saved))):
        raise ValueError('Entry source geometry is stale after its hinge or membership changed.')
    for face,triangle in depth_state['triangles'].items():
        if not np.array_equal(triangle,s.material_surface['vertices'][s.material_surface['faces'][face]]):
            raise ValueError('Entry source geometry is stale after material motion.')
    output=[];indices=[];weights=[];panels=0;evaluations=0;error=0.;entry_count=0
    corners=depth_state['corners']
    policy=s.entry_phase_depth
    for row,source,w in zip(rows,index,weight):
        face=row['face']
        if face not in corners:
            output.append(row);indices.append(source);weights.append(w);continue
        finite=depth_state['finite_pieces'].get(face)
        active=(bool(finite['triangles']) if finite is not None else np.any(corners[face]>0.))
        if not active:
            output.append(row);indices.append(source);weights.append(w);continue
        stacked=bool(len(row['upper']) or len(row['pairs']))
        mixed=stacked or abs(w-1.)>2e-10
        local_pieces=None
        if mixed and (stacked or not trace):
            local_pieces=_entered_region_triangles(depth_state,face,row['polygon'],
                float(s.material_surface.get('radius_km',6371.)))
        if stacked:
            if sum(area for area,_ in local_pieces)>1e-8:
                raise ValueError('Entry phase forcing intersects an active continental stack.')
            output.append(row);indices.append(source);weights.append(w)
            continue
        if local_pieces is not None and not local_pieces:
            output.append(row);indices.append(source);weights.append(w)
            continue
        entry_count+=1
        if trace:
            point=s.trace_xyz[source]
            outside=(face in depth_state['finite_planes'] and
                     np.any(depth_state['finite_planes'][face]@point<0.))
            depths=np.array([0. if outside else entry_depth.point_depth(
                corners[face],depth_state['triangles'][face],point)])
            local_weights=np.ones(1)
        else:
            def evaluate(depths):
                selected=np.full(len(depths),source,int)
                result,returned,contraction,*_=phase_evolution._advance_regions(state,selected,depths,
                    depths*column_density.RHO_MANTLE*1000.,controls,dt)
                return np.column_stack([result[key] for key in sorted(result)]+[returned,contraction,columns.elevation(result)])
            ordinary=state['thickness_km'][source]-state[phase.DENSE][source]/state['area_factor'][source]
            cuts=[(controls['reaction_pressure_pa']/phase_evolution.GRAVITY-ordinary*column_density.RHO_CRUST*1000.)/
                  (column_density.RHO_MANTLE*1000.),controls['reaction_pressure_pa']/phase_evolution.GRAVITY/(column_density.RHO_MANTLE*1000.)]
            if controls['geotherm_c_per_km']>0.:
                for temperature in (phase.REACTION_TEMPERATURE_C,controls['mantle_temperature_c']):
                    cuts.append((temperature-controls['surface_temperature_c'])/controls['geotherm_c_per_km']-
                                controls['thermal_depth_fraction']*state['thickness_km'][source])
            if mixed:
                face_area=float(s.material_surface['area_km2'][face])
                covered=sum(area for area,_ in local_pieces)
                if covered>face_area*w+max(face_area*2e-10,1e-8):
                    raise ValueError('Local entry phase pieces exceed their stack-free region.')
                nodes=[];shares=[];used=0
                zero=max(0.,1.-covered/(face_area*w))
                if zero>0.:
                    nodes.append(np.array([0.]));shares.append(np.array([zero]))
                for area,local_corners in local_pieces:
                    remaining=policy['max_panels']-used
                    if remaining<1:raise ValueError('Mixed entry source quadrature exceeds its panel budget.')
                    sample,amount,report=entry_depth.quadrature(local_corners,evaluate,breakpoints=cuts,
                        relative_tolerance=policy['relative_tolerance'],
                        absolute_tolerance=policy['absolute_tolerance'],max_panels=remaining)
                    nodes.append(sample);shares.append(area/(face_area*w)*amount)
                    used+=report['panels'];evaluations+=report['evaluations']
                    error=max(error,report['normalized_error'])
                panels+=used
                depths=np.concatenate(nodes);local_weights=np.concatenate(shares)
                if (not np.isfinite(local_weights).all() or np.any(local_weights<0.)
                        or abs(float(local_weights.sum())-1.)>2e-10):
                    raise ValueError('Mixed entry source partition lost material area.')
            elif finite is None:
                depths,local_weights,report=entry_depth.quadrature(corners[face],evaluate,breakpoints=cuts,
                    **{key:policy[key] for key in ('relative_tolerance','absolute_tolerance','max_panels')})
                panels+=report['panels'];evaluations+=report['evaluations'];error=max(error,report['normalized_error'])
            else:
                nodes=[];shares=[];used=0
                if finite['zero_fraction']>0.:
                    nodes.append(np.array([0.]));shares.append(np.array([finite['zero_fraction']]))
                for area,local_corners in finite['triangles']:
                    if area<=0.:continue
                    remaining=policy['max_panels']-used
                    if remaining<1:raise ValueError('Finite entry source quadrature exceeds its panel budget.')
                    sample,amount,report=entry_depth.quadrature(local_corners,evaluate,breakpoints=cuts,
                        relative_tolerance=policy['relative_tolerance'],
                        absolute_tolerance=policy['absolute_tolerance'],max_panels=remaining)
                    nodes.append(sample);shares.append(area*amount)
                    used+=report['panels'];evaluations+=report['evaluations']
                    error=max(error,report['normalized_error'])
                panels+=used
                depths=np.concatenate(nodes);local_weights=np.concatenate(shares)
                if (not np.isfinite(local_weights).all() or np.any(local_weights<0.)
                        or abs(float(local_weights.sum())-1.)>2e-12):
                    raise ValueError('Finite entry depth partition lost material reference area.')
        for depth,local_weight in zip(depths,local_weights):
            output.append(dict(row,entry_depth_km=float(depth)))
            indices.append(source);weights.append(w*local_weight)
    return output,np.array(indices,int),np.array(weights,float),dict(version=1,
        entry_columns=entry_count,panels=panels,local_source_evaluations=evaluations,
        maximum_estimated_normalized_error=error,
        pressure='uniform mantle hydrostatic increment at local translated depth',
        thermal='prescribed bath at local translated depth',
        spatial_policy='local histories then conservative face projection; marker depth is pointwise')
