"""Local pressure/thermal forcing for the retained dense-crust column model.

This is an explicit reduced constitutive model. A prescribed geotherm supplies a
conductive bath; temperature changes a Newtonian viscosity. An overstress law
admits downward drainage only when the dense root's load exceeds a specified
supporting strength, and drainage can spend only the ordinary-equivalent reserve
above the 8 km mass-equivalent floor (dense_crust.advance withholds the rest and
reports it). It is not a resolved mantle-flow or instability calculation.
"""
from copy import copy, deepcopy
import math
import numpy as np
import burial_depth
import column_density
import crustal_structure as columns
import dense_crust as phase
import eclogite_sink
import crust_inventory
from deforming_regions import IncompleteContactStepError
from exact_polygon import ExactPolygon

VERSION = 1
FLOOR_VERSION = phase.FLOOR_VERSION
FLOOR_BASIS = 'current mass-equivalent thickness >= 8 km; retained dense contraction restored'
DRAINAGE_RESERVE_POLICY = ('mantle return is limited to the ordinary-equivalent reserve above the '
                           '8 km floor; unreleased dense phase is retained in the column and reported')
SECONDS_PER_MYR = 365.25*86400.*1e6

# The saved area factor is accumulated while actual/reference is recomputed from
# geometry, so the two drift apart by roundoff. 2e-10 demanded agreement to the
# tenth significant digit and stopped a run on 5 of 15,014 faces, all about 1 km2,
# at a worst mismatch of 4.7e-10. 1e-7 is still seven significant digits of areal
# strain and still catches any genuine bookkeeping error.
AREA_FACTOR_RTOL = 1e-7
GRAVITY = 9.81
# The conductive length is the scale of the thermal perturbation that must
# relax, not the thickness of one buried sheet. Underthrusting or stacking
# displaces the geotherm of the whole thickened column, so a doubled 35 km
# crust (70 km) sets it. The first-mode time L^2/(pi^2*kappa) is then ~16 Myr,
# consistent with the 20-30 Myr that thickened crust takes to approach peak
# metamorphic temperature (England & Thompson 1984, J. Petrol. 25 894). The
# former 20 km gave 1.3 Myr, so underthrust sheets crossed the 600 C reaction
# temperature almost at once, then converted, softened and drained within
# ~10-20 Myr. Saved worlds keep the value recorded at their migration.
REFERENCE_PARAMETERS = dict(
    reaction_pressure_pa=1.5e9, reaction_tau_myr=10.,
    surface_temperature_c=0., geotherm_c_per_km=15., mantle_temperature_c=1300.,
    thermal_depth_fraction=.75, thermal_diffusivity_m2_s=1e-6, diffusion_length_km=70.,
    viscosity_reference_pa_s=1e21, viscosity_reference_temperature_c=1000.,
    activation_energy_j_mol=200000., supporting_yield_stress_pa=5e6,
    maximum_substep_myr=.25)


def parameters(value=None):
    result=dict(REFERENCE_PARAMETERS)
    if value is not None:
        if not isinstance(value,dict) or set(value)-set(result):
            raise ValueError('Unknown retained-phase constitutive parameters.')
        result.update(value)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating))
           or not np.isfinite(v) or v<0 for v in result.values()):
        raise ValueError('Retained-phase parameters must be finite nonnegative numbers.')
    for name in ('reaction_pressure_pa','reaction_tau_myr','thermal_diffusivity_m2_s',
                 'diffusion_length_km','viscosity_reference_pa_s','maximum_substep_myr'):
        if result[name]<=0:raise ValueError('Retained-phase parameter must be positive: '+name)
    if not 0<=result['thermal_depth_fraction']<=1 or result['mantle_temperature_c']<result['surface_temperature_c']:
        raise ValueError('Invalid reference geotherm/depth fraction.')
    return {k:float(v) for k,v in result.items()}


def enabled(s):
    version=getattr(s,'retained_dense_crust_version',0)
    if isinstance(version,(bool,np.bool_)) or not isinstance(version,(int,np.integer)) or version not in (0,VERSION):
        raise ValueError('Unsupported retained-phase evolution version.')
    return version==VERSION


def _validate_floor_migration(report):
    if not isinstance(report, dict):
        raise ValueError('Retained-phase floor version 2 requires an explicit migration record.')
    source = report.get('from_version')
    target = report.get('to_version')
    when = report.get('time_myr')
    if (isinstance(source, (bool, np.bool_)) or source not in (0, 1)
            or isinstance(target, (bool, np.bool_)) or target != FLOOR_VERSION
            or isinstance(when, (bool, np.bool_))
            or not isinstance(when, (int, float, np.integer, np.floating))
            or not np.isfinite(when) or when < 0
            or report.get('physical_state_changed') is not False
            or report.get('rng_changed') is not False
            or report.get('invariant') != FLOOR_BASIS):
        raise ValueError('Invalid retained-phase column-floor migration record.')
    return deepcopy(report)


def floor_version(s):
    """Return the explicit column-floor policy for an active phase world.

    Active checkpoints and historical frames without this field used the legacy
    physical 8 km floor and are identified as version 1. They may be inspected,
    but a live source step requires the explicitly migrated version 2 policy.
    """
    active = enabled(s)
    marker_present = hasattr(s, 'retained_phase_floor_version')
    raw = getattr(s, 'retained_phase_floor_version', 0)
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
        raise ValueError('Unsupported retained-phase column-floor version.')
    version = int(raw)
    if not active:
        if version != 0 or hasattr(s, 'retained_phase_floor_migration'):
            raise ValueError('A retained-phase column-floor policy requires active retained phases.')
        return 0
    if not marker_present:
        version = 1
    if version not in (1, FLOOR_VERSION):
        raise ValueError('Unsupported retained-phase column-floor version.')
    if version == 1:
        if hasattr(s, 'retained_phase_floor_migration'):
            raise ValueError('Legacy retained-phase columns cannot carry a version 2 floor migration.')
    else:
        migration=_validate_floor_migration(getattr(s, 'retained_phase_floor_migration', None))
        if hasattr(s, 't') and migration['time_myr'] > float(s.t):
            raise ValueError('Retained-phase floor migration occurs after the current world time.')
    return version


def _validate_floor_state(state, version):
    phase.validate(state)
    if (version == 1 and np.any(np.asarray(state['thickness_km'])
                                < phase.MECHANICAL_FLOOR_KM-phase.FLOOR_TOLERANCE_KM)):
        raise ValueError('Legacy retained-phase state is below its physical 8 km column floor.')


def require_current_floor(s):
    version = floor_version(s)
    if version != FLOOR_VERSION:
        raise ValueError('Legacy retained-phase columns require phase_evolution.upgrade_floor before evolution.')
    for state in (s.structure, s.trace_structure):
        _validate_floor_state(state, version)
    return version


def upgrade_floor(s):
    """Select the phase-adjusted current-mass floor without changing material."""
    if not enabled(s):
        raise ValueError('Phase-adjusted floor migration requires active retained phases.')
    previous = floor_version(s)
    if previous == FLOOR_VERSION:
        require_current_floor(s)
        return _validate_floor_migration(s.retained_phase_floor_migration)
    for state in (s.structure, s.trace_structure):
        _validate_floor_state(state, previous)
    parameters(s.retained_phase_parameters)
    report = dict(from_version=previous, to_version=FLOOR_VERSION,
                  time_myr=float(s.t), physical_state_changed=False,
                  rng_changed=False, invariant=FLOOR_BASIS)
    _validate_floor_migration(report)
    staged = copy(s)
    staged.retained_phase_floor_version = FLOOR_VERSION
    staged.retained_phase_floor_migration = report
    prepare(staged)
    s.retained_phase_floor_version = FLOOR_VERSION
    s.retained_phase_floor_migration = report
    return deepcopy(report)


def upgrade(s,initial_temperature_c,*,constitutive_parameters=None):
    """Explicit detached migration, preserving geometry, mass, relief and RNG."""
    if enabled(s):
        policy=floor_version(s)
        for state in (s.structure,s.trace_structure):_validate_floor_state(state,policy)
        parameters(s.retained_phase_parameters)
        return deepcopy(s.retained_phase_migration)
    if getattr(s,'plate_balance_version',0)!=1 or getattr(s,'material_mechanics_version',0)!=1:
        raise ValueError('Retained density requires the shared plate/sheet gravitational functional.')
    if (not hasattr(s,'collision_contacts') or not hasattr(s,'parcel_collision_sheet')
            or getattr(s,'collision_surface_version',0)!=1):
        raise ValueError('Retained density requires persistent stack order and physical collision support.')
    controls=parameters(constitutive_parameters)
    initial=np.broadcast_to(np.asarray(initial_temperature_c,float),(len(s.mass),)).copy()
    if not np.isfinite(initial).all() or np.any(initial<0):
        raise ValueError('Supply an explicit finite initial column temperature; depth cannot reconstruct thermal history.')
    staged=copy(s)
    staged.structure=deepcopy(s.structure);staged.trace_structure=deepcopy(s.trace_structure)
    for key in ('mantle_return_km3','process_totals','parcel_root_age_myr'):
        if hasattr(s,key):setattr(staged,key,deepcopy(getattr(s,key)))
    eclogite_sink.upgrade_inventory(staged)
    eclogite_sink.upgrade_depth_integration(staged)
    phase.initialize(staged.structure,initial)
    phase.initialize(staged.trace_structure,eclogite_sink.trace_values(staged,initial))
    staged.retained_dense_crust_version=VERSION
    staged.retained_phase_floor_version=FLOOR_VERSION
    staged.retained_phase_floor_migration=dict(from_version=0,to_version=FLOOR_VERSION,
        time_myr=float(s.t),physical_state_changed=False,rng_changed=False,
        invariant=FLOOR_BASIS)
    staged.retained_phase_parameters=controls
    prepare(staged)  # Validate actual regional stack geometry before committing.
    column_density.options(staged)
    import collision_surface
    collision_surface.refresh(staged)
    report=dict(from_version=0,to_version=VERSION,time_myr=float(s.t),
        initial_temperature_policy='explicit caller-supplied values; no reconstructed thermal history',
        initial_temperature_min_c=float(initial.min()) if len(initial) else None,
        initial_temperature_max_c=float(initial.max()) if len(initial) else None,
        initial_retained_dense_volume_km3=0.,historical_foundering_reconstructed=False)
    for key in ('structure','trace_structure','foundering_version','foundering_inventory_migration',
                'foundering_depth_version','foundering_depth_migration','mantle_return_km3',
                'process_totals','parcel_root_age_myr'):
        if hasattr(staged,key):setattr(s,key,getattr(staged,key))
    s.retained_dense_crust_version=VERSION
    s.retained_phase_floor_version=staged.retained_phase_floor_version
    s.retained_phase_floor_migration=deepcopy(staged.retained_phase_floor_migration)
    s.retained_phase_parameters=controls
    s.retained_phase_migration=report
    for key in collision_surface.parcel_fields(staged)+('collision_stack_diagnostics','_collision_surface_signature'):
        setattr(s,key,getattr(staged,key))
    return deepcopy(report)


def snapshot_fields(s):
    if not enabled(s):
        floor_version(s)
        return {}
    policy=floor_version(s)
    _validate_floor_state(s.structure,policy)
    result=dict(retained_dense_crust_version=VERSION,
        retained_phase_floor_version=policy,
        retained_phase_parameters=parameters(s.retained_phase_parameters),
        retained_phase_migration=deepcopy(getattr(s,'retained_phase_migration',{})),
        material_crust_temperature_c=phase.temperature(s.structure).copy())
    if policy == FLOOR_VERSION:
        result['retained_phase_floor_migration']=_validate_floor_migration(
            s.retained_phase_floor_migration)
    result.update({name:np.asarray(s.structure[field]).copy() for name,field in phase.FRAME_FIELDS.items()})
    return result


def validate_frame(frame):
    from types import SimpleNamespace
    subject=SimpleNamespace(retained_dense_crust_version=frame.get('retained_dense_crust_version',0))
    if 'retained_phase_floor_version' in frame:
        subject.retained_phase_floor_version=frame['retained_phase_floor_version']
    if 'retained_phase_floor_migration' in frame:
        subject.retained_phase_floor_migration=frame['retained_phase_floor_migration']
    active=enabled(subject)
    policy=floor_version(subject)
    fields=set(phase.ARRAY_FIELDS).intersection(frame)
    if not active:
        if (fields or 'retained_phase_parameters' in frame or 'retained_phase_migration' in frame
                or 'retained_phase_floor_version' in frame
                or 'retained_phase_floor_migration' in frame):
            raise ValueError('Retained phase fields require their explicit version.')
        return
    if frame.get('foundering_version')!=2 or frame.get('foundering_depth_version')!=1:
        raise ValueError('Retained phase frames require conservative inventory and local depth policies.')
    if fields!=set(phase.ARRAY_FIELDS):raise ValueError('Missing saved retained phase/thermal state.')
    parameters(frame.get('retained_phase_parameters'))
    if 'retained_phase_parameters' not in frame or set(frame['retained_phase_parameters'])!=set(REFERENCE_PARAMETERS):
        raise ValueError('Saved retained-phase parameters must be explicit and complete.')
    count=len(frame['material_faces'])
    state={field:np.asarray(frame[name]) for name,field in phase.FRAME_FIELDS.items()}
    for field in crust_inventory.FIELDS:
        state[field]=np.asarray(frame.get('material_'+field))
    if any(value.shape!=(count,) for value in state.values()):
        raise ValueError('Saved retained-phase state must align with all material faces.')
    _validate_floor_state(state,policy)
    if policy == FLOOR_VERSION and 'time_myr' in frame:
        when=frame['retained_phase_floor_migration']['time_myr']
        if not np.isfinite(frame['time_myr']) or when > frame['time_myr']:
            raise ValueError('Retained-phase floor migration occurs after the saved frame.')
    observed=np.asarray(frame['material_crust_temperature_c'])
    if observed.shape!=(count,) or not np.allclose(observed,phase.temperature(state),rtol=2e-12,atol=1e-9):
        raise ValueError('Saved temperature disagrees with retained mass and sensible heat.')
    if 'material_crustal_thickness_km' in frame and not np.allclose(frame['material_crustal_thickness_km'],state['thickness_km'],rtol=2e-12,atol=1e-9):
        raise ValueError('Saved phase thickness disagrees with material geometry.')
    geometry_fields=('material_actual_area_km2','material_reference_area_km2')
    if any(name in frame for name in geometry_fields):
        actual,reference=(np.asarray(frame.get(name)) for name in geometry_fields)
        if (actual.shape!=(count,) or reference.shape!=(count,) or not np.isfinite(actual).all()
                or not np.isfinite(reference).all() or np.any(actual<=0) or np.any(reference<=0)
                or not np.allclose(actual/reference,state['area_factor'],rtol=AREA_FACTOR_RTOL,atol=1e-12)):
            raise ValueError('Saved phase area factor disagrees with physical/reference material area.')



def conditions(state, upper_thickness_km, upper_mass_kg_m2, controls):
    """Pressure threshold precedes region averaging; drainage follows rheology."""
    t=np.asarray(state['thickness_km'])
    dense=state[phase.DENSE]/state['area_factor']
    ordinary=t-dense
    upper_t,upper_mass=np.broadcast_arrays(upper_thickness_km,upper_mass_kg_m2)
    if not np.isfinite(upper_t).all() or not np.isfinite(upper_mass).all() or np.any(upper_t<0) or np.any(upper_mass<0):
        raise ValueError('Phase forcing needs finite nonnegative local overburden.')
    # Converted material occupies the base. New reaction consumes the eligible
    # part of the remaining ordinary column immediately above that base.
    cutoff=(controls['reaction_pressure_pa']/GRAVITY-upper_mass)/(column_density.RHO_CRUST*1000.)
    eligible=np.clip(ordinary-cutoff,0.,ordinary)
    fraction=np.divide(eligible,ordinary,out=np.zeros_like(ordinary),where=ordinary>0)
    depth=upper_t+controls['thermal_depth_fraction']*t
    bath=np.minimum(controls['mantle_temperature_c'],
                    controls['surface_temperature_c']+controls['geotherm_c_per_km']*depth)
    tau=(controls['diffusion_length_km']*1000.)**2/(math.pi**2*controls['thermal_diffusivity_m2_s'])/SECONDS_PER_MYR
    temperature=phase.temperature(state)
    inverse_difference=1./(temperature+273.15)-1./(controls['viscosity_reference_temperature_c']+273.15)
    log_eta=math.log(controls['viscosity_reference_pa_s'])+controls['activation_energy_j_mol']/8.314462618*inverse_difference
    if np.any(log_eta>math.log(np.finfo(float).max)):
        raise ValueError('Reference rheology overflows at the supplied temperature; change the constitutive range explicitly.')
    viscosity=np.exp(log_eta)
    driving_stress=(column_density.RHO_DENSE-column_density.RHO_MANTLE)*GRAVITY*dense*1000.
    overstress=np.maximum(driving_stress-controls['supporting_yield_stress_pa'],0.)
    # A Bingham/linear-viscous drainage closure, with the uniaxial factor 3.
    # Dense rock below supporting strength is retained indefinitely by this law.
    rate=overstress/(3.*viscosity)*SECONDS_PER_MYR
    return dict(eligible_fraction=fraction,bath_temperature_c=bath,thermal_tau_myr=tau,
                drainage_rate_myr=rate,maximum_ordinary_pressure_pa=(upper_mass+ordinary*2800.*1000.)*GRAVITY,
                dense_driving_stress_pa=driving_stress,viscosity_pa_s=viscosity)


def prepare(s):
    """Disjoint spherical stack regions; geometry only, no temperature averaging."""
    require_current_floor(s)
    upper,lower,pair_area,contact=eclogite_sink._depth_pairs(s)
    surface=s.material_surface
    triangles=np.asarray(surface['vertices'])[np.asarray(surface['faces'])]
    area=np.asarray(surface['area_km2'])
    radius=float(surface.get('radius_km',6371.))
    order=np.argsort(lower,kind='stable')
    starts=np.searchsorted(lower[order],np.arange(len(area)),side='left')
    ends=np.searchsorted(lower[order],np.arange(len(area)),side='right')
    rows=[]
    for face in range(len(area)):
        selected=order[starts[face]:ends[face]]
        regions,areas,_=burial_depth.partition_face(triangles,face,selected,upper,area[face],radius)
        represented=np.zeros(len(selected));positions={int(p):i for i,p in enumerate(selected)}
        for weight,(polygon,covering) in zip(areas,regions):
            if weight<=0:continue
            indices=np.asarray(covering,int)
            if len(np.unique(s.parcel_collision_sheet[upper[indices]]))!=len(indices) and weight>area[face]*2e-10:
                # A material sheet cannot be counted twice as its own vertical
                # load.  This geometry can be created transiently by a long
                # coupled motion interval, so reject the entire transaction and
                # let adaptive_timestepping retry it at a finer interval.  Never
                # guess an order, discard material, or double the overburden.
                raise IncompleteContactStepError(
                    'Self-overlapping sheet has no unique phase overburden; '
                    'the complete coupled timestep must be retried.')
            rows.append(dict(face=face,weight=float(weight/area[face]),polygon=polygon,
                             pairs=indices,upper=upper[indices]))
            for pair in indices:represented[positions[int(pair)]]+=weight
        if not np.allclose(represented,pair_area[selected],rtol=2e-9,atol=area[face]*2e-11):
            raise ValueError('Phase region areas disagree with contact geometry.')
    result=dict(regions=rows,upper=upper,lower=lower,contact=contact,pair_area=pair_area,
        column_thickness_km=np.asarray(s.structure['thickness_km']).copy(),
        column_mass_kg_m2=(phase.mass_volume(s.structure)/s.structure['area_factor']*2800.*1000.).copy())
    import entry_phase_depth
    if entry_phase_depth.enabled(s):result['entry_depth']=entry_phase_depth.prepare(s)
    return result


def _trace_faces(s):
    ids=np.asarray(s.parcel_patch);order=np.argsort(ids)
    at=np.searchsorted(ids[order],s.trace_patch)
    valid=at<len(ids)
    valid[valid]&=ids[order[at[valid]]]==s.trace_patch[valid]
    if not valid.all():raise ValueError('Retained-phase marker has no material face.')
    return order[at]


def _region_contains(polygon, point):
    """Locate markers in the same positive fan used for region quadrature.

    A clipped polygon can retain adjacent vertices a few ulps apart. Its
    tiny edge has a poorly resolved direction and need not bound the entire
    represented fan, even though the fan's area and winding are valid. Keep
    those vertices and all area weights; use the fan when convex half-spaces
    reject. No geometric welding distance or new containment tolerance is
    introduced. Prepared regions have positive fan winding (burial_depth).
    """
    if isinstance(polygon, ExactPolygon):
        # A rounded projection can collapse an exact positive region. Test
        # its oriented source-ray halfspaces directly, keeping the existing
        # 1e-11 normalized-plane tolerance without rounding the edge normals.
        from fractions import Fraction
        ray = tuple(Fraction.from_float(float(value)) for value in point)
        tolerance_squared = Fraction.from_float(1e-11)**2
        points = polygon.homogeneous
        for a, b in zip(points, points[1:]+points[:1]):
            normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                      a[0]*b[1]-a[1]*b[0])
            signed = sum(value*coordinate for value, coordinate in zip(normal, ray))
            if signed < 0 and signed*signed > tolerance_squared*sum(value*value for value in normal):
                return False
        return True

    def inside(vertices):
        planes=np.cross(vertices,np.roll(vertices,-1,axis=0)-vertices)
        planes/=np.maximum(np.linalg.norm(planes,axis=1,keepdims=True),1e-300)
        return bool(np.all(planes@point>=-1e-11))

    if inside(polygon):return True
    if len(polygon)<=3:return False
    a=polygon[0]
    for b,c in zip(polygon[1:-1],polygon[2:]):
        # A collapsed fan triangle has no quadrature area. Do not let an
        # all-zero set of planes classify every point as inside it.
        if float(a@np.cross(b-a,c-a))>0 and inside(np.array([a,b,c])):
            return True
    return False


def _selected_regions(s,prepared,trace):
    if not trace:
        return prepared['regions'],np.array([r['face'] for r in prepared['regions']],int),np.array([r['weight'] for r in prepared['regions']])
    selected=[]
    by_face={}
    for row in prepared['regions']:by_face.setdefault(row['face'],[]).append(row)
    for face,point in zip(_trace_faces(s),s.trace_xyz):
        found=[]
        for row in by_face[int(face)]:
            if _region_contains(row['polygon'],point):found.append(row)
        if not found:raise ValueError('Retained-phase marker left its local stack region.')
        selected.append(min(found,key=lambda row:tuple(row['upper'])))
    return selected,np.arange(len(selected)),np.ones(len(selected))


def _advance_regions(state,index,upper_t,upper_mass,controls,dt):
    """Pure local source solve, shared by stack and entry-depth quadrature."""
    expanded={k:np.asarray(v)[index].copy() for k,v in state.items()}
    total_return=np.zeros(len(index));total_contraction=np.zeros(len(index))
    total_withheld=np.zeros(len(index));reserve_limited=np.zeros(len(index),bool)
    steps=max(1,int(math.ceil(dt/controls['maximum_substep_myr'])))
    if steps>10000:raise ValueError('Retained-phase step requires too many thermal/reaction substeps.')
    duration=dt/steps
    maximum_rate=0.; maximum_pressure=0.
    for _ in range(steps):
        local=conditions(expanded,upper_t,upper_mass,controls)
        maximum_rate=max(maximum_rate,float(local['drainage_rate_myr'].max(initial=0.)))
        maximum_pressure=max(maximum_pressure,float(local['maximum_ordinary_pressure_pa'].max(initial=0.)))
        report=phase.advance(expanded,local['eligible_fraction'],local['bath_temperature_c'],
            local['thermal_tau_myr'],duration,reaction_tau_myr=controls['reaction_tau_myr'],
            drainage_rate_myr=local['drainage_rate_myr'],minimum_thickness_km=columns.MIN_THICKNESS_KM)
        total_return+=report['returned_km'];total_contraction+=report['contraction_km']
        # Reserve-limited drainage: return withheld at the 8 km mass-equivalent
        # floor stays in the column as dense phase and is reported, not hidden.
        total_withheld+=report['withheld_equivalent_km'];reserve_limited|=report['reserve_limited']
    return expanded,total_return,total_contraction,steps,maximum_rate,maximum_pressure,total_withheld,reserve_limited


def evolve(s,state,dt,*,trace=False,prepared=None):
    """Return staged columns and measured reaction/return; commit is the caller's.

    Region states remain distinct during all internal substeps. At the end, face
    averages are conservatively projected back to the material mesh. This is a
    finite-volume thermal/phase discretization; refinement resolves heterogeneity
    that a coarse face cannot retain between outer timesteps. Marker columns use
    their actual containing region rather than a face-mean burial threshold.
    """
    require_current_floor(s)
    controls=parameters(s.retained_phase_parameters)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Phase evolution requires finite positive elapsed time.')
    phase.validate(state)
    prepared=prepare(s) if prepared is None else prepared
    import entry_phase_depth
    if entry_phase_depth.enabled(s)!=('entry_depth' in prepared):
        raise ValueError('Prepared phase sources do not match the explicitly selected entry-depth policy.')
    rows,index,weight=_selected_regions(s,prepared,trace)
    count=len(state['thickness_km'])
    if not count:
        return deepcopy(state),dict(returned_km=np.zeros(0),contraction_km=np.zeros(0),prepared=prepared,
                                   diagnostics=dict(version=VERSION,regions=0,substeps=0)),None
    entry_report=None
    if 'entry_depth' in prepared:
        rows,index,weight,entry_report=entry_phase_depth.expand(s,state,dt,rows,index,weight,prepared,controls,trace=trace)
    # Coeval upper-column properties are frozen at this source-step boundary,
    # so parcel/marker ordering cannot feed back a later state into this step.
    upper_t=np.array([float(prepared['column_thickness_km'][r['upper']].sum()) for r in rows])
    mass_per_area=prepared['column_mass_kg_m2']
    upper_mass=np.array([float(mass_per_area[r['upper']].sum()) for r in rows])
    depth=np.array([row.get('entry_depth_km',0.) for row in rows])
    upper_t+=depth
    upper_mass+=depth*column_density.RHO_MANTLE*1000.
    (expanded,total_return,total_contraction,steps,maximum_rate,maximum_pressure,
     total_withheld,reserve_limited)=_advance_regions(state,index,upper_t,upper_mass,controls,dt)
    # Column coalescence preserves phase/heat mass and mean height while
    # re-anchoring the reference; all other history counters use reference means.
    import phase_region_projection
    merged=phase_region_projection.coalesce_regions(
        expanded,index,weight,count,measure='reference')
    returned=np.bincount(index,weights=weight*total_return,minlength=count)
    contraction=np.bincount(index,weights=weight*total_contraction,minlength=count)
    withheld=np.bincount(index,weights=weight*total_withheld,minlength=count)
    limited_faces=np.zeros(count,bool);limited_faces[index[reserve_limited]]=True
    phase.validate(merged)
    attribution=None
    if not trace:
        area=np.asarray(s.material_surface['area_km2'])
        pair_volume=np.zeros(len(prepared['upper']));covered=np.zeros(count)
        for r,w,amount in zip(rows,weight,total_return):
            if len(r['pairs']):
                volume=area[r['face']]*w*amount
                covered[r['face']]+=volume
                load=mass_per_area[r['upper']]
                pair_volume[r['pairs']]+=volume*load/load.sum()
        volumes=area*returned
        share=np.divide(covered,volumes,out=np.zeros(count),where=volumes>0)
        attribution=dict(lower=prepared['lower'],weight=prepared['pair_area'],contact=prepared['contact'],
            covered_area=np.bincount(prepared['lower'],weights=prepared['pair_area'],minlength=count),
            covered_share=share,eligible_pair_volume_km3=pair_volume)
    result=dict(returned_km=returned,contraction_km=contraction,withheld_equivalent_km=withheld,
        reserve_limited_faces=limited_faces,prepared=prepared,
        diagnostics=dict(version=VERSION,regions=len(rows),substeps=steps,maximum_drainage_rate_myr=maximum_rate,
            maximum_ordinary_pressure_pa=maximum_pressure,projection='conservative region-to-face phase/heat means',
            return_contact_attribution='current overburden, not the unrecorded phase-formation contact',
            drainage_reserve_policy=DRAINAGE_RESERVE_POLICY,
            reserve_limited_regions=int(np.count_nonzero(reserve_limited)),
            reserve_limited_faces=int(np.count_nonzero(limited_faces)),
            reserve_withheld_return_km3=(float(np.asarray(s.material_surface['area_km2'],float)@withheld)
                                         if not trace else 0.)))
    if entry_report is not None:result['diagnostics']['entry_depth']=entry_report
    return merged,result,attribution
