"""A finite opening rate for a selected force-rift mechanism.

The onset search remains in ``plate_limit_analysis``. Here every unchanged
plate may relax its common Euler motion, and one plate may also acquire one
nonnegative daughter differential motion. ``balance_force_ledger`` projects
the *operators*, not the already solved tractions, onto that field. Thus slab
neck, mantle, anchor, interface, hinge and plastic forces react to the opening.

Uniform restriction is the original active Balance. The spatial lift of a
boundary/contact port onto native control cells is a declared approximation;
this is not a new state, committed daughter topology, or a continuum stress
solution. Current cut strength is held fixed during this instantaneous solve.
Its linear plastic cost is counted once in the generalized power closure.
Finite-time necking/strength evolution and subsequent source transactions
require their own accepted work and geometry checks.
"""
from __future__ import annotations

import math
import numpy as np

import cohesive_failure
import plate_balance as pb

VERSION = 1
_FINAL_WIDTH = pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR
_COMMON_RESTRICTION_TOLERANCE = 1e-9
_MAX_METRIC_CONDITION = 1e8


def _metric_inverse(matrix):
    """A numerically guarded upper bound for the common inverse metric.

    The congruence residual bounds the true inverse by ``B.T@B/(1-eta)``.
    Dimension-dependent arithmetic allowances also enlarge the computed Gram
    matrix. They are qualified numerical allowances, not directed-rounding
    interval proofs. Ill-resolved metrics use the full response solver instead;
    no physical drag or stiffness is added by these diagnostic bounds.
    """
    matrix = np.asarray(matrix, float)
    size = len(matrix)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError('The common quadratic metric is not finite and square.')
    eps = np.finfo(float).eps
    magnitude = float(np.linalg.norm(matrix, 2))
    symmetry = float(np.linalg.norm(matrix-matrix.T, 2))
    if symmetry > 128.*eps*size*max(magnitude, 1.):
        raise ValueError('The common quadratic metric is not numerically symmetric.')
    scale = np.sqrt(np.diag(matrix))
    if np.any(scale <= 0.) or not np.isfinite(scale).all():
        raise ValueError('The common quadratic has an unresisted mode.')
    normalized = matrix/scale[:, None]/scale[None, :]
    normalized = .5*(normalized+normalized.T)
    norm = float(np.linalg.norm(normalized, 2))
    eigenvalues = np.linalg.eigvalsh(normalized)
    spectral_allowance = 128.*eps*size*max(norm, 1.)
    lower = float(eigenvalues[0])-spectral_allowance
    condition = (float(eigenvalues[-1])+spectral_allowance)/lower if lower > 0. else np.inf
    if not np.isfinite(condition) or condition > _MAX_METRIC_CONDITION:
        raise ValueError('The normalized common inverse metric is ill-conditioned or unresolved.')
    factor = np.linalg.cholesky(normalized)
    inverse_factor = np.linalg.solve(factor, np.eye(size))
    congruence = inverse_factor@normalized@inverse_factor.T
    observed = float(np.linalg.norm(congruence-np.eye(size), 2))
    # Bounds on the two congruence products and equilibration/factor arithmetic.
    inverse_norm = float(np.linalg.norm(inverse_factor, 2))
    residual_allowance = 128.*eps*size*max(inverse_norm**2*norm, 1.)
    eta = observed+residual_allowance
    if not np.isfinite(eta) or eta >= .01:
        raise ValueError('The common inverse congruence residual is too uncertain.')
    gram = inverse_factor.T@inverse_factor
    gram_allowance = 128.*eps*size*max(float(np.linalg.norm(inverse_factor))**2, 1.)
    upper = (gram+gram_allowance*np.eye(size))/(1.-eta)
    inverse = upper/scale[:, None]/scale[None, :]
    return inverse, scale, dict(normalized_condition_bound=condition,
        normalized_positive_eigenvalue_bound=lower,
        inverse_congruence_residual=observed,
        inverse_residual_numerical_allowance=residual_allowance,
        inverse_gram_numerical_allowance=gram_allowance,
        inverse_metric_inflation=1./(1.-eta))


def _zero(reason=None):
    result = dict(predicted_opening_cm_yr_mean=0., predicted_opening_cm_yr_max=0.,
        relative_rotation_rad_myr=0., piece_rotation_rad_myr=[0., 0., 0.],
        rest_rotation_rad_myr=[0., 0., 0.], opening_mode_version=VERSION,
        opening_mode_supported=reason is None, opening_mode_reason=reason)
    return result


def _cut(s, cells, piece, axis, geometry, strength):
    """Rates and capacity for the actual oriented cut, in cm/yr Euler units."""
    edges = np.asarray(geometry['edges'], int)
    mid = np.asarray(geometry['mid'], float)
    length = np.asarray(geometry['length_m'], float)
    cut = np.asarray(geometry['cut'], bool)
    if edges.shape != (len(length), 2) or mid.shape != (len(length), 3) or cut.shape != length.shape:
        raise ValueError('Breakup cut arrays have inconsistent dimensions.')
    if (not len(length) or np.any(edges < 0) or np.any(edges >= len(cells))
            or not np.isfinite(mid).all() or not np.isfinite(length).all()
            or np.any(length < 0.) or not np.any(cut & (length > 0.))):
        raise ValueError('Breakup requires a finite resolved cut geometry.')
    strengths = tuple(np.asarray(v, float) for v in strength)
    if (len(strengths) != 3 or any(v.shape != length.shape for v in strengths)
            or any(not np.isfinite(v).all() or np.any(v < 0.) for v in strengths)):
        raise ValueError('Breakup requires finite nonnegative mixed cut strengths.')
    selected = np.flatnonzero(cut)
    a, b = edges[selected].T
    sign = piece[b].astype(float)-piece[a].astype(float)
    if np.any(sign == 0.):
        raise ValueError('Every selected breakup edge must cross the daughter partition.')
    points = np.asarray(s.xyz, float)[cells]
    radial = mid[selected]/np.linalg.norm(mid[selected], axis=1)[:, None]
    normal = points[b]-points[a]
    normal -= radial*np.sum(normal*radial, axis=1)[:, None]
    magnitude = np.linalg.norm(normal, axis=1)
    if np.any(magnitude <= 1e-14) or not np.isfinite(radial).all():
        raise ValueError('Breakup cut has a degenerate tangent normal.')
    normal /= magnitude[:, None]
    tangent = np.cross(radial, normal)
    relative = sign[:, None]*np.cross(axis, radial)
    normal_rate = np.sum(relative*normal, axis=1)
    shear_rate = np.sum(relative*tangent, axis=1)
    speed = np.linalg.norm(relative, axis=1)
    weights = length[selected]
    mean = float(np.average(speed, weights=weights))
    if mean <= 1e-14:
        raise ValueError('Breakup axis does not move its selected cut.')
    capacities = np.array([cohesive_failure.mixed_capacity_n(float(length[e]),
        tuple(float(v[e]) for v in strengths), float(n), float(t))
        for e, n, t in zip(selected, normal_rate, shear_rate)])
    return dict(edges=selected, length_m=weights, normal=normal_rate, shear=shear_rate,
                speed=speed, mean_speed=mean, capacities_n=capacities,
                capacity_n=float(capacities.sum()/mean),
                yield_w=float(capacities.sum()*pb.CM_YR_M_S))


def _basis(s, balance, ledger, plate, cells, piece, axis):
    """Common motion plus an area-centred differential daughter field."""
    selected = np.zeros(len(s.xyz), bool)
    selected[cells[piece]] = True
    support = np.asarray(balance._support(plate), float)
    area = np.asarray(s.cell_area, float)*support
    first, total = float(area[selected].sum()), float(area.sum())
    if first <= 0. or total <= first:
        raise ValueError('Both breakup daughters need positive physical support area.')
    piece_fraction = first/total
    basis = {}
    entries = ledger['plates']
    if not isinstance(entries, dict):
        raise ValueError('Force ledger plate entries must be keyed by plate slot.')
    for p, index in balance.slot.items():
        row = entries[p]
        owned = np.asarray(row['cells'], int)
        block = np.zeros((len(owned), 3, balance.size+1))
        block[:, :, 3*index:3*index+3] = np.eye(3)
        if p == plate:
            factor = np.where(selected[owned], 1.-piece_fraction, -piece_fraction)
            block[:, :, -1] = factor[:, None]*axis
        basis[p] = block
    return basis, piece_fraction


def _curvature_bound(term):
    """Global spectral upper bound for this retained convex port potential."""
    if term.shape == 'quadratic':
        return float(np.linalg.norm(term.matrix))
    rows = tuple(np.asarray(row, float) for row in term.rows)
    if term.coefficient < 0.:
        raise ValueError('A negative port coefficient cannot certify convex zero opening.')
    if term.shape == 'hinge':
        return float(term.coefficient*np.sum(rows[0]**2))
    if term.shape == 'arc_opening':
        return float(term.coefficient*(term.bounds[1]-term.bounds[0])
                     *np.sum(rows[0]**2)/_FINAL_WIDTH)
    if term.shape not in ('abs', 'negative', 'cone') or term.version != 1 or term.scale <= 0.:
        raise ValueError('An unsupported port potential cannot certify convex zero opening.')
    # |v|/one-sided Huber curvature <=1/width. For the passive cone, the
    # normal/tangent Jacobian has norm bounded by their squared row norms.
    return float(term.coefficient*sum(np.sum(row**2) for row in rows)/(_FINAL_WIDTH*term.scale))


def _common_certificate(balance, ledger):
    """Common-mode stationarity and a strong-convexity metric, cached per ledger.

    This evaluates retained port laws, without constructing a daughter model or
    running Newton. Accurate summation and an explicit local numerical allowance
    account for opposing large forces; this is not an interval proof. The common operator is a
    global lower curvature bound because every retained additional law is convex.
    """
    cached = ledger.get('_zero_kkt_common_certificate')
    if cached is not None:
        return cached
    size = balance.size
    contributions = [[] for _ in range(size)]
    allowances = np.zeros(size)
    reference = np.zeros(size)
    matrix = np.zeros((size, size))
    ports = {}
    eps = np.finfo(float).eps
    for term in ledger['local_terms']:
        if len(set(term.owners)) != len(term.owners):
            raise ValueError('Repeated common port owners are not qualified for the zero shortcut.')
        sums = []
        for port in term.ports:
            if id(port) not in ports:
                ports[id(port)] = port.uniform()
                if (np.linalg.norm(ports[id(port)]-np.eye(3))/math.sqrt(3.)
                        > _COMMON_RESTRICTION_TOLERANCE):
                    raise ValueError('A common force port does not preserve uniform motion.')
            sums.append(ports[id(port)])
        projection = np.zeros((3*len(sums), 3*len(sums)))
        for j, block in enumerate(sums):
            projection[3*j:3*j+3, 3*j:3*j+3] = block
        ids = np.concatenate([np.arange(3*balance.slot[p], 3*balance.slot[p]+3) for p in term.owners])
        local = projection@balance.x[ids]
        _, gradient, _ = term.evaluate(local, _FINAL_WIDTH)
        lifted = projection.T@gradient
        bound = _curvature_bound(term)
        # Each potential/gradient involves bounded-size port arithmetic.
        # H*|x| covers cancellation and C1 yield/closing joins as well as the
        # actual force magnitude. This is a qualified numerical allowance,
        # rather than a formal operation-by-operation interval proof.
        allowance = 128.*eps*max(float(np.linalg.norm(projection)), 1.)*(
            float(np.linalg.norm(gradient))+bound*float(np.linalg.norm(local)))
        for index, value in zip(ids, lifted):
            contributions[int(index)].append(float(value))
            allowances[index] += allowance
            reference[index] += abs(float(value))
        if term.shape == 'quadratic':
            matrix[np.ix_(ids, ids)] += projection.T@term.matrix@projection
    gradient = np.array([math.fsum(values) for values in contributions])
    matrix_error = float(np.linalg.norm(matrix-balance.stiffness)/
                         max(np.linalg.norm(balance.stiffness), 1.))
    if not np.isfinite(matrix_error) or matrix_error > _COMMON_RESTRICTION_TOLERANCE:
        raise ValueError('Projected common quadratic differs from the active Balance.')
    inverse, scale, inverse_diagnostics = _metric_inverse(matrix)
    backward = []
    for i in range(0, size, 3):
        numerator = np.linalg.norm(gradient[i:i+3]/scale[i:i+3])
        denominator = np.linalg.norm(reference[i:i+3]/scale[i:i+3])
        backward.append(float(numerator/denominator) if denominator else (0. if numerator == 0. else np.inf))
    # Strong convexity implies ||x-x*||_K <=||g(x)||_(K^-1).
    # The absolute inverse bounds uncertain force signs conservatively.
    radius = math.sqrt(max(float(gradient@inverse@gradient), 0.))
    radius += math.sqrt(max(float(allowances@np.abs(inverse)@allowances), 0.))
    result = dict(inverse=inverse, metric_radius=radius, port_sums=ports,
                  gradient=gradient, gradient_allowance=allowances,
                  backward_error=max(backward, default=0.),
                  common_matrix_relative_error=matrix_error,
                  inverse_diagnostics=inverse_diagnostics)
    ledger['_zero_kkt_common_certificate'] = result
    return result


def _zero_certificate(s, balance, ledger, plate, cells, piece, axis, cost):
    """Certify q=0 from actual differential work and a strict uncertainty margin."""
    if ledger.get('source_balance') is not balance:
        return None, 'Ledger belongs to a different Balance.'
    recorded = ledger.get('motion_coordinates')
    if recorded is None or not np.array_equal(recorded, balance.x):
        return None, 'Ledger motion does not exactly match current Balance coordinates.'
    original_gradient = balance._evaluate(balance.x, _FINAL_WIDTH)[1]
    original_residual = balance._relative_residual(original_gradient)
    if original_residual > pb.FORCE_RELATIVE_TOLERANCE:
        return None, 'Current common motion is not an accepted intact equilibrium.'
    if not ledger.get('diagnostics', {}).get('complete', False):
        return None, 'Complete retained force operators are required for a zero certificate.'
    selected = np.zeros(len(s.xyz), bool)
    selected[cells[piece]] = True
    area = np.asarray(s.cell_area, float)*np.asarray(balance._support(plate), float)
    first, total = float(area[selected].sum()), float(area.sum())
    if first <= 0. or total <= first:
        raise ValueError('Both breakup daughters need positive physical support area.')
    fraction = first/total
    field = np.where(selected[:, None], 1.-fraction, -fraction)*axis
    entry = ledger['plates'][plate]
    owned = np.asarray(entry['cells'], int)
    products = np.asarray(entry['cell_generalized_force'])*field[owned]
    local_work = math.fsum(products.ravel().tolist())
    margin = cost-local_work
    if margin <= 0.:
        return None, 'Cut yield does not strictly exceed the actual differential force.'
    try:
        common = _common_certificate(balance, ledger)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, f'Common convex zero certificate is unavailable: {exc}'
    if common['backward_error'] > pb.FORCE_RELATIVE_TOLERANCE:
        return None, 'Projected common coordinates are not at accepted equilibrium.'
    differences, curvature, arithmetic = [], 0., 0.
    eps = np.finfo(float).eps
    for term in ledger['local_terms']:
        if plate not in term.owners:
            continue
        ids = np.concatenate([np.arange(3*balance.slot[p], 3*balance.slot[p]+3) for p in term.owners])
        local = np.concatenate([common['port_sums'][id(port)]@balance.x[3*balance.slot[p]:3*balance.slot[p]+3]
                                for p, port in zip(term.owners, term.ports)])
        differential = np.concatenate([port.motion(field) if p == plate else np.zeros(3)
                                       for p, port in zip(term.owners, term.ports)])
        magnitude = float(np.linalg.norm(differential))
        if magnitude == 0.:
            continue
        _, gradient, _ = term.evaluate(local, _FINAL_WIDTH)
        differences.append(float(-gradient@differential))
        bound = _curvature_bound(term)
        # The norm of the common selector K^-1/2 is at most this Frobenius
        # bound. This covers arbitrary relaxation of every common plate rate.
        selector = math.sqrt(max(float(np.trace(common['inverse'][np.ix_(ids, ids)])), 0.))
        port_norm = max(float(np.linalg.norm(common['port_sums'][id(port)], 2)) for port in term.ports)
        curvature += bound*magnitude*selector*port_norm
        arithmetic += 128.*eps*magnitude*(float(np.linalg.norm(gradient))+bound*float(np.linalg.norm(local)))
    term_work = math.fsum(differences)
    # Cover the independently accumulated cell-field dot product, even when
    # many local carries cancel, and the observed port/cell adjoint difference.
    accumulation = 8.*eps*max(len(products.ravel()), 1)*float(np.abs(products).sum())
    uncertainty = (abs(term_work-local_work)+arithmetic+accumulation
                   +curvature*common['metric_radius'])
    if not np.isfinite(uncertainty) or margin <= uncertainty:
        return None, 'Yield margin is within the conservative force/relaxation uncertainty bound.'
    return dict(margin_w=margin, uncertainty_bound_w=uncertainty,
                certified_positive_margin_w=margin-uncertainty,
                differential_drive_w=local_work, fraction=fraction,
                original_common_residual=original_residual,
                projected_common_residual=common['backward_error'],
                common_matrix_relative_error=common['common_matrix_relative_error'],
                common_inverse_diagnostics=common['inverse_diagnostics'],
                numerical_allowance_interpretation='Qualified floating-point and common-relaxation bounds; '
                    'not a formal directed-rounding interval certificate.',
                differential_adjoint_error=abs(term_work-local_work)/max(abs(term_work),abs(local_work),1.),
                common_relaxation_metric_radius=common['metric_radius']), None


def _residual(model, x, gradient, cost):
    reference = np.asarray(model._force_reference, float).copy()
    reference[-1] += cost
    active_gradient = gradient.copy()
    if x[-1] == 0. and gradient[-1] >= 0.:
        active_gradient[-1] = 0.
    scale = np.sqrt(np.diag(model.stiffness))
    if np.any(scale <= 0.) or not np.isfinite(scale).all():
        raise ValueError('Breakup mode has an unresisted or invalid motion.')
    values = []
    for start in range(0, len(x)-1, 3):
        numerator = np.linalg.norm(active_gradient[start:start+3]/scale[start:start+3])
        denominator = np.linalg.norm(reference[start:start+3]/scale[start:start+3])
        values.append(float(numerator/denominator) if denominator else (0. if numerator == 0. else np.inf))
    numerator, denominator = abs(active_gradient[-1]), reference[-1]
    values.append(float(numerator/denominator) if denominator else (0. if numerator == 0. else np.inf))
    return max(values, default=0.)


def _minimize(model, initial, cost, *, relax_common=True):
    """Convex Newton solve with an exact one-sided plastic cut/KKT bound."""
    x = np.asarray(initial, float).copy()
    x[-1] = 0.
    iterations, backtracks = 0, 0
    def evaluate(value, delta):
        potential, gradient, hessian, _, _ = model._evaluate(value, delta)
        gradient = gradient.copy()
        gradient[-1] += cost
        return float(potential+cost*value[-1]), gradient, hessian
    def residual(value, gradient):
        if relax_common:
            return _residual(model, value, gradient, cost)
        reference = float(model._force_reference[-1])+cost
        error = 0. if value[-1] == 0. and gradient[-1] >= 0. else abs(float(gradient[-1]))
        return error/reference if reference else (0. if error == 0. else np.inf)
    for delta in np.asarray(pb.HUBER_CONTINUATION_KM_MYR)*pb.KM_MYR_CM_YR:
        for _ in range(pb.MAX_NEWTON_ITERATIONS):
            value, gradient, hessian = evaluate(x, delta)
            current = residual(x, gradient)
            if current <= pb.FORCE_RELATIVE_TOLERANCE:
                break
            step = np.zeros_like(x)
            if relax_common:
                if x[-1] == 0. and gradient[-1] >= 0.:
                    step[:-1] = pb.Balance._solve_spd(hessian[:-1, :-1], -gradient[:-1])
                else:
                    step = pb.Balance._solve_spd(hessian, -gradient)
                    if x[-1] == 0. and step[-1] < 0.:
                        step[:] = 0.
                        step[:-1] = pb.Balance._solve_spd(hessian[:-1, :-1], -gradient[:-1])
            elif not (x[-1] == 0. and gradient[-1] >= 0.):
                step[-1] = -gradient[-1]/hessian[-1, -1]
            slope = float(gradient@step)
            if slope >= 0. or not np.isfinite(step).all():
                raise ValueError('Breakup Newton direction is not a finite descent.')
            bound = -x[-1]/step[-1] if step[-1] < 0. else np.inf
            alpha = min(1., bound)
            accepted = False
            while alpha > 1e-14:
                trial = x+alpha*step
                # Represent the active bound exactly rather than leaving an
                # ulp-sized positive rate that would demand a vanishing step.
                trial[-1] = 0. if alpha == bound else max(float(trial[-1]), 0.)
                trial_value, trial_gradient, _ = evaluate(trial, delta)
                roundoff = 8.*np.finfo(float).eps*max(abs(value), abs(trial_value), 1.)
                # _evaluate updates its force reference, so compute the trial
                # backward error immediately, before another evaluation.
                trial_residual = residual(trial, trial_gradient)
                if (trial_value <= value+pb.ARMIJO*alpha*slope
                        or (-alpha*slope <= roundoff and trial_value <= value+roundoff
                            and trial_residual < (1.-pb.ARMIJO*alpha)*current)):
                    accepted = True
                    break
                alpha *= .5
                backtracks += 1
            if not accepted:
                raise ValueError('Breakup line search failed; mechanism rejected.')
            x = trial
            iterations += 1
    _, gradient, _ = evaluate(x, _FINAL_WIDTH)
    error = residual(x, gradient)
    if error > pb.FORCE_RELATIVE_TOLERANCE:
        raise ValueError(f'Breakup mode did not converge: backward error {error:g}.')
    return x, dict(iterations=iterations, backtracks=backtracks, force_relative_residual=error,
                   opening_bound_gradient_w=float(gradient[-1]))


def solve(s, plate, cells, piece, axis, cut_geometry, strength, balance, *,
          ledger=None, band_km=None, relax_common=True, allow_zero_fast_path=True):
    """Return incremental daughter rotations and a held-mode work receipt.

    ``cut_geometry`` has local ``edges``, ``mid``, ``length_m`` and ``cut``
    arrays; ``strength`` is the three already selected/scaled edge capacities
    in N/m. The scalar opening is the length-weighted mean *magnitude* of the
    differential velocity, matching existing force-rift bookkeeping. Signed
    normal/shear factors are also returned so compression/shear is not silently
    counted as tensile work. All returned daughter rotations are increments
    over the source state's parent omega, as native topology expects.
    """
    if not getattr(balance, 'uses_force_ledger', getattr(balance, 'slab_tethers', False)):
        return _zero('A complete current-force Balance is required; no basal-only fallback.')
    if pb.resistance_version(s) != 1:
        return _zero('Historical nonpassive resistance is not qualified for daughter opening.')
    cells, piece = np.asarray(cells, int), np.asarray(piece, bool)
    axis = np.asarray(axis, float)
    if (cells.ndim != 1 or piece.shape != cells.shape or len(np.unique(cells)) != len(cells)
            or np.any(cells < 0) or np.any(cells >= len(s.xyz))
            or axis.shape != (3,) or not np.isfinite(axis).all()
            or np.linalg.norm(axis) <= 0. or not piece.any() or piece.all()):
        raise ValueError('Breakup requires a finite axis and two distinct native cell pieces.')
    if plate not in balance.slot or np.any(np.asarray(s.plate)[cells] != plate):
        raise ValueError('Breakup cells must belong to the active selected plate.')
    axis = axis/np.linalg.norm(axis)
    cut = _cut(s, cells, piece, axis, cut_geometry, strength)
    import balance_force_ledger
    if band_km is None:
        band_km = float(getattr(s, 'config', {}).get('deformation_width_km',
                                                  balance_force_ledger.DEFAULT_BAND_KM))
    try:
        ledger = ledger or balance_force_ledger.export(balance, balance.x, band_km=band_km)
    except balance_force_ledger.UnsupportedForceLedger as exc:
        return _zero(f'The complete local operator mapping is unavailable: {exc}')
    certificate, fast_reason = None, 'Fast path explicitly disabled.'
    if allow_zero_fast_path and relax_common:
        certificate, fast_reason = _zero_certificate(s, balance, ledger, plate, cells, piece,
                                                     axis, cut['yield_w'])
    if certificate is not None:
        motion = np.r_[balance.x, 0.]
        solver = dict(method='certified_zero_rate_kkt', iterations=0, backtracks=0,
            force_relative_residual=certificate['projected_common_residual'],
            opening_bound_gradient_w=certificate['margin_w'],
            uncertainty_bound_w=certificate['uncertainty_bound_w'],
            certified_positive_margin_w=certificate['certified_positive_margin_w'])
        return _finish(s, plate, axis, balance, motion, certificate['fraction'], cut, solver,
            balance, balance.x, certificate['common_matrix_relative_error'],
            certificate['differential_adjoint_error'], -certificate['differential_drive_w'],
            relax_common, certificate, None)
    basis, fraction = _basis(s, balance, ledger, plate, cells, piece, axis)
    try:
        model = balance_force_ledger.compile_mode(ledger, basis)
    except (NotImplementedError, KeyError, balance_force_ledger.UnsupportedForceLedger) as exc:
        return _zero(f'The complete local operator mapping is unavailable: {exc}')
    initial = np.r_[balance.x, 0.]
    # An arbitrary native rigid field tests the entire common restriction,
    # independently of the accepted near-stationary parent motion.
    probe = np.r_[np.arange(balance.size, dtype=float)*.13-.7, 0.]
    errors = []
    # Whole-plate torque can cancel exactly at rest while separate boundary
    # ports still carry large loads. Projection then leaves ulp-scale common
    # forces. Retain their actual local source scale when checking agreement;
    # a vanishing net source is not the numerical accuracy of its summands.
    gross_drive = math.fsum(float(np.linalg.norm(term.drive))
        for term in ledger['local_terms'] if term.shape == 'quadratic')
    for motion in (initial, probe):
        got = model._evaluate(motion, _FINAL_WIDTH)
        expected = balance._evaluate(motion[:-1], _FINAL_WIDTH)
        # A solved rigid gradient is nearly zero. Normalize its agreement by
        # the actual opposing force contributions, not that vanishing net.
        for candidate, oracle, reference in (
                (got[0], expected[0], abs(expected[0])),
                (got[1][:-1], expected[1], max(np.linalg.norm(balance._force_reference), gross_drive)),
                (got[2][:-1, :-1], expected[2], np.linalg.norm(expected[2]))):
            candidate, oracle = np.asarray(candidate), np.asarray(oracle)
            errors.append(float(np.linalg.norm(candidate-oracle)/max(np.linalg.norm(oracle), reference, 1.)))
    restriction_error = max(errors)
    if restriction_error > _COMMON_RESTRICTION_TOLERANCE:
        return _zero(f'Virtual common restriction differs from the active Balance ({restriction_error:g}).')
    onset = model._evaluate(initial, _FINAL_WIDTH)[1][-1]
    # Independent adjoint check: the accepted spatial force ledger must have
    # the same differential virtual work as the retained finite operators.
    local_work = 0.
    for p, row in ledger['plates'].items():
        local_work += float(np.sum(np.asarray(row['cell_generalized_force'])*basis[p][:, :, -1]))
    onset_error = abs(float(-onset)-local_work)/max(abs(float(onset)), abs(local_work), 1.)
    if onset_error > 1e-9:
        return _zero(f'Differential force/virtual-work mapping is inconsistent ({onset_error:g}).')
    motion, solver = _minimize(model, initial, cut['yield_w'], relax_common=relax_common)
    solver['method'] = 'full_projected_convex_solve'
    return _finish(s, plate, axis, balance, motion, fraction, cut, solver, model, motion,
                   restriction_error, onset_error, onset, relax_common, None, fast_reason)


def _finish(s, plate, axis, balance, motion, fraction, cut, solver, power_model, power_motion,
            restriction_error, onset_error, onset, relax_common, certificate, fast_reason):
    plastic, passivity = power_model._resistance_work(power_motion, _FINAL_WIDTH)
    driving = float(power_model.torque@power_motion)
    resisting = float(power_motion@power_model.stiffness@power_motion)+sum(plastic.values())
    cut_power = float(cut['yield_w']*motion[-1])
    closure = driving-resisting-cut_power
    relative_closure = abs(closure)/max(abs(driving), abs(resisting)+cut_power, 1.)
    if relax_common and relative_closure > 1e-8:
        raise ValueError(f'Breakup generalized power does not close ({relative_closure:g}).')
    conversion = pb.CM_YR_M_S/pb.RADIUS_M*pb.SECONDS_PER_MYR
    common = motion[3*balance.slot[plate]:3*balance.slot[plate]+3]*conversion
    old = np.asarray(s.omega, float)[plate]
    differential = float(motion[-1])*conversion*axis
    piece_increment = common+(1.-fraction)*differential-old
    rest_increment = common-fraction*differential-old
    result = _zero()
    common_rotations = {int(p): (motion[3*i:3*i+3]*conversion).tolist()
                        for p, i in balance.slot.items()}
    result.update(predicted_opening_cm_yr_mean=float(motion[-1]*cut['mean_speed']),
        predicted_opening_cm_yr_max=float(motion[-1]*cut['speed'].max(initial=0.)),
        relative_rotation_rad_myr=float(motion[-1]*conversion),
        piece_rotation_rad_myr=piece_increment.tolist(), rest_rotation_rad_myr=rest_increment.tolist(),
        current_capacity_n=cut['capacity_n'], cut_normal_mode=(cut['normal']/cut['mean_speed']).tolist(),
        cut_shear_mode=(cut['shear']/cut['mean_speed']).tolist(),
        mode_solver=solver, common_restriction_relative_error=restriction_error,
        differential_virtual_work_relative_error=onset_error,
        differential_drive_at_rigid_w=float(-onset),
        common_motion_relaxed=bool(relax_common),
        common_rotations_rad_myr=common_rotations,
        zero_rate_fast_path=certificate is not None,
        zero_rate_fast_path_reason=fast_reason,
        zero_rate_certificate=certificate,
        mode_work=dict(driving_power_w=driving, resisting_power_w=resisting, cut_power_w=cut_power,
                       closure_absolute_w=abs(closure), closure_relative=relative_closure,
                       interpretation='Condensed generalized plate power at fixed geometry/current cut strength; '
                         'not total uncondensed slab power or finite-time qualification.'),
        mode_passivity=passivity,
        opening_mode_interpretation='Full active force operators on a virtual cell-wise daughter field. '
            'Uniform restriction is exact; boundary/contact spatial lifting is approximate. '
            'No topology, source state, force coefficient or prescribed velocity is changed.')
    return result
