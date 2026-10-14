"""Which Lite laws a newly created reviewed_v1 world selects in this fork.

This module is imported only by server.py (and the command line through it),
never by the engine. It is therefore outside the engine helper-source closure
that server.capture_auxiliary_sources hashes into every checkpoint, so editing
this table never changes a saved run's recorded helper hashes and never blocks
resume or branch. Keep it that way: do not import it from tectonics or any
engine module.

The selection applies only where a world is created (SimulationManager.start
and the new-world form defaults from GET /api/config), never on resume,
branch or migration. Those re-read a saved config whose missing keys keep the
engine's historical defaults (fixed_trench, disabled, enhanced rifting off).
Explicit choices in the request always win, and legacy or profile-less
requests are returned unchanged. The result is written explicitly into the
run's config.json, so the saved world never depends on this table again.
"""
from copy import deepcopy

REVIEWED_PROFILE = 'reviewed_v1'

# The prior detailed fresh selection remains available by explicitly disabling
# effective_subduction. Neither table is an engine or saved-checkpoint default.
DETAILED_FRESH_LAWS = dict(retained_phases='thermal_v1', subduction_response='moving_hinge_v1')
REVIEWED_FRESH_LAWS = dict(
    effective_subduction=dict(enabled=True, force_n_per_m=3e13),
    primordial_subduction=dict(enabled=True),
    primordial_ocean=dict(enabled=True),
    subduction_response='fixed_trench', retained_phases='disabled',
    continental_lifecycle=dict(enabled=False), rift_traction=dict(enabled=False),
    trench_persistence='kinematic', slab_allocation='uniform',
    force_limit_rifting=dict(enabled=False))


def fresh_world_request(config=None, initial=None):
    """Fill the Lite choices into a request that creates a new reviewed world.

    Enhanced rifting keeps enhanced_rifting.normalize's existing parameters
    (seed 37, amplitude 0.25, 350 km length scale, 150 km craton rim); only
    the enabled flag is chosen here, and only when the request names neither
    the dict nor its enabled flag. The input is never mutated.
    """
    request = dict(config or {})
    if request.get('physics_profile') != REVIEWED_PROFILE:
        return request
    effective = request.get('effective_subduction')
    if effective is None:
        request['effective_subduction'] = deepcopy(REVIEWED_FRESH_LAWS['effective_subduction'])
    elif isinstance(effective, dict):
        settings = dict(effective)
        settings.setdefault('enabled', True)
        if settings['enabled'] is True:
            settings.setdefault('force_n_per_m', REVIEWED_FRESH_LAWS['effective_subduction']['force_n_per_m'])
        request['effective_subduction'] = settings
    lite = isinstance(request['effective_subduction'], dict) and request['effective_subduction'].get('enabled') is True
    laws = REVIEWED_FRESH_LAWS if lite else DETAILED_FRESH_LAWS
    if (lite and request.get('primordial_subduction') is None and isinstance(initial, dict)
            and initial.get('initial_subduction') is not None):
        request['primordial_subduction'] = deepcopy(initial['initial_subduction'])
    declaration = request.get('primordial_subduction')
    if lite and isinstance(declaration, dict) and 'enabled' not in declaration:
        request['primordial_subduction'] = dict(declaration, enabled=True)
    if lite:
        # The ring is an explicit alternative ocean initializer. Preserve it
        # when no ocean-splitting flag was supplied; explicit conflicts still
        # reach the engine's normal validation.
        ring = request.get('supercontinent_ring')
        ocean_default = not (isinstance(ring, dict) and ring.get('enabled') is True)
        ocean = request.get('primordial_ocean')
        if ocean is None:
            request['primordial_ocean'] = dict(enabled=ocean_default)
        elif isinstance(ocean, dict) and 'enabled' not in ocean:
            request['primordial_ocean'] = dict(ocean, enabled=ocean_default)
    for key, value in laws.items():
        if request.get(key) is None:
            request[key] = deepcopy(value)
    rifting = request.get('enhanced_rifting')
    if rifting is None:
        request['enhanced_rifting'] = {'enabled': True}
    elif isinstance(rifting, dict) and 'enabled' not in rifting:
        request['enhanced_rifting'] = dict(rifting, enabled=True)
    return request
