"""Run a saved or generated starting world without opening a browser."""
import argparse
import json
import time
from pathlib import Path
from server import SimulationManager, RUNS, validate_config
import fresh_world


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, help='JSON object with engine configuration')
    result.add_argument('--initial', type=Path, help='JSON initial map: width, height, flat crust codes')
    result.add_argument('--seed', type=int)
    result.add_argument('--width', type=int)
    result.add_argument('--duration', type=float)
    result.add_argument('--physics-profile', choices=('legacy', 'reviewed_v1'),
                        help=('Physics setup for the new world. Without --config the default is reviewed_v1 '
                              '(as in the app), which selects the reviewed laws. With --config the file is '
                              'reproduced: a file without a profile stays legacy, and a law the file leaves '
                              'out keeps the engine meaning it was saved with (fixed trench, immediate '
                              'foundering, enhanced rifting off). This flag then changes only the profile; '
                              'to get the reviewed laws on an old file, drop --config or set the law keys.'))
    result.add_argument('--workers', type=int, help='Concurrent workers, from 1 through physical core count')
    result.add_argument('--effective-subduction', action=argparse.BooleanOptionalAction, default=None,
                        help='Use Lite constant trench force with declared starting trenches (default for new reviewed worlds); selects reviewed_v1, fixed trench and immediate foundering.')
    result.add_argument('--subduction-force-n-per-m', type=float,
                        help='Effective pull per metre of active trench, in N/m (default 5e12); supplying it enables effective subduction.')
    return result


def build_config(args):
    """Assemble the new-world request.

    Without --config the command line matches the app's new-world Lite default,
    including the inherited initial trench selection when available. The fresh
    law selection is explicit before SimulationManager.start receives it.

    A --config file is reproduced as saved. Its profile is never defaulted, so
    a file written before physics_profile existed stays legacy. Every law key
    fresh_world_request would fill is first made explicit with the engine's
    missing-key meaning, because the four reviewed_v1 runs of 2026-09-15 were
    saved without subduction_response or retained_phases and ran fixed_trench
    and disabled. --physics-profile, when given, then changes only the profile.
    """
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        explicit_declaration = config.get('primordial_subduction') is not None
        saved_meaning = validate_config({})  # the engine's reading of a missing key
        for key in fresh_world.REVIEWED_FRESH_LAWS:
            if config.get(key) is None:
                config[key] = saved_meaning[key]
        ocean = config.get('primordial_ocean')
        if isinstance(ocean, dict) and 'enabled' not in ocean:
            config['primordial_ocean'] = dict(ocean, enabled=False)
        rifting = config.get('enhanced_rifting')
        if rifting is None:
            config['enhanced_rifting'] = {'enabled': False}
        elif isinstance(rifting, dict) and 'enabled' not in rifting:
            config['enhanced_rifting'] = dict(rifting, enabled=False)
    else:
        config = {'physics_profile': fresh_world.REVIEWED_PROFILE}
        explicit_declaration = False
    if args.physics_profile is not None:
        config['physics_profile'] = args.physics_profile
    if args.seed is not None:
        config['seed'] = args.seed
    if args.width is not None:
        config.update(width=args.width, height=args.width // 2)
    if args.duration is not None:
        config['duration_myr'] = args.duration
    effective = getattr(args, 'effective_subduction', None)
    force = getattr(args, 'subduction_force_n_per_m', None)
    if force is not None and effective is False:
        raise ValueError('--subduction-force-n-per-m cannot be combined with --no-effective-subduction.')
    if effective is True or force is not None:
        if args.physics_profile == 'legacy':
            raise ValueError('Effective subduction requires reviewed_v1; remove --physics-profile legacy.')
        settings = dict(config.get('effective_subduction') or {}, enabled=True)
        if force is not None:
            settings['force_n_per_m'] = force
        config.update(effective_subduction=settings, physics_profile='reviewed_v1',
                      subduction_response='fixed_trench', retained_phases='disabled',
                      trench_persistence='kinematic', slab_allocation='uniform')
        config['continental_lifecycle'] = dict(config.get('continental_lifecycle') or {}, enabled=False)
        config['rift_traction'] = dict(config.get('rift_traction') or {}, enabled=False)
        declaration = config.get('primordial_subduction')
        if not explicit_declaration and args.initial is not None:
            initial = json.loads(args.initial.read_text(encoding='utf-8'))
            declaration = initial.get('initial_subduction')
        config['primordial_subduction'] = dict(declaration or {}, enabled=True)
    elif effective is False:
        config['effective_subduction'] = dict(config.get('effective_subduction') or {}, enabled=False)
    if args.config is None:
        initial = json.loads(args.initial.read_text(encoding='utf-8')) if args.initial is not None else None
        config = fresh_world.fresh_world_request(config, initial=initial)
    return config


def main():
    args = parser().parse_args()
    config = build_config(args)
    initial = json.loads(args.initial.read_text(encoding='utf-8')) if args.initial else None
    manager = SimulationManager(workers=args.workers)
    try:
        result = manager.start(config, initial)
        previous = -1
        while True:
            status = manager.status()
            if status['time_myr'] != previous:
                print(f"{status['time_myr']:7g} Myr / {status['duration_myr']:g}   {status['frame_count']} saved frames", flush=True)
                previous = status['time_myr']
            if status['state'] != 'running':
                break
            time.sleep(.5)
    finally:
        manager.prepare_shutdown()
        if manager.worker and manager.worker.is_alive():
            manager.worker.join()
        manager.parallel.close()
    if status['state'] == 'error':
        raise SystemExit(status['error'])
    print(f"{status['state'].capitalize()} in {status['elapsed_seconds']} seconds. Files: {RUNS / result['run_id']}")


if __name__ == '__main__':
    main()
