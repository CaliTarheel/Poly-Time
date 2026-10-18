"""Resumable artist scenarios with actual snapshots, source receipts and a local replay.

Example: python -m poly.artist_run start --output runs/artist --duration 1000
An unsuccessful run retains its last accepted checkpoint and reports its true end time.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from poly import artist, dynamics, sphere
from poly.checkpoint import atomic_json, load_checkpoint, save_checkpoint
from poly.run import advance, diagnostics, provenance


def _snapshot(world, output, history, width):
    name = 'map-%010.3f.png' % world.time_myr
    metrics = artist.draw(world, output / name, width=width)
    metrics['image'] = name
    if (output / 'checkpoint.json').exists():
        states = output / 'states'
        states.mkdir(exist_ok=True)
        state_name = 'states/state-%010.3f.json' % world.time_myr
        shutil.copyfile(output / 'checkpoint.json', output / state_name)
        metrics['checkpoint'] = state_name
    metrics['boundaries'] = [dict(kind=kind, points=points.tolist())
                             for points, kind in world.boundary_lines()]
    metrics['features'] = [dict(kind=f['kind'], time_myr=f['time_myr'],
                               strain_km=f['cumulative_strain_km'],
                               points=sphere.rotate(f['points'], world.frame(f['plate'])).tolist())
                           for f in world.geology['features']]
    metrics['events'] = [event for entry in world.ledger
                         if entry['time_myr'] > (history[-1]['time_myr'] if history else -1.)
                         for event in entry['events']]
    history.append(metrics)
    atomic_json(output / 'history.json', history)
    replay(output, history)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ('boundaries', 'features', 'events')}), flush=True)


def replay(output, history):
    """One portable HTML file, embedding only accepted output frames."""
    frames = []
    for entry in history:
        entry = dict(entry)
        entry['image'] = 'data:image/png;base64,' + base64.b64encode((output / entry['image']).read_bytes()).decode()
        frames.append(entry)
    template = Path(__file__).with_name('replay.html').read_text(encoding='utf-8')
    data = json.dumps(frames, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (output / 'replay.html').write_text(template.replace('/*__HISTORY__*/[]', data), encoding='utf-8')


def _execute(output, world, metadata, duration):
    settings, config, phase = metadata['artist'], metadata['config'], metadata['progress']
    count = duration / config['dt_myr']
    if not math.isclose(count, round(count), abs_tol=1e-8) or duration < 0.:
        raise ValueError('Duration must be a nonnegative whole number of timesteps.')
    history_path = output / 'history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    manifest = dict(format='poly-artist-run', version=1, source=metadata['provenance'],
                    settings=settings, status='running', time_myr=world.time_myr,
                    target_myr=(phase['completed_intervals'] + round(count)) * config['dt_myr'],
                    terrain='Material-attached output relief over resolved crustal thickness',
                    limitations=['Reduced plate mechanics; no resolved mantle convection',
                                 'Continental extension uses a conservative finite-band approximation',
                                 'No separate evolving sediment layer or internal material deformation',
                                 'Sub-mesh relief is procedural amplification',
                                 'Continental bodies below 20,000 km2 are coarsened into regional crust or passive inventory',
                                 'Duration reached is recorded; a target is not validation'])

    def commit(current, progress):
        metadata['progress'] = dict(progress)
        save_checkpoint(current, output / 'checkpoint.json', metadata)
        atomic_json(output / 'diagnostics.json', diagnostics(current))
        manifest['time_myr'] = current.time_myr
        atomic_json(output / 'manifest.json', manifest)

    atomic_json(output / 'manifest.json', manifest)
    try:
        commit(world, phase)
        if not history:
            _snapshot(world, output, history, settings['width'])
        remaining = round(count)
        while remaining:
            # Output cadence does not alter the force or rift decision cadence.
            next_time = (math.floor((world.time_myr + 1e-8) / settings['snapshot_myr']) + 1) * settings['snapshot_myr']
            steps = min(remaining, max(1, round((next_time - world.time_myr) / config['dt_myr'])))
            advance(world, config, phase, steps, commit)
            remaining -= steps
            _snapshot(world, output, history, settings['width'])
            commit(world, phase)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        atomic_json(output / 'manifest.json', manifest)
        # A rift decision may precede a rejected step. Read the accepted checkpoint instead
        # of depicting the partially attempted world as an accepted state.
        if (output / 'checkpoint.json').exists():
            accepted, _ = load_checkpoint(output / 'checkpoint.json')
            manifest['time_myr'] = accepted.time_myr
            atomic_json(output / 'manifest.json', manifest)
            if not history or accepted.time_myr > history[-1]['time_myr'] + 1e-9:
                try:
                    _snapshot(accepted, output, history, settings['width'])
                except Exception as snapshot_error:
                    manifest['snapshot_error'] = f'{type(snapshot_error).__name__}: {snapshot_error}'
        atomic_json(output / 'manifest.json', manifest)
        raise
    manifest.update(status='complete', time_myr=world.time_myr)
    atomic_json(output / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start')
    start.add_argument('--seed', type=int, default=1)
    start.add_argument('--plates', type=int, default=8)
    start.add_argument('--land', type=float, default=.55)
    start.add_argument('--continents', type=float, default=.65)
    start.add_argument('--layout', choices=('simple', 'varied'), default='varied')
    start.add_argument('--spacing', type=float, default=150.)
    start.add_argument('--samples', type=int, default=20000)
    start.add_argument('--dt', type=float, default=.5)
    start.add_argument('--snapshot', type=float, default=5.)
    start.add_argument('--width', type=int, default=1200)
    for command in (start, commands.add_parser('resume')):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--duration', type=float, default=1000.)
    render = commands.add_parser('render')
    render.add_argument('--checkpoint', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    render.add_argument('--width', type=int, default=4096)
    render.add_argument('--boundaries', action='store_true')
    args = parser.parse_args()
    if args.command == 'render':
        world, _ = load_checkpoint(args.checkpoint)
        if args.width < 256:
            parser.error('Map width must be at least 256.')
        print(json.dumps(artist.draw(world, args.output, args.width, args.boundaries)), flush=True)
        return
    if args.command == 'start':
        if args.dt <= 0. or args.snapshot <= 0. or args.width < 256:
            parser.error('Positive dt/snapshot and map width >= 256 are required.')
        if not math.isclose(args.snapshot / args.dt, round(args.snapshot / args.dt), abs_tol=1e-8):
            parser.error('Snapshot cadence must be a whole number of timesteps.')
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('Output is not empty; use resume or a new directory.')
        args.output.mkdir(parents=True, exist_ok=True)
        world, rotations = artist.make_world(args.seed, args.plates, args.land, args.continents, args.spacing,
                                             layout=args.layout)
        settings = dict(seed=args.seed, plates=args.plates, land_fraction=args.land,
                        continental_fraction=args.continents, spacing_km=args.spacing,
                        snapshot_myr=args.snapshot, width=args.width, layout=args.layout,
                        initial_contacts='local_crust')
        config = dict(dt_myr=args.dt, motion='forces', parameters=asdict(dynamics.Parameters(sample_points=args.samples)),
                      rifting=asdict(dynamics.Rifting()), rotations_per_myr=rotations)
        metadata = dict(artist=settings, config=config,
                        progress=dict(completed_intervals=0, remaining_myr=0.), provenance=provenance())
    else:
        world, metadata = load_checkpoint(args.output / 'checkpoint.json')
        if set(metadata) != {'artist', 'config', 'progress', 'provenance'}:
            parser.error('This is not an artist run checkpoint.')
        if metadata['provenance'] != provenance():
            parser.error('Source or numerical environment changed; resume with the recorded version.')
    _execute(args.output, world, metadata, args.duration)


if __name__ == '__main__':
    main()
