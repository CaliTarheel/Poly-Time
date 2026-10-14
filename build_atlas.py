"""Create a six-epoch overview image from a saved Deep Time experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server import color_image


def build(run_path, target):
    run_path, target = Path(run_path), Path(target)
    manifest = json.loads((run_path / 'manifest.json').read_text())
    frames = manifest['frames']
    selected = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    image = Image.new('RGB', (1600, 1560), '#101a21')
    draw = ImageDraw.Draw(image)
    large = ImageFont.load_default(size=42)
    normal = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=20)
    draw.text((32, 28), 'DEEP TIME / A WORLD IN MOTION', fill='#e8d9b6', font=large)
    config = manifest['config']
    draw.text((34, 88), f"Seed {config['seed']}  |  {config['width']} x {config['height']} spherical grid  |  elapsed geological time", fill='#92a8b1', font=normal)
    colors = {1: '#77d4dc', 2: '#ef806c', 3: '#c5a1de', 4: '#f4d38a', 5: '#ed9be2', 6: '#ffffff'}
    labels = {1: 'Spreading', 2: 'Subduction', 3: 'Transform', 4: 'Collision', 5: 'Rift', 6: 'Converging, no trench'}
    for slot, index in enumerate(selected):
        metadata = json.loads((run_path / f'frame_{index:04d}.json').read_text())
        with np.load(run_path / f'frame_{index:04d}.npz', allow_pickle=False) as data:
            w, h = metadata['width'], metadata['height']
            terrain = color_image(data['elevation'], w, h).resize((752, 376), Image.Resampling.BILINEAR)
            line = ImageDraw.Draw(terrain)
            owners, boundaries = data['plate'].reshape(h, w), data['boundary'].reshape(h, w)
            sx, sy = 752 / w, 376 / h
            for y in range(h):
                for x in range(w):
                    code = int(boundaries[y, x])
                    if not code:
                        continue
                    if x + 1 < w and owners[y, x] != owners[y, x+1]:
                        line.line(((x+1)*sx, y*sy, (x+1)*sx, (y+1)*sy), fill=colors[code], width=1)
                    if y + 1 < h and owners[y, x] != owners[y+1, x]:
                        line.line((x*sx, (y+1)*sy, (x+1)*sx, (y+1)*sy), fill=colors[code], width=1)
        left, top = 32 + (slot % 2) * 784, 154 + (slot // 2) * 440
        draw.text((left, top), f"{metadata['time_myr']:,.0f} MILLION YEARS", fill='#e8d9b6', font=normal)
        image.paste(terrain, (left, top+38))
    for j, code in enumerate(colors):
        x, y = 35 + j*255, 1498
        draw.line((x, y+13, x+32, y+13), fill=colors[code], width=3)
        draw.text((x+42, y), labels[code], fill='#b4c1c4', font=small)
    draw.text((35, 1532), 'Approximate worldbuilding scenario. Lines indicate sampled tectonic boundaries; terrain detail is limited by the grid.', fill='#92a8b1', font=small)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(build(args.run, args.output))
