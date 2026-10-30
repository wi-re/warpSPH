#!/usr/bin/env python3
"""Render saved Lobovsky pressure snapshots with a fixed signed-log scale."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .render import read_snapshot
    from .export_viewer import snapshot_clock
except ImportError:
    from render import read_snapshot
    from export_viewer import snapshot_clock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', default='dynGPUSPH FP64')
    ap.add_argument('--bounds', type=float, nargs=4, default=(-.005, 1.615, -.005, 1.05),
                    metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    ap.add_argument('--time', type=float, nargs=2, default=(0., float('inf')))
    ap.add_argument('--size', type=int, nargs=2, default=(2560, 1800))
    ap.add_argument('--fps', type=float, default=12.)
    ap.add_argument('--pressure-limit', type=float, default=8000.)
    args = ap.parse_args()
    if args.fps <= 0 or args.pressure_limit <= 0 or min(args.size) <= 0:
        ap.error('FPS, pressure limit and dimensions must be positive')
    if any(s % 2 for s in args.size):
        ap.error('Video dimensions must be even')
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = sorted((args.run / 'snapshots').glob('*.bin'))
    frames = [(snapshot_clock(p), p) for p in paths]
    frames = [(t, p) for t, p in frames if args.time[0] <= t <= args.time[1]]
    if not frames:
        raise ValueError('No snapshots in the requested time window')
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from vispy import app, scene
    from matplotlib import colormaps
    import imageio.v2 as imageio
    app.use_app('egl')
    canvas = scene.SceneCanvas(size=tuple(args.size), show=False, bgcolor='#030e18')
    grid = canvas.central_widget.add_grid(spacing=8)
    title = scene.Label('', color='white', font_size=18)
    title.height_max = 55
    grid.add_widget(title, row=0, col=0)
    legend = scene.Label(
        f'Pressure: blue = -{args.pressure_limit:g} Pa | white = 0 | red = +{args.pressure_limit:g} Pa'
        f' | signed log(1+|p|), clipped | saved snapshots, {args.fps:g} playback fps',
        color='white', font_size=12)
    legend.height_max = 35
    grid.add_widget(legend, row=1, col=0)
    view = grid.add_view(row=2, col=0, border_color='#53616a')
    view.camera = scene.PanZoomCamera(aspect=1)
    xmin, xmax, ymin, ymax = args.bounds
    view.camera.set_range(x=(xmin, xmax), y=(ymin, ymax), margin=0.)
    marker = scene.visuals.Markers(parent=view.scene, antialias=1)
    # Coplanar overlapping sprites must blend, including their soft edges.
    marker.set_gl_state('translucent', depth_test=False)
    cmap = colormaps['coolwarm']
    # Slight overlap prevents subpixel lattice aliasing from looking like holes.
    pixels_per_m = min(args.size[0] / (xmax-xmin), (args.size[1]-105) / (ymax-ymin))
    diameter = max(2., 1.3 * .00075 * pixels_per_m)
    receipt = dict(source=str(args.run.resolve()), label=args.label,
        bounds=args.bounds, pressureRangePa=[-args.pressure_limit, args.pressure_limit],
        normalization='sign(p)*log1p(abs(p))', size=args.size, fps=args.fps,
        interpolation=False, frames=[])
    with imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=None,
            macro_block_size=2, ffmpeg_params=['-crf', '17', '-preset', 'medium',
                                              '-movflags', '+faststart']) as writer:
        for i, (t, path) in enumerate(frames):
            data = read_snapshot(path)
            p = data['p']
            value = .5 + .5*np.sign(p)*np.log1p(abs(p))/np.log1p(args.pressure_limit)
            colors = cmap(np.clip(value, 0., 1.)).astype('f4')
            colors[data['bc'] != 1] = [.4, .47, .53, 1.]
            marker.set_data(data['x'].astype('f4'), face_color=colors,
                            edge_width=0, size=diameter)
            title.text = f'{args.label} | Lobovsky | t = {t:.6f} s | snapshot {i+1}/{len(frames)}'
            rendered = canvas.render()
            writer.append_data(rendered)
            receipt['frames'].append(dict(frame=i, time=t, source=str(path.resolve())))
            if i in (0, len(frames)-1) or abs(t-1.44) < .005:
                imageio.imwrite(args.out.with_name(f'{args.out.stem}-frame-{i:04d}.png'), rendered)
            if i % 20 == 0 or i == len(frames)-1:
                print(f'RENDER {i+1}/{len(frames)} t={t:.6f} {args.out.name}', flush=True)
    canvas.close()
    args.out.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
