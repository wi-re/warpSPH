"""Extract published Fig23's 2D vector trace, then overlay saved diagnostics.

Run extract with pdfplumber available; run plot with numpy/matplotlib available.
The extracted curve is figure geometry, not the authors' original numerical data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def extract(pdf, output):
    import pdfplumber
    with pdfplumber.open(pdf) as document:
        page = document.pages[33]  # Published page193, Fig23.
        axes = [r for r in page.rects if 120 < r['x0'] < 125
                and 70 < r['top'] < 80 and 145 < r['width'] < 155
                and 145 < r['height'] < 155][0]
        candidates = [c for c in page.curves
                      if c.get('stroking_color') == (0., .502, 0.)
                      and c['x0'] < 200 and c['width'] > 140]
        assert len(candidates) == 1
        curve = candidates[0]
        points = []
        previous = None
        for item in curve['path']:
            op, *coords = item
            if op in ('m', 'l'):
                previous = coords[0]
                points.append(previous)
            elif op == 'c':
                start = previous
                c1, c2, end = coords
                for i in range(1, 41):
                    t = i / 40
                    points.append(tuple((1-t)**3*start[j] + 3*(1-t)**2*t*c1[j]
                                        + 3*(1-t)*t*t*c2[j] + t**3*end[j]
                                        for j in range(2)))
                previous = end
            else:
                raise ValueError('Unexpected path command: ' + op)
        x0, x1, top, bottom = (axes[k] for k in ('x0', 'x1', 'top', 'bottom'))
        data = [(8*(x-x0)/(x1-x0), (bottom-y)/(bottom-top)) for x, y in points]
        assert all(b[0] >= a[0] for a, b in zip(data, data[1:]))
        assert abs(data[-1][0]-8) < .001 and abs(data[-1][1]) < 1.e-6
        output.mkdir(parents=True, exist_ok=True)
        with (output/'paperMassFraction.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_over_tau_KH', 'cold_dense_mass_over_initial_cloud_mass'])
            writer.writerows(data)
        metadata = dict(source=str(pdf.resolve()), sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        doi='10.1016/j.jcp.2016.12.004', figure=23, panel='2D, green CRKSPH',
                        pdfPage=34, printedPage=193, axesPDFPoints=dict(x0=x0, x1=x1, top=top, bottom=bottom),
                        axisRanges=dict(timeKH=[0, 8], massFraction=[0, 1]),
                        method='PDF vector path; cubic Bezier segments sampled at40 equal parameter intervals; no time shift, rescaling, extrapolation or curve fitting.',
                        caveat='Published curve geometry, not underlying author simulation data. Tiny negative values at zero are retained PDF coordinate rounding.',
                        vectorPath=curve['path'], samples=len(data))
        (output/'paperMassFractionExtraction.json').write_text(json.dumps(metadata, indent=2)+'\n')


def plot(raw, output):
    import numpy as np
    import matplotlib.pyplot as plt
    history = json.loads((raw/'diagnosticHistory.json').read_text())
    kh = json.loads((raw/'settings.json').read_text())['tKH']
    t = np.array([v['tKH'] for v in history])
    mass = np.array([v['cloudMassFraction'] for v in history])
    reference = np.genfromtxt(output/'paperMassFraction.csv', delimiter=',', skip_header=1)
    plt.rcParams.update({'font.size': 11, 'axes.spines.right': False})
    for filename, limit in [('massFractionOverlay', 4.3), ('massFractionOverlayFull', 8.15)]:
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        ax.plot(reference[:, 0], reference[:, 1], color='#008000', lw=1.8,
                label='Published CRKSPH, Fig23 (vector trace)')
        ax.plot(t, mass, color='#2166ac', lw=1.7, label='WarpSPH, 512 × 128 ambient')
        ax.scatter([t[-1]], [mass[-1]], color='#2166ac', s=22, zorder=3)
        ax.set(xlim=(0, limit), ylim=(0, 1.03), xlabel=r'Time $t / \tau_{\mathrm{KH}}$',
               ylabel=r'Cold, dense mass $M(t)/M_0$', title='2D blob: WarpSPH and published CRKSPH')
        ax.legend(frameon=False, loc='upper right')
        ax.grid(alpha=.2)
        sec = ax.secondary_xaxis('top', functions=(lambda x: x*kh, lambda x: x/kh))
        sec.set_xlabel('Physical simulation time')
        ax.annotate(f'Warp stopped at {t[-1]:.3f} KH', xy=(t[-1], mass[-1]),
                    xytext=(2.65 if limit < 5 else 4.5, .14), fontsize=10,
                    arrowprops={'arrowstyle': '-', 'color': '#555'})
        fig.text(.5, -.022, 'Reference extracted from published PDF vector paths; not original numerical data. No time alignment applied.',
                 ha='center', fontsize=9)
        for extension in ('png', 'svg'):
            fig.savefig(output/(filename+'.'+extension), dpi=190, bbox_inches='tight')
        plt.close(fig)
    comparison = {str(x): dict(paperFigure=float(np.interp(x, reference[:, 0], reference[:, 1])),
                               warp=float(np.interp(x, t, mass))) for x in (.5, 1., 2., 3., 4.)}
    (output/'paperOverlayValues.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('extract', 'plot'))
    parser.add_argument('--pdf', type=Path, default=Path('literature/frontiere2017_crksph.pdf'))
    parser.add_argument('--raw', type=Path, default=Path('results/materialEOS/blob_paper_long_2026-10-21'))
    parser.add_argument('--output', type=Path, default=Path('reports/materialEOS/blobPaperLong'))
    args = parser.parse_args()
    if args.mode == 'extract':
        extract(args.pdf, args.output)
    else:
        plot(args.raw, args.output)
