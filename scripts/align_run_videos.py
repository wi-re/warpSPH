"""Time-align several run videos (frame k = step k*interval, t from each run's record.json)
and stack them. usage: align.py out.mp4 dtVideo interval label:dir [label:dir ...]"""
import glob, json, os, subprocess, sys, tempfile
out, dtv, interval = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
runs = []
for arg in sys.argv[4:]:
    label, d = arg.split(':', 1)
    vid = sorted(glob.glob(os.path.join(d, '*', 'output.mp4')))[-1]
    ts = [r['t'] for r in json.load(open(os.path.join(d, 'record.json')))]
    n = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                     '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', vid]))
    tf = [ts[min(k * interval, len(ts) - 1)] for k in range(n)]
    runs.append((label, vid, tf))
tEnd = max(tf[-1] for _, _, tf in runs)
grid = [i * dtv for i in range(int(tEnd / dtv) + 1)]
tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)))
streams = []
for r, (label, vid, tf) in enumerate(runs):
    fdir = os.path.join(tmp, f'r{r}'); os.makedirs(fdir)
    subprocess.check_call(['ffmpeg', '-v', 'error', '-i', vid, '-vf', 'scale=1600:-2', os.path.join(fdir, '%06d.png')])
    seq = os.path.join(tmp, f's{r}'); os.makedirs(seq)
    for i, t in enumerate(grid):
        k = min(range(len(tf)), key=lambda k: abs(tf[k] - t)) if t <= tf[-1] else len(tf) - 1
        os.symlink(os.path.join(fdir, f'{k + 1:06d}.png'), os.path.join(seq, f'{i:06d}.png'))
    ended = ' (ended t=%.2f)' % tf[-1] if tf[-1] < tEnd - dtv else ''
    streams.append((seq, label + ended))
inputs, filt = [], []
for r, (seq, label) in enumerate(streams):
    inputs += ['-framerate', '25', '-i', os.path.join(seq, '%06d.png')]
    filt.append(f"[{r}:v]drawtext=text='{label}':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6[v{r}]")
filt.append(''.join(f'[v{r}]' for r in range(len(streams))) + f'vstack=inputs={len(streams)}[v]')
subprocess.check_call(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex', ';'.join(filt), '-map', '[v]',
                       '-c:v', 'libx264', '-crf', '24', '-preset', 'slow', '-pix_fmt', 'yuv420p', out])
subprocess.call(['rm', '-rf', tmp])
print(out, len(grid), 'frames, t 0 ..', tEnd)
