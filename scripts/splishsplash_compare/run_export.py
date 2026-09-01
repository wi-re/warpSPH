"""Clean SPlisHSPlasH run (no callback) with VTK export; parse the frames
afterward to get the |v|max / density time series without touching live sim
objects (which segfault after base.run())."""
import sys, glob, os, struct, numpy as np

SCENE = sys.argv[1]
T_STOP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "/tmp/splish_out"

import pysplishsplash as sph
base = sph.Exec.SimulatorBase()
base.init(sceneFile=SCENE, useGui=False, initialPause=False, useCache=True,
          outputDir=OUTDIR)
base.setValueFloat(base.STOP_AT, T_STOP)
base.run()
print("run finished; parsing", flush=True)

# --- parse VTK frames (legacy binary or XML) written under OUTDIR ---
def read_vtk_points_vel(path):
    with open(path, "rb") as f:
        raw = f.read()
    # legacy VTK: "POINTS n float" then big-endian float32 xyz
    i = raw.find(b"POINTS ")
    j = raw.find(b"\n", i)
    npts = int(raw[i + 7:j].split()[0])
    start = j + 1
    pts = np.frombuffer(raw[start:start + npts * 12], dtype=">f4").reshape(-1, 3).astype(np.float64)
    v = None
    k = raw.find(b"velocity")
    if k != -1:
        kk = raw.find(b"\n", k) + 1
        v = np.frombuffer(raw[kk:kk + npts * 12], dtype=">f4").reshape(-1, 3).astype(np.float64)
    return pts, v

frames = sorted(glob.glob(os.path.join(OUTDIR, "**", "*.vtk"), recursive=True))
print(f"{len(frames)} vtk frames under {OUTDIR}", flush=True)
for fp in frames[:: max(1, len(frames) // 15)] + (frames[-1:] if frames else []):
    try:
        p, v = read_vtk_points_vel(fp)
        vm = np.linalg.norm(v, axis=1).max() if v is not None else float("nan")
        print(f"  {os.path.basename(fp):40s} n={len(p):5d} "
              f"x[{p[:,0].min():.3f},{p[:,0].max():.3f}] y[{p[:,1].min():.3f},{p[:,1].max():.3f}] "
              f"|v|max={vm:.4g}", flush=True)
    except Exception as e:
        print("  parse fail", fp, e, flush=True)
