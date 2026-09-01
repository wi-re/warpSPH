"""Side-by-side of the hydrostatic-column INITIAL STATE in the two libraries,
plus the SPlisHSPlasH DFSPH |v|max/density time series parsed from its VTK
export."""
import glob, os, re, json, numpy as np

HERE = os.path.dirname(__file__)
sp = np.load(os.path.join(HERE, "splish_initial.npz"), allow_pickle=True)
wp = np.load(os.path.join(HERE, "warpsph_initial.npz"), allow_pickle=True)
spi = json.loads(str(sp["info"]))
wpi = json.loads(str(wp["info"]))

sp_pos, sp_m = sp["positions"][:, :2], sp["masses"]
wp_pos, wp_m = wp["positions"][:, :2], wp["masses"]

def spacing(pos):
    # nearest-neighbour distance median (robust to jitter)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pos).query(pos, k=2)
    return float(np.median(d[:, 1]))

try:
    sp_dx, wp_dx = spacing(sp_pos), spacing(wp_pos)
except Exception:
    sp_dx = 2 * spi["particleRadius"]
    wp_dx = wpi["dx"]

rows = [
    ("fluid particle count",      spi["nFluid"],                wpi["nFluid"]),
    ("particle spacing dx",       f"{sp_dx:.6f}",               f"{wp_dx:.6f}"),
    ("rest density rho0",         spi["density0"],              wpi["rho0"]),
    ("particle mass",             f"{sp_m.mean():.4e}",         f"{wp_m.mean():.4e}"),
    ("  mass / (rho0*dx^2)",      f"{sp_m.mean()/(spi['density0']*sp_dx**2):.4f}",
                                  f"{wp_m.mean()/(wpi['rho0']*wp_dx**2):.4f}"),
    ("kernel support radius h",   f"{spi['supportRadius']:.6f}", f"{wpi['support_h']:.6f}"),
    ("  h / dx",                  f"{spi['supportRadius']/sp_dx:.3f}",
                                  f"{wpi['support_h']/wp_dx:.3f}"),
    ("kernel",                    f"CubicSpline2D (id {spi['kernelId']})", wpi["kernel"]),
    ("W(0)",                      f"{spi.get('Wzero', float('nan')):.2f}", "-"),
    ("neighbours (mid column)",   spi.get("neighbors_midColumn_t0", "?"),
                                  wpi.get("neighbors_midColumn_t0", "?")),
    ("fluid x-extent",            f"[{sp_pos[:,0].min():.3f}, {sp_pos[:,0].max():.3f}]",
                                  f"[{wp_pos[:,0].min():.3f}, {wp_pos[:,0].max():.3f}]"),
    ("fluid y-extent",            f"[{sp_pos[:,1].min():.3f}, {sp_pos[:,1].max():.3f}]",
                                  f"[{wp_pos[:,1].min():.3f}, {wp_pos[:,1].max():.3f}]"),
    ("initial summation density", "(not computed at t0)",
     f"min {wpi['density_t0_min']:.3f}  mean {wpi['density_t0_mean']:.3f}  "
     f"max {wpi['density_t0_max']:.3f}"),
    ("boundary model",            "Bender2019 volume map (mapInvert box)",
     f"{wpi['nBoundary']} particles, 5-layer band"),
    ("gravity",                   9.81, wpi["gravity"]),
]
w = max(len(r[0]) for r in rows)
print("=" * 92)
print("HYDROSTATIC COLUMN -- INITIAL STATE  (matched: L=1 box, dx=1/128, bottom-half fill)")
print("=" * 92)
print(f"{'quantity':<{w}}   {'SPlisHSPlasH (DFSPH ref)':<32}   warpSPH (hydrostaticColumn)")
print("-" * 92)
for name, a, b in rows:
    print(f"{name:<{w}}   {str(a):<32}   {b}")
print("-" * 92)

# --- SPlisHSPlasH VTK time series ---
def read_vtk(path):
    raw = open(path, "rb").read()
    i = raw.find(b"POINTS ")
    j = raw.find(b"\n", i)
    n = int(raw[i + 7:j].split()[0])
    off = j + 1
    p = np.frombuffer(raw[off:off + n * 12], dtype=">f4").reshape(-1, 3).astype(float)
    v = None
    k = raw.find(b"velocity")
    if k != -1:
        kk = raw.find(b"\n", k) + 1
        v = np.frombuffer(raw[kk:kk + n * 12], dtype=">f4").reshape(-1, 3).astype(float)
    d = None
    k = raw.find(b"density")
    if k != -1 and k != raw.find(b"velocity"):
        kk = raw.find(b"\n", k) + 1
        d = np.frombuffer(raw[kk:kk + n * 4], dtype=">f4").astype(float)
    return p, v, d

for outdir, label in [("/tmp/splish_out_long", "t->1.0"), ("/tmp/splish_out", "t->0.3")]:
    fr = glob.glob(os.path.join(outdir, "**", "ParticleData_Fluid_*.vtk"), recursive=True)
    if not fr:
        continue
    fr.sort(key=lambda s: int(re.search(r"_(\d+)\.vtk$", s).group(1)))
    print(f"\nSPlisHSPlasH DFSPH run ({label}, {len(fr)} frames, dataExportFPS=100):")
    print(f"  {'frame':>5} {'~t':>6} {'|v|max':>9} {'|v|mean':>9} {'y_top':>8} "
          f"{'x_in_box':>9}")
    for fp in fr[:: max(1, len(fr) // 18)] + fr[-1:]:
        idx = int(re.search(r"_(\d+)\.vtk$", fp).group(1))
        p, v, d = read_vtk(fp)
        vn = np.linalg.norm(v, axis=1) if v is not None else np.array([np.nan])
        inbox = (np.abs(p[:, 0]) < 0.5) & (p[:, 1] > -0.5) & (p[:, 1] < 0.5)
        print(f"  {idx:>5} {idx/100:>6.2f} {vn.max():>9.4g} {vn.mean():>9.4g} "
              f"{p[:,1].max():>8.3f} {100*inbox.mean():>8.1f}%")
    break
