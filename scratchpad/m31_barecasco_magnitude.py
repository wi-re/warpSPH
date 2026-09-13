import h5py, numpy as np, glob, os, re

DIG = '/home/lu26029/dev/warpSPH/scratchpad/m31_dig/3-dambreak_2026-09-13_08-02-37/trajectory'

files = sorted(glob.glob(os.path.join(DIG, 'state_*.h5')),
               key=lambda f: int(re.search(r'state_(\d+)\.h5', f).group(1)))

def loadFrame(f):
    with h5py.File(f, 'r') as h:
        t = h.attrs['time']
        uid = h['state/UIDs'][:]
        p = h['state/pressures'][:]
        fs = h['state/surfaceIndicators'][:]
        pos = h['state/positions'][:]
        kinds = h['state/kinds'][:]
        supports = h['state/supports'][:]
        ai = h['adjacency/i'][:]
        aj = h['adjacency/j'][:]
        qpos = h['adjacency/queryPositions'][:]
        rpos = h['adjacency/referencePositions'][:]
        edgeOffsets = h['adjacency/edgeOffsets'][:]
        numNeighbors = h['adjacency/numNeighbors'][:]
    return dict(t=t, uid=uid, p=p, fs=fs, pos=pos, kinds=kinds, supports=supports,
                ai=ai, aj=aj, qpos=qpos, rpos=rpos, edgeOffsets=edgeOffsets, numNeighbors=numNeighbors)

def coverVectorMagnitudes(fr):
    """Recompute the raw (unnormalized) Barecasco cover vector magnitude per query particle
    from the stored adjacency + positions, matching wp_barecasco.py's n_ij = x_ij/(r_ij+eps*h)."""
    n = fr['qpos'].shape[0]
    cov = np.zeros((n, fr['qpos'].shape[1]), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    ai, aj = fr['ai'], fr['aj']
    xij = fr['qpos'][ai] - fr['rpos'][aj]
    r = np.linalg.norm(xij, axis=-1)
    valid = ai != aj  # exclude self-pair (r=0), matches kernel's w_ij>0 support-radius filter approximately
    hi = fr['supports'][ai]
    nij = xij / (r[:, None] + 1e-14 * hi[:, None] + 1e-300)
    np.add.at(cov, ai[valid], nij[valid])
    np.add.at(cnt, ai[valid], 1)
    mag = np.linalg.norm(cov, axis=-1)
    return mag, cnt

# ---- Correlate cross-frame switch flips with cover-vector magnitude ----
print("=== Barecasco raw cover-vector magnitude vs. Antuono-switch flip (cross-frame) ===")
prev = None
allFlipMag = []
allStableMag = []
for f in files:
    fr = loadFrame(f)
    mag, cnt = coverVectorMagnitudes(fr)
    order = np.argsort(fr['uid'])
    uid_s = fr['uid'][order]; p_s = fr['p'][order]; fs_s = fr['fs'][order]
    kinds_s = fr['kinds'][order]; mag_s = mag[order]; cnt_s = cnt[order]
    if prev is not None:
        pt, puid, pp, pfs, pkinds, pmag, pcnt = prev
        common, ia, ib = np.intersect1d(puid, uid_s, return_indices=True)
        switchPrev = (pp[ia] >= 0.0) | (pfs[ia] >= 1)
        switchNow  = (p_s[ib] >= 0.0) | (fs_s[ib] >= 1)
        flipped = switchPrev != switchNow
        fluidHere = kinds_s[ib] == 0
        # relative magnitude: cover-vector magnitude as a fraction of neighbor count (isotropy measure)
        relMag = mag_s[ib] / np.maximum(cnt_s[ib], 1)
        f_flipped = fluidHere & flipped
        f_stable  = fluidHere & (~flipped)
        allFlipMag.append(relMag[f_flipped])
        allStableMag.append(relMag[f_stable])
    prev = (fr['t'], uid_s, p_s, fs_s, kinds_s, mag_s, cnt_s)

flipMag = np.concatenate(allFlipMag)
stableMag = np.concatenate(allStableMag)
print(f"n flipped={flipMag.size}  n stable={stableMag.size}")
for name, arr in [("FLIPPED", flipMag), ("STABLE", stableMag)]:
    print(f"{name}: relMag (|coverVector|/nNeighbors) mean={arr.mean():.4f} median={np.median(arr):.4f} "
          f"p10={np.percentile(arr,10):.4f} p90={np.percentile(arr,90):.4f}")

# Also: what fraction of ALL fluid particles ever have a tiny relMag (near-isotropic, "should be bulk")
# vs how many are flagged surface at that same instant
print()
print("=== Does surfaceIndicators==1 correspond to large relMag, or is it noise-driven at small relMag? ===")
relMagsAll = []
fsAll = []
for f in files[::20]:
    fr = loadFrame(f)
    mag, cnt = coverVectorMagnitudes(fr)
    relMag = mag / np.maximum(cnt, 1)
    fluidMask = fr['kinds'] == 0
    relMagsAll.append(relMag[fluidMask])
    fsAll.append(fr['fs'][fluidMask])
relMagsAll = np.concatenate(relMagsAll)
fsAll = np.concatenate(fsAll)
for label, mask in [("fs==1 (flagged surface)", fsAll >= 1), ("fs==0 (flagged bulk)", fsAll < 1)]:
    arr = relMagsAll[mask]
    print(f"{label}: n={arr.size} relMag mean={arr.mean():.4f} median={np.median(arr):.4f} p90={np.percentile(arr,90):.4f}")
