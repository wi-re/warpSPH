import h5py, numpy as np, glob, os, re

DIG = '/home/lu26029/dev/warpSPH/scratchpad/m31_dig/3-dambreak_2026-09-13_08-02-37/trajectory'

files = sorted(glob.glob(os.path.join(DIG, 'state_*.h5')),
               key=lambda f: int(re.search(r'state_(\d+)\.h5', f).group(1)))

def load(f):
    with h5py.File(f, 'r') as h:
        t = h.attrs['time']
        uid = h['state/UIDs'][:]
        p = h['state/pressures'][:]
        fs = h['state/surfaceIndicators'][:]
        pos = h['state/positions'][:]
        kinds = h['state/kinds'][:]
        stages = []
        for k in range(4):
            gp = f'stages/stage_{k}/aux/1_state'
            su = h[gp + '/UIDs'][:]
            sp = h[gp + '/pressures'][:]
            sfs = h[gp + '/surfaceIndicators'][:]
            stages.append((su, sp, sfs))
    return t, uid, p, fs, pos, kinds, stages

# ---- Part A: intra-step (RK4 sub-stage) switch chatter, one committed step per file ----
print("=== Part A: intra-step (single real timestep, RK4 sub-stages) switch chatter ===")
print(f"{'file':>14} {'t':>8} {'nFluidFlip>=1':>14} {'nFluidFlip>=2':>14} {'nFluid':>8} {'fracFlip>=1':>12}")
rows = []
for f in files:
    t, uid, p, fs, pos, kinds, stages = load(f)
    fluidMask = kinds == 0
    nFluid = fluidMask.sum()
    # switch per stage, aligned by UID (assume same ordering across stages == state order; verify once)
    switchStages = []
    for (su, sp, sfs) in stages:
        assert np.array_equal(su, uid), "UID order differs between stage and committed state!"
        sw = (sp >= 0.0) | (sfs >= 1)
        switchStages.append(sw)
    switchStages = np.stack(switchStages, axis=0)  # (4, N)
    # number of times the switch value changes across the 4 stages, per particle
    nChanges = np.sum(switchStages[1:] != switchStages[:-1], axis=0)  # (N,)
    nFlip1 = np.sum((nChanges >= 1) & fluidMask)
    nFlip2 = np.sum((nChanges >= 2) & fluidMask)
    rows.append((os.path.basename(f), t, nFlip1, nFlip2, nFluid))

for name, t, n1, n2, nf in rows:
    print(f"{name:>14} {t:8.3f} {n1:14d} {n2:14d} {nf:8d} {n1/max(nf,1)*100:11.2f}%")

# ---- Part B: across consecutive SAVED frames (~100 steps apart), does the committed switch value
#      change for the same UID, and does that correlate with a big pressure jump? ----
print()
print("=== Part B: cross-frame (100-step) committed switch flips vs pressure jumps, by particle ===")
prev = None
for f in files:
    t, uid, p, fs, pos, kinds, stages = load(f)
    order = np.argsort(uid)
    uid_s, p_s, fs_s, kinds_s, pos_s = uid[order], p[order], fs[order], kinds[order], pos[order]
    if prev is not None:
        pt, puid, pp, pfs, pkinds, ppos = prev
        # intersect UIDs (particle count should be constant, but be safe)
        common, ia, ib = np.intersect1d(puid, uid_s, return_indices=True)
        dp = p_s[ib] - pp[ia]
        switchPrev = (pp[ia] >= 0.0) | (pfs[ia] >= 1)
        switchNow  = (p_s[ib] >= 0.0) | (fs_s[ib] >= 1)
        flipped = switchPrev != switchNow
        fluidHere = kinds_s[ib] == 0
        bigJump = np.abs(dp) > 50.0  # arbitrary "anomalous" pressure jump threshold in this run's units
        nBig = np.sum(bigJump & fluidHere)
        nBigAndFlip = np.sum(bigJump & flipped & fluidHere)
        nFlip = np.sum(flipped & fluidHere)
        nFluid = np.sum(fluidHere)
        if nBig > 0 or nFlip > 200:
            print(f"{os.path.basename(f):>14} t={t:7.3f}  nFluid={nFluid:6d}  nFlip={nFlip:6d} ({nFlip/nFluid*100:5.1f}%)  "
                  f"nBigJump(|dP|>50)={nBig:5d}  nBigJump&Flip={nBigAndFlip:5d}  "
                  f"P(flip|bigJump)={ (nBigAndFlip/nBig*100 if nBig else float('nan')):5.1f}%  "
                  f"P(flip|~bigJump)={ ((nFlip-nBigAndFlip)/max(nFluid-nBig,1)*100):5.1f}%")
    prev = (t, uid_s, p_s, fs_s, kinds_s, pos_s)
