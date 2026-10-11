# Stage 3 follow-up (2026-10-11): closing the "Left to close" numbers

Follow-up to `stage3_validation_2026-10-10/`: three of the open numbers were run out. Each is still a comparison of the warpSPH port against the boundaries repo's oracle (`DFSPH2D`) **on the identical lattice**, per the plan's rule. Raw lines: `results_extract.txt`; full logs in this folder. The fourth open number (the wake Re 100 lift-amplitude resolution check, D/dx = 20) was still running when this was written; its result lands in the follow-up commit.

## 1. The outer-torque oracle number (Taylor–Couette) — resolved

The 2026-10-10 validation never stored the oracle's **outer** wall torque. Running `DFSPH2D` on the warpSPH `taylorCouette` cut lattice (N = 1384, H = 2.505 dx, closed preset, the same construction `probe_taylorCouette.py` runs) books it:

| | profile / exact | inner torque / exact | outer torque / exact |
|---|---|---|---|
| warpSPH omni, closed preset (2026-10-10) | 0.9994 | 0.9733 | 0.8878 |
| oracle `DFSPH2D`, same lattice, auto cal | 0.9990 | 0.9732 | **0.8867** |
| oracle `DFSPH2D`, same lattice, morris cal 0.9431 | 0.9974 | 0.9717 | 0.8809 |

The ~0.89 outer-torque deficit is **a lattice / closure artifact the oracle shares** (0.8867 / 0.8809 vs the port's 0.8878, within 0.1 %), not a port bug — the curved no-slip closure is not exactly torque-conserving on the cut lattice, exactly as the δ⁺ outer torque is 0.762 against the oracle's 0.766. The inner torque and the profile match to < 0.1 %.

## 2. The omni square array at t = 60 — steady, not falling

The 2026-10-10 t = 20 value (K/K_ref 0.9863) was read as "still falling." Running to t = 60:

| | K / K_ref (25.91) | load / body force | history at t/4, t/2, t |
|---|---|---|---|
| warpSPH omni, square a = 1/3, t = 20 (2026-10-10) | 0.9863 | 0.9991 | 1.048, 0.996, 0.988 |
| warpSPH omni, square a = 1/3, **t = 60** | **0.9856** | **1.0000** | 0.986, 0.989, 0.985 |

It is **steady** (0.986 at t = 20 and t = 60, the momentum balance exactly 1.0): the ~1.4 % K deficit is real, not a transient. Caveat: the oracle's square is **0.6 % smaller** (side 0.331174, c = 0.109676 vs 1/3; N = 2048 both), so the K comparison is muddied by the size mismatch — the oracle's own K/K_ref is 0.9920 and its balance 1.0012. The physics check (the momentum balance) matches; the K gap is mostly the 0.6 % size difference, not the scheme.

## 3. The divergenceFree disk deficit is a scheme effect, not the support

The 2026-10-10 comparison (port K/K_ref 0.991 / balance 0.982 vs the oracle's 1.0034 / 0.9986) was **confounded by the support**: the port's disk probe never set `n_h`, so it ran at the case default **4.0 dx**, while the oracle uses **2.505 dx** (`dx / PACKING`). Running the port's `divergenceFree` disk at the oracle's support (n_h = 2.505, nu re-calibrated on that lattice):

| setup, support | balance (t = 20..60) | K / K_ref (t = 20..60) | behaviour |
|---|---|---|---|
| warpSPH divergenceFree, **n_h = 4.0** (2026-10-10 support) | ~0.985 (0.979–0.999) | ~0.989 (0.975–1.000) | small persistent oscillation, period ~10–15 |
| warpSPH divergenceFree, **n_h = 2.505** (the oracle's) | **~0.885** (0.85–1.06) | **~0.95** (0.90–1.15) | large persistent oscillation, not settling |
| oracle `DFSPH2D`, n_h = 2.505 (2026-10-10) | 0.9986 | 1.0034 | smooth, settled |

At the **same support** the port's `divergenceFree` is clearly below the oracle (balance 0.885 vs 0.9986, K ~0.95 vs 1.0034) with a persistent oscillation the oracle does not have: the deficit is a **scheme effect** (the port's `divergenceFree` vs `DFSPH2D`), not a support mismatch. Two things to note: (a) the port is actually *better* at n_h = 4.0 than at the oracle's 2.505 — a support-dependent issue in the port's `divergenceFree` worth a separate look; (b) the oscillation is persistent (it does not damp from t = 10 to 60), so a longer run will not settle it.

## 4. The wake Re 100 lift-amplitude resolution check — done: a resolution effect

`probe_cylinderWake.py --Re 100 --res 20 --time 150` (D/dx = 20, delta+, the graph + compiled closure on, N = 179684, t = 150):

| Re 100 | C_D | C_L amplitude | St | L_w / D |
|---|---|---|---|---|
| warpSPH res = 10 (2026-10-10) | 1.520 | 0.426 | 0.1705 | (unsteady) |
| **warpSPH res = 20 (this)** | **1.4252** | **0.3198** | **0.1739** | **1.785** |
| oracle res = 10 (2026-10-10) | 1.5278 | 0.3755 | 0.1717 | 1.878 |
| oracle res = 20 | – | 0.333 | – | – |
| unconfined literature | 1.33–1.35 | 0.32–0.34 | 0.164–0.166 | 2.2–2.35 |

The Re 100 lift-amplitude gap was a **resolution effect**: at D/dx = 20 the warpSPH C_L amplitude drops from 0.426 (res = 10) to **0.3198**, converging to the oracle's own res = 20 (0.333) and the low end of the unconfined literature (0.32–0.34); the res = 10 value was under-resolved. C_D also drops (1.520 → 1.4252) toward the literature (1.33–1.35), and St = 0.1739 is close to the literature (0.164–0.166), slightly high. The +13 % res = 10 lift gap is closed — it was the lattice, not the scheme. Raw lines: `wake_Re100_res20.clean.txt`.
