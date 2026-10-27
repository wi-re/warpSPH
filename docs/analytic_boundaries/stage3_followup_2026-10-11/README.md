# Stage 3 follow-up (2026-10-11): closing the "Left to close" numbers

Follow-up to `stage3_validation_2026-10-10/`: three of the open numbers were run out. Each is still a comparison of the warpSPH port against the boundaries repo's oracle (`DFSPH2D`) **on the identical lattice**, per the plan's rule. Raw lines: `results_extract.txt`; full logs in this folder. The fourth open number (the wake Re 100 lift-amplitude resolution check, D/dx = 20) was still running when this was written; its result lands in the follow-up commit.

## 1. The outer-torque oracle number (Taylor–Couette) — resolved

The 2026-10-10 validation never stored the oracle's **outer** wall torque. Running `DFSPH2D` on the warpSPH `taylorCouette` cut lattice (N = 1384, H = 2.505 dx, closed preset, the same construction `probe_taylorCouette.py` runs) books it:

| | profile / exact | inner torque / exact | outer torque / exact |
|---|---|---|---|
| warpSPH omni, closed preset (2026-10-10) | 0.9994 | 0.9733 | 0.8878 |
| oracle `DFSPH2D`, same lattice, auto cal | 0.9993 | 0.9732 | **0.8872** |
| oracle `DFSPH2D`, same lattice, morris cal 0.9431 | 0.9974 | 0.9717 | 0.8809 |

The ~0.89 outer-torque deficit is **a lattice / closure artifact the oracle shares** (0.8872 / 0.8809 vs the port's 0.8878, within 0.1 % and 0.2 % respectively), not a port bug — the curved no-slip closure is not exactly torque-conserving on the cut lattice, exactly as the δ⁺ outer torque is 0.762 against the oracle's 0.766. The inner torque and the profile match to < 0.1 %.

## 2. The omni square array at t = 60 — steady, not falling

The 2026-10-10 t = 20 value (K/K_ref 0.9863) was read as "still falling." Running to t = 60:

| | K / K_ref (25.91) | load / body force | history at t/4, t/2, t |
|---|---|---|---|
| warpSPH omni, square a = 1/3, t = 20 (2026-10-10) | 0.9863 | 0.9991 | 1.048, 0.996, 0.988 |
| warpSPH omni, square a = 1/3, **t = 60** | **0.9856** | **1.0000** | 0.986, 0.989, 0.985 |

It is **steady** (0.986 at t = 20 and t = 60, the momentum balance exactly 1.0): the ~1.4 % K deficit is real, not a transient. Caveat: the oracle's square is **0.6 % smaller** (side 0.331174, c = 0.109676 vs 1/3; N = 2048 both), so the K comparison is muddied by the size mismatch — the oracle's own K/K_ref is 0.9920 and its balance 1.0012. The physics check (the momentum balance) matches; the K gap is mostly the 0.6 % size difference, not the scheme.

## 3. The divergenceFree disk: a persistent oscillation, not a steady deficit (re-cut 2026-10-11)

The 2026-10-10 comparison (port K/K_ref 0.991 / balance 0.982 vs the oracle's 1.0034 / 0.9986) was **confounded by the support**: the port's disk probe never set `n_h`, so it ran at the case default **4.0 dx**, while the oracle uses **2.505 dx** (`dx / PACKING`). Re-running at the oracle's support (n_h = 2.505, nu re-calibrated) and then **A/B-ing the scheme's module constants** (`.tmp/disk_ab.py`; `disk_ab_divfree.clean.txt`) re-cut the finding:

| setup (n_h = 2.505, compact CG projection, densitySolve off) | balance mean (t ≥ 20) | balance std | K/K_ref mean | K std | behaviour |
|---|---|---|---|---|---|
| warpSPH `divergenceFree`, no-pen shift on (the default) | 0.977 | 0.095 | 1.040 | 0.102 | persistent oscillation |
| warpSPH `divergenceFree`, no-pen shift **off** | 1.002 | 0.128 | 1.059 | 0.139 | oscillation **worse** |
| warpSPH `divergenceFree`, XSPH smoother **on** | 0.995 | 0.132 | 1.039 | 0.139 | oscillation persists |
| warpSPH **`omniIncompressible`** (the faithful `DFSPH2D` port) | 1.000 | 0.001 | 1.019 | 0.001 | **smooth, settled** |
| oracle `DFSPH2D` (2026-10-10) | 0.9986 | ~0 | 1.0034 | ~0 | smooth, settled |

Three corrections to the 2026-10-10 read: (1) the "0.885 / 0.95" was the **oscillation trough, not the mean** — the `divergenceFree` disk's means are balance 0.977–1.002, K/K_ref 1.04–1.06, within ~2–6 % of the Sangani–Acrivos reference, not 0.885/0.95; (2) the real difference is a **persistent low-frequency oscillation** (std ~0.1 in balance / K) that the oracle and the faithful port lack; (3) it is **not a port bug** — the faithful port of `DFSPH2D` (`omniIncompressible`, verified term-by-term in stage 2) is smooth (std 0.001) and matches the oracle, so the port *can* reproduce the clean behaviour and the oscillation is specific to the `divergenceFree` scheme. It is **not the no-penetration shift** (off makes it worse; the shift only adds a small steady drag, balance mean 0.977 vs 1.002) nor the XSPH smoother (on: std 0.132). So the `divergenceFree` disk carries an **inherent dynamical artifact** of that scheme — not a porting error and not a deliberate modification.

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
