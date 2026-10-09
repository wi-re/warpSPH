"""CEILING_STICKING_PLAN.md: read ceiling_run.py's per-step trace and list
(a) riders: fluid rows staying within `--near` dx of the ceiling for a long
time, and (b) kicks: the largest per-step speed changes of rows near the
ceiling. Prints per-UID summaries; checkpoint to resume from = the last
stored state before the event."""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('trace')
ap.add_argument('--near', type=float, default=1.5)
ap.add_argument('--top', type=int, default=15)
ap.add_argument('--uid', type=int, nargs='*', default=None)
a = ap.parse_args()

d = np.load(a.trace)
dx, yC = float(d['dx']), float(d['yCeil'])
uid, step, t = d['uid'], d['step'], d['t']
y, vx, vy, rho, p, fs = d['y'], d['vx'], d['vy'], d['rho'], d['p'], d['fs']
s = (yC - y) / dx          # distance below the ceiling in dx
spd = np.hypot(vx, vy)
tstar = t * np.sqrt(9.81 / 0.6)

if a.uid:
    for u in a.uid:
        m = uid == u
        o = np.argsort(step[m])
        print(f'--- uid {u}')
        for k in o[::max(1, m.sum() // 60)]:
            i = np.nonzero(m)[0][k]
            print(f'  step {step[i]:6d} t* {tstar[i]:.3f} s {s[i]:5.2f}dx x {d["x"][i]:+.4f} '
                  f'v ({vx[i]:+6.2f},{vy[i]:+6.2f}) rho {rho[i]:.4f} p {p[i]:+8.2f} fs {fs[i]}')
    raise SystemExit

near = s < a.near
print(f'dx={dx:.5f} yCeil={yC:.5f}; rows near ceiling (<{a.near}dx): {near.sum()} over {np.unique(step[near]).size} steps')
# riders
rows = []
for u in np.unique(uid[near]):
    m = near & (uid == u)
    st = np.sort(step[m])
    # longest run of consecutive steps
    br = np.nonzero(np.diff(st) > 1)[0]
    starts = np.r_[st[0], st[br + 1]]; ends = np.r_[st[br], st[-1]]
    L = ends - starts + 1; j = np.argmax(L)
    mm = m & (step >= starts[j]) & (step <= ends[j])
    rows.append((L[j], u, tstar[mm].min(), tstar[mm].max(), np.mean(vy[mm]), np.mean(np.abs(vx[mm])),
                 rho[mm].min(), rho[mm].max(), p[mm].min(), p[mm].max(), np.mean(fs[mm]), starts[j], ends[j]))
rows.sort(reverse=True)
print('\nlongest ceiling residences: steps uid  t*[a,b]  <vy> <|vx|>  rho[min,max]  p[min,max]  fsFrac  steps')
for r in rows[:a.top]:
    print(f'  {r[0]:6d} {r[1]:6d}  [{r[2]:.2f},{r[3]:.2f}]  {r[4]:+.3f} {r[5]:.2f}  '
          f'[{r[6]:.4f},{r[7]:.4f}]  [{r[8]:+7.1f},{r[9]:+7.1f}]  {r[10]:.2f}  {r[11]}-{r[12]}')

# kicks: per-uid consecutive-step speed jump among rows within 4 dx
o = np.lexsort((step, uid))
u2, st2, sp2, s2 = uid[o], step[o], spd[o], s[o]
same = (u2[1:] == u2[:-1]) & (st2[1:] == st2[:-1] + 1)
dv = np.where(same, np.abs(sp2[1:] - sp2[:-1]), 0)
dvv = np.where(same, np.hypot(vx[o][1:] - vx[o][:-1], vy[o][1:] - vy[o][:-1]), 0)
idx = np.argsort(dvv)[::-1]
print('\nlargest per-step velocity jumps |dv| (rows in the trace):  uid step t* s(dx) |v| before->after')
seen = set()
for k in idx:
    if len(seen) >= a.top: break
    key = (u2[k + 1], st2[k + 1] // 200)
    if key in seen: continue
    seen.add(key)
    print(f'  {u2[k+1]:6d} {st2[k+1]:6d} {tstar[o][k+1]:.3f} {s2[k+1]:5.2f}  |dv|={dvv[k]:.3f}  {sp2[k]:.2f}->{sp2[k+1]:.2f}  rho {rho[o][k]:.4f}->{rho[o][k+1]:.4f}')
