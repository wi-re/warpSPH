import sys, json, time, numpy as np, pysplishsplash as sph
SCENE = "/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/hydrostatic_2d.json"
T_STOP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
OUT = "/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/splish_initial.npz"
LOG = "/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/splish_result.txt"
lf = open(LOG, "w", buffering=1)
def log(*a): print(*a, file=lf); lf.flush()

base = sph.Exec.SimulatorBase()
base.init(sceneFile=SCENE, useGui=False, initialPause=False, useCache=True)
base.initSimulation()
sim = sph.Simulation.getCurrent(); fluid = sim.getFluidModel(0)
tm = sph.TimeManager.getCurrent(); n = fluid.numActiveParticles()
def fb(x): return np.array(fluid.getFieldBuffer(x), copy=False)[:n]
pos0, vel0, den0 = fb("position").copy(), fb("velocity").copy(), fb("density").copy()
masses = np.array([fluid.getMass(i) for i in range(n)])
info = dict(nFluid=int(n), density0=float(fluid.getDensity0()),
           particleRadius=float(sim.getParticleRadius()),
           supportRadius=float(sim.getSupportRadius()),
           dt0=float(tm.getTimeStepSize()), kernelId=int(sim.getKernel()),
           Wzero=float(sim.W_zero()))
log("INIT " + json.dumps(info))
log(f"pos bbox x[{pos0[:,0].min():.6f},{pos0[:,0].max():.6f}] y[{pos0[:,1].min():.6f},{pos0[:,1].max():.6f}]")
log(f"mass mean {masses.mean():.6g}  density(t0) mean {den0.mean():.5f} min {den0.min():.5f} max {den0.max():.5f}")
np.savez(OUT, positions=pos0, velocities=vel0, masses=masses, density=den0, info=json.dumps(info))
log("saved initial npz")

hist = []; t0 = time.time()
def cb():
    v = fb("velocity")
    t, vm, dt = float(tm.getTime()), float(np.linalg.norm(v,axis=1).max()), float(tm.getTimeStepSize())
    hist.append((t, vm, dt))
    if len(hist) % 10 == 1 or dt < 1e-6:
        log(f"  step {len(hist):5d}  t={t:.5f}  |v|max={vm:.5g}  dt={dt:.2e}  wall={time.time()-t0:.0f}s")
    if time.time() - t0 > 600 or len(hist) > 20000:
        raise SystemExit("stop: wall/step budget")
base.setTimeStepCB(cb)
base.setValueFloat(base.STOP_AT, T_STOP)
try:
    base.run()
except SystemExit as e:
    log(str(e))
vmaxes = [x[1] for x in hist] or [float('nan')]
log(f"\nDONE steps {len(hist)}  |v|max min {min(vmaxes):.4g} max {max(vmaxes):.4g} final {vmaxes[-1]:.4g}")
log("HOLDS" if (vmaxes and max(vmaxes) < 0.25) else f"DOES NOT HOLD (peak {max(vmaxes):.4g})")
