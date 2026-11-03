# Isambard 128³ four-GPU turbulence test

Index: [PLANS.md](PLANS.md). User request, 2026-11-03: inspect storage policy,
run 128³ 3D subsonic turbulence to t=10 on one node/four physical GPUs, and
retrieve results to the workstation. Existing 3D CRKSPH weak-driving pilot is
the proposed configuration; user confirmed CRKSPH and the existing pilot setup.

- Check actual scratch/project quotas and published retention/transfer policies.
- Stage the smaller frozen 0.7.1 ARM64 bundle on personal shared scratch, verify
  its pins, and keep the previously qualified `current` release unchanged.
- Run/inspect short single-GPU and four-GPU site checks before the capacity run.
  Renew restart coverage as applicable; no inter-node qualification is claimed.
- Reuse the finite-gamma seed1234, B7/n_h4, acceleration0.3, OU0.5/0.05,
  relaxation cooling0.2 physical model. Use eager native distributed operators.
- Prepare bounded output/checkpoint cadence and one-node/four-GPU Slurm launch;
  validate resources, record exact source/configuration/runtime and timings.
- Inspect endpoint, finite/positive fields and diagnostics; pull output with
  resumable rsync, verify hashes locally and keep remote originals initially.

Published Phase2 quotas: home100GiB, personal scratch5TiB, shared project200TiB.
No storage is backed up. Retention table says AI scratch persists to project
expiry; generic scratch section also mentions60-day cleanup, so copy results
promptly. Compute-local storage is ephemeral; authoritative outputs stay shared.
Source: https://docs.isambard.ac.uk/user-documentation/information/system-storage/
Transfer: https://docs.isambard.ac.uk/user-documentation/guides/file_transfer/local/

Actual account: /scratch/b6ez/wire.b6ez available and empty, reported5TiB;
/projects/b6ez reported200TiB. Workstation currently235GB free.
Runtime staged and all1594 files/242 links verified. Actual scratch quota5TiB,
1,024,000 inodes; usage3.267GiB. Single-GPU site check8040786 passed (6m38s); all24 sealed artifacts verify
locally. Four-GPU same-node check8041029 submitted after that inspection.
Configuration resolves without sampling particles: CRKSPH,128³,t10,Morton.
Native output every0.25 physical units at the next accepted state plus final; contract-mode inner checks
retain forced exact initial/search/accepted/checkpoint replica checks.
No turbulence job submitted yet; inspect site gate results first.

Output adapter delegates to the frozen launcher and native checkpoint writer,
without changing state/timesteps. The job pins a private copy of the harness and
adapter. Validate its cadence and final-state parity before the capacity run.

The first two local cadence fixtures stalled before process-group creation on
container hostname resolution with networking disabled. Stopped those exact
containers and retained failure logs; retry uses hostname localhost and Gloo
loopback only. No simulation states were produced by those failed launches.

Timed-output CPU/Gloo control passed:4 steps,3 samples at targets0.00025,
0.00050,0.00075 with bounded accepted-step lag. Final native state/forcing/clock/
context compare bitwise against the unsampled two-process control. Adapter and
job script uploaded; production still waits for inspected same-node and
cross-job/rank-count restart gates. Slurm8041029 is waiting for resources.
Final workstation destination: results/isambard/2026-11-03/<run-name>/; use
rsync without caches and verify the job-created SHA256 inventory plus native
checkpoint checksums/finite-positive fields and UID/mass consistency.

Controller started at2026-11-03T12:11Z; the detached process was cleaned up
with its launch shell before any submissions. Restarted at12:12Z in persistent
execution session18690, host PID3925077, with an exclusive process lock. It waits for
same-node8041029, retrieves/verifies the sealed stage locally, then submits and
verifies the one-node/four-GPU cross-job/rank-count restart gate. It verifies
staged input hashes before submitting the confirmed128³/t10 case, then retrieves
and audits outputs automatically. No inter-node jobs or physical-model changes.
Status/log/scripts: .tmp/isambard-turbulence128-20261103/controller-status.json,
controller.log, controller.py. Remote submission receipts avoid repeated dispatch
on ordinary retries. Any Slurm/checkpoint/audit failure stops the controller.
The four-GPU gate remains queued for Resources; the large case has not started.

Status follow-up2026-11-03: same-node8041029 completed in4m49s and all25
sealed artifacts verified locally. Cross-job/rank-count restart8043577 also
passed,37 artifacts verified. Controller submitted production job8044505 at
14:49UTC; it remains pending, with current Slurm start estimate15:04:43UTC.
Remote/local output roots are recorded in controller-status.json.

At15:06UTC, Slurm confirms8044505 RUNNING, actual start15:05:23UTC.
Earlier rolling start estimates were not reliable ETAs. Cluster backfill
configuration includes bf_min_prio_reserve=2 while this job priority is1;
this ordinarily prevents a future backfill reservation (association/QoS overrides
may apply). Report actual state/progress instead of those rolling timestamps.

User requested stopping the run because observed runtime was unexpectedly long.
Controller3925077 terminated; cancellation of production8044505 requested.
No restart/continuation or further GPU submissions. Existing checkpoints/logs
are retained. Last inspected state near t2.79 had zero mass drift and positive
density, pressure and energy; the t10 target was not reached.

Local128³ reference found at /home/lu26029/.codex/worktrees/chunked-neighbor-planner/warpSPH/.tmp/crksph-3d-overnight-20261030/runs/flow-128-t4.
31Oct2026 completed CRKSPH/B7/nh4/seed1234/OU0.3/bath on one RTX PRO6000
Blackwell Workstation GPU. Exact loop8286.316925955s (2h18m06.3s) to t4,
4079 accepted steps,2031.46ms/step; build2.27749s. Its history reaches t2.79040
after5063.74336s (84m23.7s),2769 steps. Local supports/chunks/runtime/output
policies differ from the new distributed test; compare as context, not a
controlled scaling measurement. The cancelled Isambard main srun lasted1h15m08s
near the same physical time; full allocation1h18m22s including3m11s preflight.
No further simulation or continuation is authorized. Cancellation confirmed.

Cancelled output retrieval completed. Eleven native128³ checkpoints verify
content checksums, UID uniqueness, finite fields and positive thermodynamics,
with latest checkpoint t2.750517845. Last accepted diagnostic t2.830689669.
Local partial-audit.json records the cancelled/incomplete status explicitly.
Remote output remains intact and no jobs/controller continue.

User follow-up: defer the comprehensive performance campaign until C++ native
backend integration. Operator and full-step strong/weak scaling, hardware
baselines and synchronization/communication accounting are specified in
[PERFORMANCE_SCALING_PLAN.md](PERFORMANCE_SCALING_PLAN.md). No new GPU run.
