# Whole-row neighbour chunks and multi-GPU execution

Plan index: [PLANS.md](PLANS.md). Parent physics programme: [driven compressible turbulence](docs/roadmap/driven_compressible_turbulence.md).

Status, 2026-11-02: **Bounded local L1–L7, allocation recovery and CRK work-launch fix qualified; compressible/delta/native incompressible compatibility qualified in bounded fixtures; local ARM64 build and Isambard single/two-GPU, cross-job/rank-count restart and two-node/CXI correctness accepted. Four GPUs, broader site cases, capacity/performance and held long-flow runs remain open.**
The count/scan/ownership design is agreed in discussion. Core/main and frontend/dev
are integrated; see the [integration report](docs/chunked_integration_2026-10-30.md).
The planner and opt-in standalone single-device chunk construction/lifecycle are
qualified on CPU/CUDA. Explicit row-mapped shared operators and a bounded eager
Monaghan path pass flat parity. Native CRKSPH/CompSPH now retain compatible work
in owned chunks and preserve changed-list replay in bounded CPU/CUDA fixtures.
The bounded replicated-state Gloo/NCCL process path is now qualified locally;
native GH200 single/two-GPU and two-node/CXI correctness now pass. Four-GPU,
broader capacity/performance and long-flow gates remain open.
This document is the canonical implementation/handoff plan for both repositories;
the core carries a short companion pointer, `CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md`.

### Portable container and optional Remote companion (2026-11-02)

User decisions: ship a configurable simulation-stack image for Linux x86-64 and
ARM64/CUDA, including pinned public dependencies and exact local/unpushed source
snapshots. Bundle the Remote companion as an optional service so Docker users
can connect from their own browser. Keep Isambard transport and distributed
viewer/control qualification separate; MPI/ROCm remain deferred.

Implemented: configurable OCI recipes and [build/use instructions](containers/README.md),
public distributions or exact local source snapshots with file hashes, packaged
launch scripts and an optional Remote target. A viewer-only nginx proxy bridges
Docker port publishing to the unchanged private loopback daemon; producer APIs
remain private. Persistent state/cache/output mounts and service lifecycle are
explicit. No distributed solver admission or rank-local controls changed.

Validation: five focused checks pass, including real nginx/companion HTTP auth,
Host/Origin checks, private producer registration, shutdown and replacement of
an installed distribution by different private source at the same release number.
All five actual local stack snapshots build as wheels without installing into or
changing the shared environment. The import scanner passes576 modules and6282
imports across1335 files. Temporary evidence/context/wheels are indexed under
`.tmp/container-packaging-20261102/`.

The local Docker gate is now qualified below. Native ARM64/Hopper, Apptainer,
Isambard Slingshot and physical multi-GPU execution remain open. The actual
image, base digest and resolved dependency constraints are retained locally;
no registry publication occurred.

### Local Docker image qualification (2026-11-02)

The user has installed Docker29.9.2/NVIDIA Container Toolkit1.20.1 and enabled
Docker group access. The private x86-64/CUDA image now builds and passes the
bounded runtime and companion gate. Frozen source/context/image identities,
build logs and dependency inventories/constraints are retained under
`.tmp/container-packaging-20261102/`; `docker-qualification.json` is the result index.
Do not claim ARM64/Hopper, Slingshot or physical multi-GPU qualification from
these single-GPU checks. Keep runs small/sequential; no overnight simulation.

First actual build passed, but a non-root run exposed inherited private permissions
on copied image code. Normalize image application read/traverse permissions while
retaining the host build context's privacy; qualify using an arbitrary non-root UID.
The next arbitrary-UID import exposed Torch's getpass-based default compiler
cache lookup (host UID absent from image /etc/passwd). Supply an explicit writable
Torch compiler cache and stable USER/LOGNAME environment, and include installed
stack import/cache initialization in the non-root image regression. Also remove
root-owned build-time pip caches from /work in both targets, so a fresh work
volume lets the runtime UID create its own NumPy/Numba/Torch/Warp cache tree. This is
packaging only; no solver model or application source changes.

Passed local checks:

- Image `warpsph:local-cuda-20261102`, identity
  `sha256:9314ee4823a1435efa13c0f831f117c3a0cc747aa244a1e10f077599e6095b45`,
  approximately32.3GB uncompressed. Pinned NGC base digest38ed2ecb2c16...41007e;
  exact Torch2.10.0a0/NVIDIA26.1, CUDA13.1, NCCL2.29.2 and Warp1.17.0 versions recorded.
- Six packaging regressions pass; the arbitrary UID65001 absent from /etc/passwd
  can run the entrypoint, import the installed stack and create fresh-volume
  caches. Source scanner passes576 modules/6282 imports/1335 files.
- CUDA Warp/Torch interoperability kernel passes. CRKSPH8² runs two NCCL steps
  on one physical GPU, checkpoints, then resumes for two more steps with changed
  row cap8→4. Two CPU Gloo ranks independently complete two steps. All run
  manifests complete; this is not a two-GPU or shared-GPU multi-rank NCCL check.
- The real companion/proxy through a localhost-published Docker port passes
  token/Origin checks, private producer isolation, installed case discovery,
  Gresho/Monaghan8² live WPR1 frames (2832 dynamic +1008 static bytes), acknowledged
  pause/resume and four-step completion to t.04. Pairing and run history persist
  across recreation with the same hostname/user/work mount. Test containers stop.
  Browser graphics were not retested; this validates the live HTTP/protocol path.
- Timings are startup/JIT/contended correctness diagnostics: cold CRK two steps
  57.87s container /51.78s runner (first step47.038s compilation; next132.1ms);
  checkpoint continuation7.93s /1.071s runner; CPU Gloo21.52s /15.77s runner.
  Remote Monaghan four steps45.97s, mean11354.9ms, min7.1ms/max45396.4ms, dominated
  by fresh kernel compilation. No matched performance claim or overnight flow.

Keep actual base dependencies intact; NCCL2.29.2 here does not satisfy the >=2.30
experimental same-GPU multi-rank debug gate. MPI/ROCm, native ARM64/Hopper,
Apptainer/Isambard networking and physical multi-GPU remain separate work.

### Staged Isambard deployment qualification (2026-11-02)

User requests a deployable Slurm test suite for single GPU, consistency across
physical GPUs, restart, data retrieval and multiple nodes, progressing one stage
at a time. SSH is the login/transfer route. Implement a self-contained harness
with explicit stage submission, an ARM64 source-container build job, rank/device
and collective probes, one-rank vs distributed state comparisons, separate-process
restart with changed rank/row layouts, stable artifact hashes and offline retrieval
verification. Stop between stages; do not auto-submit a whole campaign. Direct
SSH read-only diagnostics were subsequently authorized and verified below. Multi-node must verify distinct hosts and the expected
Slingshot NCCL transport. Native ARM64/Slurm/fabric validation stays pending until
user execution; exercise the harness locally and test failure detection first.

Implemented [staged harness and deployment guide](scripts/isambard/README.md):
ARM64 source archive/native Apptainer build job; manual `single`, `same-node`,
`restart`, `multi-node` submissions; physical rank/GPU/storage/collective probe;
seeded CRKSPH four-step comparisons; previous-job checkpoint continuation and
1↔N rank/row-cap/offset-width changes; SHA256 and native-checksum retrieval checks.
The multi-node gate requires per-rank AWS Libfabric/CXI evidence and distinct
hosts; socket fallback cannot pass. Optional Remote is packaged but no service
is exposed by these jobs. No model changes, SSH connection or remote job submitted.

Local evidence under `.tmp/isambard-qa-20261102/`:

- Eleven focused acceptance/failure-detection tests pass. Four existing packaging
  tests also pass; optional Docker/proxy tests were skipped in this focused run
  (their six-test qualification is recorded above). Shell syntax and full imports
  pass: 576 modules, 6283 first-party imports across1340 source files.
- Actual CPU/Gloo `single`, two-rank `same-node` and `restart` stages all pass:
  3/2/4 solver launches, 1/1/2 comparisons, 23/20/32 sealed files respectively.
  Solver launch wall-time sums are51.60/17.77/34.06s; the latter controllers take
  23.04/39.40s including probes and checks. These tiny cold/JIT fixtures are
  correctness evidence, not performance measurements. Video omitted deliberately.
- Independently copied `single` output verifies all23 stable artifacts and native
  checksums offline. Unit failures detect corrupted transfer/checksum, nonfinite
  state, physical/RNG drift, socket transport and wrong topology.
- The shipped probe passes in `warpsph:local-cuda-20261102` with one physical
  RTX PRO6000 Blackwell GPU/NCCL2.29.2, including float64 Warp work on a nondefault
  Torch stream and real NCCL collectives. An initially untyped Warp scalar literal
  failed compilation; explicit `wp.float64(2.0)` fixes it and the final probe passes.

Returned jobs7986156/7988602 qualify native ARM64/Apptainer/GH200 single-
and same-node two-GPU execution, both cross-job/rank-count restart directions,
peer communication and retrieval (audits below). Four GPUs and Slingshot remain
unqualified. Custom OFI build 7999853 and one-GPU preflight 8000996 now pass
with the unchanged SIF/NCCL/CUDA; next is two nodes/one GPU each, after review.
No two-node job or automatic campaign has been launched.

Path configuration follow-up (2026-11-02): use `QA_ROOT` as the single remote
base (default `$HOME`, with the user's Isambard HOME at
`/home/b6ez/wire.b6ez`), deriving bundle, image, result and build-log paths.
Local SSH/transfer examples use `ISAMBARD_SSH`, `ISAMBARD_ROOT` and a selectable
archive/download directory. The already-uploaded archive remains valid: these
are guide/environment changes only, with no runtime or image rebuild required
because of the path update. Keep the original frozen upload artifact unchanged.

Slurm account follow-up (2026-11-02): user submission with `wire` or `wire.b6ez`
failed before job creation. Official site troubleshooting recommends
`sacctmgr show user "$(id -un)" withassoc`; returned user output confirms default
and associated account **`brics.b6ez`**, cluster displayed `gracehopp+`, no
partition-specific association, QOS `normal`. Use that account with the default
partition initially; do not infer the scheduler account from the login name or
home path. Updated guide includes discovery and the confirmed account. The old
archive remains compatible; actual build/execution is still pending.

Results-path follow-up (2026-11-02): user confirms the running campaign uses
`QA_RESULTS_ROOT=/home/b6ez/wire.b6ez/qualification`. Inspected submit/job,
previous-checkpoint binding and retrieval paths: later stages honor the explicit
results root and `QA_PREVIOUS`; no runtime path fix or relocation is required.
Keep that value in subsequent login shells. Local download examples must set
`ISAMBARD_RESULTS` to the same directory instead of the fresh-campaign default.
The build log under `$QA_ROOT/logs` and test logs under `$QA_RESULTS_ROOT/logs`
are separate paths. User home listing confirms `images`, `logs`, `qualification`
and `warpsph-isambard` are directly in HOME, so QA_ROOT is HOME itself. Guide
defaults now match that layout on both local transfers and remote submissions.

Returned Isambard single-GPU evidence review (2026-11-02, complete):

- Job7986156, `I-AI_P2`, host `nid010010`, native `aarch64`, GH200120GB,
  Torch2.10.0a0/NVIDIA26.1, CUDA13.1, Warp1.17 and NCCL2.29.2. Native NCCL
  library `/usr/lib/aarch64-linux-gnu/libnccl.so.2.29.2`; SIF SHA256
  `9f0bb994ce8c2e6aebaf99fcab0c392422c2afa937ed22b7e599db77b2c7c04d`.
- Independent offline verification passes23 artifacts. Continuous step4 versus
  2+restart+2, and continuous step2 versus independent prefix step2, each have
  **23 physical/clock/forcing/context datasets byte-identical**, maximum
  difference0. Diagnostics match step-for-step; mass1 unchanged; all state
  finite, positive density/pressure/energy/support and advancing clocks.
  Maximum absolute relative energy-budget residual3.346e-12 for this tiny fixture.
- Checked904 installed source files against the frozen ARM64 build context for
  each of the three solver launches. Same source digest/device across all runs;
  real resumed int32→int64 offset and maxRows4→3 transition confirmed.
- Solver process wall times: reference141.085s, prefix32.396s, resumed122.604s
  (296.085s total, excluding probe/check/seal). Cold first steps117.634s and
  103.292s for the changed layout; warm steps418–525ms. Tiny/JIT correctness
  evidence only; video deliberately omitted. No CUDA errors, NCCL WARN/ERROR,
  nonfinite diagnostics or velocity/stall alarm in returned logs.
- NCCL initializes successfully but selects `Socket`; INFO reports failed
  optional IB plugin initialization. One rank cannot exercise inter-GPU traffic.
  This is **not** Slingshot/RDMA evidence; multi-node still requires per-rank
  AWS Libfabric/CXI proof. Next gate is one node/two distinct GPUs.
- Six repeated HOME warnings trace to our redundant `--env HOME=/work` argument.
  Removed that unsupported override from current `node.sh`; the existing
  container bootstrap already sets HOME and caches before Python. Existing
  wrapper argument regression now rejects the unsupported HOME argument.
  Eleven focused tests, shell syntax and full imports pass (576 modules,
  6283 imports,1340 files). Deployed old wrapper may still warn; no SIF rebuild
  or rerun of this accepted single stage is needed for that warning alone.
- Missing `fuse2fs`/`gocryptfs` are optional EXT3/encrypted-filesystem helper
  notices, not failures mounting this SIF. Repeated CUDA compat `rm` errors
  match `/etc/shinit_v2` in the same local NGC release: it tries removing/replacing
  a compatibility symlink and explicitly falls back to adding `lib.real` to
  LD_LIBRARY_PATH when the image is not writable. Direct source inspected was
  the matching amd64 image, so the precise ARM startup implementation is inferred
  from matching errors; successful actual Warp/NCCL/solver execution is verified.
  Do not bypass compat setup, delete CUDA libraries, add writable images or
  suppress stderr to hide those messages. Barrier device-context warning is
  informational; local device binding and GPU UUID are verified.

Read-only retrieved campaign is preserved in the main workspace at
`.tmp/isambard-retrieved/single-7986156/`. Adjacent `audit_single.py`,
`audit-single-7986156.json` and `ngc-shinit_v2-amd64.txt` reproduce/index the audit
without modifying sealed results. Original uploaded source archive unchanged.

Returned same-node two-GPU review (2026-11-02, accepted):

- Job7988602, host `nid010156`, two independent ranks/processes and distinct
  GH200 UUIDs (`29de7ce5-…`, `79b8c5a3-…`). Native ARM64/NCCL2.29.2, same
  source/runtime fingerprints and SIF SHA256 as accepted job7986156. All20
  transferred artifacts/native checkpoints independently reverified.
- Recomputed full23-field comparisons locally at completed steps2 and4.
  Step2 is byte-identical across every field. Step4: all particle arrays,
  clocks, forcing/RNG and physical context are byte-identical; the only saved
  difference is global `sourceEnergyBudget[0]`, one float64 ULP (4.136e-25).
  The new one-rank reference is byte-identical across all23 fields to the
  earlier job's one-rank reference on a different node.
- Diagnostic times, particle/thermodynamic statistics and mass match exactly.
  Only last-step `forceWork` and relative energy-budget residual differ, by
  4.136e-25/4.596e-25 respectively. All values finite, density/pressure/energy/
  support positive, mass1 unchanged. Max absolute relative energy residual
  3.346e-12. Checked904 source files per solver invocation against frozen context.
- Probe logs16 P2P/CUMEM channel entries per rank; distributed solver logs8 per
  rank, with successful NCCL init/rings and no NCCL WARN/ERROR or CUDA/stall/
  nonfinite failure. Optional network plugin selection still says Socket;
  actual local peer-channel evidence is present. No inter-node RDMA claim.
- Solver launch wall times: reference148.657s, distributed33.029s. Reference
  first step118.761s, warm465.5–471.6ms; distributed first step6.484s,
  warm295.8–299.4ms. Shared compilation/cache state and tiny four-step fixture
  prevent treating these differences as a performance/scaling benchmark.
  Video omitted for this repeated correctness fixture.
- Independent CPU audit and summary: main workspace ignored
  `.tmp/isambard-retrieved/audit_same_node.py`, `audit-same-node-7988602.json`
  and `.out`, alongside unchanged sealed campaign `same-node-7988602/`.
  No solver/harness/image changes or extra GPU runs needed for this acceptance.

Next manual gate: `QA_PREVIOUS=$HOME/qualification/same-node-7988602`,
`QA_GPUS_PER_NODE=2`, stage `restart`. It resumes the preceding job's distributed
step2 checkpoint on one GPU in a new allocation and tests a fresh one→two rank
continuation; both compare to a new uninterrupted reference. Four GPUs, longer/
larger runs and multi-node AWS Libfabric/CXI remain subsequent gates.

Returned restart review (2026-11-02, accepted with RAS advisory):

- Job7989645 on `nid010330`/two distinct GH200s, different from checkpoint
  producer job7988602 on `nid010156`. Same validated SIF, packages and source
  fingerprints. All32 artifacts/native HDF5 checksums independently verify.
- Five independent23-field comparisons are **byte-identical**: uninterrupted
  reference vs prior-job two→one continuation; vs fresh one→two continuation;
  prior-job two-rank checkpoint vs new reference at step2; new prefix vs step2;
  previous-job one-rank reference vs new reference. Clocks, forcing/RNG, all
  particle/physical context fields and work budgets agree with zero difference.
- Previous native checkpoint is complete step2/sourceRanks2 at t0.0004, SHA256
  `ceca76a7873b681a9deb8ae71a7e798aef57851147c20742fbe9de4067c435a5`.
  Config points to `/previous/runs/distributed/checkpoint-00000002.h5`; initial
  resumed diagnostics restore completed2/time0.0004 and finish4/time0.0008.
  One→two uses the new prefix native checkpoint. Both change int32→int64 offsets
  and maxRows4→3; output sourceRanks1/2 and native rank reports confirm transitions.
- All diagnostic histories exactly match reference by completed step; mass1
  unchanged, positive thermodynamic/support state, no nonfinite/CUDA/collective
  execution failures. Max absolute relative energy-budget residual3.346e-12.
  Checked904 installed source files per four solver invocations (3616 checks).
- Three NCCL WARN lines are specifically RAS client listener bind failures at
  default `localhost:28028`, with corresponding failed-listener INFO. Successful
  NCCL init/rings and P2P/CUMEM channels follow. Those diagnostic endpoints were
  unavailable; solver correctness passes. The audit explicitly records only
  this reviewed warning pattern and rejects other warnings, rather than claiming
  the logs are warning-free. NVIDIA documents per-job RAS addresses when nodes
  are shared; the owner of the conflicting listener is not established here.
- Launch wall times: reference139.958s, cross-job139.199s, prefix21.007s,
  two-rank continuation39.513s. Reference cold first step115.313s, warm420–428ms;
  cross-job changed-layout first step102.257s, warm522.4ms; prefix1105.5/426.6ms;
  two-rank continuation4431.0/353.2ms. No performance claim from this tiny/JIT
  fixture; video deliberately omitted. No new GPU runs performed locally.
- Independent audit: main workspace ignored `.tmp/isambard-retrieved/audit_restart.py`
  and `audit-restart-7989645.json`, alongside unchanged sealed campaign.

Networking prerequisite (2026-11-02): user module inventory shows
apptainer-multi-node0.3.2 injecting host NCCL2.26.6, OFI plugin1.8.1 and CUDA12.6
libraries, while the accepted image uses NCCL2.29.2/CUDA13.1. Actual host Libfabric
prefix is `/opt/cray/libfabric/2.3.1`, not the documentation's example1.22.0.
Added standalone `scripts/isambard/network_preflight.slurm`: one GPU, host/custom
profile, same SIF/fingerprint requirement, unchanged loaded NCCL version, real
Warp/Torch/NCCL probe and AWS Libfabric/CXI log evidence. It fails explicitly on
runtime replacement, socket fallback or probe failure, before two-node allocation.
RAS remains enabled with a job-derived loopback port; QA_RAS_PORT can override
it. This qualifies library/provider loading only, not inter-node transfers.
The runtime extension queries `ncclGetVersion` on the already-loaded library via
RTLD_NOLOAD and records `/proc/self/maps` library paths; it does not infer the
runtime from a Torch build-time/package version. Negative regression keeps the
Torch-reported version unchanged while replacing the actual NCCL runtime.
Sixteen focused tests, shell syntax and full imports pass; a real single-GPU
Docker probe also validates the exact appended runtime capture (22902 and one
mapped native NCCL library). Site execution is recorded below. Only this small
Slurm file needs copying into the existing harness;
no container rebuild or replacement of the accepted source archive is required.

Direct SSH authorized/verified (2026-11-02): user supplies `b6ez.aip2.isambard`.
BatchMode/strict host-key read-only commands succeed on login40 as wire.b6ez;
`sbatch` and `squeue` resolve to /usr/bin. No remote job or write was performed
in the connection check. Directly inspected the site's adapt.sh: it prepends
/host/nccl/lib (so the older host NCCL can replace the validated one) and ends
with `exec "$@"`. Existing container.sh sourced it while probe arguments were
still positional parameters, which could execute those options. Current wrapper
now sources it in an empty-argument function scope; regression confirms exported
environment and caller arguments survive. The standalone preflight repairs this
only in its private copy of old uploaded wrappers, preserving original/image
sources. No additional source-bundle transfer is needed.

Direct network preflight dispatch completed (2026-11-02): user authorized one
host-profile/one-GPU submission over SSH. Job 7998167 under brics.b6ez ran on
nid010291 for 37 s, then exited FAILED/1:0. Uploaded/private script SHA256 matches
`2fb6e18b8a332f6ba1ef50006751d4302abc01df774bb9ba98f8646fb9d4d293`.
Submission receipt, final Slurm accounting, probe/Slurm logs and failure report
were retrieved and independently audited. SSH upload/hash verification → sbatch
→ compute execution → failure reporting → local retrieval all work.

The host profile fails at Torch import with `undefined symbol:
ncclCommWindowDeregister`. This is consistent with the inspected adapter
prepending NCCL 2.26.6 to an image qualified with 2.29.2; the runtime-capture probe
was not reached, so no successful GPU/provider test or loaded-version capture is
claimed. Adapter argument handling reached the intended Torch entrypoint.
Report correctly retains profile=host, previousJobId=7989645, probeExitCode=1 and
status=failed. No two-node job launched, no image/archive or prior stage replaced.
Compatible OFI/custom follow-up is now completed below, retaining the image's
NCCL/CUDA; two-node execution remains pending. Earlier
single/two-GPU/restart qualifications remain accepted. Ignored local evidence:
`.tmp/isambard-retrieved/network-preflight-7998167/`, submission receipt,
`audit_network_preflight.py` and `audit-network-preflight-7998167.json` in the
main workspace; audit status `dispatcher-passed-preflight-failed`. Hash inventory
covers local retrieved files, not a sealed remote solver-stage inventory.

Custom communication continuation completed (2026-11-02): native build 7999853
and corrected one-GPU custom preflight 8000996 both pass, with independent local
retrieval/audit. Build:3m45s, nid010043; preflight:1m12s, nid010515, one physical
GH200. Exact accepted SIF SHA and application/package fingerprints retained;
actual loaded NCCL22902 at `/usr/lib/aarch64-linux-gnu/libnccl.so.2.29.2`, image
CUDA13.1. No NCCL warnings in the final preflight. Warp/Torch nondefault-stream
kernel work, collectives, placement and shared-file visibility pass.

Added `build_network.slurm`/`build_network.sh` and a published-checksum source
lock: aws-ofi-nccl1.18.0 (86ebeaf59a2348b22780023effdb682503747de7),
hwloc2.13.0 (32b399232a27ee04f317d743db44e7ba3e154d05). Built inside the same
ARM64 SIF against host Libfabric2.3.1, without rebuilding NCCL/CUDA or replacing
the image/archive. All3 installed shared-library hashes independently verified.
Final live maps prove custom OFI, bound host Libfabric/libcxi and image NCCL/CUDA.
Logs prove net plugin v11, AWS Libfabric, CXI/4 discovered NICs and SENDRECV.
This establishes provider loading only; no inter-node traffic, RDMA bandwidth
or throughput qualification is claimed. Next is two nodes/one GPU each.

Remote deployment for the next gate:
`QA_COMM_ROOT=/home/b6ez/wire.b6ez/communication/ofi-7999853`,
`QA_NETWORK_SETUP=$QA_COMM_ROOT/network-setup.sh`,
`QA_LIBFABRIC_HOST=/opt/cray/libfabric/2.3.1`, `QA_NETWORK=custom`;
previous accepted restart7989645 remains the solver-stage predecessor.
No two-node job was launched in this continuation. Four GPUs, broader schemes,
capacity and held overnight simulations remain open.

Two retained harness failures were repaired before acceptance: build 7999738
(FAILED2:0,0s,nid010111) resolved helpers relative to Slurm's spool; explicit
QA_BUILD_HARNESS now selects the uploaded harness. Preflight8000492 (FAILED1:0,
1m04s,nid010028) passed GPU/OFI/CXI work but took its library snapshot after
process-group teardown unloaded the plugin. Capture now runs before teardown,
with finally-protected cleanup and positive/failing-capture lifecycle regressions.
The actual executed preflight Slurm copy is retained in the private harness.
Twenty-five focused harness checks pass; shell/embedded Python and guide snippets
parse. Full paired import scan:576 modules/6283 imports/1340 files pass with paired
PYTHONPATH and writable Numba/Matplotlib caches. Initial unpaired/default
read-only-cache attempts are recorded as environment failures, not library fixes.

Ignored main-workspace evidence: `.tmp/isambard-retrieved/network-build-7999853/`,
`communication-ofi-7999853/`, `network-preflight-8000996/`, both failed attempts,
submission receipts and `audit_custom_network.py`. Reproduce with arguments
`7999853 8000996`; `audit-custom-network-7999853.json` reports passed. Pinned
archives, initial scanner logs and build-source notes are indexed in
`.tmp/isambard-ofi-sources/README.md`. The audit checks receipts/source hashes,
shared-library hashes, final accounting, fingerprint, live maps and raw logs.

Inter-node continuation (2026-11-02, completed; final acceptance below): user explicitly authorizes
recording all accepted site setup requirements for later simulations and running
the two-node/one-GPU qualification. Retain image/build/profile/checksum provenance,
exact launch/environment/bind settings, then dispatch one bounded multi-node
stage against accepted restart7989645. Retrieve and independently audit topology,
actual CXI traffic evidence, numerical parity, restart and transfer integrity.
Do not auto-start a larger GPU count or long simulation campaign.
Added pinned `site-profile.json`, parametric `site-profile.sh`, and stdlib
`deployment.py` receipts: SIF/app/runtime identity, OFI/hwloc library hashes and
provenance, full setup-script text/hash, host Libfabric/CXI binary hashes,
explicit binds and selected launch settings. Qualification seals the deployment
receipt and records its digest; a documented real-simulation batch recipe uses
the same bootstrap. Slurm network disable_rdzv_get reaches allocation and steps;
node wrapper uses job-derived RAS address while preserving an explicit override.
Thirty-two focused checks, shell/embedded Python/guide syntax and full paired
imports (576 modules/6283 imports/1341 files) pass. Tiny four-step qualification
fixture omits video deliberately; no physical/model/source-image changes.
Initial inter-node job8001638 submitted using harness e43340f. Primary CXI
documentation requires per-step SLINGSHOT_VNIS/DEVICES/SVC_IDS for explicit
Slurm-assigned authorization; the original cleanenv dropped them. The repaired
wrapper preserves each node's own lists, verified on two hosts in8005442.
User decision: future ARM64 OCI/SIF builds must run on the local development
machine, with ready-built artifacts uploaded to Isambard. Use Buildx ARM64
emulation first (or a verified cross toolchain), then local SIF conversion without
executing target GPU code. Bring pinned host Libfabric headers/libraries into a
local ARM64 sysroot for OFI; no build-only Slurm allocation by default. Preserve
the current accepted SIF for this ongoing test. Local build plumbing is now completed below; hardware execution/fabric
qualification still runs on Isambard.
Job8001638 FAILED1:0 after3m28s on nid011241/nid011264. Two-host Warp/Torch/
NCCL collective/stream/storage probe passes with AWS Libfabric/CXI; one-GPU
reference completes4 steps. Distributed solver fails before stepping at
launcher:rank-start: rank1 cannot open /work/runs/distributed/.rank-00001.json.tmp
after rank0 creates the fresh directory. Investigate/repair bounded shared-file
visibility at launcher startup, retaining failure and numerical source baseline.
A deployable source fix requires a fresh locally built ARM64 image and renewed
source-fingerprint qualification; do not silently patch the accepted SIF or
loosen existing source guards. The initial tool session lacked the new Docker group. Using existing membership
via sg docker restored access; no daemon/socket/group changes were needed.

### Static import scanner issue resolved (2026-10-30)

The previously recorded wp_alpha.computeAlpha `find_spec` traceback is fixed.
Missing symbols on regular modules now produce normal diagnostics; only packages
are searched for child modules, and child-lookup errors do not abort the scan.
Five focused scanner regressions pass. The full L6 check now imports567 modules
and checks6092 first-party imports across1316 files with no failures.
The five real stale imports exposed after fixing the exception were resolved:
two old pressure-solver notebooks are clearly labeled archival source listings
(their retired numerical APIs were not revived), and the Kidder regression calls
the accepted-boundary helper now used by finalizers. Five affected Kidder checks
also pass. Original notebook files are retained in ignored L3 evidence. Do not
re-investigate the old scanner traceback; older notes below describe history.
L3 control/hash and particle-packing work is now qualified locally. Evidence: `.tmp/l3-communication-20261030/` and `.tmp/l3-payload-20261030/`.

## Current checkpoint: bounded I1 complete (2026-10-31)

The authorized IISPH → DFSPH → predictor-first → shift/resample sequence is
complete in the bounded periodic scope below, alongside C2 and W1. Preserve native pressure iterations,
stopping norms, warm starts and projection ordering. Qualify dependent field
exchanges, accepted-state allocation recovery and exact managed restart before
admitting each family. Eager periodic single-material execution remains the
initial scope; long overnight holds and ACSPH exclusion remain unchanged.

## Authorized continuation sequence: remaining scheme families (2026-10-31)

The user authorizes extending this setup through the remaining compressible
schemes, then delta-SPH, then incompressible schemes, explicitly excluding
ACSPH. The completed recovery/CRK launch work is no longer a local blocker.
Keep full164³/196³t4 and96³→t40 as overnight holds while development remains;
physical multi-GPU/site/performance and MPI/ROCm builds remain separate gates.

1. **C1 PESPH:** preserve both PressureEnergy and PressureEntropy authoritative
   variables, support calibration, dissipation and native fresh-stage integration.
   Port custom gathers/force with explicit query/reference roles and global self
   identity; preserve chunk residency during initialization. Qualify flat/chunked
   operators and native steps before widening startup admission. Driven forcing,
   native restart and failure recovery require their own state/source audit.
2. **C2 MFM/MFV:** preserve ConservedLagrangian/SSPRK stage geometry, face fluxes
   and MFV mass limiting. Audit geometry and flux ownership before porting; do
   not route these through the SPH RHS driver or change flux formulas.
3. **W1 delta-SPH / Sun2017:** use the native periodic case and density state;
   preserve Sun2017 frozen diffusion and fresh RK stages. Audit custom operators,
   initialization and restart rather than reusing ideal-gas driven sources.
4. **I1 IISPH, then DFSPH / predictor / shift:** preserve pressure iterations,
   projection ordering, warm starts and stopping norms. Qualify dependent particle
   exchanges and accepted-state rollback before claiming failure recovery.

Initial admission for each family is eager, periodic, single-material 2D/3D.
Walls, free surfaces, materials, AD and graphs remain explicit later gates.
Per-family completion requires native flat/chunked parity, compatible saved
state/restart, allocation-failure controls and serial Gloo/NCCL qualification.
An operator port alone does not admit an entire family to the distributed CLI.
No numerical model, support defaults or ACSPH code changes are authorized by
this compatibility work.

### C1 driven admission continuation (2026-10-31)

User asks to keep going. Implement a native total-energy source adapter and
retain source objects/budgets through PESPH stage cloning/finalization. Admit
PressureEnergy to the existing finite-time bath; PressureEntropy initially
requires adiabatic driving, retaining its Guarded thermal policy (no new entropy
source law or automatic experimental-policy selection). Keep NumberDensity
supports/grad-h defaults. Combine native signal-speed and case forcing/cooling
limits instead of replacing the driven timestep hook. Qualify independent body
source controls, native managed save/resume and accepted allocation rollback
before extending the distributed CLI allowlist. Continue MFM/MFV afterward.

### C1 driven admission result (2026-10-31)

PESPH is admitted to the explicit eager periodic driven launcher and managed
checkpoint/recovery path. PressureEnergy retains the existing finite-time bath;
PressureEntropy requires explicit dissipation=False and adiabatic driving under
its default Guarded policy. No entropy-bath policy/default was changed.
Dedicated driven E/A state classes retain the two-entry source ledger, the
source object survives native stage clones, and finalization updates only the
ledger after native pressure-volume closure (no mass-density EOS overwrite).
Both native signal and case forcing/cooling timestep limits are retained.

Independent uniform acceleration/bath controls and exact flat/chunked stage
clocks pass9 CPU checks perFP32/FP64; forcingInterval is deliberately tighter
than maxDt. Shared source/PESPH regression set passes142 CPU checks. Actual
Gloo launcher manifests, rank-owned output and1→2→3-rank continuation pass for
both variants; the explicit native worker also requalifies in exact/contract.
Managed failure tests preserve source generators/clocks, E/A, optional scalar
switch histories and particle tensors after neighbour/force-output rejection,
then resume across row budget, local offset width and process-count changes.
Managed restore skips the extra finite-tolerance support solve on validated
accepted state; ordinary non-native PESPH restart keeps its existing behavior.

All20 serial NCCL debug jobs pass: both variants,2D/3D, independent one-rank
four-step reference, two-rank step2 save, changed-row-budget continuation, force
allocation failure after step2 and recovery continuation. Particle/stage/time/
forcing fields match exactly; only the existing source-ledger2e-12 allowance
applies. Very fine chunks make these correctness controls launch/control bound;
job wall times include startup/compilation and are not flow performance figures.
The temporary NCCL driver initially reused a checkpoint name for one fault
control; the expected no-overwrite rejection is retained, the driver is fixed,
and all successful jobs are retained rather than rerun. No solver fix for that
orchestration error was needed. Evidence: `.tmp/scheme-compatibility-20261031/`.

C1 bounded admission is complete within these thermal/eager/single-material
limits. Final CUDA source/clock controls pass9 checks perFP32/FP64; scope/budget
guards pass7 checks; the affected prior native/Godunov distributed source control
also passes. Core MFM moments are committed as `4fbac54`:24 ordinary/prepared
geometry controls perprecision across CPU/CUDA,69 existing CPU regressions
(78 expected device/reference skips), four2/3-rank Gloo jobs and the complete
two-rank NCCL ownership/error/packing/filter worker with four new moment fixtures
all pass. Both NCCL ranks audit runtime2.32.3 and the same GPU UUID. No GPU job
remains active after qualification. The frontend scanner passes570 modules,
6149 first-party imports and1320 files. No held overnight job was launched.
Default chunked meshless closure still rejects.
Continue owned closure weights/matvec/preconditioner/coarse stencil, gradients,
limiting and canonical flux identity before MFM/MFV solver admission.

### C2 implementation and qualification (2026-10-31)

MFM/MFV now keep global particle identity through moments, gradients, limiters,
closure residuals, timestep and canonical pair fluxes. Projected closure retains
CG, the existing auto/explicit multigrid hierarchy and its coarse stencil, warm
starts, convergence norms and bounded outer passes. Kappa and capped-closure
scale use dedicated owned pair buffers; dependent particle matvec/stencil results
are exchanged before reuse. Replicated CG vector norms are evaluated once on the
replicated field, never summed once per rank. The chunk path is eager; the flat
CSR graph optimization remains unchanged. A flat capped-closure padding mismatch
was repaired without changing any live-pair arithmetic.

The frontend uses an explicit neighbourhood factory through native state clones,
retains the ConservedLagrangian stages and each mode's temporal default, and
keeps accepted geometry/search history attached to the accepted state. Sources
use current conserved-stage primitives and uniquely owned work/heat sums. Managed
restart reconstructs moments and restores the native closure warm start without
an extra closure solve. Failure recovery retains accepted Q, supports, warm start,
forcing history and clocks and clears trial stage scratch.

CPU/CUDA native 2D/3D parity, sources, restart and neighbour/weight allocation
failures pass. Default MFM SSPRK / MFV Hancock controls pass. CPU FP64 checks and
actual Gloo CLI save/resume across1→2→3 ranks pass. Core CG/MG/capped-closure
controls pass on CPU/CUDA and two Gloo/NCCL ranks. All 24 independent managed
reference/save/resume jobs pass across Gloo/NCCL, MFM/MFV, none/project closure
and 2D/3D, changing1→2→3 ranks, row budgets and local offset widths. Eight more
managed neighbour-failure/recovery jobs pass across both transports and modes.
Closure-weight allocation failures are coordinated in the core worker and
preserve accepted native state in frontend CPU/CUDA tests. The affected CPU
regression set passes159 checks (three device-only skips); native FP64 CPU
controls pass 18 checks, and core FP64 CUDA controls pass 40. Actual Gloo CLI
output/manifests/save/resume pass for both modes. C2 bounded admission is
complete. Final FP64 CUDA frontend checks pass 21 controls. Evidence: `.tmp/meshless-delta-20261031/`.

### W1 delta/Sun2017 continuation (2026-10-31)

Proceed with the native periodic `tgv-wc` case. Retain the native initialization
relaxation, explicit shifting constants, density-continuity stages, pressure
sign switch and Sun2017 frozen diffusion. Port density/velocity diffusion,
surface-aware pressure and delta shifting with explicit row roles. Initial
shuffle must use the explicit chunk factory, coordinate its random perturbation
across ranks and retain ordinary random/default behavior. Saved accepted density
and surface history are separate from ephemeral per-step frozen diffusion;
restart must recompute stage 0 diffusion for the next real step. Qualify failure
inside post-step shifting as well as the RHS; do not let shared constant
thermodynamic diagnostics mutate the accepted input state on a failed finalize.
Keep walls/free surfaces/materials/graphs/AD and ACSPH outside W1 admission.

### W1 bounded result and remaining gates (2026-10-31)

Delta-SPH and Sun2017 now retain native density continuity, pressure sign switch,
explicit shifting and their RK stage policy on owned rows. Density/velocity
diffusion, surface-aware pressure and delta shifting declare query/reference
identity. The pressure pass carries the global field/dummy mask distinction into
one-row chunks, preserving the original singleton-field behavior. Native
nearest-neighbour diagnostics now reduce per row, without global pair expansion.

The periodic `tgv-wc` launcher uses the explicit chunk factory through initial
relaxation. A fresh post-jitter search and a fresh final seed preserve the native
per-row neighbour order; density during initial relaxation retains its original
pair-list-free grid traversal on each replica. The ordinary RNG/defaults are
unchanged, optional shuffleSeed makes initialization reproducible, and rank0's
jitter is replicated for unseeded distributed initialization. The default 128
relaxation iterations and full Sun shifting strength pass an exact CPU A/B.
No solver parameter or default was weakened for admission.

The dedicated accepted delta checkpoint saves native density/particle fields,
clock and edge-free frozen search history, plus accepted mask history when used.
It never saves the ephemeral frozen RHS: the next step computes fresh stage 0
diffusion. Pressure may be signed; accepted density must be positive and every
particle tensor finite. Failure during an RHS row allocation or post-step shift
preserves the accepted state, including its diagnostic tensors, and publishes
an atomic resumable checkpoint. The chunked finalizer owns diagnostic destinations
before copying, so a rejected shift cannot overwrite the input's constant fields.
Not-applicable wall-only diagnostic statistics appear as JSON null in launcher
output, retaining the native unavailable value without allowing nonfinite state.

All 28 independent managed Gloo/NCCL jobs pass: both schemes, one-rank reference,
two-rank save, changed rows/offsets and up to three Gloo ranks on resume, injected
RHS/shift failure after step 2 and recovery continuation. Native final fields,
fresh stage rates, time and next dt match exactly. Four extra two-rank jobs
qualify native 3D steps against the flat path on Gloo/NCCL. Native 2D/3D CUDA
controls pass 13 checks in FP32 and15 in FP64 (one CPU-only default-initialization
skip); FP64 CPU native controls pass 13, plus separate reverse-AD and initialization
controls. Flat pressure/viscosity reverse derivatives match finite differences;
their existing unregistered-JVP rejection is retained, and chunked AD rejects.
Actual Gloo CLI manifests/output/save/resume pass for both schemes. The import
scanner passes 574 modules and 6231 first-party imports across 1327 source files.
Evidence: `.tmp/meshless-delta-20261031/`.

**Known pre-existing CPU test failure; do not re-investigate as this port:**
`test_deltaSPHDiffusion.py::test_theUnrenormalizedVariantCancelsToo` measures
corrected/uncorrected RMS 0.2218983918428421 against a 0.2 threshold. The same
failure and exact ratio reproduce using isolated pre-W1 frontend source
`f700c56` with core`dcc62cf`; only row metadata changed in that diffusion
operator. Its formula/tolerance were not altered. The other affected delta
regressions pass 27 checks (six device-only skips). Configuration/catalogue,
padded diagnostics and meshless integration regressions pass 126 checks on the
GPU-capable stack; prior CRK neighbour-failure recovery passes two checks. Logs include the isolated
baseline and current result. This is an existing numerical validation issue,
separate from storage parity and continuation qualification.

W1 admission is eager, periodic, one fluid material, native delta shifting;
the managed launcher/checkpoint case is 2D `tgv-wc` with regular sampling/native
relaxation. Native 3D operators/steps are qualified separately. Walls, free
surfaces, other shifting schemes, AD/graphs and additional case adapters remain
explicit gates. C2 and W1 bounded continuation are complete. I1 incompressible
compatibility is qualified in the following I1 receipt; ACSPH remains excluded
and all overnight holds remain.

### I1 native projection continuation (2026-10-31)

IISPH's C/F, wall/total gradient and exact diagonal now opt into owned whole
rows. Particle extras retain global reference indexing and receive the explicit
row origin for self terms; local query/CSR/output indexing stays local. Each
particle result is exchanged before a dependent pressure pass. Native Jacobi,
independent DF/CD histories, iteration budgets, residual norms, pressure clamp
and all three projection orderings remain unchanged. Replicated vector norms
and periodic source means are evaluated once per replica, without multiplying
sums by rank count. Neighbour counts and post-drift velocity resampling use
owned rows too; no global pair compatibility tensor is introduced.

The explicit managed case is periodic 2D `tgv`, through the same caller-owned
Gloo/NCCL interface. Its versioned parameter/solver descriptors and optional
`shuffleSeed` are now present. Initial pressure relaxation retains chunk
residency, rank0 jitter agreement and native search ordering; ordinary unseeded
sampling/default32 initial iterations are unchanged. Native 3D maps are checked
separately. Admission excludes walls, free surfaces, materials, external
shifting, experimental Band pressure geometry, AD and graphs; ACSPH is excluded.

`projectionRowChunkCheckpointVersion=1` saves accepted native particle fields,
physical IISPH pressure, DF/CD pressures and timestep-scaled impulses (including
optional fields), exact edge-free search history, time, next dt and the nominal
timestep ceiling. It checks configuration/state class/content before rebuilding
owned rows. Restart does not apply the adaptive growth rule twice. Ordinary
flat resume refuses this file. Source-less recovery now preserves only the
family's explicit metadata; projection histories are already immutable accepted
state fields. One-rank failures in the dependent C(F(q)) pass and post-drift
shift retain the accepted state and publish an atomic checkpoint.

CPU qualification passes the native field/iteration/source/order comparisons,
FP64 3D and five reverse-derivative controls, changing-timestep restart and
checkpoint rejection guards. Actual Gloo CLI save/resume passes all four
variants over1→2→3 ranks. Separate Gloo controls pass46 jobs:26 native stress/
3D/empty-owner/recovery jobs plus20 active-pressure continuation/fault jobs.
Local rows and offsets change across restart (including int32→int64); all
recorded particle fields, pressure iterates, stopping statistics, dt/time and
warm histories match the uninterrupted oracle exactly. Final CUDA/NCCL gates pass. The process receipt contains124 successful jobs:
58 Gloo (including12 tightened dependent-pressure fault/replay controls) and66
shared-GPU NCCL2.32.3 (26 native stress/3D/empty-owner/recovery,20 active FP32
and20 active FP64 continuation/fault jobs). Every live rank reports success;
physical fields, independent pressure impulses, iterative outputs/stopping
statistics and dt/time replay exactly. Nondefault CUDA streams are covered.
No new base NCCL installation or physical multi-GPU claim is made.

Additional final checks: actual delta/projection Gloo CLI6 passes; CUDA delta
storage/recovery6, CUDA native rest-density10, corrected CUDA pressure-fault
recovery4, CPU FP64 final storage/recovery/scope13, configuration metadata95,
CUDA-only configured-flow/manifest9 and native one-sweep descriptor3 passes.
The initial broad affected sweep had285 passes/18 expected device skips; its
five validation-priority failures pass after repair, and six CUDA-required
fixtures pass on the CUDA-visible rerun. CPU/CUDA FP32/FP64 native/3D/reverse
and active-pressure controls pass, retaining the independently reproduced
native FP64 stress stop described below. Final import scanner:576 runtime
modules,6282 first-party imports across1331 files, all clean. Raw attempts,
commands, source/runtime pins and diagnostic wall/step timings remain under
`.tmp/projection-20261031/`; timings include debug checks/contention and are
not performance results.

**Restart ordering repair found by FP64 CUDA active control:** the disposable
seed omitted I, whereas native projection/delta RHS requires I by default.
Restore inherited the seed setting, so the next RHS forced a fresh spatial
search and changed J reduction order. The first density changed4.44e-16,
subsequently amplified to~1.19e-9 pressure. All saved particle bits and
reconstructed per-row J were exact before that unnecessary rebuild. Both
family adapters now reconstruct using the next native RHS's storeI contract;
the launcher seeds that contract too. FP64 CUDA active four-family continuation
now matches bitwise. Focused tests inspect storage before the first resumed RHS.
No pressure formula/tolerance or rounding allowance was changed.

**Native numerical limits retained, not fixed by this storage port:** the tiny
nx8 absolute-source/default-relaxation control can hit native100-iteration caps
and has degenerate IISPH pressure geometry after relaxation. Active pressure
coverage therefore additionally uses the existing explicit periodicMeanFree
control, nx12, no initial relaxation and native IISPH omega=.25; all four
variants converge with nonzero pressure and exact restart. These are labelled
controls, not new defaults. On CUDA FP64 the nx8 IISPH stress stops at the
physical dt guard (1.29715e-11 < minDt1e-8). The same failure/value reproduces
on isolated pre-I1 frontend7fa1ba7 with the unchanged paired core. Preserve it
as a strict expected failure in that stress fixture; healthy active/3D gates
stay independent. Evidence includes the source archive receipt and both logs
under `.tmp/projection-20261031/`. This admits bounded execution/communication,
not long-time TGV convergence or pressure accuracy.

### C1 initial native checkpoint (`ea8534e`; superseded by driven admission above)

Pressure sums and variational force now declare row roles. The force keeps
local query and global reference velocity/weight/factor/grad-h views separate,
with global self identity supplied by the row base; formulas/order are fixed.
Initialization retains chunk policy/communication and lets the existing support
solver rebuild Gather to SuperSymmetric, rather than discarding it for flat J.
The default dissipative energy variant also needed the shared Cullen-Dehnen
matrix and signal-speed passes ported. Signal-speed support masking now receives
the global row ID, preserving the existing flat formula and reverse AD split.

Both variants pass exact flat/chunked operator and two-step native field
controls in2D/3D with fixed/adaptive support. Four-step two-process Gloo and
shared-GPU NCCL controls pass in exact/contract modes: step2 native particle HDF5
roundtrip, edge-free search reconstruction with changed row budget, continuation
to4 and injected force-output allocation failure. Native accepted fields match
flat storage bitwise. The failure preserves accepted particles and switch
history and leaves the communicator usable. Explicit guards reject chunked
sources/boundaries/materials/graphs/AD; flat PESPH differentiation remains tested.
These are tiny repeated controls with video deliberately omitted; transport
wall times include compilation/reference work and are not simulation benchmarks.

At `ea8534e`, C1 was **partially qualified**: the driven source adapter,
managed checkpoint and automatic recovery still required qualification. The
generic driven source adapter evolves dudt and refreshes a mass-density EOS;
reusing it directly would overwrite authoritative PESPH E/A. The dedicated
adapter and admission gates have since passed as recorded above. This initial
native evidence remains indexed under `.tmp/scheme-compatibility-20261031/`.

C2 implementation starts with ordinary/prepared row-chunk geometry moments and
explicit global self identity, retaining flat formulas and forward-only guards.
Empty/single/nonuniform2D/3D populations pass ordinary/prepared exact moments/
inversion parity on CPU/CUDA in FP32/FP64 and2/3-rank Gloo controls. Default chunked face closure explicitly rejects.

C2 audit: frontend canonical-pair flux is already directed per row (no shared
force scatter). Core geometry moments/gradients/limiters/timestep still need
row roles; closure kappa is a global per-edge output and the optimized CSR
matvec directly reads global J/kappa. Port kappa as owned chunks, exchange
matvec particle outputs before the next CG read, and keep replicated-global
stopping norms from being summed once per rank. Preserve default auto multigrid,
coarse-stencil ownership and ConservedLagrangian/SSPRK lifecycle. Do not disable
closure/MG or silently select Jacobi merely to get a small fixture running.

## Start here / next action

Priority decision (user,2026-11-02): broader scheme/case coverage that exercises
core functionality comes before larger GPU counts, capacity campaigns or more
long trajectories. Establish solid local behavior first. Existing bounded
C1/C2/W1/I1 ports and Isambard single/two-GPU/two-node receipts stay accepted;
do not repeat those investigations or mistake stale historical checkboxes for
missing implementations. ACSPH and numerical-model/default changes remain outside
this compatibility scope.

1. Audit the actual supported scheme/case/configuration matrix against the
   existing regressions. Separate implemented bounded coverage, genuinely
   unsupported paths and documentation drift; MFM/MFV are already locally admitted.
2. Broaden short local native flat-versus-chunked and multi-rank cases across
   compressible, delta and native incompressible families. Exercise meaningful
   2D/3D configurations and scheme-specific stage state, dependent exchanges,
   switches/shifting/projection warm starts and convergence, as applicable.
3. Cover moving/rebuilt lists, filtering/repartition, uneven/empty owners,
   changed row budgets/offset widths/rank counts, native checkpoint continuation
   and injected allocation failures preserving the last accepted state. Preserve
   method/stage/timestep semantics and use existing native references.
4. Keep qualification local: CPU/Gloo and bounded one-device CUDA checks,
   including the already-supported shared-GPU NCCL debug mode when appropriate.
   Any newly exposed unsupported operator/case needs explicit admission and
   focused qualification; do not silently widen wall/material/AD/graph support.
5. Once local scheme support is consistent, integrate this worktree with the
   fixed-count particle-reordering work described in PARTICLE_LAYOUT_PLAN.md
   in the main development checkout. Pin the frontend/core/integrator inputs
   and preserve both accepted baselines before merging; do not merge unfinished
   support or modify the other active worktree prematurely.
6. Debug and qualify the combined implementation locally before any new physical
   multi-GPU/site testing. Check persistent UIDs versus storage rows, global
   reference J indices, history/warm-start/source permutations, layout-generation
   invalidation, deterministic row repartition and owned W lifetime, accepted
   state/restart and failure rollback. Initially rebuild/repartition after a
   reorder, without gathering global J/W. Use UID-aligned native references and
   declare reduction-order tolerances explicitly.
7. Only after the merged implementation passes local checks, resume physical
   multi-GPU constellations, then capacity/performance and held overnight flows.
   Existing Isambard qualification stays a frozen baseline; no new site jobs or
   early physical multi-GPU detour. Changed merged sources require a fresh image
   and renewed site gates. MPI/ROCm and ACSPH remain deferred/excluded as above.

## Active implementation workspace (2026-10-30)

One paired workspace, used for the full sequential effort:

|Repository|Checkout|Branch|Integrated base|
|---|---|---|---|
|Frontend|`/home/lu26029/.codex/worktrees/chunked-neighbor-planner/warpSPH`|`codex/chunked-neighbor-planner`|`8b58008`|
|Core|`/home/lu26029/.codex/worktrees/chunked-neighbor-planner/warpSPHCore`|`codex/chunked-neighbor-planner`|`04e126e`|

The originally prepared `chunked-neighbors-multigpu` workspace is owned by
another chat and was left intact. The user requested a new worktree; this chat
created the active managed frontend workspace and its paired core checkout from
the same integrated commits. The frontend managed worktree is attached to this
chat. Both imports resolve to
these paths when their `src` directories are supplied explicitly in `PYTHONPATH`;
core reports0.6.3. Bootstrap does not select checkouts. For restricted read-only
import checks, use writable `NUMBA_CACHE_DIR`/`MPLCONFIGDIR` under`/tmp` rather than
requiring cache writes inside a new checkout. No editable installs were changed.
Set int64 offsets before imports when starting the wide-offset implementation;
the ordinary integration baseline retained its default int32 offsets.

The previous `meshless-boundary-balance` frontend/core worktrees remain intact
on their existing branches for the smaller-turbulence physics track. No worker
was launched there during this integration. The two phases do not get separate
implementation worktrees.

## Pre-integration repository snapshot (2026-10-29; historical)

|Repository|Current feature checkout / branch|Last code/scope commit when surveyed|Shared checkout|
|---|---|---|---|
|warpSPH|`/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH`, `codex/meshless-boundary-balance`|`6ad5d2b` (with this subsequent documentation handoff)|`/home/lu26029/dev/warpSPH`, `dev` at`5d4f41d`; `main` at`66492a2`|
|warpSPHCore|`/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPHCore`, `codex/compressible-csr-only`|`3c56e2f` (plus companion handoff)|`/home/lu26029/dev/warpSPHCore`, `main` at`98081c1`|

At that snapshot the shared frontend had uncommitted ACSPH code/docs/tests. Its `dev`
and the turbulence branch have diverged (2 versus18 commits at the survey).
The shared core was clean and its feature branch was one commit ahead before
adding the companion plan. Re-evaluate all of this when executing Phase 0.
Frontend requires core>=0.6.3. Do not merge the frontend against the old core0.6.2.
The 2026-10-30 integration supersedes these checkout/head observations. The
user's Focus and ACSPH follow-up text are preserved.

Evidence already committed: CSR-only neighbour allocation and core0.6.3,
native CRK/CompSPH recomputation, retained-state lifetime cleanup, memory sweeps,
and a bounded CPU tile-view ABI probe. Read the reports rather than repeating
those investigations:

- [CSR-only implementation/parity](docs/compressible_csr_neighbors_2026-10-29.md)
- [CompSPH work/lifetime results](docs/compsph_work_lifetime_2026-10-29.md)
- [CRK retention/capture results](docs/crksph_retention_2026-10-29.md)
- [Seven-scheme memory sweep](docs/compressible_memory3d_2026-10-29.md)
- [Storage scope and row-chunk discussion](docs/tiled_edge_storage_scope_2026-10-29.md)

## User decisions and boundaries

- 2026-11-02: keep this continuation local until scheme support is consistent;
  then merge with particle-reordering support, debug/qualify the combination
  locally, and only then return to physical multi-GPU constellations. Preserve
  current accepted branches/images as baselines until the integration gate.

- 2026-11-02: prioritize broader local scheme/case functionality and lifecycle
  coverage before larger hardware/scaling/long-run campaigns; bounded CUDA/NCCL
  checks remain part of functional qualification. Preserve the accepted site baseline.

- Integrate the current paired work into the main development checkouts after
  ACSPH reaches a mergeable checkpoint; develop this architecture separately.
- Keep whole particle rows together; partition by stored edge/byte budget.
  Each GPU holds the full particle/reference fields and only its owned J/W.
  Never gather the complete neighbour list or CRK pair-work tensor on one GPU.
- Reuse the existing separate count pass: distributed counting, synchronized
  int32 counts, replicated int64 prefix scan and deterministic partitioning,
  followed by direct local allocation/collection. Lists stay resident until
  the existing validity rules call for a coordinated rebuild.
- Preserve the numerical method: fresh stages, integrator/tableau, compatible
  work, forcing, EOS, support/Verlet policy and acceptance rules. No stage reuse
  or reduced RHS evaluations as a side effect of partitioning.
- Ordinary turbulence up to128^3 remains a separate physics track in the current
  worktree.96^3 has measured startup evidence;128^3 is a candidate, not a completed
  fit/performance/stationarity qualification.192^3/256^3 are later capacity gates.

Terminology: **row chunk** is a storage/ownership partition. Existing **neighbour
lanes** (`autograd/lanes.py`, `setNeighborLanes`) are CUDA threads cooperating on
one particle's reduction. They are independent axes. Freeze the existing lane
setting for chunk parity, then qualify combinations; do not rename or replace
that setting. Uniform tile/slot views remain optional, not a prerequisite.

## Data and execution contract

At rebuild generation g, all ranks agree on particle ordering, reference state,
supports, domain/periodicity, grid/search configuration and acceptance rule.
Each rank counts its disjoint counting interval. All-gather count slices into
C:int32[N]; do not sum duplicated full counts. Every GPU computes P:int64[N+1],
with P[0]=0, P[1:]=cumsum(C), and the same deterministic partition/owner map.
Validate a generation/config/partition signature before allocating pair buffers.

For a chunk owning particles[a,b):

```text
rowStart=a; rowCount=b-a; liveEdges=P[b]-P[a]
counts=C[a:b]; localOffsets=P[a:b]-P[a]
J[capacity] holds global reference particle IDs
W[capacity,dim] holds only this chunk's retained compatible work
local row r -> global particle i=a+r
```

Chunks contain variable particle counts for an edge budget; any optional row cap
is an additional bound. Include empty rows, non-divisible rank counts and empty
ranks. Define an explicit oversized-row error/fallback; never truncate neighbours.
Capacity headroom, byte strides and all associated buffers must fit the per-chunk
Warp limits. A global edge count may exceed INT_MAX without any allocation doing
so. Keep global offsets/prefixes int64 and IDs/counts int32; do not narrow a global
address before subtracting its chunk base. Account for the stored Verlet list,
not only physical-support neighbours. Count and collect use the same generation.

J and W reside together on their owner; no implicit flatten/concatenate is allowed
in production dispatch. Source fields use global indices. Query corrections,
extras and outputs require explicit local/global mapping: slicing positions alone
is insufficient. Single-GPU chunks write disjoint output ranges, without an
additional sum across chunks. Distributed particle outputs are gathered between
dependent passes, while pair data stays local. Reuse capacities until invalidated.

At256^3, replicated C/P cost64MiB/~128MiB per GPU. Derived particle matrices,
stage states, grids and communication staging must also be counted. Resident
chunking alone does not reduce total single-GPU J/W bytes; scratch lifetimes can
be bounded chunk-by-chunk, and distribution reduces owned pair bytes per GPU.

## Phase 0 — paired integration and isolation

- [x] Commit the scope/handoff in both feature repositories. Core companion is
  `f6c5900`; frontend handoff is the commit containing this plan. Production
  baselines remain `3c56e2f` (core) and `6ad5d2b` (frontend).
- [x] Obtain a mergeable committed ACSPH checkpoint through the existing user/task
  workflow. A clean tree is necessary but not evidence that numerical work is done.
  Preserve known failures and the ACSPH qualification record; do not message other
  sessions or invent a readiness signal.
- [x] Inspect fresh status/diffs and freeze paired baseline provenance. At the
  clean checkpoint merge `codex/compressible-csr-only` into the core's actual
  integration branch (currently`main`) first; run the bounded core gates.
- [x] Merge `codex/meshless-boundary-balance` into frontend`dev`. Reconcile shared
  runner/config, PLANS, roadmap and ACSPH changes deliberately; preserve both
  histories and avoid wholesale file replacement. Run paired import/config,
  CSR, shared-work and affected ACSPH gates. Promote frontend`dev` to`main` only
  according to the existing main/dev integration cadence.
- [x] Record merged frontend/core SHAs, core version, test results and remaining
  failures. Do not change global editable installs under active physics jobs.
- [x] Create one paired implementation workspace from those integrated commits. Suggested
  branch names are `codex/chunked-neighbors-multigpu` in both repositories; choose
  one frontend checkout and one sibling core checkout; record their absolute paths.
  Use both sequentially for chunks and multi-GPU, not one worktree per phase.
  Use managed frontend worktree creation/attachment and a paired core checkout;
  these names/paths are proposals, not already-created resources.
- [x] Verify imports explicitly: `warpSPH.__file__`, `warpSPHCore.__file__`,
  versions, offset dtype and both source hashes. Use explicit per-process import
  paths or an isolated environment; bootstrap selects precision, not repo paths.
  Keep plotting/integrator dependencies pinned as well.

Exit: committed merged baseline, bounded integration results, isolated paired
implementation checkouts, and the existing turbulence checkout still available.
Do not reverse-sync or archive it while a source-pinned physics worker is active.

## Phase 1 — deterministic planner (core, CPU first)

- [x] Implement immutable generation/partition metadata and deterministic
  whole-row boundary selection from C/P, with separate counting and compute owners.
  Ownership policy/version, edge/byte budget and capacity rules are explicit.
- [x] Test empty populations/rows, zero-work ranks, strongly uneven counts,
  exact-budget boundaries, oversized rows, capacity padding and deterministic
  replica agreement. Exercise total prefixes>INT_MAX with metadata-only fixtures,
  without allocating billions of edges. Test exact local/global reconstruction.
- [x] Planner-local `RowChunkPolicy` exposes edge/byte budgets, exact rational
  headroom, alignment, scalar work extents and optional row caps. Compute owners
  are logical ranks. Production backend/device selection remains a Phase2/4 task;
  the flat API, global defaults and existing neighbour lanes are unchanged.

Exit: exhaustive row coverage once, no row split, bounded local capacity,
identical partition on all replicas. This phase is independently reviewable.

### Qualified implementation and limits (2026-10-30)

Core `src/warpSPHCore/radiusSearch/rowChunks.py` provides `RowChunkGeneration`,
`RowChunkPolicy`, `RowChunkPlan` and `buildRowChunkPlan`. Rebuild configuration
signatures are supplied by the caller and must cover ordering, reference state,
supports, domain, grid/search settings and stored-list acceptance. The plan digest
also includes every count/prefix, generation, capacity policy and ownership map;
`validateSignature` rejects disagreement before future local allocation.

The CPU reference consumes an explicitly assembled CPU int32 count vector,
snapshots its prefix into immutable Python integers and materializes CPU int64
global/local offsets regardless of the flat default. It allocates no edge arrays.
Counting ownership balances contiguous row intervals; compute ownership balances
contiguous chunk counts (`contiguous-balanced-chunks-v1`), with empty owners
allowed. This is deterministic, not a measured optimal load balancer. Greedy
chunks retain zero-neighbour rows and never split an oversized row; such a row raises an
explicit error. Budgets include ceil-rounded rational headroom and alignment,
and conservatively bound the largest flattened scalar work array by INT_MAX.
Callers must account for every colocated pair buffer in bytesPerEdge. Particle
fields, row metadata, grids and communication staging need separate accounting.

At the Phase1 checkpoint this was a host reference for the ownership contract.
Phase2 adds standalone search and resident adjacency below. The production
replicated device planner, large-N memory/performance, native solver dispatch,
graph/AD and distributed transport remain open; physical CUDA accesses above
INT_MAX remain unqualified.

Validation: 59 planner controls (including 1024 exhaustive small count layouts)
pass in both offset configurations. Combined with the existing index-width,
CSR-only, split-collection, query-order and Verlet-validity suites, the int64 run
passes 160/skips 89 in 2.75 s; default int32 passes 159/skips 90 in 2.76 s, zero failures.
Skips are CUDA controls and int64-only cases. These are unit fixtures, not
simulations; no video or GPU workload was started. Imports are explicitly pinned
to this paired workspace; core remains 0.6.3, with shared installs unchanged.

Reproduce from the core checkout with `/home/lu26029/miniconda3/envs/warp/bin/python`,
`PYTHONPATH` set to the two active checkout `src` paths, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`, writable `NUMBA_CACHE_DIR` and
`MPLCONFIGDIR`, and `warpSPHCore_OFFSET_DTYPE=int32` or `int64` set before import:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/radiusSearch/test_rowChunks.py tests/radiusSearch/test_indexDtypes.py \
  tests/radiusSearch/test_csrOnly.py tests/radiusSearch/test_splitCollection.py \
  tests/radiusSearch/test_queryOrder.py tests/radiusSearch/test_verletValidity.py
```

Local evidence/provenance: frontend `.tmp/row-chunk-planner-20261030/`.
Core implementation commit: `9a1057e` (on integrated base `04e126e`). Paired
source pins are recorded in `workspace-imports.json`; frontend's handoff commit
follows this core commit. The next direct count/collect and lifecycle milestone
is qualified below.

## Phase 2 — direct chunk collection and lifecycle (core, one GPU)

- [x] Add owned-row/global-row-base inputs to count/collect without changing
  neighbour acceptance or traversal order. Count first; allocate local buffers;
  collect directly. No monolithic intermediate J or reconstructed I.
- [x] Add a chunked adjacency container/iterator and connect validity, filters,
  capacity reuse, rebuild flags and generation tracking. Preserve the flat API
  for existing callers. Explicit diagnostics may request bounded/chunked export;
  do not hide a giant concatenation behind a compatibility property.
- [x] Force tiny budgets on ordinary2D/3D cases. Compare every live J/count/offset
  against the existing flat oracle, including periodic seams, unequal query/ref
  populations, all support modes and both stored-I/CSR-only contexts where supported.
  Use allocation/ownership checks to prove local construction and stable reuse.

Exit: exact order/content and count/collect consistency, correct invalidation,
no full-edge allocation, visible temporary/retained memory accounting.

### Qualified standalone API and limits (2026-10-30)

Core implementation: `31bc041`, following Phase1 `9a1057e`. New module
`radiusSearch/chunked.py` exports `ChunkedAdjacency`, `OwnedRowChunk`,
`radiusSearchRowChunks[_]`, `buildChunkedVerletList[_]`,
`filterChunkedVerletList[_]` and `chunkedVerletStaleFlag_`. These APIs and the
planner types are exported from the core root; they are explicitly opt-in.
There is no frontend backend switch or automatic solver dispatch yet.

The existing count/J-only collect/filter kernels receive query-position,
query-support and CSR-row slices together; references and J IDs stay global.
Host row bases define each owned interval. The only new Warp kernel fills optional
stored I with global query IDs. Traversal, acceptance, periodic wrapping, support
rules and per-row J ordering remain unchanged. Each generation freezes private
particle/domain snapshots before counting and collection. Counts synchronize to
the host reference planner at rebuild; the device also scans C into int64 P.
Local offsets subtract their global base in int64 before conversion to the
configured local offset width. Qualification uses int32 particle IDs/counts,
int32/int64 local offsets, FP32 CPU/CUDA and FP64 CPU.

`ChunkedAdjacency` has no global paddedJ/paddedI or implicit flattening property.
Iterate its owned chunks explicitly. Chunk adjacency row metadata is local;
source IDs and optional stored I are global. Do not pass these small adjacency
objects to unmodified global-row operators. `numOwners`/`owner` on explicit
search provide a single-device logical-owner fixture: all count intervals are
evaluated on that device, and only the selected owner's J is allocated. This
does not implement distributed counting or communication. Resident Verlet
lifecycle and filtering currently require one complete owner.

Resident lists return the same object while the existing motion/support validity
reduction accepts them. Forced rebuilds, motion/support changes beyond the validity buffer, and changes
to domain, support mode, scale, population, layout, budget or ordering key advance
generation/configuration metadata. Default rebuilds preserve old buffers;
explicit inPlace reuses pair/local-offset storage only for identical row intervals,
capacities and compatible layouts. It obsoletes old pair/CSR views; snapshots are
fresh. There is no maximal-capacity retention/shrink heuristic yet. Native W
accumulation/replay/migration across a rebuild is still Phase3 work.

Filtering checks validity first, then counts directly over source chunks and
collects each source/destination row overlap into the destination's local J.
It handles both merged and split chunk boundaries without assembling global pairs
or reconstructing CSR-only I. Filtered views record the source signature and
cannot be reused as resident Verlet supersets. A stale source fails before collect.
Callers must advance orderingKey when externally changing particle identity/order.
AD, deferred checks and CUDA graph capture fail explicitly; no flat fallback is
selected. Tensor-valued Verlet scales and distributed lifecycle remain unsupported.

Memory diagnostics report owned live/capacity pair bytes, policy-reserved pair
bytes, replicated/per-chunk row metadata, unique snapshots, host prefix bytes and
the grid/wrapped-query scratch frontier. These are logical storage measurements,
not allocator/process peaks or every intermediate of grid construction. Allocation
spies prove direct bounded J allocation and absent CSR-only I; resident and forced
rebuild controls prove reuse. Single-GPU resident chunking still retains all its
owned pair bytes; large-resolution fit and throughput remain Phase6 gates.

Validation (zero test failures/errors in the final gates):

|Gate|Passed|Skipped|Wall seconds|
|---|---:|---:|---:|
|Full core radiusSearch, CPU FP32/int32 offsets|487|521|5.10|
|Full core radiusSearch, CPU FP32/int64 offsets|488|520|5.01|
|New chunk search/lifecycle, CPU FP64/int64 offsets|186|187|7.19|
|CUDA chunk matrix, int32 offsets|173|1|21.91|
|CUDA chunk matrix, int64 offsets|173|1|40.80|
|Final fresh-field/wrapper extension, CPU+CUDA/int32|28|0|4.12|
|Final fresh-field/wrapper extension, CPU+CUDA/int64|28|0|3.84|
|Frontend CSR/native-work/lanes/driving compatibility, CPU/int64|73|2|383.86|

The CUDA matrix plus final extension cover 186 distinct CUDA cases per offset
setting (the extension repeats one wrapper control). CPU-only suites skip CUDA
controls; the CUDA matrix also selects the CPU variant of the CUDA-stream test,
which intentionally skips. All six support modes, 2D/3D, periodic seams and
out-of-domain images, unequal populations, empty rows/populations/owners,
headroom, global row identity, input freezing, signature disagreement, source
invalidity, capacity changes, fresh physical filter fields and nondefault CUDA
streams are exercised. Existing flat operator/gradient contexts remain covered
by the frontend compatibility controls. The new kernel is integer I fill; no
floating numerical kernel or model formula changed. Chunked AD is rejected.

These are bounded automated fixtures, without videos; no new flow campaign or
solver throughput benchmark was run. CUDA jobs were serial and the GPU was shared
with an existing workload; wall times include JIT/cache/test overhead and are
not comparable solver performance measurements.

Runtime import verification passes all 561 frontend modules against the paired
sources. The combined/static scanner still crashes at
`warpSPH.modules.incompressible.wp_alpha.computeAlpha` (`find_spec` on a module
without __path__). The identical crash reproduces with untouched core `04e126e`;
it is an inherited validation limitation, not a passing static check or a fix in
this change. Its baseline traceback and the initial fixture-construction failure
are retained with the final evidence.

Reproduce with explicit paired source PYTHONPATH, the conda Python above,
`OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`, a writable `WARP_CACHE_PATH`,
`NUMBA_CACHE_DIR` and `MPLCONFIGDIR`. Select offsets/precision before imports;
set `CUDA_VISIBLE_DEVICES=''` for CPU-only gates. Main core commands:

```bash
python -m pytest -q -p no:cacheprovider tests/radiusSearch
python -m pytest -q -p no:cacheprovider tests/radiusSearch/test_chunkedSearch.py -k cuda
python -m pytest -q -p no:cacheprovider tests/radiusSearch/test_chunkedSearch.py \
  -k 'fresh_particle_fields or particle_wrapper'
```

Frontend gates: `test_compressibleCSR.py`, `test_crkSharedWork.py`,
`test_compSharedWork.py`, `test_compSPHPairWork.py`, `test_neighborLanes.py` and
`test_drivenTurbulence3D.py`, plus `scripts/check_imports.py --runtime`.
Evidence/provenance/commands: frontend `.tmp/chunked-search-20261030/`.
Shared editable installs and the physics worktrees remain unchanged by this work.
That Phase2 handoff led to the shared operators and native compatible-work
checkpoints recorded below.

## Phase 3 — shared operators and native compatible work (both repos)

- [x] Map row ownership through host contexts before canonical extraction/state
  bundles and prepared launchers. Reuse unchanged local-row adjacency/range/kernel
  ABIs. Slice all query fields; keep references and J IDs global. Grid sortIndex
  remains a particle permutation. Reject unported specs explicitly.
- [x] Bring up shared support/density/corrections/gradients and an explicit eager
  Monaghan path. Fixed lanes/tableau and full stored-field parity qualify the
  bounded checkpoint below, not every scheme/boundary/graph configuration.
- [x] Bring up native CRKSPH/CompSPH solver dispatch and compatible work. A CRK
  correction-construction pass is not a CRKSPH compatible-energy solver gate.
- [x] Allocate/accumulate W per owned chunk across stages. First complete force
  outputs for all rows; then form stage half velocities and run work sweeps.
  Final energy reduction uses local W and global final half velocities.
- [x] Pin pair identity/ordering and ownership during W accumulation. At changed
  lists or repartition, preserve existing replay semantics; explicitly migrate
  matching work if needed. Never silently discard work or use stale fields.
- [x] Force mid-stage list changes, multiple integrator weights and chunk-layout
  changes; compare full live W and all stored fields against the flat baseline.
  Require bitwise per-row parity where arithmetic order is unchanged. Keep earlier
  recomputation/lane roundoff failures visible; do not relax screens to force a pass.

Exit: native fresh-stage correctness, energy/source budgets, positivity, no extra
or missing RHS, and independently measured scratch/retained-memory changes.

### Shared-operator and Monaghan checkpoint (2026-10-30)

Core `autograd/row_dispatch.py` implements an explicit host adapter, without
changing floating-point kernels or the canonical Warp struct ABI. For each
resident chunk, positions, supports, masses, densities and kinds are sliced as
query fields. Reference particle arrays remain global. Volumes, CRK A/B/gradA/
gradB, grad-H and renormalization fields receive the same role mapping. Self
aliases resolve before slicing. A single query CRK state is permitted for
query-only consistency density; operators that read reference corrections
validate their reference population independently.

`OperatorSpec.rowChunked` opts a particle-output kernel into this contract.
`ExtraSpec.rowRole` declares QUERY, REFERENCE or REPLICATED, optionally through a
resolver following an actual enabled/disabled kernel branch. It never guesses
identity from population size. Disabled reconstruction dummies stay replicated;
active velocity Jacobians, Balsara factors and reconstructed state gradients
are mapped as particle fields. Pair-shaped outputs, unannotated extras, partial
owners, AD and deferred/captured execution are rejected. Per-query outputs are
assembled by disjoint copies into O(N) particle arrays, without cross-chunk sums
or a concatenated pair list. Local output scratch is released after each chunk.
This host loop is a correctness implementation, not a throughput qualification.

Prepared launchers hold a canonical prefix per chunk and use the same mapping.
They must be rebuilt when state tensor bindings change, as for flat prepared
prefixes. A private storage lease now invalidates old operator/prepared/filter/
Verlet consumers when an in-place rebuild actually reuses CSR/J buffers. Default
rebuilds leave old independent buffers valid. This makes buffer obsolescence
explicit; it does not implement W migration or replay.

Qualified core consumers: density, interpolation, gradient, divergence, curl,
Brookshaw Laplacian, covariance (including count output), CRK volume/moments/
consistency density and renormalization construction. Tests include self queries,
distinct equal-sized and unequal populations, 2D/3D, full correction roles,
periodic support/direction masks, empty populations, prepared extras refreshed
between calls, unsafe-reference validation and obsolete-buffer failures. Scalar
lanes are fixed for the full matrix; density/covariance also pass fixed 32-lane
chunk-versus-flat checks on a nondefault CUDA stream. This does not qualify every
Laplacian scheme, lane combination or AD path for chunks.

Frontend enables Omega/Psi0 support operators and classic Monaghan dissipation
operators. The ParticleState `buildVerletList` wrapper preserves an explicitly
seeded chunked representation/policy through subsequent validity checks and
rebuilds, rejecting flat capacity/grid-factory overrides. For an eager, wall-free
Monaghan case, select the path in its Python `buildSystem` hook:

```python
from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy
config.compute.neighborOmitI = True
system.adjacency = buildChunkedVerletList(
    system.state, config.domain,
    policy=RowChunkPolicy(edgeBudget=1_000_000, bytesPerEdge=4),
    verletScale=config.verletScale, storeI=False)
```

No default or runner CLI selection changes. The fixtures use `store=False` and
`storeMode='trajectory'` to avoid the checkpoint serializer. Chunked native
checkpoint export raises explicitly; checkpoint/restart remains Phase5 work.
Monaghan wall closure and per-side pressure reject at entry. The latter has a
direct `i == j` check needing global query identity before it can be opted in.
Other unported solver/spec contexts remain gated.

Five two-step eager trajectory fixtures cover fixed support/RK2, Monaghan support
with grad-H/RK4, 3D Owen support/RK2, active velocity/Balsara reconstruction, and
active Riemann-state reconstruction. They compare every stored tensor exactly,
time, expected 4/8 fresh RHS calls, positivity and the existing source/energy
budget gate. Monolithic pair search is forbidden after the explicit seed. These
are tiny unit A/B fixtures: no video or long flow campaign, with progress/stall
monitoring enabled. They do not establish stationary turbulence or scaling.

Qualification, using source-pinned paired worktrees and fresh offset processes:

| Gate | Result |
| --- | --- |
| New core operators, CPU float32/int32 | 53 passed, 55 skipped |
| New core operators, CPU float32/int64 | 53 passed, 55 skipped |
| New core operators, CPU float64/int64 | 53 passed, 55 skipped |
| New core operators, CUDA float32/int32 | 54 passed, 1 skipped; CPU cases deselected |
| New core operators, CUDA float32/int64 | 54 passed, 1 skipped; CPU cases deselected |
| Monaghan CPU float32, int32/int64 | 5 trajectories passed per offset |
| Monaghan CUDA float32, int32/int64 | 5 trajectories passed per offset |
| Existing core CPU/int64 focused regressions | 834 passed, 849 skipped |
| Existing frontend CPU/int64 focused regressions | 73 passed, 2 skipped |
| Frontend runtime import checker | 561 modules passed |

The CUDA selection also matches the CPU parametrization of the test whose name
contains "cuda"; that one CPU stream-test case skips as intended. All 54 actual
CUDA parametrizations pass. CPU-only processes skip those 54 CUDA cases plus
their CPU stream-test case.
The existing static import-scanner crash from Phase2 remains recorded; runtime
imports pass. One initial import attempt omitted NUMBA_CACHE_DIR and failed on a
non-writable source cache; the corrected, source-pinned environment passes.
Development also corrected a test return-type assumption, then declared CRK
consistency density's genuinely query-only correction role; numerical tolerances
were not relaxed. The initial native checkpoint AttributeError was replaced by
the explicit unsupported export gate above.

Evidence/commands/source provenance: frontend `.tmp/chunked-operators-20261030/`.
The bounded tests run GPU suites sequentially. Timings include JIT and shared GPU
contention; no throughput, peak-memory saving, large-edge addressing or physical
multi-GPU claim is made. Single-device resident chunks retain total J storage;
this checkpoint bounds per-chunk addressing and operator scratch, not total J.
The next step at that checkpoint was chunk-owned native compatible work; it is
now qualified in the following section. Paired implementation provenance: core c0b8cf7 (frontend is the commit containing this checkpoint).

### Native chunk-owned compatible work (2026-10-30)

This checkpoint adds eager, wall-free native recomputation for CRKSPH and CompSPH,
with the original compatible-energy formula and fresh Butcher stages. It does
not change forcing, support iteration/acceptance, EOS or integrator weights.
The old flat path remains the default. Native special energy/material/AD/graph/
wall controls are gated at entry for chunks rather than allocating expanded pairs.

Core `radiusSearch/pairField.py` introduces `ChunkedPairField`: a frozen tuple of
dedicated bounded buffers associated with one adjacency object/generation and
its storage lease. It validates capacity, dtype, device, stride, disjoint storage
and policy extents. Hidden larger backing allocations and shared buffers across
chunks are rejected. There is no flat tensor view. `ExtraKind.CHUNK_TENSOR` and
`RowRole.CHUNK` bind each work input to its matching chunk; `ROW_START` supplies
global query identity for reads of replicated stage/final half velocities.
Prepared launches use the same mapping. Only explicit one-element unread ABI
scratch outputs are allowed beside query outputs; the first output must still
define the query thread count.

Force-only dispatch reuses the existing scalar/tiled force kernels with bounded
one-element pair placeholders. It completes all query-force outputs before
forming stage half velocities and starting the work sweep. Each chunk then
accumulates its own W with the same pair formula, fraction and stage weight.
Final reduction walks each local CSR row in order, reading global final half
velocities and writing its disjoint global particle result. It never assembles W
or edge-level velocity gathers on a device. Allocation/order audits in the test
fixtures check these statements against actual launches and tensor storage.

A changed adjacency identity or repartition triggers the existing `RerunRequest`:
the first attempt computes the global final half velocity, then the integrator
repeats all fresh stages and reduces each stage on its current list directly.
The replay retains no W, only a one-element unread placeholder and particle
increments. No implicit work migration or silent discard-and-accept was added.
Tests force a generation/layout change at stage 1, including a row cap of 5 -> 2.
Each two-step RK4 fixture performs the same 16 RHS calls as its flat replay;
normal RK2/RK4 controls perform exactly 4/8 RHS calls.

Policies must budget both J and W, including the scalar work extent. Example
for a native compatible scheme in a Python case build hook:

```python
from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
dim = system.state.positions.shape[1]
policy = RowChunkPolicy(
    edgeBudget=1_000_000,
    bytesPerEdge=4 + dim * system.state.positions.element_size(),
    maxElementsPerEdge=dim)
config.compute.neighborOmitI = True
system.adjacency = buildChunkedVerletList(
    system.state, config.domain, policy=policy,
    verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
    storeI=False)
```

The four-byte ID term assumes omitted I; include another four bytes for stored I.
A J-only policy fails before work allocation. This is a single-owner selection;
partial-owner execution remains gated until the distributed exchange schedule
exists. Native checkpoint/HDF5 export also fails explicitly, before creating a
partial output file. Use the qualified `store=False, storeMode='trajectory'`
fixture configuration until Phase5 implements a chunk-local checkpoint format.

The new CUDA nondefault-stream control initially exposed an existing flat-path
dependency bug: CRK terms and flat work reductions launched outside Torch's
current stream. The first screen differed by one ULP; an isolated run could
read uninitialized values and fail positivity even on the flat control. Those
helpers now use the existing `torchWarpStream` context. Their floating kernels
are unchanged. Strict 32-lane chunk/flat parity then passes for both schemes.
Delayed-producer tests independently exercise terms, flat/chunk work reduction
and reference entropy work; repaired flat helpers also capture/replay correctly.
Chunked capture remains explicitly rejected. Failed initial logs/XML are kept;
no numerical tolerance was relaxed to remove the failure.

Qualification (fresh offset/precision processes, paired source paths):

| Gate | Result |
| --- | --- |
| Core pair-field/row-operator matrix, CPU float32 int32/int64 | 59 passed per offset, 61 expected skips |
| Core pair-field/row-operator matrix, CPU float64/int64 | 59 passed, 61 expected skips |
| Core pair-field/row-operator matrix, CUDA float32 int32/int64 | 60 passed per offset, 1 CPU stream-test skip |
| Native scalar-lane trajectory/work/replay cases, CPU int32/int64 and FP64/int64 | 8 cases pass per configuration |
| Native trajectory/work/replay plus fixed 32 lanes/nondefault stream, CUDA int32/int64 | 10 cases pass per offset |
| Padding/empty/generation/policy/unsupported-context/export/stream contracts | 15 CPU checks plus 16 CUDA checks per offset; capture/export included |
| Existing core CPU/int64 focused regressions | 834 passed, 849 expected skips |
| Existing frontend work/pair-work/Monaghan/graph-selection CPU controls | 57 passed, 18 CUDA skips |
| Existing native flat work CUDA controls | 33 passed |
| Runtime frontend imports | 562 modules pass |

Every live W vector, its J identity/order, stage force/heating/increment and all
stored final tensor fields match the native flat recomputation control bitwise.
The native recomputation/expanded-control regressions retain their existing
tolerances; they were not changed. All bounded accepted steps pass positivity
and the existing source/energy-budget gate. The largest flow fixture is 216
particles; these are two-step unit A/Bs with progress/stall monitoring, with the
documented video exception. No new developed-flow campaign was launched.

Measured storage example, 64-particle 2D RK4: both schemes retain 36,040 bytes of
flat W capacity versus 36,048 bytes across chunks (padding), with identical live
work. Largest force-kernel output storage is 784 bytes flat versus 76 bytes per
chunk at the debug row cap of 5. The chunk adapter additionally assembles the
full O(N) particle output, so this is **not a peak-memory saving**. Single-device
resident chunks keep total J/W; distribution will reduce ownership per GPU.
The allocation audit reports actual tensor storage separately from allocator/
native/JIT/process memory. Tiny CPU RK4 examples took roughly 24 -> 112 ms/step
(CRK) and 19 -> 82 ms/step (Comp); host-loop overhead is visible and expected at
this debug chunk size. Logs include all wall/step times. JIT and shared workload
contention prevent throughput conclusions; no performance default is promoted.

Commands, test XML/logs, allocation/timing audits, failure records and paired
source fingerprints: frontend `.tmp/chunked-compatible-work-20261030/`.
The existing static-scanner issue remains recorded in Phase2. No physical
>INT_MAX edge-buffer access, large-memory fit or physical multi-GPU claim follows
from these fixtures. Next action is the Phase4 replicated-state process backend.
Implementation provenance: core 85b5ef9; frontend is the commit containing this checkpoint.

## Phase 4 — replicated-state multi-GPU runner (two GPUs, then four)

User scope (2026-10-30): start the process proof of concept with two separate
ranks sharing the existing GPU using isolated NCCL 2.32.3. ACSPH continues to
use the device; keep fixtures bounded, qualify correctness/stability and treat
all timings as diagnostic. Do not wait for an idle device or infer performance,
physical interconnect behavior or multi-device memory fit from these runs.

- [x] Explicit c10d process-group binding with NCCL CUDA and Gloo CPU under the
  same interface; torchrun process correctness/failure tests. Production retains
  one process per GPU; shared-GPU NCCL is an experimental debug exception.
- [x] Disjoint local counts -> replicated int32 counts / int64 scan -> identical
  plan/signature -> allocate/collect owned chunks only; coordinated resident
  reuse/rebuild and reference-grid ordering agreement.
- [x] Eager Monaghan/native CRKSPH/CompSPH dependent particle-field exchanges,
  owned compatible W, global half velocities and changed-generation replay.
- [x] Global timestep/rebuild/positive/failure and unique-owner source/work
  reductions; exact replicated OU clock/history/mean-removal contract.
- [x] Uneven/empty owners, unequal populations, nondefault streams, local capacity
  changes, deliberate mismatches, two-/three-process CPU tests and two real
  shared-GPU NCCL ranks. Fatal rank termination is tested with torchrun/Gloo.
- [ ] Physical two-/four-GPU execution, actual cross-device neighbours and memory,
  topology/Isambard Slurm bootstrap/network configuration, device-fault failure
  behavior, conservation/positivity and short-step reference comparisons.
- [ ] Large physical edge storage/capacity fit and target-network performance;
  declare physical collective tolerances and measure wait/compute/device memory
  on the target topology without shared-workload timing assumptions.

Exit: no full J/W on any rank, correct synchronization and failure handling,
per-device memory measured, physical conservation/positivity gates passed.

### Transport decision and local debugging (2026-10-30)

User preference: NCCL is the preferred communication library on Isambard and
PyTorch provides the intended interaction layer. Production execution will use
`torch.distributed`/c10d with NCCL, one process per GPU. Warp continues to compute
through Torch-backed device buffers, with explicit stream/event ordering before
communication and dependent kernels. Collectives exchange counts and owned
particle outputs plus global timestep/rebuild/failure/source reductions; pair
J/W stay on their owners. The bounded eager process checkpoint below is now
implemented; physical multi-GPU/site qualification remains open.

Shared base package inspection: Torch 2.13.0+cu130, NCCL 2.29.7, both NCCL
and Gloo backends available. Torch pins the NCCL wheel exactly to 2.29.7; that
base does not support sharing one GPU among multiple ranks of the communicator.
The user subsequently authorized an NCCL upgrade if useful for debugging.
An isolated NCCL 2.32.3 runtime overlay is now installed and smoke-qualified below;
the shared conda environment and existing physics jobs retain their original
libraries. Production retains one process per GPU.

NCCL **2.30+** documents an experimental exception:
`NCCL_MULTI_RANK_GPU_ENABLE=1`. Each separate process must explicitly select the
shared device (e.g. cuda:0), instead of selecting LOCAL_RANK as a device ordinal.
NVIDIA expects up to two ranks per GPU with the default resource configuration;
more/larger ranks may require lower NCCL_MAX_CTAS and can exhaust resources or
hang. NVLS is incompatible with same-GPU ranks; the default auto setting disables
it, while forcing NVLS on fails initialization. Separate rank threads in one
process sharing that GPU are not supported by this feature. The isolated
2.32.3 runtime override below passes the bounded PyTorch/Warp smoke; this is not
a general compatibility qualification of every PyTorch/NCCL operation. No shared
package installation is changed.

Local debug choices now include the tested two-process/one-GPU NCCL 2.32.3
runtime and a qualified Gloo CPU transport for device-independent process/failure
tests. A single-rank NCCL smoke checks API plumbing only. Real two-/four-GPU NCCL
qualification is still required for device separation, transport, stream ordering
and performance on the target topology.

Sources: [PyTorch distributed process groups](https://docs.pytorch.org/docs/stable/distributed.html),
[NVIDIA same-GPU rank support and limits](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable).
The specific experimental switch qualifies the otherwise general prohibition in
the communicator/PyTorch documentation; do not assume it exists in older stacks.

### Isolated NCCL upgrade and shared-GPU smoke (2026-10-30)

Installed the official `nvidia-nccl-cu13==2.32.3` wheel with `--no-deps --target`
into frontend `.tmp/nccl-shared-gpu-20261030/overlay`. No shared Python or editable
package was replaced. `with-nccl-2.32.sh` prepends that overlay to PYTHONPATH and
preloads its `libnccl.so.2` for the launched Python process and its children.
Both settings matter: PyTorch eagerly preloads native libraries by searching
sys.path, so preloading alone also mapped the original NCCL. The final wrapper
verifies/uses a single mapped NCCL library from the overlay. It selects the native
runtime only; continue to set the paired frontend/core source paths explicitly
in PYTHONPATH when running SPH code.

`torch.cuda.nccl.version()` still reports the **build-time** 2.29.7 for this Torch
binary. The real loaded runtime was independently verified with `ncclGetVersion`
(23203), /proc/self/maps (one overlay library), and NCCL's communicator-init log
(2.32.3). A fresh process without the wrapper still loads only base NCCL 2.29.7.
The shared package dependency pin remains satisfied; this is an explicitly
selected experimental native-runtime override, not a replacement Torch build.

Ran torchrun with two separate OS processes, both explicitly selecting cuda:0.
Distinct PIDs and the same physical GPU UUID are recorded. Settings were
`NCCL_MULTI_RANK_GPU_ENABLE=1`, `NCCL_MIN_CTAS=1`, `NCCL_MAX_CTAS=2`, and
`NCCL_NVLS_ENABLE=0`, with a 20-second process-group timeout and 60-second outer
timeout. No MPS configuration or system library changes were made.

Both ranks passed all seven check groups:

- Uneven int32 count all-gather and replicated int64 prefix reconstruction.
- Padded owned particle-field all-gather.
- Global MIN timestep, MAX rebuild flag, SUM over unique owned particles.
- Broadcast.
- Int64 reduce-scatter.
- Batched point-to-point send/receive.
- Warp producer -> asynchronous NCCL all-reduce -> Warp consumer on a nondefault
  Torch CUDA stream, five exact repetitions.

Per-rank probe time was about 1.38 seconds excluding Python/torchrun startup;
this is a correctness fixture, not a benchmark. Torch allocator counters in the
reports exclude NCCL/native/driver allocations and are not process memory peaks.
Local optional IB/RMA plugin warnings fell back successfully; this run does not
test Isambard networking. No SPH solver or new flow campaign was run. The Phase4
runner/collective scheduling and physical multi-GPU gates remain unimplemented.

Re-run using the executable frontend
`.tmp/nccl-shared-gpu-20261030/run-nccl-shared-gpu-smoke.sh`.
The same directory contains the wrapper, smoke/verifier/summary scripts,
smoke.log, per-rank JSON, base/overlay provenance and summary.json with a library
SHA256. This runtime is now available for the subsequent distributed backend work.
The package source is [NVIDIA NCCL CUDA13 on PyPI](https://pypi.org/project/nvidia-nccl-cu13/2.32.3/).

### Bounded replicated-state process checkpoint (2026-10-30)

The opt-in eager process proof of concept now runs separate OS processes with
Gloo CPU or NCCL CUDA tensors. Two real NCCL ranks sharing the existing GPU
pass solver correctness and repeated rebuild/replay checks under ACSPH
contention. This qualifies the process protocol locally; the physical two-/four-
GPU and Isambard gates above remain open.

Core `warpSPHCore.distributed.RowChunkProcessGroup` binds an existing, explicitly
initialized c10d group to a device. The caller owns group initialization with a
finite timeout, device selection and destruction. Construct `communication = RowChunkProcessGroup(dist.group.WORLD, device)`
and pass `communication=communication` to `radiusSearchRowChunks` / `buildChunkedVerletList`; subsequent resident builds
infer it from the prior adjacency. Logical owners without a group remain test
fixtures and still cannot produce distributed particle results.

Each rank counts only its balanced counting interval. Disjoint broadcasts
assemble the replicated int32 counts; every rank builds its int64 prefix and
the same deterministic plan. Configuration, generation, complete plan signature
and reference-grid ordering agree **before owned pair allocation/collection**.
Only owned flat J/I and W buffers exist in the distributed path; no pair gather
or flattened compatibility buffer was added. Changing capacity/row boundaries
rebuilds directly on the new owners. Filtering an existing partial-owner list
remains explicitly unsupported; it would need a separate pair-migration design.

The existing row-dispatch ABI runs only local chunks, then broadcasts disjoint
particle-output intervals to every replica before returning. Query/reference
roles, canonical corrections, output layouts, scalar controls and operation
sequence are explicit; reference inputs are hashed exactly in this debug path.
Prepared launches follow the same protocol. Unread one-element force scratch
and chunk-local work buffers never enter particle collectives.

The qualified eager dependency schedule is:

1. Support/density outputs are exchanged before the next support iteration or
   correction pass. CRK volume -> moments -> density, gradients/limiters and
   optional grad-H fields therefore have complete reference inputs. Pointwise
   correction algebra runs on complete replicas and is checked before use.
2. Force/heating outputs are exchanged before either global half-velocity reads
   or the compatible-work sweep. Native CRKSPH/CompSPH accumulate W only on their
   owned chunks, in the existing row/edge/stage order.
3. Final half velocities are complete replicas. Each owner reduces its W directly
   to particle energy increments, which are exchanged before accepted updates.
   A midstage generation change coordinates a rebuild and preserves the existing
   full fresh-stage replay, including repartitioning.
4. Global MIN selects the timestep, MAX coordinates rebuild/invalid-state flags,
   and FP64 SUM counts uniquely owned source/work contributions once. Replicated
   FourierOU fields retain the original global mean-removal summation order;
   their clock, coefficients, generator and resulting fields must agree exactly.
   No duplicated full-replica SUM is used. Distributed body sources currently
   qualify `ConservedSources` with `FourierOU`; custom sources remain gated.

The ordinary eager runner recognizes an explicitly bound adjacency. It validates
configuration/initial replicas, picks a global timestep, and checks positive,
finite, equal accepted states and time. An optional `neighborhoodFactory(state,
config)` in the compressible sampler and driven case's `buildSPHSystem` lets the
initial density/support setup use owned chunks too. The process solver fixture
forbids monolithic search from before that setup. There is no default/CLI change,
automatic process-group initialization, output/checkpoint support or promotion
of unqualified AD/graph/material/wall/scheme contexts.

Control handshakes propagate local exceptions and detect differing sequence,
operation, layout, generation or replicated reference data before payload
collectives. Hard process death/device faults rely on c10d's finite timeout and
torchrun peer termination. Real two-/three-process CPU tests include fatal rank
termination; two-rank NCCL tests cover eight recoverable contract failures.

The longer run found a real time-precision bug in the first process adapter:
converting a float32 tensor dt to Python float changed accumulated time after
three steps (0.0005999999848427251 versus 0.0005999999702908099), with the same
24 RHS evaluations. The global MIN now preserves the original tensor dtype and
scalar representation. The three-step regression, longer run and forcing-clock
crossings retain exact time/stage/particle/W/history checks; no tolerance was
relaxed. Failed longer-run logs and the short CPU reproduction are retained.

An existing source-replay regression still patched only legacy `accumulateStage`
even though native recomputation already bypassed it before this work. Its hook
now spoils both storage paths, so the default actually performs the requested
replay. Both 20-step source controls and all 57 focused existing native/source
checks pass; this repair changes the test injection, not solver arithmetic.

Qualified controls and evidence:

|Control|Result|
|---|---|
|Separate-process CPU regression drivers, FP32/int32 and FP64/int64|5 pass each: two-/three-rank core, fatal peer death, two-/three-step solver comparisons|
|Existing chunk/planner/operator/native/Monaghan CPU controls, each local offset width|332 pass, 251 expected CUDA skips per width|
|Extended eager solver matrix|8 fixtures / 68 fresh RHS per rank, with 2D/3D, RK2/RK4, adaptive supports, source forcing and forced replay; Gloo FP32/FP64 and shared-GPU NCCL offset controls recorded|
|NCCL 32-lane controls|Native work/replay remains bitwise against the corresponding single-process lane configuration|
|Longer NCCL control, int64 FP32|16 accepted steps and 128 RHS per native scheme per rank; 16 forced rebuilds per scheme; 15 OU knots at interval 0.00025|
|Existing native/source CPU compatibility|57 pass, including repaired source replay injection|
|Runtime imports|563 frontend modules pass; inherited static scanner `wp_alpha.computeAlpha` failure remains unchanged|

Particle fields, live owned W/J order, fresh stage rates, replay increments,
simulated time and complete OU knot/generator histories match the serial oracle
bitwise. Only regrouped FP64 global source-budget sums use the predeclared
`rtol=atol=2e-12`. Accepted positivity and the existing relative energy-budget
gate of 2e-6 pass. Longer-run residuals were approximately 5.04e-9 (CRKSPH) and
3.51e-9 (CompSPH), with both ranks identical. The largest fixture is 216 particles;
videos are omitted for these tiny A/B loops, with progress/stall monitoring kept.

For all 16 accepted samples, per-rank Torch allocated bytes were constant within
each scheme (about 64.1 MiB). These counters exclude native/driver allocations
and are not process peaks or large-memory-fit evidence. Logical J/W ownership
is recorded separately: in the 64-particle initial CRK fixture, rank 0 owns 35
rows and 9,856 J-capacity bytes / 19,712 W-capacity bytes; the other rank owns the
remaining rows. Aggregate same-GPU memory still includes every owner's pairs
plus extra process/communication resources. No capacity scaling claim follows.

Telemetry includes cumulative logical particle payload/control bytes, generation,
partition and a bounded last-512-event ring of host local/enqueue and control/
collective wall times. Exact replica hashes add CPU transfers and synchronization;
the debug broadcasts prioritize correctness and bounded particle scratch. These
times are diagnostic under ACSPH contention, not GPU compute/transport benchmarks.

Maintained regressions live in core `tests/distributed/` and frontend
`tests/{test_distributed.py,distributed/_solverWorker.py}`. Tiny flat references
are explicit test oracles, released before the distributed solver phase; only
small CPU oracle snapshots remain. Logs, per-rank manifests, source fingerprints,
runtime-library provenance, failure records and reusable launch commands are in
frontend `.tmp/distributed-poc-20261030/`. This includes the isolated NCCL 2.32.3
wrapper used without changing the shared environment or ACSPH sources.

Implementation provenance: core `44bccb9`; frontend is the commit containing this checkpoint.

Next: physical multi-GPU/device/topology and Isambard launch/network qualification,
then larger storage/capacity and performance gates. Distributed filtering,
checkpoint/restart, other schemes, AD/graphs and walls/materials remain gated.


## Local continuation sequence (user-approved, 2026-10-30)

Complete these in the existing paired workspace while physical multi-GPU access
is unavailable. Keep CPU/Gloo and bounded two-process shared-GPU NCCL controls;
ACSPH contention means timing remains diagnostic. The ordering is deliberate.

1. [x] **L1 — accepted-step checkpoint/restart (bounded native/legacy v1 scope qualified).** Save canonical particle,
   exact time/timestep, forcing/generator/budget state and sufficient frozen search
   history to reconstruct owned J without exporting global J/W. Qualify changed
   chunk budgets and 1/2/3 CPU ranks, plus 1/2 NCCL ranks on one GPU. Compare
   uninterrupted versus restarted fresh stages, physical fields, pair order and
   forcing history; cover interrupted/corrupt writes and physical-policy mismatch.
   The versioned native recipe and changed int32-to-int64 local offsets now pass.
   Version-1 COO/CSR import and ordering/offset conversion now pass within the
   qualified scope below; other legacy/search policies and physical devices remain gated.
2. [x] **L2 — local launcher/output (site validation open).** Explicit rank/device/chunk configuration, rank-zero
   diagnostics/checkpoint output, source-pinned run manifests, meaningful failure
   reporting and locally checked Isambard submission templates. Final site launch
   validation still requires Isambard. Keep storage and solver code independent
   of the communicator. NCCL is implemented; optional MPI should require adapter/
   capability/failure tests rather than SPH kernel changes. The current Torch
   build has no MPI backend. User decision (2026-10-30): defer MPI-enabled and
   ROCm-enabled Torch builds; continue the local sequence on the existing stack.
   Distinguish MPI rank launching with NCCL payloads
   from an MPI data backend; do not replace/build shared packages for this task.
3. [x] **L3 — communication overhead (bounded local scope qualified).** Separate exhaustive replica hashing from
   normal validation, combine exchanges where dependencies allow, reuse metadata/
   particle scratch and reduce unnecessary host waits. Prove correctness, buffer
   lifetimes and fewer operations/allocations locally; defer throughput claims.
   Cached controls/metadata, explicit exact/contract validation, compatible output
   packing and one bounded particle buffer now pass the qualification below.
4. [x] **L4 — distributed filtering/redistribution (bounded local scope qualified).** Filter resident partial-owner
   lists and handle changed boundaries without a global pair intermediate. Preserve
   pair identity during accumulation or coordinate the existing fresh-stage replay.
5. [x] **L5 — stronger stability/failure controls (bounded local scope qualified).** Randomized partitions, skewed
   counts, empty owners, repeated capacity changes, longer forcing histories,
   failures during solver/IO phases and bounded memory checks; use more CPU ranks.
6. [x] **L6 — compatibility expansion (bounded local scope qualified).** GSPH and
   InutsukaGSPH join the eager five-scheme matrix, including prepared operators and
   native restart. Explicitly test rejections for boundaries/materials, graphs,
   differentiation and unported schemes; retain native CRKSPH/CompSPH controls.
   This completes the bounded local milestone, not every Phase5 compatibility port.
7. [x] **L7 — actual wide-edge device access (bounded debug scope qualified).** A total edge
   count above INT_MAX across bounded chunks is testable on one physical GPU.
   Require a memory/headroom check and actual device accesses; metadata/aliases
   do not qualify it. User authorizes large single-node/single-GPU multi-process
   debugging on2026-10-30; perform a fresh free-memory check before allocations.

Physical inter-device transport/topology, aggregate multi-GPU capacity and
scaling remain separate gates. This sequence does not claim to complete them.

### Accepted-step native restart checkpoint (2026-10-30)

The first L1 slice now implements an explicit native checkpoint format for eager
driven Monaghan/CRKSPH/CompSPH with resident row chunks and ConservedSources /
FourierOU. The same API works with no process group, Gloo CPU, or NCCL CUDA.
Legacy flat-file import, automatic output cadence and physical multi-GPU restart
remain separate gates. No public defaults, CLI or shared packages changed.

Every rank calls `warpSPH.io.chunkCheckpoint.writeChunkedCheckpoint(path,
result.ctx, result.state)` after an accepted runner step. A weak identity/time/
completed-step token rejects stage/intermediate saves without retaining the
previous system or W. Rank zero writes a versioned HDF5 file to a temporary sibling,
checksums all datasets/metadata, closes/fsyncs it and atomically publishes it.
Existing files reject by default; `overwrite=True` explicitly replaces atomically.
Recoverable IO errors propagate through the existing local-phase protocol. A hard
process kill may leave an unpublished temporary file; no fault-tolerant filesystem
or physical-device failure claim follows from the injected exception tests.

The file stores canonical particle fields, accepted step count, exact time and next
dt representation/dtype, initial mass/energy reference, source energy budget, and
OU parameters/knots/generator. It also stores a versioned deterministic search
recipe: frozen query/reference positions and scaled supports, domain, int32 counts,
generation and ordering metadata. Frozen inputs matter: reconstructing only from
the current particle state could change a still-valid resident list and replay
semantics. Query/reference aliases are preserved. No J, I, W, saved local offset
array, communicator or device ordinal is serialized.

`radiusSearch.rowChunkCheckpointState` produces an independent inert CPU recipe;
`restoreRowChunkCheckpointState` validates it and recounts the frozen geometry.
It derives the target plan from the saved counts and a newly selected policy/
owner count, agrees on the complete plan before pair allocation, then collects
only locally owned J/I. Prefixes are reconstructed in int64; local offsets use
the target process's configured int32/int64 type. Particle precision must match.
The traversal algorithm is explicitly versioned; arbitrary older search ordering
is not assumed compatible. Restore rejects checksum, algorithm, precision,
shape/count, budget and recounted-plan mismatches.

For a continuation, seed the target runner with the existing explicit chunked
`neighborhoodFactory` and set `simulation['resumeFrom'] = path`. The target factory
selects the new policy and process group; the runner loads this format before
distributed startup, restores source/time/dt/context, discards transient pair work,
and rebuilds local adjacency. This bounded path still initializes a seed list
before restore, so seed and restored storage overlap until installation; remove
that startup duplication before large-capacity qualification. Physical/numerical configuration must match; chunk
budget, rank count and offset width may differ. The old flat checkpoint branch
and automatic-output guards remain in place. There is no monolithic fallback.

Qualification (tiny 64-particle, four-step uninterrupted versus two-step save +
two-step continuation controls; video omitted for this bounded A/B matrix,
progress/stall/velocity controls enabled):

|Control|Result|
|---|---|
|Separate Gloo jobs, CPU FP32|3 scheme drivers pass: CRKSPH 2→3 ranks, CompSPH 1→2, Monaghan 3→1; all change maxRows 3→1 and int32→int64 offsets|
|Separate Gloo jobs, CPU FP64|Same 3 drivers pass plus 14 core recipe cases; 22 independent rank reports / 168 RHS evaluations, with 14 expected CUDA skips|
|Shared-GPU NCCL 2.32.3, FP32|10 serial jobs / 14 rank reports / 136 RHS evaluations pass: CRKSPH 2→1 and 2→2, CompSPH 1→2, Monaghan 2→1|
|Empty owners|CRKSPH large-budget restore creates one chunk with two empty Gloo owners or one empty NCCL owner, including empty W storage|
|Failure/publication controls|Mid-stage save, existing-file overwrite, injected write failure/cleanup/retry, corrupt state checksum and changed forcing configuration reject; live peers remain coordinated|
|Core recipe CPU FP32/int32, CPU FP64/int64|14 pass each, 14 expected CUDA skips; self/unequal/empty populations, alias/order/reuse and malformed/re-signed recipes before pair allocation|
|Core recipe CUDA FP32/int32 and int64|14 pass each; non-CPU cases selected|
|Existing search/operators/native-work CPU FP32/int64|344 pass, 263 expected skips; includes the initial 12 CPU recipe cases, before two additional recipe guards|
|Legacy native checkpoint regressions|7 pass, preserving the old runner branch|
|Runtime imports|564 modules pass; static scanner still aborts on the inherited wp_alpha.computeAlpha `find_spec` issue documented in Phase2|

Continuation compares all particle tensors, exact accepted time/next dt, each
fresh RHS dvdt/dudt, per-row J and compatible W hashes, complete OU history and
generator. Only the two-component globally reduced source budget uses the existing
2e-12 tolerance; other checks are exact. Test oracles contain particle fields and
per-row hashes, never pair arrays. All three schemes retain positive finite states.
The FP32 process-driver aggregate is 100.25 seconds and the FP64 driver/recipe
aggregate is 100.03 seconds, including launches; GPU logs
retain per-step timings. Hashing, host waits, process startup and ACSPH contention
dominate these fixtures, so they establish correctness rather than throughput.

Evidence, XML/logs, files/oracles, failure history, reproduction scripts and paired
source fingerprints are in frontend `.tmp/chunked-restart-20261030/`. The first
standalone recipe test failed because its fixture had not called `wp.init()`;
the corrected fixture and the raw failure are retained. No floating kernel or
numerical formula changed. Next L1 work is legacy COO/CSR import with explicit
ordering/offset conversion; L2–L7 and physical multi-GPU/site/capacity gates remain
open. This checkpoint does not qualify cross-device-type restarts, AD/graphs,
walls/materials or schemes beyond the stated native driven scope.
Implementation provenance: core `e740b76`; frontend is the commit containing this checkpoint.

### L1 legacy import and L2 local launcher checkpoint (2026-10-30)

User decision: MPI-enabled and ROCm-enabled Torch builds are deferred. The local
sequence continues on the existing CPU/Gloo and isolated NCCL 2.32.3 stack; no
shared packages or ACSPH sources/installs changed.

L1 now imports version-1 driven SPH COO and CSR files into owned row chunks.
`rowChunkCheckpointRecipe` encodes explicit frozen inputs without accepting pair
arrays. The frontend legacy adapter reads O(N) snapshots/counts/offsets, validates
the offset scan in int64, reconstructs under the target policy/group, and compares
each owned J slice with the file before installing the state. Optional COO row IDs
and padding sentinels are checked too. No global pair tensor, flat compatibility
list or pair migration is constructed. Raw whole-file fingerprints use fixed
host blocks to detect cross-rank/file changes during import; legacy files have
no original checksum, so this does not detect arbitrary finite particle-state
corruption that occurred before import. Expanded per-edge stage-state files,
missing history, unknown formats and incompatible reconstructed ordering reject.

These old files lack mode/Verlet-scale metadata. The caller must supply the
matching original physical/search configuration, as in the existing flat restore
path; target config supplies mode/scale and saved scaled supports remain frozen.
Qualification uses the default search policy in bounded 2D controls. Other legacy
policies/versions, physical multiple-device restart and broad 3D import remain
separate qualifications. The seed-list startup overlap noted above remains.

L2 adds `warpSPH.distributedRun.runDistributed(configuration, communication,
outputDirectory=..., edgeBudget=..., maxRows=..., checkpointEvery=...)` and
`scripts/run_distributed.py`. The Python API accepts an existing explicit group;
the CLI initializes/destroys c10d with a finite timeout. Ordinary runner defaults
and existing run documents remain unchanged. The adapter currently selects eager
driven Monaghan/CRKSPH/CompSPH, with owned initialization/search from the start.
Native numerical/support choices come from the run document, including the
default Owen support policy; the adapter does not force a numerical preset.

An optional appended `Case.acceptedStep` output hook runs after the case's
postStep/timestep hooks and distributed accepted-state validation. Periodic and
final native checkpoints use the existing accepted-boundary writer. Rank zero
owns diagnostics JSONL and the global manifest; ranks write separate runtime,
communication and failure reports. Output directories must be fresh and shared.
Manifests include active frontend/core/integrator Python file hashes, available
Git heads, input configuration, rank/device/PID/host placement, dtype/offset width,
Torch/Warp/CUDA and actual loaded NCCL provenance. Git heads alone do not assert
that an uncommitted working tree matches that commit; active file hashes identify
what ran. The actual entrypoint path/hash is recorded in rank runtime metadata.
Accepted-step hooks explicitly reject pipelined graph output rather than being
silently skipped. Logical adjacency storage and sampled Torch allocation are not native/
driver/process peaks. Hard rank death still relies on c10d timeouts and torchrun;
no new collectives are entered in the failure-reporting exception path.

Example local CPU launch, with paired `src` paths in PYTHONPATH and a versioned
JSON RunConfiguration that disables ordinary store/plot/video and selects eager:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=3 \
  scripts/run_distributed.py --configuration case.json --backend gloo \
  --output /absolute/fresh/shared-output --edge-budget 1048576 \
  --max-rows 512 --checkpoint-every 100 --offset-dtype int64
```

Production NCCL maps LOCAL_RANK to a distinct visible GPU. The explicit
`--shared-gpu` debug exception selects cuda:0 and requires NCCL >=2.30, the
experimental enable environment variable and at most two local ranks. Continue
to use the isolated overlay and low CTA/NVLS settings above for local shared-GPU
checks. Set the input document's `simulation.precision` before launch (float32 or
float64); the CLI selects it and the offset width before importing the stack.
Set `simulation.resumeFrom` to a checkpoint and use a fresh output directory to
continue with another budget/rank count. Launcher output overrides use its own
checkpoint path and rank-zero progress, keeping the legacy serializer disabled.

`scripts/slurm/chunked_driven.slurm` is a template with explicit environment
inputs for paired source paths, target Python, configuration/output, GPU count
and edge budget. It launches one torchrun parent per allocated node, quotes paths
and derives rendezvous from the Slurm node list/job ID. It passes bash syntax and
mocked remote argument-expansion checks including paths with spaces. No submission,
site module/resource/network setup or Arm64/GH200 execution has been qualified.

|Control|Result|
|---|---|
|Legacy COO/CSR CPU FP32|6 scheme/layout drivers pass; serial legacy files -> 3 owned ranks, int32/int64 source offsets; malformed scan/J/I reject|
|Legacy CPU FP64|CRKSPH both layouts and 14 recipe checks pass; 14 expected CUDA skips|
|Legacy shared-GPU NCCL FP32|6 serial jobs / 8 rank reports / 80 RHS evaluations pass; CRK CSR int32→int64 and Comp COO int64→int32, 1→2 ranks|
|Recipe CUDA after encoder refactor|14 pass per offset width|
|Native-format restart nonregression|3 scheme drivers + 14 core cases pass, 14 CUDA skips|
|Launcher CPU FP32|3 scheme lifecycle drivers pass: 1-rank uninterrupted, 2-rank save, 3-rank resume under changed budgets; exact final particles/time/dt/OU, one diagnostics row per accepted step plus initial row|
|Launcher shared-GPU NCCL FP32|9 serial jobs / 12 rank reports pass for all 3 schemes: 1-rank uninterrupted, 2-rank save, 1-rank resume; exact final particles/time/dt/OU; runtime23203 and shared UUID recorded|
|Output/failure controls|Periodic accepted checkpoints, final save, preserved existing directory and coordinated capacity failure with rank error reports and no final checkpoint pass|
|Existing chunked search/operators/native work|332 CPU passes, 251 expected skips|
|Final hook / legacy old-runner regression|8 pass (launcher CRKSPH lifecycle and 7 old checkpoint tests); Case hook appended to preserve existing positional fields|
|Runtime imports|566 modules pass; inherited static-scanner issue remains documented above|

Tiny 64-particle A/Bs omit video, retain progress/stall/velocity controls and run
serially on the GPU. CPU legacy drivers took154.88s total, launcher drivers94.68s;
native restart/core82.23s and legacy FP64/core54.22s are overlapping controls.
NCCL launcher rank-zero wall times are2.85–31.24s including setup/IO; all timings
are diagnostic under contention, hashing and process/IO overhead. No throughput
or physical capacity claim follows.

L3 baseline: the two-step shared-GPU launcher fixtures recorded980 CRKSPH,
740 CompSPH and1069 Monaghan control handshakes per rank at the report snapshot,
including initialization, exhaustive hashing and checkpoint/output controls.
Counters precede final report publication. This confirms substantial control/
hash overhead to address next; timings must not be used to rank transports.
Next: separate exhaustive replica hashing from normal validation, then combine
metadata/particle exchanges and reuse buffers while retaining exact-mode failure
and numerical controls. L4–L7 and physical-device/site gates remain open.

Evidence and reproducible scripts: frontend `.tmp/local-continuation-20261030/`.
Implementation provenance: core `7c09724`; frontend is the commit containing this checkpoint.

### L3 control/validation checkpoint (2026-10-30)

The first L3 slice is qualified on CPU/Gloo and bounded shared-GPU NCCL. It
reduces debug/coordination work without changing numerical kernels, payload
broadcast order, scalar reduction arithmetic, Butcher stages or owned J/W.

`RowChunkProcessGroup(..., replicaChecks='exact')` remains the default. Exact
mode retains the existing operator-input fingerprints (owned pair buffers and
unread scratch remain excluded). Explicit
`replicaChecks='contract'` (CLI `--replica-checks contract`, Slurm
`REPLICA_CHECKS=contract`) validates operator tensor layout/device/AD and operation
contracts without copying every field to CPU for a content hash. Both modes
force exact agreement at search/rebuild, initial/accepted-state and checkpoint
boundaries, plus the existing source clock/generator contract. Select the same
mode on every rank when constructing the group; the mode is part of every
control digest. Checkpoints can resume under a different mode in a fresh job.

Contract mode assumes the dependency schedule maintains replica contents during
an RHS evaluation. It deliberately does not detect every transient corruption
of a reference/correction field before owner outputs are mixed; exact mode is
the debugging choice for that guarantee. Exact boundary checks are not a claim
of equivalent fault detection inside each stage. AD/capture and other previously
unsupported contexts remain rejected.

Control send/receive storage is allocated once per group; CUDA also retains
pinned host staging. Gathered packets share a contiguous receive tensor, with
cached peer views and one explicit blocking device-to-host packet copy, replacing
one allocating host copy per peer. The host read completes before a buffer can
be reused on another current stream. This is a serial eager interface, not a
concurrent-thread/async-control API. Persistent packet storage is1.5KiB on CPU or
3KiB across CUDA/host for two ranks (native/backend workspace excluded).

A fused local-validation/JSON-contract handshake preserves coordinated exceptions
while removing redundant validation acknowledgements. Scalar validation and
clone allocation now precede one handshake; operator metadata is similarly
combined. All particle outputs from an operator are validated in one ordered
batch before any broadcast. Payload broadcasts themselves retain their previous
field/owner order; no floating reduction, dtype conversion or pair gather was
introduced. An invalid last output cannot leave earlier fields partly exchanged.

Matched measurement uses a frozen core source tree from `7c09724`, the same
frontend numerical code/fixtures, two ranks, 8 solver variants / 68 RHS per rank.
The frozen adapter shim only accepts the new exact keyword and counts existing
hash input bytes; its original control/search/dispatch implementation runs.
Explicit Python Torch factories inside control calls are instrumented in both
implementations. Timings include instrumentation/contention and are diagnostic.

|Per-rank quantity, FP32 CPU and NCCL (same counts)|Frozen exact|New exact|New contract|
|---|---:|---:|---:|
|Control handshakes|6,756|3,288|3,288|
|Replica bytes hashed|7,860,720|7,860,720|222,528|
|Explicit control Tensor factory calls after construction|20,268|0|0|
|Logical particle/scalar payload bytes|608,960|608,960|608,960|

This is51.3% fewer controls in either mode and97.2% less hashed data in contract
mode. The factory count excludes native/backend allocations and other solver
Tensors; the new group has two initial packet storage allocations on CPU/four on
CUDA. CPU FP64/int64 repeats the same handshake/factory counts, with hashed bytes
14,727,472 ->411,008 in contract mode. No throughput speedup is inferred.

Qualification (tiny A/Bs omit video, preserve progress/stall/velocity controls;
GPU jobs run serially under the existing isolated NCCL 2.32.3 runtime):

|Control|Result|
|---|---|
|Real CPU process drivers|5 pass: exact/contract × 2/3 ranks, plus fatal-peer termination|
|Core protocol fixtures|Self/unequal/empty populations, prepared density, covariance/gradients/CRK corrections, owned sums/rebuild/repartition and 8 existing recoverable failures pass; contract reference-content failure explicitly uses a forced exact boundary|
|Added packet/batch failures|Forced exact mismatch, differing live-rank modes, invalid scalar and invalid last batch output reject coherently; next controls recover with stable buffer addresses|
|Allocation/lifetime audit|12 warm controls invoke no explicit Tensor factories; cached addresses remain stable; GPU int64 cases alternate two CUDA streams for12 further controls in both modes|
|Frozen/new solver comparisons|CPU FP32/int32, CPU FP64/int64 and NCCL FP32/int32: 3 variants of implementation/check policy × 8 solver fixtures pass exact stages/owned W/particles/time/OU; existing2e-12 source-budget tolerance only|
|Long NCCL forcing/replay|16 accepted steps /128 fresh RHS per native scheme per rank, 256 RHS combined; every step forced changed-list replay, OU crossings, exact history/fields/work/time pass|
|NCCL 32-lane control|3 steps /48 RHS per rank across native schemes, forced replay and nondefault stream pass|
|CPU solver/launcher regressions|7 pass: two-/three-step exact/contract solver controls plus all 3 scheme launcher lifecycles, 1/2/3 ranks; exact→contract→exact across jobs|
|Native checkpoint drivers|3 pass: changed budgets/ranks/int32→int64 offsets, contract continuation, exact fresh RHS/J/W hashes/forcing and coordinated IO/corruption/policy controls|
|NCCL launcher/checkpoint lifecycle|9 serial jobs /12 rank reports, all 3 schemes exact→contract→exact with changed budgets/1/2 ranks and exact final particle/time/dt/OU parity|
|Existing focused chunk regressions|349 CPU passes, 265 expected skips, including the initial3 scanner tests|
|Static scanner follow-up|5 scanner regressions pass, including parent finder exceptions and installed-optional missing children; full runtime/static check passes566 modules /6063 imports /1314 files|
|Slurm template|Syntax/mock remote quoting passes with explicit contract mode; actual submission/site qualification remains open|

Core CPU process-driver wall total23.23s, solver/launcher134.79s, native checkpoint
77.80s and overlapping focused regression15.31s. Raw progress and per-step times
remain in the logs; the shared GPU and debug instrumentation invalidate transport
performance conclusions. All numerical/protocol controls pass. Failed pre-fix
scanner output and original retired notebooks are retained as evidence.

At this control/hash checkpoint, particle payload packing/workspace reuse remained
open; the following checkpoint qualifies them. These earlier changes batch output
contracts, not data payloads. Avoid
removing exact boundaries or failure controls as a shortcut. L4 filtering,
L5 stability expansion, L6 compatibility, L7 actual wide-edge accesses and physical
multi-GPU/site/performance remain separate gates. MPI/ROCm builds remain deferred.
Evidence/reproduction/source snapshots: `.tmp/l3-communication-20261030/`.
Implementation provenance: core `3742f10`; frontend is the commit containing this checkpoint.

### L3 particle packing/workspace checkpoint (2026-10-30)

L3 is qualified locally in the existing bounded eager process scope. Compatible
nonempty operator outputs share dtype-specific row packets; particle fields keep
their own storage and J/W remain owned. Packing/unpacking uses only same-dtype
copies, preserves signed-zero/NaN bits and changes no floating kernels, scalar
reduction arithmetic, fresh Butcher stages or exact-boundary policy. Exact input
checks remain the default, with the same explicit contract-mode limits above.

`RowChunkProcessGroup(..., particleWorkspaceBytes=4*1024*1024)` lazily allocates
one fixed-capacity byte buffer for packing. The capacity is read-only after
construction and included in the output contract on every rank; zero disables
packing. Whole-row tiles fit the capacity. Singleton fields, rows wider than the
buffer and groups whose tiling would not reduce broadcast calls use the direct
path. No per-shape/tag/dtype cache, buffer growth or borrowed output storage is
introduced. The default costs 4 MiB per process after first packed exchange, zero
before it; a 127-byte audit exercises mixed types, alignment and multi-tile owners.
This is additional communication scratch, separate from computational outputs,
owned pair buffers and backend/allocator workspace. Releasing the group releases
its retained scratch; the Torch caching allocator can retain reserved memory.

Whole-batch layouts, exact-alias topology, capacity and allocation agree before
any payload. Disjoint views of one storage and matching exact aliases are valid;
partial/cross-row overlap, retyped overlap and different alias topology across
ranks reject coherently. Allocation failure rejects before any field exchange;
subsequent controls/exchanges recover. CUDA records an event after the final
unpack and waits on it before reusing scratch, including another current stream.
`record_stream` additionally registers allocator lifetime. This adds no explicit
host synchronization; callers still own external producer/consumer dependencies.
The interface remains serial eager, not concurrent-thread/async/capture/AD safe.

Matched two-rank fixtures freeze the preceding control/hash core `3742f10`, use the
same frontend and 8 solver variants / 68 fresh RHS per rank, then run packed exact
and packed contract modes. A wrapper counts actual `dist.broadcast` calls; it
changes no solver/transport logic. Both ranks show the same counts:

|Per-rank quantity|Frozen 3742f10 exact|Packed exact|Packed contract|
|---|---:|---:|---:|
|Payload broadcasts, CPU FP32/FP64 and NCCL FP32|1,424|968|968|
|Control handshakes|3,288|3,288|3,288|
|Logical payload bytes, FP32|608,960|608,960|608,960|
|Logical payload bytes, FP64|1,202,464|1,202,464|1,202,464|
|Packed batches|0|116|116|
|Persistent packing storage / allocations|0 /0|4 MiB /1|4 MiB /1|

This removes 456 broadcasts per rank (32.0%) on these fixtures, without reducing
logical data volume. Logical bytes include each rank's own interval and scalar
payloads; they are not wire traffic. The earlier control/hash reductions remain
in place. No native/backend allocation or throughput reduction is inferred.

|Qualification|Result|
|---|---|
|CPU final process drivers|5 pass: exact/contract × 2/3 ranks plus fatal-peer termination; existing 8 recoverable faults and packet/batch audit retained|
|New packing audit|All 5 payload dtypes, scalar/vector/matrix tails, exact aliases/disjoint storage views, signed zero/NaN bits, zero-channel fields, zero populations, empty owners, tiled rows and oversized/direct fallbacks pass bitwise|
|Packing failures|Injected allocation failure, partial overlap, differing alias topology, retyped overlap and rank-budget mismatch reject before any payload; following controls recover|
|Bounded storage/lifetime|One 127-byte storage allocation across changing shapes/types and repeated warm calls, no explicit Tensor storage factories in warmed exchanges, stable pointer and retained outputs; disabled/uneconomical packing allocate zero bytes|
|Matched solver controls|CPU FP32/int32, CPU FP64/int64 and NCCL FP32/int32: frozen/exact/contract all pass exact stages, owned J/W, fields/time/OU; source-budget tolerance remains 2e-12|
|NCCL streams/offsets|Both check modes and both offset widths pass 12 alternating streams with delayed producers/unpacks, retained outputs and event-fenced scratch reuse; actual NCCL 23203 /same shared GPU UUID verified|
|NCCL replay/lanes|16-step native forced-replay controls retain 128 fresh RHS per scheme/rank and OU crossings; 3-step 32-lane controls also pass; 7 cohorts /14 rank reports /1,016 distributed RHS in the main GPU sequence|
|CPU solver/launcher/native restart|10 final passes, including all 3 schemes,1/2/3 ranks, changed budgets/int32→int64 offsets and exact→contract→exact mode changes|
|Broader first pass|16 solver/launcher/native/legacy driver passes (includes 6 COO/CSR imports); 351 focused chunk/scanner passes and 265 expected skips. These began before the final alias/schedule guard; final transport drivers/cohorts above qualify that guard|
|NCCL launcher/restart|9 serial jobs /12 rank reports, all 3 schemes, changed budgets/1/2 ranks, exact→contract→exact; exact final particle/clock/dt/OU parity and source hashes pass|
|Full import scanner|566 modules /6063 first-party imports /1314 files, no failures|

CPU final process suite 24.78s and solver/launcher/native drivers 209.08s; focused
chunk/scanner first pass 15.23s. Tiny A/Bs omit video and retain progress, stall and
velocity controls. Timings include debugging/control/contended execution and are
diagnostic only. For the 8 matched cases (16 accepted steps per rank), rank-zero
aggregate wall time /ms per accepted step was:

|Runtime|Frozen exact|Packed exact|Packed contract|
|---|---:|---:|---:|
|CPU FP32|3.66s /228.8ms|3.64s /227.4ms|3.30s /206.1ms|
|CPU FP64|3.45s /215.8ms|3.40s /212.7ms|3.34s /208.7ms|
|Shared-GPU NCCL FP32|17.34s /1083.8ms|16.35s /1021.8ms|15.65s /978.2ms|

These figures do not establish a transport speedup or a regression under ACSPH
contention. Physical devices/topology, aggregate capacity and scaling remain open.
L3 does not recycle computational output tensors retained by later stages, reduce
checkpoint seed/restored adjacency overlap, or migrate pair arrays. Next is L4
distributed filtering/redistribution; L5 stability expansion, L6 compatibility and
L7 actual wide-edge accesses remain ordered gates. MPI/ROCm builds stay deferred.

Evidence/reproduction/frozen sources: `.tmp/l3-payload-20261030/`; draft results
before the final alias/schedule guard are retained separately. Final core/front
heads are recorded in its `commits.json` after the paired checkpoint commits.
Implementation provenance: core `0513736`; frontend is the commit containing this checkpoint.

### L4–L6 continuation authorization (2026-10-30; sequence locally qualified)

User authorizes L4, then L5 and L6 in order if unblocked while away. L4 will count
filtered rows on existing owners, agree the target partition, and stream bounded
filtered J intervals directly into target chunk storage through c10d point-to-point
transfers. One old-chunk temporary at a time; no global pair tensor. Generate I
locally, preserve source leases/J/W, and qualify existing fresh-stage compatible
replay on list changes. Each local allocation/kernel/transfer contract must agree
before payload, including empty/skewed owners, changed boundaries and failures.
L5 extends randomized partition/capacity, memory, forcing and fault controls.
RHS fault probes also cover native W/half-velocity/dt/material preparation; local
allocation errors must agree before the work operator enters a payload.
L6 records and tests an explicit support/rejection matrix, then expands compatible
contexts as separately qualified changes. Physical devices/site/scaling, large
L7 allocations and MPI/ROCm builds remain deferred. Tiny serial GPU controls only;
no shared install/source changes and no throughput conclusions under ACSPH load.

### L4 distributed filtering/redistribution checkpoint (2026-10-30)

Distributed resident filtering now counts only existing owner rows, exchanges
O(N) counts, agrees a new whole-row plan and allocates only new owned J/I.
Existing count/filter kernels preserve source neighbour order. Each source owner
filters one original chunk into a bounded live-J temporary, then `transferOwned`
copies its contiguous row intersections directly into target chunks, with c10d
send/recv between changed owners or a local copy for unchanged owners. Stored
I is generated locally. No global pair tensor, pair gather or W transfer exists.

Every live rank validates each transfer's source/target/count/dtype/plan before
payload. Allocation, filtering and view failures propagate before their payload;
zero-edge intervals skip traffic. Source lists and work buffers remain valid,
independent and unchanged. New filtered adjacency identity triggers the existing
native fresh-stage replay when used during compatible-work accumulation. It does
not remap retained W or change integrator stages/reduction arithmetic. Filtering
is explicit; ordinary eager native steps still use resident acceptance kernels.

Peak storage includes the old and newly allocated owned lists, replicated current
snapshots/counts/prefix, plus at most one original source chunk's filtered J and
local offsets. `searchScratchBytes` reports that logical temporary bound, not
allocator/backend peaks; scan/cast temporaries and reserved memory remain separate.
Local filtering retains its original linear interval walk. Dynamic group size,
physical multi-GPU transfer and asynchronous recovery remain unqualified.

Qualification: final CPU core/search suite191 pass /187 expected skips (including
5 process drivers:2/3 ranks ×exact/contract and fatal-peer termination);4 new
frontend replay drivers pass across2/3 ranks ×exact/contract. Each core worker
adds27 source/filter fixtures across all support modes,2D/3D, stored-I changes,
source/target chunk splits/merges, periodic fields, zero populations/references
and empty owners. Tiny flat filters verify exact counts, prefix, each owned J/I
and padding. Retained J/W and leases survive; old W rejects the new list identity.
Temporary logical bytes stay below one original owned chunk's live J/offset bound.

Six added recoverable failures cover mismatched policy, capacity, target/source
allocation, stale source and malformed transfer layout; no payload occurs after
invalid preparation, and a subsequent valid filter recovers. CPU FP64/int64 and
NCCL FP32/int32 plus FP32/int64 each run both modes for core/filter and3-step
native filtering/replay:8 rank reports /108 filtering fixtures /192 distributed
RHS per configuration. Actual shared-GPU NCCL23203 verified. Native CRKSPH and
CompSPH each retain8 fresh RHS per accepted replay step and exact work/fields/
time/OU parity, with the existing2e-12 source-budget tolerance only.

Final core suite28.29s, frontend replay suite33.80s; rank-zero contract-mode filter
matrix0.65s CPU FP64 and6.86/6.89s NCCL int32/int64. Contract solver totals for
6 accepted steps/rank:1.92s CPU (320ms/step),9.90/10.00s NCCL (1649/1667ms/step).
These tiny A/Bs omit video, retain progress/stall/velocity controls and run GPU
jobs serially. Timings are diagnostic under ACSPH contention, with no speedup or
physical-device claim. Full scanner passes566 modules /6065 imports /1314 files.

An initial broad run failed only the obsolete single-owner rejection-message
assertion; the corrected clean run passes. The pre-fix XML/log and the restored
linear local-walk source variant are retained in evidence. Distributed code paths
are unchanged by that local-walk correction. Evidence/reproduction/frozen files:
`.tmp/l4-filtering-20261030/`; final paired heads in `commits.json`. L4 is locally
qualified. Proceed to L5 seeded partition/capacity/lifetime and forcing/fault
stress, then L6 compatibility. MPI/ROCm and large-memory L7 remain deferred.

### L5 bounded stability/failure checkpoint (2026-10-30)

A seeded process worker now changes populations (including zero/unequal sets),
2D/3D, all support modes, skewed/isolated rows, periodicity, headroom/alignment,
chunk capacities/row caps, stored-I layout and ownership repeatedly. Guaranteed
transitions move all rows into one owner and split them back across owners.
Every cycle compares exact filtered counts/prefix/owned J/I to a tiny independent
flat filter, retains and checks old W/J, exercises safe in-place lease invalidation
and bounded packet reuse, then uses weak references/GC to require that obsolete
pair tensors have no remaining Python owners. One source-chunk scratch bound is
checked each cycle. Packet scratch is a single stable 4 KiB allocation.

Injected failures after target allocation but before a transfer contract's payload
reject coherently, including an error on a nonparticipating rank. Retrying keeps
source leases/work valid. Fatal sender and receiver probes exit after the transfer
contract and require torchrun to terminate peers; ordinary Gloo peer death and
all earlier protocol/filter/IO failure controls remain covered. No arbitrary
post-device-fault recovery is promised; fatal/asynchronous faults still rely on
process-group deadlines and launcher termination. NCCL fatal-participant probes
were deliberately not added on the shared busy GPU.

The RHS fault probe found native W/half-velocity/dt/material preparation outside
a coordinated phase. `workInputs` now runs that local preparation behind one
`localPhase` before either native work operator starts. A work allocation failure
on only the last rank therefore reaches all live ranks before work payload, and
a fresh solver attempt with the same communicator recovers with exact flat parity.
This adds one control handshake per prepared weighted native work stage (also for
replay), with no floating-function/formula, stage, reduction or hash-policy change.
AST fingerprints of every existing Warp kernel/function in both touched work files
are unchanged. Flat execution runs the same local preparation without a collective.

|Qualification|Result|
|---|---|
|CPU process drivers|9 pass: prior2/3-rank controls plus4/5 ranks ×exact/contract stress,32 cycles/cohort,2 seeds,18 stress rank reports; fatal sender/receiver adds2 passing drivers|
|CPU FP64/int64|5 ranks ×48 seeded cycles pass;3-rank native RHS allocation failure/retry passes both native schemes|
|RHS/replay/compatible-work regressions|14 pass,2 expected skips:3-rank exact/contract work failure/retry,2/3-rank filtering replay, retained single-owner native controls|
|Shared-GPU NCCL stress|Exact/contract ×2 ranks ×48 generations; no obsolete pair tensors, one stable4 KiB packet buffer,24 KiB span in sampled allocated memory across changing tiny populations|
|Shared-GPU RHS faults|Both modes and both native schemes reject one-rank work allocation, then recover exact RHS/fields/time/forcing|
|Shared-GPU long forcing/replay|64 steps each CRKSPH/CompSPH,512 fresh RHS per scheme/rank,54 OU knots; exact stage/W/fields/clock/history parity and declared2e-12 source-budget tolerance; steady allocated-memory span0 after step8|
|Expanded checkpoint/fault lifecycle|Reference1 rank →save4 ranks with midstage/overwrite/interrupted-write cleanup and retry →resume5 ranks/int64 with empty owners; exact fresh rows/work/clock/forcing continuation passes,10 rank reports|
|Full imports|566 modules /6066 first-party imports /1314 source files, all resolve|

CPU process driver total81.83s, added fatal drivers and FP64 controls retain their
raw wall logs; frontend RHS/replay/work suite66.01s. NCCL48-cycle stress rank-zero
wall19.27s exact /21.27s contract. The64-step forced-filter/replay controls take
108.00s CRKSPH (1687.5ms/step) and92.80s CompSPH (1450.0ms/step) on rank zero.
The GPU sequence has10 rank reports and2,112 successful distributed solver RHS,
plus deliberately rejected attempts and flat reference RHS. These are correctness
controls under ACSPH contention, with tiny A/B video exceptions and progress/stall/
velocity guards; no throughput claim follows. Reserved memory is sampled, not
required to shrink. Logical bounds, weak-reference lifetime and Torch allocations
do not measure all backend/driver/process memory or establish multi-hour stability.

L5 is locally qualified within this bounded scope. L6 will expand compatible
Godunov contexts and record/test supported and rejected schemes, walls/materials,
AD/graphs and lane/prepared behavior, keeping native controls. Physical devices,
site/topology/scaling, MPI/ROCm builds and large-memory L7 remain separate gates.
Evidence/reproduction/frozen sources: `.tmp/l5-stability-20261030/`; paired heads
are recorded in its `commits.json` after checkpoint commits.

### L6 bounded compatibility checkpoint (2026-10-30)

The maintained [support matrix](docs/chunked_distributed_support.md) now qualifies
five eager schemes: Monaghan, CRKSPH, CompSPH, GSPH and InutsukaGSPH. The last two
use explicit query/reference/dummy roles and a global row base solely for
self-pair identity. Gaussian density and both Godunov force operators also work
through prepared launches. Inutsuka's Gaussian reach is converted to an agreed
host scalar only for chunked eager planning; the flat device-tensor path is
unchanged. Both list variants remain distinct. No floating formula, reconstruction
choice, integrator, fresh-stage rule or default was changed. Normalized Warp ASTs
match L5 after removing only integer rowStart plumbing; all16 existing flat
Godunov/Gaussian/Inutsuka FP64 reverse-gradchecks pass.

The shared scheme gate now admits five schemes at distributed startup, the
explicit launcher and native checkpoint IO. Native checkpoint/restart includes
both Godunov forms; legacy version-1 CSR/COO remains qualified for the original
three schemes only, with explicit Godunov rejection. Actual capture guards reject
before callbacks/control/collectives, including native work preparation. Forward
AD search inputs reject before detached snapshots. The operator contract now
agrees the effective selected kernel and lanes, so different tiled reduction
settings reject before compute; identical scalar fallbacks remain compatible.

Qualification uses tiny serial A/B controls without video, retaining progress,
stall and velocity monitoring. Every solver fixture compares fresh RHS values,
J/W as applicable, accepted particle fields, time/dt and OU histories against the
flat path; only the existing two-component source-budget comparison uses2e-12.

| Gate | Result |
| --- | --- |
| CPU formal driver suites | 292 passed,245 expected CUDA skips across core search/operators,2/3-rank context/solver matrices,5 native checkpoint drivers,5 launcher lifecycles,legacy/scanner regressions and core process/stress/fatal controls; overlapping driver selections are counted as executions |
| Existing flat Godunov reverse AD | All16 FP64 gradchecks pass after the integer ABI addition |
| CPU FP64/int64 process cohort | Exact/contract × first/second order;12 rank reports including operators and544 solver RHS evaluations |
| Shared-GPU NCCL FP32/int32 | Exact/contract × both orders plus global32-lane second-order control;14 rank reports and680 solver RHS evaluations |
| Shared-GPU NCCL FP32/int64 | Exact/contract × both orders;12 rank reports and544 solver RHS evaluations |
| Final prepared/operator/rejection matrix | 12 rank reports: CPU FP64/int64 and NCCL FP32/both offsets, exact/contract. Each report has24 configurations (2D/3D ×0/1/11 rows ×2 orders ×time-centred on/off), both stored-I layouts and all 3 prepared operators. CPU rejects15 contexts; CUDA rejects26, including effective-lane mismatch and actual capture before callbacks |
| Final effective-lane admission | Repeated full nine-variant second-order32-lane NCCL solver:2 rank reports,136 RHS; native controls and both Godunov forms pass. Godunov/Gaussian use scalar fallback, existing core/native specs select their tiled kernels |
| Final-source launcher/native IO | All5 schemes pass15 serial NCCL jobs /20 rank reports: uninterrupted1-rank4-step reference versus2-rank2-step save +1-rank2-step resume; row cap3→1 and contract→exact, native files/state/clock/OU agree |
| Runtime/static imports | 567 modules,6092 first-party imports,1316 source files; full check passes |

Every two-step compatibility solver report includes nine variants and68 fresh
RHS: the original three schemes, both native forced-filter replay controls, and
GSPH/Inutsuka in2D/3D. Two matrix axes vary reconstruction order; scalar/global32
controls compare each to its corresponding flat setting, not to a different
reduction tree. There is no new tiled Godunov kernel or claim of bitwise parity
between different lane settings. Native restart qualification here remains the
small2D driven case;3D solver parity does not qualify3D restart.

The formal checkpoint/process cohorts precede the final effective-lane metadata
addition, and some initial CPU IO runs precede the last capture guard. Their
manifests retain those source variants. Final operator/solver checks and all15
repeated NCCL launcher jobs use the final source; their package file hashes are
verified against the committed source contents. The first test helper-path error
and its repair are retained in raw evidence. No numerical tolerance was relaxed.

Wall times are diagnostic under ACSPH contention. Formal context/solver drivers
were80.28s, context/native retry41.20s, core2.44s, checkpoint/launcher259.41s,
legacy/scanner126.20s and core process/stress79.80s. After lane admission the four
CPU FP64/int64 solver drivers took54.05s. Representative rank-zero contract/order2
GSPH2D/3D takes0.82/1.67s for2 steps (408/834ms per step) on NCCL/int32;
Inutsuka takes1.10/1.73s (551/863ms per step). These are correctness fixtures with
control handshakes and a shared busy GPU, not throughput/scaling benchmarks.

MFM/MFV/PESPH, walls/materials, chunked reverse/forward AD/JVP/HVP and graphs,
physical multi-GPU/rank-to-device changes, site topology and capacity/scaling
remain open. MPI/ROCm builds and L7 allocation/access gates remain deferred.
L4–L6 have reached their bounded local exits. Broader Phase5/6 checkboxes below
are deliberately still open. Evidence, commands, manifests, normalized ASTs,
source snapshots and paired commit receipt: `.tmp/l6-compatibility-20261030/`.

## Phase 5 — remaining schemes and compatibility

- [x] Bounded Inutsuka list variants and MFM/MFV geometry/kappa/closure workspaces,
  including independent closure reserve and iterative matvec/vector exchanges/
  global reductions (C2). Wider boundary/material/temporal contexts stay gated.
- [x] Native accepted-step checkpoint/restart across chunk budgets and local
  process counts: canonical particle/forcing/time state, reconstructed owned
  adjacency and int32-to-int64 local offsets, within the L1 scope above.
- [x] Bounded version-1 COO/CSR import with owned pair-order verification and
  int32/int64 local offsets. Other legacy policies/versions remain unqualified.
- [ ] Physical GPU-count changes and larger/broader restart qualification; persist
  only needed history metadata, never create a global pair tensor for checkpointing.
- [ ] Legacy flat and all scheme families remain covered. Qualify graph cache keys,
  pointer stability, rebuild recapture, backward/JVP/HVP and prepared launches
  separately. Initially reject unsupported chunked AD/graph/wall contexts clearly
  or retain a safe explicitly selected flat backend; no silent giant allocation.
- [ ] Combine chunks with existing1/32 neighbour lanes after chunk-only parity;
  lane changes alter reduction order and are a separate comparison.

Exit: an explicit support matrix (scheme, eager/graph/AD, boundaries, lane count,
restart) and preserved old-path regressions; unsupported paths are visible.

## Phase 6 — capacity/performance and Isambard handoff

- [ ] Short fixed-step controls at32^3/64^3/96^3 before larger allocation gates;
  compare rebuild/steady time, kernels/host syncs, communication and memory against
  the integrated flat baseline. Separate allocator capture from timing runs.
- [x] Validate real physical device accesses for a total edge count>INT_MAX spread
  across bounded chunks: L7 dense debug fixture qualifies2,209,000,000 real J/W
  entries on one GPU, standalone and two NCCL ranks, both offset widths. Larger
  spatial/flow capacity and physical multi-GPU remain separate unchecked gates.
- [ ] Size128^3, then160^3/192^3 and ultimately256^3 from measured per-rank terms,
  leaving headroom for developed-flow neighbour growth and overlapping generations.
  Do not assume four GPUs make every live allocation exactly one quarter as large.
- [ ] On Isambard inspect actual node peer topology and measure transport/collective
  bandwidth. NVLink-C2C names a CPU-GPU link, not proof of a four-GPU NVLink domain.
  Record Arm64, CUDA/Warp/Torch, device/HBM, compiler/JIT, ranks and paired source
  pins. Qualify on GH200/Hopper rather than extrapolating local Blackwell timings.
- [ ] Require numerical/energy/positivity/source gates and a bounded memory run
  before any long trajectory. Report wall time, steady ms/step, edges/particle,
  live/reserved/process peaks, imbalance and communication fractions per rank.

Exit: reproducible node submission/run manifests and measured fit/throughput;
256^3 is a target, not a promised capacity or stationary turbulence result.

## Existing entry points and checks for pickup

Core anchors: `radiusSearch/csr.py`, `radiusSearch/compactHash/{search,wp_collectNeighbors}.py`,
`radiusSearch/verlet/{build,filter}.py`, `dataTypes/adjacency_t.py`,
`autograd/{arg_extract,prepared,operator_spec,lanes}.py`, `util/stateBundle.py`,
`mfm/{wp_closure,warpBackend}.py` (all under`src/warpSPHCore/`).
Frontend anchors: `modules/{crk,compSPH}/{accel,work}.py`,
`modules/compSPH/{pairWork,workReduction}.py`, `modules/meshless/flux.py`,
`utils/cudaGraph.py`, runner ownership and source/restart paths (under`src/warpSPH/`).

Useful existing focused suites (verify environment/device requirements first):
core`tests/radiusSearch/{test_csrOnly,test_splitCollection,test_indexDtypes,test_queryOrder,test_verletValidity}.py`;
frontend`tests/{test_compressibleCSR,test_crkSharedWork,test_compSharedWork,test_compSPHPairWork,test_neighborLanes,test_drivenTurbulence3D}.py`;
actual graph coverage`test_crkGraph.py`/`test_compSPHGraph.py` plus affected ACSPH
checkpoint tests selected by that owner. New focused suites now include core
`tests/radiusSearch/test_{rowChunks,chunkedSearch,chunkedState}.py`,
`tests/operations/test_chunked{PairField,_operators}.py`, both repositories’
`tests/distributed/` workers, and frontend `tests/test_chunkedCheckpoint.py` /
`test_distributed.py` / `test_chunkedCompatibleWork.py`. Extend these where useful.

Use fresh processes with`warpSPHCore_OFFSET_DTYPE=int64` before imports for the
wide-offset backend. Explicit source paths and provenance must be checked, not
assumed from an editable install. `scripts/profile_compressibleMemory.py` is the
current short memory driver (needs runtime/configuration/output); it and
`scripts/benchmark_crkWork.py` may need explicit chunk/transport instrumentation.
Do not use the older CRK profiler unchanged: it still inspects `paddedI.numel()`.

Every simulation follows the existing video/progress/watchdog/source/positivity
rules, with a documented exception only for tiny A/B fixtures. No long run is
needed to qualify an addressing/ownership change. Keep failed artifacts, frozen
inputs/source hashes, per-rank logs/status and reviewed videos. Respect shared
GPU workloads; contention invalidates throughput comparisons even when pass/fail
or per-process memory measurements are permitted concurrently.

## Parallel physics track and handoff completion

The current turbulence checkout remains the place for smaller single-GPU physics
runs (up to128^3 after readiness checks), with sparse full-state exports, recorded
flow and richer scalar/source diagnostics. Keep that campaign's source runtime
frozen; development of chunks or package installation must not modify it mid-run.
The physics programme's time horizon/seed choices remain independent decisions.
No turbulence run is launched by writing or picking up this infrastructure plan.

For each phase, update this checklist, the one PLANS row, paired commit IDs,
commands/configuration, passed/failed gates and the exact next action. First
reviewable implementation deliverable: Phase1 planner plus Phase2 direct
single-GPU chunk collection; first useful solver deliverable includes Phase3
CRK retained work. Full multi-GPU completion requires real multiple GPUs.

### L7 capacity/access authorization and qualification (2026-10-30)

User reports the GPU is free and authorizes large local runs, multiple instances
on the node/GPU and a genuine total neighbour count above INT_MAX for debugging.
Initial nvidia-smi reports92,094 MiB free of97,887 MiB,3% utilization and no
listed compute process. Start with a small independent-oracle control, then
physically collect >2^31 neighbours into distinct bounded chunks, read every J,
write/read real chunk-owned pair fields and execute production particle-output
operators. Repeat with both local offset widths and two NCCL shared-GPU ranks,
then exercise filtering/changed ownership within measured memory headroom.
Record exact source hashes, counts, allocation extents, boundary probes, timings
and memory; retain normal-size regressions. Synthetic all-neighbour geometry is
a device addressing/debug test; physical flow capacity and multi-GPU scaling
remain separate. No numerical formula/default changes are planned.

The actual-device gate passes without a production library change. The durable
core worker `tests/distributed/_wideRowChunkWorker.py` uses47,000 coincident
particles with self-inclusive supports, producing2,209,000,000 stored neighbours
(61,516,353 above INT_MAX). Search/count/scan/collection use the actual library.
The int64 global prefix spans33 dedicated J chunks; all local capacities and
three-component work scalar extents remain within the policy's int32 limit.
The stored J allocations occupy8.229 GiB and the actual owned work24.687 GiB
globally. No alias, metadata-only extent or global pair view stands in for them.

Each row is checked against an independent single-row flat search: that O(N)
reference must itself be a complete permutation of0..46,999. A CUDA audit reads
every J and every written work vector. Work is stamped through prepared row
dispatch with the global row, reference J and an exactly representable residue
of the int64 global edge index. Actual device probes report:

| Global edge index | J | Global query row |
| --- | --- | --- |
| 2,147,483,647 | 6,647 | 45,691 |
| 2,147,483,648 | 6,648 | 45,691 |
| 2,208,999,999 | 46,999 | 46,999 |

The production Wendland2 density operator runs eager and prepared over every
owned row and matches the independent flat reference exactly. Fixed scalar
lanes qualify int32 local offsets; fixed32 lanes qualify int64 offsets. This
cohort does not isolate an offset-width performance effect from lane choice.

| Case | Result / Torch allocated peak |
| --- | --- |
| Standalone CUDA FP32/int32, scalar lanes | All2.209b J/work checked; filtering/repartition/source preservation and post-filter density pass;41.182 GiB |
| Standalone CUDA FP32/int64,32 lanes | All2.209b J/work and eager/prepared density pass;32.940 GiB |
| Two shared-GPU NCCL ranks FP32/int32, exact | All2.209b J/work across17+16 original chunks; filtering/redistribution/source preservation and density pass;21.380/20.306 GiB per rank |
| Two shared-GPU NCCL ranks FP32/int64, contract,32 lanes | Same wide/filter coverage passes;21.380/20.307 GiB per rank |
| Small CPU/Gloo oracle regression | Standalone/int32,2 ranks/int32,3 ranks/int64 pass;289 real neighbours, changed owner boundaries and retained work |
| Small NCCL bring-up |2 ranks,1,048,576 real neighbours, empty-to-active owner and filtering pass |
| Complete core process suite |14 passed in87.77s, including earlier stress/fatal cases and3 new oracle drivers |
| Frontend runtime/static imports |567 modules,6092 imports,1316 source files; full scan passes |

Large filtering changes the row cap1427→713, creating66 target chunks. With2
ranks the owner boundary moves24,259→23,529:730 rows /34,310,000 real edges
move from rank0 to rank1 through137,240,000 NCCL payload bytes. The other pair
intervals copy locally. All98 intersection transfers are validated; both old
J/work and every new J remain correct. W stays associated with its original
adjacency, preserving the existing fresh-stage replay contract.

The large cases ran serially through isolated NCCL2.32.3 on the same GPU UUID.
Summed phase wall times (including setup/JIT/synchronization) were11.99s local
int32,2.59s local int64,12.70s NCCL int32 and5.75s NCCL int64 at rank0. These
are diagnostic debug costs; different lanes/filtering choices prevent a direct
performance comparison. Per-rank peaks are Torch allocator peaks, not sampled
aggregate process peaks. The worker checks an explicit64 GiB budget and8 GiB
reserve before large allocations; all GPU contexts are released at process exit.
No simulation/video or flow integration is involved in this device-access test.

Three initial CPU driver failures were fixture issues (metadata versus container
access); raw failures and successful reruns are preserved. No numerical failures
or overflow were reproduced in the qualified paths. Every successful wide
report pins all core package files and the worker source. The library contents
are unchanged from the L6 checkpoint79944e7. Evidence/commands/source snapshots
and final paired receipt: frontend `.tmp/l7-wide-20261030/`.

L1–L7 meet their stated local exits. Realistic spatial/developed-flow capacity,
full solver trajectories at wide edge counts, large single-chunk extents near
INT_MAX, physical multi-GPU/site/scaling, wider physics/AD/graph ports and MPI/ROCm
builds remain open. The maintained support guide distinguishes this debug gate
from those broader qualifications.

A second boundary fixture uses50,000 particles /2,500,000,000 real edges. Its
38 original chunks include a final global edge base2,482,700,000, itself above
INT_MAX. Standalone and two NCCL ranks pass with both offset widths, including
full J/work audits, eager/prepared density and filtering to75 chunks. This
explicitly checks a whole chunk starting above the signed32-bit limit, in
addition to the earlier limit crossing inside a chunk. Boundary device probes
report J33,647/33,648 at global edges2^31−1/2^31 and J49,999 at the final edge.
The original J/work occupy37.253 GiB globally; standalone allocated peak46.578
GiB including retained source storage and filtered J. All8 large cohorts /12
rank reports pass. Source package bytes remain identical to L6; the raw report
and `summary.json` retain separate costs/peaks for every case.

### Overnight real-flow campaign (authorized2026-10-30; launched2026-10-31)

User requests serial CRKSPH32³/64³/96³/128³ and an actual flow above INT_MAX
neighbours. User chooses t=4 at every resolution, sparse continuation checkpoints
and a conditional t=40 continuation at96³ or128³ if measured runtime fits a
10–12+hour window. Use the established B7/nh4, seed1234, OU0.3 bands1–2, finite
gamma cooling bath and fresh native midpoint model. Source copies are frozen
under `.tmp/crksph-3d-overnight-20261030/runtime/`; no shared install changes.
Use one process on the single GPU with chunk-owned J/W, int64 local offsets and
normal qualified kernel lane defaults; full particle/video output has no peer
collectives. Count-only metadata/memory gates precede large pair allocation.
Require measured initial total edges>INT_MAX for the final resolution, selecting
160³/164³/168³ from actual counts, without changing supports to force the count.
Qualify3D resume on a small fixture before queue launch. Keep z=0 slice video,
per-step scalar diagnostics, fresh RHS checks and sparse native checkpoints.
A sequential detached supervisor monitors GPU/process memory and progress,
stops failed dependencies, and chooses a t40 continuation only from passed
96³/128³ runs when a1.5× measured-cost forecast fits the remaining12.5h budget.

The detached serial supervisor launched at2026-10-31 00:01:30 UTC, PID3878944,
with a12.5hour wall window. Both source packages and integrators/plotting are
frozen in941 hashed files; subsequent checkout edits cannot affect these jobs.
Small 3D controls pass exact flat/chunked and changed-budget restart fields
(existing2e-12 source-budget tolerance), including a time-limited endpoint save.
Only its explicit endpoint time representation differs: the case snaps to its
declared Python time; all particle fields still compare exactly. The driver
saves the retained positive uncapped physical next dt for continuation, restores
the capped runner dt afterward, and does not modify library/numerical sources.
The vispy/EGL96² z=0 density/speed slice smoke video passes and was inspected.

32³ already reaches t4 with no bad RHS; the full run writes native midpoint and
final files plus a slice video.64³ is active at this checkpoint. Measured initial
counts are16,875,520 at32³ and135,004,160 at64³ (515 stored neighbours/particle).
At that occupancy160³ has2,109,440,000 edges, below INT_MAX;164³ is expected to
qualify at2,271,636,160, subject to the scheduled actual count/memory gate. The
wide flow must also report full native RHS calls above INT_MAX, rather than
only a wide initialization recipe. Raw RHS positivity/finite checks and accepted
energy/thermodynamic gates remain active throughout every stage.

Storage policy: one midpoint and one final native checkpoint per simulation,
with at most one overwritten `latest.h5` every20wall minutes; no global J/W or
dense particle export sequence. Scalars are logged per accepted step. Videos
use0.25 physical-time spacing to t4 and1.0 during a continuation. Watchdogs
require accepted progress within20min and a graceful accepted-step checkpoint
at the wall budget; only the supervisor's own process group can be terminated.
No unrelated GPU process is terminated. Larger dependent runs stop on failure.

After all five stages pass, the supervisor prefers128³→t40 if its measured t4
loop cost×9×1.5 plus30min reserve fits the remaining window; otherwise it tests
96³ against the same rule. It resumes the saved flow/OU clock. The choice and
its forecast are recorded, including a clear decision if neither fits.
Campaign configuration, executable scripts, pinned copies, controls, live
status, GPU telemetry, results/videos/checkpoints: `.tmp/crksph-3d-overnight-20261030/`.
This queue is running independently of the chat; it has not finished and does
not yet qualify the larger real-flow capacity or statistical stationarity.

Initial32³ measured loop: 38.715s, 39.304ms/step, 985accepted steps; Torch allocated peak0.506GiB. All run costs include declared observers and are retained in per-run JSON.

### Overnight outcome audit (2026-10-31)

The queue terminated03:17:38UTC after3h16min.32³/64³/96³/128³ all reached t4;
164³ failed at its work-allocation headroom check near t0.104. The supervisor
blocked the conditional t40 continuation after that failure; no continuation
was launched. GPU is now idle. Original status/logs/results are preserved,
with an independent audit in `.tmp/crksph-3d-overnight-20261030/outcome.json`.

| Resolution | Accepted steps | Loop wall | ms/step | Torch allocated peak GiB |
| --- | ---: | ---: | ---: | ---: |
|32³|985|38.71s|39.30|0.506|
|64³|2011|480.57s|238.97|3.817|
|96³|3040|2475.63s|814.35|12.756|
|128³|4079|8286.32s|2031.46|30.158|

All four completed runs have zero invalid RHS states, unchanged mass and maximum
relative energy-ledger residual below2.60e-8. Their native midpoint/final/rolling
checkpoints were independently checked for completed publication, finite fields,
positive thermodynamics/supports/masses and positive continuation dt. Final
files have exact t4 and the correct accepted-step count. Videos and all941
frozen executable/source hashes remain intact. Total campaign disk use is
about951 MiB, including runtime copies and small controls.

164³ physically built2,271,636,160 neighbours and executed245 full CRKSPH RHS
evaluations above INT_MAX with no invalid RHS;122 accepted steps reached
t=0.1037968248. Its maximum ledger residual was2.97e-8 and Torch allocated peak
60.014 GiB. The recorded Python error is the campaign guard rejecting a new
owned compatible-work allocation, before an actual CUDA allocation OOM was
reported. Exit status is alsoSIGSEGV(-11), following the recorded guard error;
cleanup requires investigation as well. This is an incomplete wide-flow run,
not a passed t4 endpoint. No wide checkpoint was written before failure: it
occurred before both the physical midpoint and the20min rolling-save cadence.

The next investigation is peak work-buffer lifetime/allocation and the guard's
accounting of free versus cached memory; the existing log does not identify
which of its two headroom conditions fired. Do not claim a confirmed memory
leak, actual device OOM or numerical instability from this trace alone. Preserve
the failed runtime while reproducing a short large run with detailed allocation
receipts and safe cleanup before retrying the full wide endpoint.

For cost context, the old3D pilot loop times were35.12/376.19/1919.04s at
32³/64³/96³, versus38.71/480.57/2475.63s here (about10/28/29% higher). This
campaign uses chunked/int64 addressing and exhaustive raw-RHS observers, with
different source revisions; these are not matched measurements isolating
chunking cost. Larger flow costs and memory must be measured separately from
the dense L7 debug fixture. No model/default was changed.

### Local continuation after overnight campaign (2026-10-31)

User authorizes this order: allocation receipts and work/list lifetime accounting;
graceful allocation failure with last-accepted particle checkpoint and safe
CUDA/renderer shutdown; short164³ retry beyond122 accepted steps with early
save/restart, then t4; matched flat/chunked profiling; longer96³ flow and real
shared-GPU NCCL flow/restart. Preserve failed frozen runtime and raw evidence.
An allocation failure must not publish a partially advanced state as accepted.
Qualify recovery at neighbour/work allocation sites and retain the original error
if synchronization or checkpoint publication also fails. Physical multi-GPU
performance and MPI/ROCm builds remain deferred. Work in isolated paired trees.

User clarification2026-10-31: hold full large t4 and96³→t40 for overnight
when development work is exhausted.196³ projects~13h (capacity unqualified);
96³ continuation~6–8h. During development use short wide runs, fault injection,
restart controls and matched profiling only. Do not automatically launch either
long campaign while there is useful development work.

Allocation receipt reproduction: at accepted step122 the requested W is
29,932,714,800bytes, live23,597,358,592, reserved65,678,606,336, cached
42,081,247,744 and device free30,071,390,208. The live78GiB limit passes;
raw-free-only reserve check fails. This reproduces the original rejection
without a leak or allocator OOM. Revised experiment admits free+cached while
keeping the live cap/reserve and real allocator-error handling; no cache flush.

A32³ injected failure reproducesSIGSEGV in tqdm.__del__ →close →format_meter
at interpreter teardown even after CUDA/plot cleanup. The runner now explicitly
closes progress on failure and preserves the primary exception if timer, frame,
renderer or trajectory cleanup also fails. Repaired32³ publishes its accepted
step2 native checkpoint and exits2 normally. Production code adds failure-only
J/W receipts and typed allocation propagation through existing control packets.
CPU2D/3D CRK/Comp work/neighbour rejection, forcing rollback and restart controls
pass; coordinated two-rank errors on either owner and three-rank continuation
pass. Bounded164³ fault/restart and matched profiling are in progress.

Bounded wide recovery now passes:164³ crosses the old step122 rejection, then
an injected work rejection after124 accepted steps atomically saves the
accepted state at t0.1054982841 and exits2 withoutSIGSEGV. Peak allocated
60.014GiB;249 completed wide RHS with zero invalid states. Continuation for
6 steps from that recovery file and10 steps from the earlier step120 checkpoint
both reach step130/t0.11060260981321335. Independent HDF5 comparison of29
datasets finds identical particles, forcing, clocks, context and neighbour
history, including exactly equal source budget. This qualifies bounded
wide3D failure/restart/rebuild; t4 stability remains an overnight gate.
CPU regressions:55 frontend passes/1skip;270 core passes/212skip; six focused
core failure/control passes (including4 real Gloo jobs); scanner passes568
runtime modules/6100 first-party imports/1317 files. Eight CUDA2D/3D CRK/Comp
work/search failure/restart controls pass. Shared-GPU NCCL recovery/profile
qualification follows serially.

User flags the compatible-work timing increase for investigation (2026-10-31).
The9.8% matched64³ total overhead is single process versus single process,
serial on an idle GPU, not shared-rank contention. Raw CUDA kernels show work
38.88→51.47ms/step (+32.4%),8→32 launches in the8-step profile; force is
59.94→59.52ms. Vary chunk granularity with fixed physics/observers and exact
endpoint checks, then inspect indexing/memory behavior if the gap persists.
Keep this separate from diagnostic shared-GPU NCCL timings.

Granularity diagnosis:8/4/3/1-chunk64³ profiles use the same compiled work
kernel; one chunk recovers38.43ms/step versus flat38.88. CUDA traces report
149registers/thread,256threads/block and one resident block/SM (register
limited). Four65,536-row chunks each launch256blocks on188SMs, wasting much
of the second wave; the flat1,024block launch amortizes that tail. This is
GPU kernel utilization, not just CPU launch latency or NCCL contention.
Test smaller work-kernel blocks without changing formulas, stages or row order.

### Compatible-work launch diagnosis and fix (2026-10-31)

The controlled32k/64k/128k/256k row-limit runs (8/4/3/1 chunks at64³)
all preserve every saved particle tensor bitwise. Work GPU time is
53.21/51.47/45.10/38.43ms per step. One chunk matches the flat38.88ms.
CUDA traces identify149registers/thread and256threads/block, allowing only
one resident block/SM; the small-chunk grid has poor wave utilization.
This supports a launch-geometry explanation without an arithmetic/indexing
change. The initial module-option experiment proved ineffective (traces still
show256threads); those raw controls remain and are excluded from block tuning.

Explicit128/64-thread trials both reduce work to~31ms/step and retain bitwise
particle/source-budget parity.128threads is selected: the trace permits three
resident blocks/SM. Core OperatorSpec now exposes an optional validated
blockDim, forwarded through ordinary/prepared launches and included in the
effective distributed contract. None preserves the256-thread scalar default;
tiled operators retain their cooperating-lane block size. Only the frontend
CRK work spec selects128; floating kernel AST/formulas/stages are unchanged.

| Matched64³ fixture | Steady wall ms/step | Work kernel ms/step |
| --- | ---: | ---: |
| Flat,256threads |168.54|38.88|
| Four chunks,256threads |185.08|51.47|
| Flat,128threads (controlled trial) |162.22|30.45|
| Four chunks,128threads (production path) |163.90|30.47|

All cases use the same early-flow100-step physical model/diagnostics/video;
first20steps are excluded from steady timings, and separate8-step profiles
provide raw CUDA kernel times. Production128-thread fields match the original
256-thread baseline bitwise. This removes the suspicious work-kernel gap and
reduces this fixture's steady step cost~11.4%; late t4/t40 and physical multi-GPU
performance are not remeasured. No chunk-size/global-pair expansion is needed.

Production wide recovery requalifies at164³: inject failure after accepted
step131, save safely and resume to132/t0.11230403185. All29 native datasets
match the original256-thread two-step control exactly. Frozen old/optimized
runtimes and hashes remain intact. Independent evidence: ignored
`.tmp/allocation-recovery-20261031/work-launch-audit.json`.

Real distributed flow gate:64³ adaptive3D CRKSPH reaches64accepted steps on
two NCCL ranks sharing one GPU, with a step32 checkpoint/restart and changed
chunk edge budget. Independent comparison to a one-rank64-step reference
passes23 datasets; particle/forcing/clock fields are exact, source-budget
roundoff is1.09e-19. Loaded runtime2.32.3 and a single GPU UUID are audited.
This is correctness/stability only. Full large-resolution t4/t40 runs remain
held for overnight, per user; no long job is active. Structured launcher
failure manifests now record the recovery path, accepted step/time and any
secondary write error; recovery files stay inside the run output directory.

Final admission: CUDA block ABI17passes (ordinary/prepared/reverse and invalid
declarations); native compatible-work/replay8passes in each FP32/FP64, plus
two nondefault-stream/32-lane passes in each precision. Two NCCL ranks pass
the core ownership/operator/error matrix including effective block mismatch;
native NCCL reference/failure/restart passes with the128-thread work selection.
The scanner resolves568modules/6105imports across1317files. Final primary-error
cleanup controls pass. Ordinary pipelined ownership also passes its existing
eager fallback; this does not newly qualify graphs. Both paired trees are
ready to commit. Full164³/196³t4 and96³→t40 remain overnight-only;196³ also
needs an independent capacity gate. No long simulation is launched.

Paired core implementation is committed as f4fe38c on
codex/chunked-neighbor-planner. Frontend commit carries accepted-state recovery,
launcher recovery manifests, the CRK128-thread selection and regressions.

### I1 regression admission notes (2026-10-31)

Initial affected CPU sweep found guard ordering before native invalid-driver
validation (three DFSPH controls), and the earlier L6 Godunov guard preceding
native unsupported finite-step energy validation (two controls). Restore native
validation priority before backend inspection; no formulas/defaults change.
Six rest-density fixtures unconditionally allocate CUDA; their CUDA-visible
rerun passes10 controls. CPU-hidden failures are not numerical evidence. Final
metadata review also preserves native one-iteration divergence budgets in the
new configuration descriptors; density/IISPH retain their native two minimum.

### I1 completion / next gate

Bounded eager compatibility now covers eight compressible schemes, both delta
variants and all four public native incompressible variants. No ACSPH changes.
Additional case adapters, walls/free surfaces/materials, experimental pressure
geometry, AD/graphs and physical multi-GPU/site/capacity/performance remain
explicit gates. Native pressure accuracy/long-time robustness is not upgraded
by storage parity. Full164³/196³t4 and96³→t40 remain overnight-only; none was
launched during this continuation. MPI/ROCm builds remain deferred.


Local repair/build progress (2026-11-02): shared-path ENOENT visibility waits
bounded at120s now cover output directory and peer runtime JSON publication;
permission/format/other IO failures remain immediate. Fault-injected two-rank
CPU/Gloo regression and serial/two/three-rank CRKSPH continuous/restart/freshness/
allocation-failure tests pass (5 targeted checks). First sandbox attempt blocked
TCP sockets; approved local-network rerun passes. No kernel/model change.
Docker access on it111636 works with sg docker after stale session groups were
refreshed. Isolated BuildKit builder warpsph-arm64-20261102 uses host networking
for DNS and bundled userspace qemu helpers; no host binfmt registration. Official
Apptainer1.5.5 non-setuid package and libsubid4 extracted into owned .tmp tools,
no system installation. Local ARM64 image and SIF pipeline in preparation.
Communication profiles now distinguish build-image provenance from target-image
identity to permit explicitly pinned reuse of identical OFI binaries; actual
application/runtime fingerprint is checked against the profile even on single.

Local build prerequisites verified: BuildKit's ARM64 ELF/uname smoke runs under
its bundled qemu, and the x86_64 userspace Apptainer converts that OCI archive
into a SIF whose system partition reports arm64. Scratch-only smoke cannot
supply ELF architecture evidence and was replaced by an Alpine ARM64 smoke.
No system Apptainer installation, host binfmt change or remote build allocation
is needed. Harness35 checks plus5 targeted distributed IO/CRKSPH checks pass;
full imports576/6285/1341 pass. Freeze only the IO-fix frontend delta over the
accepted ARM64 package snapshot, pin the resolved NGC ARM64 digest, and build
locally. The previous inter-node stage stays failed until a new image qualifies.

NFS mount inspection confirms qualification is on nfs4 /home, with no acdir
override. The Linux nfs-utils defaults allow cached directory attributes for
up to60s, including negative lookups. Extend the bounded visibility wait from
30s to120s before qualification; source: https://man7.org/linux/man-pages/man5/nfs.5.html .
Per-release QA_DEPLOYMENT_PROFILE overrides are preserved by site-profile.sh.

Local ARM64 release candidate completed (2026-11-02): commit2ecb382, only
distributedRun.py differs from frozen accepted app; target fingerprint e44ba6e.
BuildKit QEMU validates ARM64/CUDA13.1/NCCL22902, then userspace Apptainer1.5.5
converts locally to9,494,970,368-byte SIF. SHA256:
4ef5027e120d8d9065965405077ea942517403e54b7c50fb07b9421e24c1d518 .
Uploaded in2m10s and remotely checksum-verified under fresh images/warpsph-arm64-io-repair-2ecb382.sif;
old image unchanged. Full context,249 constraints, metadata, tool hashes, archive
and SIF retained in development checkout .tmp/arm64-local-20261102 with index.
No build-only Slurm allocation; no system Apptainer/binfmt installation.
New immutable harness warpsph-runtime-2ecb382-local-arm64 records candidate
profile separately from old OFI build-image provenance. Renew strict single,
same-node2GPU, restart, then2node/1GPU inter-node, retrieving/independently auditing
each before successor; failed job/audit/warnings stop. No four-GPU or long run.

Renewed single8004381 COMPLETED0:0 in5m55s on nid010476. Local audit verifies24
stable artifacts, runtime/image/harness pins and all23 reference fields bit exact
against old7989645. Checkpoint continuation exact, no NCCL/error warnings.
Reference launch136.413s, prefix37.554s, int64-offset resumed launch133.070s; cold
JIT/import/checkpoint qualification, not a benchmark. Same-node2GPU8004781 running.

Renewed same-node8004781 COMPLETED0:0 in4m19s,21 artifacts locally verified.
Two GH200s on nid010516; solver/probe logs prove P2P/CUMEM channels. All particle
fields exact; sourceEnergyBudget differs only4.1359030627651384e-25 (oneULP),
matching prior accepted partition rounding. No NCCL warnings. Reference launch
144.158s; two-rank four-step launch25.658s (warm cache; not comparable throughput).
Restart8005076 COMPLETED0:0 in7m20s on nid010754,33 artifacts locally verified,
2→1 cross-job and fresh1→2 transitions exact in all23 fields; no warnings.
Renewed two-node/oneGPU-per-node8005442 running on nid010558/nid010626.

### Renewed Isambard multi-node acceptance (2026-11-02)

All four strict stages pass on local-built ARM64 SIF4ef5027e...c1d518: single
8004381 (5m55s,24 artifacts), same-node8004781 (4m19s,21), restart8005076
(7m20s,33), two-node8005442 (8m31s,35). All113 stable artifacts and native
checkpoint checksums verified locally. Every fresh reference matches the old
accepted image bit-for-bit in all23 fields. Fresh two-rank same/inter-node
solves preserve particle fields exactly; only global sourceEnergyBudget differs
4.1359030627651384e-25 (oneULP). All restart continuations are exact.

Job8005442 used nid010558/nid010626 and two distinct GH200 UUIDs. Probe and both
distributed solver processes evidence real NET/AWS Libfabric/CXI channels. Each
node receives its own Slingshot service-ID list through cleanenv; comma lists
are preserved. All solver ranks load image NCCL22902/CUDA13.1. No NCCL warnings
or errors found. Reference/distributed/multi-prefix/single-resume launch walls
164.675/56.122/77.139/140.155s include startup/JIT/NFS waits and are correctness
measurements only, not throughput comparisons.

The original8001638 failure,15 sealed partial artifacts and failed status remain
untouched. New image changes only frontend distributedRun.py over frozen app
snapshots; no kernel/model change. Actual two-node startup/checkpoint publication
now passes, as do injected visibility/error and full CRKSPH restart regressions.
Reused OFI7999853 artifacts retain their original build-image hash separately.

Accepted parametric profile/defaults and actual-simulation recipe:
scripts/isambard/site-profile.{json,sh} and scripts/isambard/README.md. Remote
release directory: $HOME/warpsph-deployment-arm64-2ecb382; source its profile to
select images/warpsph-arm64-io-repair-2ecb382.sif. Source/image/runtime changes
require renewed qualification. Four GPUs, additional site scheme/boundary/3D
cases, capacity/performance, MPI/ROCm and held overnight flows remain open.
No larger topology or long simulation was launched. Evidence/build contexts,
receipts, raw logs, failed result, per-field audits and complete native states
are indexed in development .tmp/arm64-local-20261102/README.md and
.tmp/isambard-retrieved/README.md. Focus remains user-owned.

Final deployment activation verified over SSH without another Slurm allocation:
all release-file hashes/default paths, final profile/SIF and exact communication
hashes match. Receipt retained locally and under remote release provenance/.
No jobs remain queued/running. Final35 harness checks pass, all21 guide shell
examples/helpers parse, default and explicit profile overrides pass. Source
scanner576/6285/1341 and IO/CRKSPH regressions passed before source freeze; only
docs/profile metadata changed subsequently. Harness commit e7627d3; isolated
frontend branch clean after coherent commits. No push or source/core merge.

### Local support step1: native empty-owner lifecycle audit (2026-11-02)

User authorizes step1; retain local-only scope, no particle-layout integration
or new physical GPU/site campaign. Audit confirms bounded C1/C2/W1/I1 native
stage/restart coverage exists. Genuine gap: core empty-owner transport tests
are not an end-to-end empty-owner scheme qualification. Current managed solver
tests use small row caps, leaving every rank active. Add native flat versus
all-work-on-one-rank checks, checkpoint, and finer-partition resume that activates
the idle rank across admitted compressible/delta/projection families. Check
particle state/history, forcing/source clocks and ownership, without global J/W
fallbacks or model/default changes. Tight repeated fixtures omit video and keep
progress/timing. Evidence indexed in development .tmp/local-scheme-support-20261102/.

Local CUDA empty-owner audit exposes a cold Owen LUT producer/consumer stream
race: positions match but supports diverge across ranks before the next operator.
wp_psi.generatePSILut_warp allocates/launches on implicit Warp device/stream and
immediately clones through Torch current stream; cache warmup can hide it.
A default-stream warmup followed by nondefault flat run passes, while two cold
nondefault ranks produce different supports. Preserve raw CUDA contract/field
captures. Fix only device/current-stream ordering, not LUT kernel/formulas; add
a cold-cache local regression and repeat affected scheme coverage.

Cold LUT stream regression fails before the repair and passes afterwards;
wp_psi.computePsi numerical kernel is unchanged. The generator now selects
Torch current CUDA device (instead of fixed cuda0) and bridges Warp allocation/
launch plus consuming Torch clones through torchWarpStream. No device-wide fence
or method/default change. All 28 configurations pass CPU/Gloo and shared-GPU
CUDA/NCCL in both FP32 and FP64: 112 configuration/backend/precision controls,
224 rank-level continuations. The matrix contains 17 managed 2D configurations
and the 11 compressible configurations in 3D; it does not widen managed delta/
projection cases beyond 2D. State/history tensors, time/next dt and OU parameters/
knots/RNG match exactly; only the existing global sourceEnergyBudget comparison
uses tolerance 2e-12. Every idle owner has zero J capacity/native W buffers and
becomes active after restart. Exact replica checks remain enabled.

Existing native checkpoint/stage/recovery/filter/support regressions pass 208
with two expected skips; CUDA support regressions pass 104 with one optional-table
skip, and the focused cold lookup-stream regression also passes in FP64. Full
imports pass 576 runtime modules, 6298 imports and 1342 source files.
Independent verify_receipts.py checks all 32 rank receipts against the exact
matrix, source digests, worker hash, ownership and actual runtime. Results and
phase wall times/derived ms per step are indexed in development
.tmp/local-scheme-support-20261102/{README.md,verified-summary.json,timings.json}.
Timings include cold compilation and shared-device contention; no performance
claim follows from these tiny correctness fixtures. The actual NCCL runtime is
23203; both ranks use the same GPU UUID. Explicit PYTHONPATH pins this frontend,
paired core179df45 and the accepted pre-layout Integrators snapshot1fea2a3, whose
source digest is f4c1cb30…. Do not accidentally pick up the concurrently advanced
editable Integrators before the planned integration.

Step1 is complete within the documented eager periodic single-material support
boundary. Existing operator, moving/filter/repartition, rank/offset-change and
allocation-recovery gates remain accepted; no further port is needed for the
currently admitted families. Next is the separately ordered source-pinned merge
with fixed-count particle layout and local combined debugging; physical GPU/site,
long-run and explicitly unsupported wall/material/graph/AD gates remain separate.
No merge, main-checkout edit, site job or image replacement occurred in step1.
Documentation audit also removes obsolete claims that MFM/MFV are unported and
that physical CRKSPH rank changes remain unqualified; site baseline is frozen.

## Paired multi-GPU / particle-layout integration (2026-11-03)

User authorizes checkpointing the current repository states, then merging the
qualified multi-GPU core into core/main and frontend into dev. Qualify the
combined fixed-count layout, restart/ownership and current integrator contracts
locally before merging the core baseline into codex/native-cluster-pairs.
Accepted Isambard artifacts remain frozen; this integration starts no site jobs.

Starting pins: frontend dev 6a067df, multi-GPU frontend 2b57b59; core/main
04e126e, multi-GPU core 179df45; integrators/main cdbbc52; native backend
430aef7 plus the existing untracked operator survey to checkpoint separately.
Native source and survey artifacts are preserved; the unrelated Isambard archive
in the frontend checkout is not part of the code checkpoint.

The paired integration now combines the accepted particle layout with the
owned-row infrastructure against integrators cdbbc52. Core/main merge a2c85b5
is qualified; frontend and native-worktree conflicts are resolved, with final
precision/CUDA checks in progress. The native tree retains its Warp path for
chunked execution; this does not claim C++ distributed support. Results and
remaining limits: [integration report](docs/particle_layout/MULTIGPU_INTEGRATION_2026-11-03.md).

Final local gates pass: 54 FP64 CPU runner/history controls, 13 combined CUDA
controls against the native tree's Warp backend, 1,751 native C++ CPU and 1,752
native C++/CUDA checks, and the actual Morton CompSPH/CRKSPH launcher with
1→2→3-rank reference/save/resume (2 controls). Accepted Isambard deployment is
unchanged. Frontend merge85ae913 and native development merge82dac97 are committed; no
further qualification is pending within this bounded local integration scope.
