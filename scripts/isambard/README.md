# Staged Isambard acceptance checks

Plan: [whole-row chunks and multi-GPU](../../CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).

Submit **one stage, retrieve its results, verify them locally, then continue**.
This suite checks deployment and correctness with a tiny deterministic CRKSPH
fixture. It does not benchmark performance or qualify every scheme/boundary case.
Video is omitted for these repeated four-step fixtures; progress, diagnostics,
native checkpoints and rank/transport reports are retained.

| Stage | Default allocation | Required checks |
|---|---|---|
| `single` | 1 node × 1 GPU | ARM64/runtime, real Warp/Torch CUDA stream ordering, NCCL collectives, shared output, continuous four steps versus 2+restart+2 with changed row cap/offset width |
| `same-node` | 1 node × 2 GPUs | Distinct physical devices/processes, collectives, full particle-state consistency with a fresh one-GPU baseline; also supports `QA_GPUS_PER_NODE=4` |
| `restart` | 1 node × 2 GPUs | Checkpoint from the **preceding Slurm job** resumes on one GPU; a fresh one-GPU prefix resumes on multiple GPUs; both agree with uninterrupted four steps |
| `multi-node` | 2 nodes × 1 GPU | Distinct hosts/devices, shared storage, AWS Libfabric/CXI transport proof, one-GPU versus distributed consistency and multi-node checkpoint continuation on one GPU |

Later stages require the previous stage's `report.json` to say `passed`, with
matching image, application/runtime fingerprint and precision. No chain of jobs
is submitted automatically. Results are a new directory for every job. A failed
probe/comparison exits nonzero and retains a failure report and available output.

## 1. Prepare the native ARM64 bundle locally

The existing `warpsph:local-cuda-20261102` image is **x86-64**. Do not transfer it
to Isambard or reuse its architecture-specific base digest for ARM64. Export the
same local/unpushed source using the existing builder, without Docker execution:

```bash
python scripts/build_container.py --prepare-only --platform linux/arm64 \
  --base-image nvcr.io/nvidia/pytorch:26.01-py3 \
  --context .tmp/isambard-arm-context \
  --source warpSPHCore=/path/to/paired/warpSPHCore \
  --source warpSPHIntegrators=/path/to/warpSPHIntegrators \
  --source warpSPHPlotting=/path/to/warpSPHPlotting \
  --source warpSPHRemote=/path/to/warpSPHRemote --with-remote

python scripts/isambard/prepare_bundle.py --context .tmp/isambard-arm-context \
  --output .tmp/isambard-bundle --archive .tmp/warpsph-isambard.tar.gz
sha256sum .tmp/warpsph-isambard.tar.gz
```

Use an ARM64-capable base compatible with the site's driver. The tag above is
the example in Isambard's newer-container NCCL guide; prefer a resolved ARM64
digest for a release. `--constraints` can pin a **target-appropriate** dependency
set; do not apply the local amd64/NVIDIA wheel lock blindly to ARM64.

The archive contains the package snapshots/provenance, a generated Apptainer
definition, and this harness. There is no Conda environment, x86 binary/JIT cache,
SSH credential or large prebuilt image. The optional Remote daemon is installed
by the generated definition when the context selects `--with-remote`; the site
tests below do not expose a daemon or claim live-viewer connectivity.

## 2. Upload over your normal SSH connection

All remote working paths derive from one base directory, `QA_ROOT`. On your
Isambard account, `$HOME` is `/home/b6ez/wire.b6ez`, so the default below becomes
`/home/b6ez/wire.b6ez` itself. Set a different absolute base to use another
shared filesystem location. Bundle, image, results and build logs can also be
overridden individually.

**The previously uploaded archive remains usable.** These path settings already
work with its scripts. Skip the upload commands if it is already on Isambard;
set `QA_ROOT` to the directory containing that archive. For example, use
`export QA_ROOT="$HOME"` if you copied it directly into your home directory.

For a fresh upload, on your local machine with your normal Clifton certificate
and SSH configuration:

```bash
export ISAMBARD_SSH="${ISAMBARD_SSH:-b6ez.aip2.isambard}"
export ISAMBARD_ROOT="${ISAMBARD_ROOT:-/home/b6ez/wire.b6ez}"
export QA_ARCHIVE="${QA_ARCHIVE:-.tmp/warpsph-isambard.tar.gz}"
ssh "$ISAMBARD_SSH" "mkdir -p '$ISAMBARD_ROOT'"
scp "$QA_ARCHIVE" "$ISAMBARD_SSH:$ISAMBARD_ROOT/"
ssh "$ISAMBARD_SSH"
```

On the login node, check your scheduler account before submitting:

```bash
sacctmgr show user "$(id -un)" withassoc
```

For your `wire.b6ez` login, the returned default and associated Slurm account is
**`brics.b6ez`**. Use the `Account` column from this command if using another
allocation; the login name and home-directory name do not identify a Slurm
account. Your returned association has no partition-specific entry, so start
with the default partition. If an explicit partition is needed, check `sinfo`
and the association output before setting `QA_PARTITION` for the test wrapper
or adding `--partition=...` to the build submission.

Use the same base as `ISAMBARD_ROOT` above, or the location of your existing
upload:

```bash
export QA_ROOT="${QA_ROOT:-$HOME}"
export QA_BUNDLE="${QA_BUNDLE:-$QA_ROOT/warpsph-isambard}"
export QA_SIF="${QA_SIF:-$QA_ROOT/images/warpsph-arm64.sif}"
export QA_IMAGE="${QA_IMAGE:-$QA_SIF}"
export QA_RESULTS_ROOT="${QA_RESULTS_ROOT:-$QA_ROOT/qualification}"
export QA_BUILD_LOGS="${QA_BUILD_LOGS:-$QA_ROOT/logs}"
export QA_ACCOUNT=brics.b6ez             # confirmed by your sacctmgr association
mkdir -p "$QA_BUILD_LOGS"
cd "$QA_ROOT"
sha256sum warpsph-isambard.tar.gz       # compare with the local archive hash
tar -xzf warpsph-isambard.tar.gz       # extract once; keep an existing extraction
sbatch --account="$QA_ACCOUNT" --output="$QA_BUILD_LOGS/build-%j.out" \
  "$QA_BUNDLE/harness/build.slurm"
```

Keep these exports in your shell or in a small settings file you source after
logging in. All derived directories are siblings under `QA_ROOT`; changing the
base requires re-evaluating the derived exports in a fresh shell, or explicitly
setting their overrides. An existing extracted bundle can live elsewhere: set
`QA_BUNDLE` to its actual location before the block. The SIF output must be a
fresh filename; keep a previously successful image for subsequent test jobs.

For your current uploaded campaign, the confirmed results directory is
`/home/b6ez/wire.b6ez/qualification`. Keep it explicitly in subsequent login
shells:

```bash
export QA_RESULTS_ROOT="$HOME/qualification"
```

Later stage submissions and `QA_PREVIOUS` examples use this variable directly;
changing `QA_ROOT` alone does not change an already-exported results root.
Build logs are under `QA_BUILD_LOGS`; test Slurm logs are under
`$QA_RESULTS_ROOT/logs`. Keep the running job's existing output directory.

The build job verifies all uploaded files, requires native `aarch64`, then builds
with Apptainer fakeroot **on a compute allocation**. Its defaults are one GPU,
16 CPU cores and one hour; override account/partition/time as needed. It retains
the SIF, SHA256 and inspection metadata. GPU access for the build allocation is
not proof that the final image can execute; stage `single` provides that gate.
Build scratch uses a fresh directory under `QA_BUILD_TMP_ROOT`, `SLURM_TMPDIR`,
`TMPDIR`, or `/tmp`, in that order, and is removed when the build job exits.
Host orchestration needs Python3.8+ (`QA_PYTHON` selects it); all NumPy/h5py/Torch
analysis for submitted jobs runs inside the image.

Watch `squeue -u "$USER"` and the build log. Do not run image builds/simulations
on login nodes. Review commands before running them yourself; these tools do
not establish or retain an SSH connection automatically. Direct read-only SSH
execution from the local workspace has also been verified through
`b6ez.aip2.isambard`; site commands can use that alias when authorized.

## 3. Submit only the single-GPU stage first

After the build succeeds, on the login node:

```bash
export QA_NETWORK=none
bash "$QA_BUNDLE/harness/submit.sh" single
```

This returns a job ID. `squeue -j JOB_ID` shows its state; follow
`$QA_RESULTS_ROOT/logs/single-JOB_ID.out`. The result directory is
`$QA_RESULTS_ROOT/single-JOB_ID`. `QA_PARTITION`, `QA_TIME` (default45 minutes),
`QA_CPUS_PER_NODE` (default16) and `QA_PRECISION=float32|float64` are configurable.
Keep the same precision throughout a campaign. Start with the default8² fixture;
larger capacity/scaling studies belong to a later gate.

The rank probe performs all-reduce, int64 all-gather/broadcast, reduce-scatter,
ring send/receive for multiple ranks, a nondefault-stream Warp/Torch operation,
and per-rank shared-filesystem visibility checks. It records GPU UUIDs, hosts,
PIDs, package versions, loaded source identities and network settings.

## 4. Retrieve and verify before progressing

From your local machine, set the SSH alias and remote base as in step2. Match
`ISAMBARD_ROOT` to the `QA_ROOT` actually used on the login node; if you overrode
`QA_RESULTS_ROOT` remotely, set `ISAMBARD_RESULTS` to that same absolute path.

For the current campaign under your home directory, set these **locally** before
the retrieval commands, overriding any earlier fresh-campaign defaults:

```bash
export ISAMBARD_ROOT=/home/b6ez/wire.b6ez
export ISAMBARD_RESULTS="$ISAMBARD_ROOT/qualification"
```

```bash
export ISAMBARD_SSH="${ISAMBARD_SSH:-b6ez.aip2.isambard}"
export ISAMBARD_ROOT="${ISAMBARD_ROOT:-/home/b6ez/wire.b6ez}"
export ISAMBARD_RESULTS="${ISAMBARD_RESULTS:-$ISAMBARD_ROOT/qualification}"
export QA_DOWNLOADS="${QA_DOWNLOADS:-$PWD/downloaded-qa}"
export QA_RUN=single-JOB_ID             # replace with the actual stage/job ID
mkdir -p "$QA_DOWNLOADS"
rsync -a "$ISAMBARD_SSH:$ISAMBARD_RESULTS/$QA_RUN/" "$QA_DOWNLOADS/$QA_RUN/"
rsync -a "$ISAMBARD_SSH:$ISAMBARD_RESULTS/logs/$QA_RUN.out" \
  "$QA_DOWNLOADS/$QA_RUN/logs/slurm.out"
python scripts/isambard/check.py verify "$QA_DOWNLOADS/$QA_RUN"
```

Verification needs only NumPy/h5py, not a GPU or Torch. `artifacts.json` checks
every stable result/config/probe/checkpoint by SHA256; HDF5 files additionally
need a valid native completion marker/content checksum and finite state. Live
Slurm stdout and NCCL transport logs are retrieved but excluded from this sealed
set because the surrounding job can still finish writing them.

Inspect `report.json`, `probes/probe.json`, `checks/*.json`, `runs/*/manifest.json`
and `runs/*/diagnostics.jsonl`. `runs/*/finalState.h5` contains real particle data;
the sparse `checkpoint-00000002.h5` is deliberately retained for the restart
stage. You can inspect these locally with h5py and use the existing viewer's
standalone-state HDF5 workflow for a final snapshot. Keep failed job directories.

Comparisons align particle arrays by UID, require exact identities/forcing RNG
and clocks, and compare all saved physical state/context fields. Default numeric
tolerances are float32 `rtol=2e-5, atol=2e-8` and float64 `rtol=1e-10, atol=1e-12`;
every field's maximum absolute difference is reported. Ownership/search caches
are not required to be byte-identical. No tolerance is automatically relaxed.

## 5. Subsequent stages, separately

On the login node, after inspecting/downloading each predecessor:

```bash
export QA_PREVIOUS="$QA_RESULTS_ROOT/single-SINGLE_JOB_ID"
export QA_GPUS_PER_NODE=2
bash "$QA_BUNDLE/harness/submit.sh" same-node

# After same-node passes:
export QA_PREVIOUS="$QA_RESULTS_ROOT/same-node-SAME_JOB_ID"
bash "$QA_BUNDLE/harness/submit.sh" restart
```

Use four GPUs by setting `QA_GPUS_PER_NODE=4` for a fresh same-node/restart
campaign. `restart` binds the predecessor read-only and resumes its saved step2
checkpoint in a **new Slurm allocation**. It also tests a new 1→N rank-count
transition and offset/row-cap changes. An independent uninterrupted reference
within the job makes the numerical comparison independent of checkpoint bytes.
This is a distributed implementation consistency check: the reference uses the
same solver on one rank and is not an independent physics oracle.

## 6. Multi-node networking is a separate prerequisite

For GPU Slingshot RDMA, Isambard requires the NCCL `aws-ofi-nccl` plugin and CXI
Libfabric provider. A newer CUDA/NGC image may require communication libraries
built against that image's runtime. Follow the site's current NCCL container
guide before this stage; native MPI-enabled Torch remains unnecessary here.

Your returned module inventory shows host NCCL2.26.6/CUDA12.6 and OFI plugin1.8.1,
versus the tested image's NCCL2.29.2/CUDA13.1. Direct inspection confirms
`adapt.sh` prepends the host NCCL library directory. The installed host Libfabric prefix
is `/opt/cray/libfabric/2.3.1`. Check actual runtime/provider loading before two
nodes; do not assume that injecting the host module preserves the tested stack.

### One-GPU network preflight for this campaign

This new file works with the existing uploaded bundle and unchanged SIF. Copy
only the small job file from this checkout on your local machine:

```bash
scp scripts/isambard/network_preflight.slurm \
  "$ISAMBARD_SSH:/home/b6ez/wire.b6ez/warpsph-isambard/harness/"
```

On the Isambard login node:

```bash
export QA_ROOT="$HOME"
export QA_BUNDLE="$QA_ROOT/warpsph-isambard"
export QA_IMAGE="$QA_ROOT/images/warpsph-arm64.sif"
export QA_RESULTS_ROOT="$QA_ROOT/qualification"
export QA_PREVIOUS="$QA_RESULTS_ROOT/restart-7989645"
export QA_PREFLIGHT_NETWORK=host
sbatch --account=brics.b6ez --output="$QA_ROOT/logs/network-preflight-%j.out" \
  "$QA_BUNDLE/harness/network_preflight.slurm"
```

The job uses one GPU for at most15 minutes. It forces AWS Libfabric/CXI and runs
the real GPU probe, requiring the same SIF, source/package fingerprint and loaded
NCCL runtime version as the accepted restart stage. It queries the already-loaded
library directly before process-group teardown and saves its mapped path plus
CUDA/fabric library paths; Torch
package/build metadata alone is insufficient. Runtime downgrade, import/plugin
failure or Socket fallback fails; inspect `network-preflight-JOB_ID/report.json`
and logs. A successful preflight certifies library/provider loading only.
Its private harness copy also handles the site's `adapt.sh` trailing `exec "$@"`
without consuming the probe arguments; the original uploaded scripts stay intact.
Keep its output to review before submitting two nodes. It is a preparatory job,
not a sealed solver stage; do not use `check.py verify` on this directory.

It preserves RAS and chooses a job-derived loopback diagnostic port instead of
fixed28028; `QA_RAS_PORT` can select another unused port. For an already-prepared
custom communication stack, set `QA_PREFLIGHT_NETWORK=custom` and the three
custom-path variables below. A failed host preflight identifies whether a
compatible OFI plugin/build is needed; it does not modify the image or libraries.

The host-profile trial **7998167** completed on `nid010291` in 37 s and failed at
Torch import with `undefined symbol: ncclCommWindowDeregister`. The inspected
adapter prepends host NCCL 2.26.6 over the accepted image's 2.29.2. Runtime/provider
capture was not reached. Upload, submission, failure reporting and retrieval
worked; this host-profile result does not qualify GPU networking. The custom
profile described below now passes without replacing the accepted image or
prior solver-stage results.

### Build only the compatible OFI plugin

`build_network.slurm` builds aws-ofi-nccl 1.18.0 and hwloc 2.13.0 natively inside
this campaign's unchanged SIF. The [site guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/)
describes this approach; the [OFI release](https://github.com/aws/aws-ofi-nccl/releases/tag/v1.18.0)
supports NCCL 2.29.2. Published release checksums and source revisions are pinned
in `communication-sources.json`. The build verifies checksums before extraction,
uses host Libfabric 2.3.1, disables AWS-specific platform tuning, records dependency
paths/library hashes and checks the loaded NCCL against the accepted restart.
It does not rebuild NCCL, CUDA or the SIF. Build tools must already exist in the
image; failures retain logs and `network-build-JOB_ID/report.json`.

Download the pinned release archives on the local upload machine:

```bash
export QA_COMM_SOURCES_LOCAL=/fresh/local/communication-sources
python - "$QA_COMM_SOURCES_LOCAL" <<'PY'
import hashlib, json, sys, urllib.request
from pathlib import Path
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
lock = json.loads(Path('scripts/isambard/communication-sources.json').read_text())
for package in lock.values():
    with urllib.request.urlopen(package['url'], timeout=90) as response:
        data = response.read()
    algorithm = 'sha512' if 'sha512' in package else 'sha256'
    if hashlib.new(algorithm, data).hexdigest() != package[algorithm]:
        raise ValueError('Release checksum mismatch: '+package['file'])
    (root/package['file']).write_bytes(data)
PY
ssh "$ISAMBARD_SSH" 'mkdir -p "$HOME/communication-sources" "$HOME/warpsph-communication-build/harness"'
scp "$QA_COMM_SOURCES_LOCAL/"*.tar.gz "$ISAMBARD_SSH:$ISAMBARD_ROOT/communication-sources/"
scp scripts/isambard/{build_network.slurm,build_network.sh,communication-sources.json,network-custom.example.sh,network_preflight.slurm} \
  "$ISAMBARD_SSH:$ISAMBARD_ROOT/warpsph-communication-build/harness/"
```

On the login node, with the usual QA paths exported and the incompatible host
injection module unloaded:

```bash
export QA_PREVIOUS="$QA_RESULTS_ROOT/restart-7989645"
export QA_BUILD_HARNESS="$QA_ROOT/warpsph-communication-build/harness"
export QA_COMM_SOURCES="$QA_ROOT/communication-sources"
export QA_LIBFABRIC_HOST=/opt/cray/libfabric/2.3.1
sbatch --account=brics.b6ez --output="$QA_ROOT/logs/network-build-%j.out" \
  "$QA_ROOT/warpsph-communication-build/harness/build_network.slurm"
```

Inspect the build report before the next submission. Its `communicationRoot`
defaults to `$QA_ROOT/communication/ofi-BUILD_JOB_ID`. After a successful build:

```bash
export QA_COMM_ROOT="$QA_ROOT/communication/ofi-BUILD_JOB_ID"
export QA_NETWORK_SETUP="$QA_COMM_ROOT/network-setup.sh"
export QA_PREFLIGHT_NETWORK=custom
sbatch --account=brics.b6ez --output="$QA_ROOT/logs/network-preflight-%j.out" \
  "$QA_ROOT/warpsph-communication-build/harness/network_preflight.slurm"
```

The custom preflight additionally requires the mapped plugin from this build,
bound host Libfabric and a loaded libcxi, alongside unchanged application/runtime
fingerprints and actual NCCL version. Build/load success alone does not qualify
collectives or inter-node traffic. Retrieve and review the one-GPU result before
two nodes. Keep failed builds in their separate directories for diagnosis.

For this campaign, **build 7999853** and **custom preflight 8000996** now pass
(3m45s and 1m12s). The retrieved audit verifies all 3 built shared-library hashes,
unchanged SIF/application fingerprint and actual NCCL2.29.2/CUDA13.1. Live maps
show the new OFI plugin, host Libfabric2.3.1 and libcxi; NCCL logs select AWS
Libfabric/CXI, SENDRECV and discover 4 NICs with no NCCL warnings. One-GPU kernel,
stream and collective checks pass. Inter-node traffic/performance remain untested.
Use `$QA_ROOT/communication/ofi-7999853` and its `network-setup.sh` for the next
reviewed two-node/one-GPU gate; no second build is needed for this accepted SIF.

The first build 7999738 failed before compilation because Slurm spools scripts;
`QA_BUILD_HARNESS` now resolves the helper directory explicitly. First custom
preflight 8000492 passed GPU/transport work but its final map check ran after
NCCL unloaded the plugin. Current preflight captures before teardown, always
cleans up the group, and retains the actual executed batch script. Both failed
results remain preserved; final acceptance uses 8000996.

### Host and custom profiles

Two explicit profiles are supported:

- `QA_NETWORK=host`: loads `brics/apptainer-multi-node` and sources
  `/host/adapt.sh` in the container. Use only with a compatible image/host stack;
  the site documents GLIBC/runtime incompatibilities for some combinations.
- `QA_NETWORK=custom`: set `QA_COMM_ROOT` to the compatible ARM64 communication
  build directory, `QA_LIBFABRIC_HOST` to the host Libfabric prefix, and
  `QA_NETWORK_SETUP` to a reviewed copy of `network-custom.example.sh`. The wrapper
  binds these to `/opt/slingshot`, `/host/libfabric`, `/network-setup.sh` and
  mounts `/usr/lib64` for libcxi. Adjust provider paths/settings to the installed
  site versions. This suite does not silently rebuild or replace network libraries.

Set `QA_COMM_ROOT` to an existing compatible build and `QA_NETWORK_SETUP` to
your reviewed setup file; the defaults below place them under `QA_ROOT`. Then:

```bash
export QA_PREVIOUS="$QA_RESULTS_ROOT/restart-7989645"
export QA_NETWORK=custom
export QA_COMM_ROOT="${QA_COMM_ROOT:-$QA_ROOT/communication/ofi-7999853}"
export QA_LIBFABRIC_HOST=/opt/cray/libfabric/2.3.1
export QA_NETWORK_SETUP="${QA_NETWORK_SETUP:-$QA_COMM_ROOT/network-setup.sh}"
export QA_NODES=2 QA_GPUS_PER_NODE=1
bash "$QA_BUNDLE/harness/submit.sh" multi-node
```

Start with two nodes/one GPU each; later use `QA_GPUS_PER_NODE=4`. There is one
Slurm task/torchrun launcher per node, and one process per physical GPU. GPU lists
are preserved literally through Apptainer's clean environment. Step failures kill
peer tasks; c10d has a finite timeout and Slurm bounds total runtime.

Multi-node forces `NCCL_NET="AWS Libfabric"` and `FI_PROVIDER=cxi`, and requires
each rank's NCCL log to evidence both that network and CXI provider. A missing
plugin, socket fallback, duplicate GPU, wrong node count, local-only storage,
source drift, corrupt checkpoint or numerical mismatch **fails** the stage.
All NCCL logs remain available when a site/plugin version uses a different log
format; review the evidence rather than treating absence of proof as a pass.

## Reusing the accepted deployment for simulations

`site-profile.json` pins the accepted SIF SHA256, application/package fingerprint,
CUDA/NCCL versions, OFI/hwloc sources and installed shared-library hashes. Its
companion `site-profile.sh` exports paths relative to `QA_ROOT` (default `$HOME`),
account `brics.b6ez`, the custom profile and Slurm network option. Keep the whole
harness together in a versioned shared directory; source the profile from that
directory so Slurm spool paths never select helper files. The current image uses
NGC26.01, CUDA13.1, NCCL2.29.2, OFI1.18.0/hwloc2.13.0 and host Libfabric2.3.1.

```bash
source "$HOME/warpsph-runtime-COMMIT/site-profile.sh"
export QA_PREVIOUS="$QA_RESULTS_ROOT/restart-7989645"
export QA_NODES=2 QA_GPUS_PER_NODE=1
export QA_TIME=00:30:00 QA_CPUS_PER_NODE=16
bash "$QA_HARNESS_ROOT/submit.sh" multi-node
```

`disable_rdzv_get` is passed to both the allocation and each `srun` step, following
the [site NCCL examples](https://docs.isambard.ac.uk/user-documentation/guides/nccl/).
The custom profile adds only OFI/hwloc/host fabric paths ahead of the image's
libraries, keeping `/host/usr/lib64` last. Do not load the incompatible
`brics/apptainer-multi-node` module or inherit implicit Apptainer bind paths.
`network-setup.sh` contains every NCCL/FI tuning variable; its hash and full text
are captured, alongside explicit bind destinations, GPU visibility handling,
rendezvous/RAS rules, host fabric/CXI binary hashes and selected Slurm settings.
No SSH credentials or unrelated environment variables are recorded.

With this profile selected, qualification saves `probes/deployment.json`, pins
its digest in `report.json`, and seals it with the other retrieved artifacts.
Image, communication-library, setup-script or Slurm-network drift fails before
the GPU probe. `deployment.py` can write the same receipt for real simulations.
New application code requires a new source-pinned image and qualification; moving
intact accepted artifacts to another shared root is supported through path exports.

For an actual simulation, save a supported eager distributed `RunConfiguration`
JSON and use this batch-body pattern with the same node/container wrappers. Set
`SIM_CONFIGURATION` to its absolute host path, `SIM_OUTPUT` to a fresh shared
directory, `SIM_EDGE_BUDGET` to the chosen edge budget, `SIM_GPUS_PER_NODE` to the
allocated GPU count, and `SIM_CHECKPOINT_EVERY` to the desired checkpoint cadence.
Select node/GPU/time/CPU resources at submission, matching those worker settings;
include `--network=disable_rdzv_get`. Source the profile on the login node and
export these variables before `sbatch`; the exported QA_HARNESS_ROOT locates it
from Slurm's spooled batch script.

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${QA_HARNESS_ROOT:?Source the versioned site-profile.sh before sbatch}"
source "$QA_HARNESS_ROOT/site-profile.sh"
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${SIM_CONFIGURATION:?Absolute saved RunConfiguration JSON}"
: "${SIM_OUTPUT:?Fresh shared output directory}"
: "${SIM_EDGE_BUDGET:?Choose the edge budget}"
: "${SIM_GPUS_PER_NODE:?Match the Slurm GPU allocation}"
: "${SIM_CHECKPOINT_EVERY:?Choose checkpoint cadence in steps}"
mkdir "$SIM_OUTPUT"
cp -- "$SIM_CONFIGURATION" "$SIM_OUTPUT/configuration.json"
export QA_NODES=$SLURM_NNODES QA_GPUS_PER_NODE=$SIM_GPUS_PER_NODE
export QA_MASTER_ADDR QA_MASTER_PORT=$((20000 + SLURM_JOB_ID % 35000))
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
QA_MASTER_ADDR=${hosts[0]}
"$QA_PYTHON" "$QA_HARNESS_ROOT/deployment.py" --profile "$QA_DEPLOYMENT_PROFILE" \
  --output "$SIM_OUTPUT/deployment.json"
srun --network="$QA_SLURM_NETWORK" --nodes="$QA_NODES" --ntasks="$QA_NODES" \
  --ntasks-per-node=1 --gpus-per-task="$SIM_GPUS_PER_NODE" --kill-on-bad-exit=1 --unbuffered \
  bash "$QA_HARNESS_ROOT/node.sh" "$QA_IMAGE" "$SIM_OUTPUT" torchrun \
  "$QA_NODES" "$SIM_GPUS_PER_NODE" simulation run_distributed.py \
  --configuration /work/configuration.json --output /work/run --backend nccl \
  --edge-budget "$SIM_EDGE_BUDGET" --offset-dtype int64 --replica-checks exact \
  --checkpoint-every "$SIM_CHECKPOINT_EVERY" --timeout-seconds 600
```

Keep `deployment.json`, copied configuration, native manifests/checkpoints and
logs with the simulation data. The example reuses the qualified bootstrap; larger
resolutions, longer integrations and additional GPU counts require their own
capacity/numerical/performance checks. `--shared-gpu` is a local debugging mode
and is excluded from physical-node deployment.

## Startup messages observed on Isambard

Returned job7986156 passes native ARM64/GH200 single-GPU execution, checkpoint
continuation and offline artifact/state checks. Its repeated startup messages
have separate causes:

- `Overriding HOME ... not permitted`: the original wrapper redundantly passed
  HOME using `--env`. Current `node.sh` removes that unsupported argument;
  `container.sh` already sets HOME and explicit cache paths before Python.
  An already-uploaded old wrapper can still print this warning. No image rebuild
  or rerun of a passing stage is needed solely for this message.
- Missing `fuse2fs`/`gocryptfs`: optional helpers for EXT3 and encrypted filesystems.
  The tested ordinary SIF does not use these formats; no installation is needed
  for the passing fixture.
- `rm ... /usr/local/cuda/compat/lib ... Read-only file system`: consistent with
  NGC CUDA compatibility startup attempting to replace its symlink in a read-only
  SIF. The matching base release's startup script explicitly supports a non-writable
  compatibility directory by prepending its real library directory. Successful
  GPU kernels, collectives and solver/checkpoint checks establish that these
  messages did not prevent this run. Keep CUDA compatibility setup and stderr
  visible; do not make the image writable merely to remove the messages.
- RAS `Call to bind failed: Address already in use`, accompanied by failed client
  listening socket at localhost28028: observed three times in restart7989645.
  The RAS diagnostic listener was unavailable; initialization, GPU transfers and
  exact continuation comparisons still completed. The network preflight uses
  a job-derived loopback RAS address, as recommended for jobs sharing nodes,
  and retains warning logs. This is distinct from a failed collective.
- NCCL INFO failed IB plugin initialization followed by `Using network Socket`:
  no network qualification is claimed for one rank. Multi-node must still prove
  AWS Libfabric/CXI; socket fallback cannot pass that stage.

## Validation boundary and local development

Local qualification on 2026-11-02 passed all three available CPU/Gloo stages
(`single`, two-rank `same-node`, and cross-job/rank-count `restart`), including
checkpoint comparisons and sealed-artifact verification. The shipped probe also
passed inside the existing Docker image on one CUDA GPU with NCCL. Focused tests
cover corrupt checkpoints, numerical/RNG drift, retrieval damage, wrong topology,
socket transport fallback and Slurm submission/environment handling.

Returned Isambard job7986156 now qualifies native ARM64/Apptainer/GH200
single-GPU execution and restart:23 artifacts verified,23 saved physical/clock/
forcing/context fields byte-identical after continuation, and diagnostic histories
matching step-for-step. Returned job7988602 now also qualifies two physical GH200s
on one node:20 artifacts verified, NCCL P2P/CUMEM channel evidence, and all particle/
clock/forcing/context fields byte-identical to one rank. The global work budget
alone differs by one float64 ULP (4.136e-25). The new one-rank reference also matches
the earlier job byte for byte. These tiny cold/cache-dependent fixtures do not
qualify performance.

Returned job7989645 now also passes both prior-job two→one and fresh one→two
continuation on a new node:32 artifacts verified and all23 physical/clock/forcing/
context fields byte-identical to uninterrupted execution. Diagnostics exactly
match. Three reviewed RAS listener-port conflicts are recorded as diagnostic
advisories; there are no failed GPU/solver comparisons.

For this campaign, the custom one-GPU compatibility gate has passed. Next is
the two-node/one-GPU stage using the accepted communication root above. Four
GPUs and multi-node fabric remain pending. New campaigns still start with the
build/single gates.

```bash
python -m pytest tests/test_isambard_qualification.py
python scripts/isambard/qualify.py --launcher local --backend gloo \
  --expected-architecture x86_64 --stage single --output /fresh/local-output
```

Local reports explicitly say `local-cpu-debug` and cannot substitute for site
reports. This suite checks checkpoint/file retrieval; interactive daemon routing
through Isambard SSH/Jupyter remains a later, separate connection test.

Primary references: [Slurm submission troubleshooting](https://docs.isambard.ac.uk/user-documentation/guides/slurm-troubleshooting/),
[Isambard Apptainer multi-node](https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/),
[native Apptainer builds](https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer/),
[NCCL/custom runtime builds](https://docs.isambard.ac.uk/user-documentation/guides/nccl/#build-nccl-in-a-container),
[SSH file transfer](https://docs.isambard.ac.uk/user-documentation/guides/file_transfer/local/),
[Slurm](https://docs.isambard.ac.uk/user-documentation/guides/slurm/),
[Apptainer environment](https://apptainer.org/docs/user/latest/environment_and_metadata.html),
[filesystem helpers](https://apptainer.org/docs/admin/main/configfiles.html),
[HOME override restriction](https://github.com/apptainer/apptainer/issues/388),
[NCCL RAS address](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ras-addr).
