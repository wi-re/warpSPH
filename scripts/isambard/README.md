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
`/home/b6ez/wire.b6ez/warpsph-qa`. Set a different absolute base to use another
shared filesystem location. Bundle, image, results and build logs can also be
overridden individually.

**The previously uploaded archive remains usable.** These path settings already
work with its scripts. Skip the upload commands if it is already on Isambard;
set `QA_ROOT` to the directory containing that archive. For example, use
`export QA_ROOT="$HOME"` if you copied it directly into your home directory.

For a fresh upload, on your local machine with your normal Clifton certificate
and SSH configuration:

```bash
export ISAMBARD_SSH=PROJECT.aip2.isambard  # your existing SSH alias
export ISAMBARD_ROOT="${ISAMBARD_ROOT:-/home/b6ez/wire.b6ez/warpsph-qa}"
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
export QA_ROOT="${QA_ROOT:-$HOME/warpsph-qa}"
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
not establish or retain an SSH connection.

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

```bash
export ISAMBARD_SSH=PROJECT.aip2.isambard
export ISAMBARD_ROOT="${ISAMBARD_ROOT:-/home/b6ez/wire.b6ez/warpsph-qa}"
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
export QA_PREVIOUS="$QA_RESULTS_ROOT/restart-RESTART_JOB_ID"
export QA_NETWORK=custom
export QA_COMM_ROOT="${QA_COMM_ROOT:-$QA_ROOT/communication}"
export QA_LIBFABRIC_HOST=/opt/cray/libfabric/1.22.0
export QA_NETWORK_SETUP="${QA_NETWORK_SETUP:-$QA_ROOT/network-setup.sh}"
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

## Validation boundary and local development

Local qualification on 2026-11-02 passed all three available CPU/Gloo stages
(`single`, two-rank `same-node`, and cross-job/rank-count `restart`), including
checkpoint comparisons and sealed-artifact verification. The shipped probe also
passed inside the existing Docker image on one CUDA GPU with NCCL. Focused tests
cover corrupt checkpoints, numerical/RNG drift, retrieval damage, wrong topology,
socket transport fallback and Slurm submission/environment handling.

Actual Isambard ARM64, Apptainer, multi-GPU and multi-node execution can
only be certified from your returned site reports. The first action on Isambard
should be the ARM64 build followed by **only** `single`.

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
[Apptainer environment](https://apptainer.org/docs/user/latest/environment_and_metadata.html).
