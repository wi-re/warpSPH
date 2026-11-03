# Coordinated 0.7.1 GitHub checkpoint

User decision, 3 November 2026: align warpSPH, warpSPHCore and
warpSPHIntegrators at 0.7.1 and publish the qualified main-branch checkpoint to
GitHub. C++ native development remains on its separate branch. No PyPI upload.

- Update package/runtime versions, frontend dependency bounds and release notes.
- Build and check distributions; exercise all three installed wheels together
  outside the source checkouts and the integrator isolated-wheel consumer gate.
- Commit dependency releases and frontend/dev; merge dev into frontend/main.
- Tag each main v0.7.1, push dependency main branches then frontend main/dev,
  verify remote refs, and inspect triggered CI. Preserve source worktrees,
  accepted Isambard artifacts, unrelated archive and user Focus.

Starting pins: frontend/dev 705ea6d, frontend/main 66492a2, core/main a2c85b5,
integrators/main cdbbc52. Native C++ branch 82dac97 is excluded.

## Release validation

- Three source archives and wheels build from clean staging and pass Twine
  metadata checks. Wheel Python files match current source exactly.
- All three installed distributions/runtime imports report0.7.1;14 installed
  combined solver/restart/recovery controls pass outside the source checkouts.
- Integrator full CPU:8,851 passed,639 skipped (592.10s). Installed consumer
  gate:49 passed. Catalog,1896 import bindings, repository links,57 published
  examples and production documentation build pass. Installed frontend CLI works.
- Evidence: .tmp/release-0.7.1/, including artifact-hashes.json and wheel report.

## Container follow-up (user,3 November2026)

After release checks, build a general0.7.1 Docker image and a fresh deployable
Isambard ARM64 SIF/bundle using the established local recipes. Include the optional
Remote companion, retain accepted image/profile artifacts, pin source/base/runtime
provenance, and qualify local smoke behavior. No registry push, site deployment
or Slurm submission is requested. Renewed GH200/fabric checks remain separate.

## Completed checkpoint and artifact handoff

Main branches and annotated v0.7.1 tags are pushed for frontend c231bb7,
core c06ea96 and integrators33df9b9; frontend dev is also pushed.
Docker image warpsph:0.7.1 and its reload-verified archive are built.
The fresh locally built ARM64 SIF and self-contained Isambard deployment tar
include checked source/runtime/image/communication pins and a fresh profile
marked pending-site-qualification. Previous accepted deployment is preserved.
See [artifact handoff](containers/releases/0.7.1.md) for files, checksums,
checks, local Docker networking limits and remaining site gates.

Hosted documentation builds passed. Hosted test workflows were still running
at the artifact handoff; local full integrator and paired image gates passed.
No PyPI/registry upload or site jobs. C++ development remains separate.

## Runtime image size follow-up (3 November 2026)

User rejected the initial download size as impractical for hosting. The first
Docker archive is 9.63 GB compressed; Docker initially reported 32.29 GB of
local image storage, dominated by its NGC development base. Build a fresh runtime-only final stage, retain
CUDA/Torch/NCCL provenance, measure transfer size and rerun deployment controls.
Keep previous artifacts and the accepted site baseline. Do not silently reuse
the Isambard communication payload with an incompatible CUDA/NCCL runtime.

The replacement recipe copies the pinned NGC Torch/CUDA/NCCL binaries into a
clean Ubuntu 24.04 final stage. Keep matching functorch/Triton, CUDA/Python
headers and g++ for compilation operators, visualization/video and Remote.
Preserve NVIDIA's forward-compat bundle and activation check: accepted Isambard
communication provenance maps its libcuda. Exclude that bundle from global
ldconfig and leave its driver selection to NVIDIA's check. Strip other native
symbols only when allocated ELF section bytes/addresses remain identical; retain
original and runtime SHA256 hashes. Physical site qualification stays pending.

Runtime follow-up completed: general Docker archive4.031GB (58.1% smaller),
ARM64 SIF3.443GB (63.7% smaller), self-contained Isambard tar3.506GB. Same
CPU/CUDA/resumed numerical states bit-for-bit across29datasets. Runtime compiler,
graph, NCCL/Gloo and real companion lifecycle gates pass;7packaging tests pass
(2optional host HTTP tests skipped). ARM64 QEMU source/runtime and SIF metadata,
bundle files/links and archive contents are checked. Preserve original numerical
flags (notably TF32), CUDA aliases/forward compatibility and copied venv Python.
The reusable exporter is also exercised against the full source image.
Receipts and failures are in .tmp/runtime-slim-20261103/. Smaller artifacts are
primary local tags; previous images/accepted site baseline remain intact.
See containers/releases/0.7.1-runtime.md. Physical site gates remain pending.

## Remote layout follow-up (3 November 2026)

User authorized reorganizing the idle Isambard home under `$HOME/warpsph/`:
versioned immutable releases, explicit environment selectors, separate tools,
inputs, qualification results, runs, staging and archive. Preserve the qualified
2ecb382 deployment as `current`; 0.7.1 remains pending site qualification.
Completed 15 same-filesystem, inode-preserving moves. Image/profile/communication
hashes match before and after; all 113 accepted artifacts/checkpoints verify.
Submission paths pass with sbatch replaced by a stub; queue remains empty.
Original Conda interpreter retained for batches; fresh Python3.11 maintenance
venv added. Conda files live under tools/miniforge3 with a home compatibility
symlink preserving embedded prefixes; ~/dev is unchanged. User follow-up adds
an interactive-only .bashrc selector with backup, preserving existing Conda/mamba
initialization and noninteractive exported paths; fresh login checks pass.
Historical receipts/results remain unchanged; move map and verification records
are under archive/migration-20261103. No jobs or 0.7.1 upload.
See [remote layout](scripts/isambard/REMOTE_LAYOUT.md).
