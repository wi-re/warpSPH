# Portable simulation images

Plan: [whole-row chunks and multi-GPU](../CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).

The image recipes package the paired simulation stack for Linux x86-64/CUDA or
ARM64/CUDA. The optional `with-remote` target installs warpSPHRemote's Python
companion, allowing the existing browser viewer to launch/view ordinary local
simulations inside a Docker allocation. Rendering happens in the browser; the
image does not need a desktop or a Node development server.

**Qualification:** source snapshot/wheel packaging and native daemon/proxy HTTP
checks can run without Docker or a GPU. No Docker/Apptainer engine is installed
in the development environment, so complete image builds, container GPU access,
ARM64 execution and Isambard networking remain target-machine gates. A prepared
context is not evidence that an image has built or run. ROCm and an MPI-enabled
Torch build remain deferred.

## Build from private local repositories

Run with Python 3.11 or newer from the frontend repository:

```bash
python scripts/build_container.py \
  --context .tmp/image-context --prepare-only \
  --platform linux/amd64 \
  --base-image nvcr.io/nvidia/pytorch:26.01-py3 \
  --source warpSPHCore=/path/to/warpSPHCore \
  --source warpSPHIntegrators=/path/to/warpSPHIntegrators \
  --source warpSPHPlotting=/path/to/warpSPHPlotting \
  --source warpSPHRemote=/path/to/warpSPHRemote \
  --with-remote
```

`warpSPH` defaults to this checkout. Other packages default to same-named sibling
checkouts when present. All source overrides must name Git repository roots.
Uncommitted edits, new non-ignored package files and tracked deletions are honored.
Only package source (`src/`, `python/`) and build metadata/licenses are copied;
Git history, notebooks, run outputs and unrelated files are not bundled. Symlinks
are rejected. Review the generated file inventory before transferring a context.
No repository or package publication occurs.

Replace `--source NAME=/path` with `--package NAME==VERSION` to use a pinned public
distribution for that package. Every unspecified package still requires a local
checkout. Local wheels are installed together so a paired local package is not
silently substituted by its public release. The distributed launch script comes
from this frontend checkout and is recorded separately; public stack versions
must actually support that script before using distributed execution.

The base tag above is an **example from Isambard's NCCL documentation**, not a
qualified choice for this stack. Select a compatible Torch/CUDA/driver combination
and pin the base by OCI digest for releases. `--warp-version` defaults to 1.17.0;
`--constraints target-constraints.txt` adds dependency constraints for the selected
platform. The base's exact Torch version is constrained during installation so
pip cannot silently replace its CUDA-enabled Torch build. Builds fail on dependency
conflicts. Other transitive dependencies are resolved at build time unless the
constraints file pins them: the recipe alone is not yet a dependency lock.

To build immediately on a Docker-capable machine, omit `--prepare-only` and add
`--image warpsph:local`. To build a transferred context:

```bash
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 \
  --build-arg WARP_VERSION=1.17.0 \
  --target with-remote -f .tmp/image-context/containers/Dockerfile \
  -t warpsph:local .tmp/image-context
```

Use the same base/platform/Warp version recorded by context preparation. For
ARM64, change both platform arguments and use a base supporting ARM64. Prefer
native builds and run validation on the target GPU architecture. Remove
`--with-remote` and select target `simulation` for an image without the installed
companion/proxy. Each architecture produces a separate image; validated variants
can subsequently be published under a multi-platform OCI manifest. Publication
is a separate action.

Local checks (nginx plus a Remote checkout are needed for the HTTP check):

```bash
WARPSPH_TEST_REMOTE_ROOT=/path/to/warpSPHRemote \
  python -m pytest tests/test_container_packaging.py
```

`WARPSPH_TEST_NGINX=/path/to/nginx` can select a non-installed test binary. Without
the optional dependencies, the HTTP check is explicitly skipped. These tests
exercise dirty-source capture, excluded files/symlinks, mixed public/local context
preparation, same-version private-wheel replacement, authenticated viewer access,
Origin/Host gates, private producer registration and service shutdown.

The image retains `/opt/warpsph/stack-manifest.json` (source commits, dirty flags,
per-file hashes, recipe hashes and requested base/platform), `wheel-hashes.json`
and `installed-*.txt` dependency inventories. Source snapshots remain in the
image for reconstruction. For a release, retain the prepared context, resolved
base digest and final image digest as well. Uncommitted source identity comes
from the shipped file hashes, not merely the Git commit.

## Local Docker viewer

The host needs a compatible NVIDIA driver and Docker GPU integration. Start:

```bash
docker volume create warpsph-work
docker run --rm --init --gpus all --name warpsph \
  --hostname warpsph-local --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:8766:8767 \
  --mount source=warpsph-work,target=/work \
  warpsph:local companion
```

Open [warpSPHRemote](https://fluids.dev/warpSPHRemote/) in a WebGL-capable browser,
choose **Connect daemon → This device / SSH tunnel**, and enter
`http://127.0.0.1:8766` and the pairing token printed by the container. The static
viewer is hosted separately; its Python companion is bundled. Browser local-network
permission may be required. The address printed by the underlying daemon is its
private container endpoint; use the published host address above in the viewer.

The companion listens privately on `127.0.0.1:8766` **inside the container**.
nginx listens on container port 8767, allows only loopback Host names and forwards
`/v1/` viewer requests. Producer registration is not proxied. Pairing tokens and
Origin checks remain enforced; the upstream Host port is normalized to the
private daemon port. Publish to host `127.0.0.1`, as shown, to keep access local.
An older Docker release may not isolate localhost-published ports completely;
use a current engine (Docker documents the fix from 28.0.0).

To retrieve the token without restarting:

```bash
docker exec warpsph warpsph-companion --show-token
```

`/work` holds pairing state, run history/exports and architecture-specific caches.
Keep the same user and hostname to retain pairing across container recreation.
For bind mounts, create a directory writable by that user; existing private
state must retain its owning UID and private permissions. Mount input directories
read-only under `/inputs` as needed. For another simultaneous container, use a
different host port, hostname and work volume. A different published host port
is accepted; no daemon configuration change is needed.

`docker stop warpsph` stops the proxy and daemon, then terminates remaining
container processes. The image supervisor propagates termination to both service
groups and exits if either service dies. Use `--init` to reap orphaned workers.
This is container-local process management, not a Slurm submission service.

## Simulation commands and HPC scope

The entrypoint accepts:

- `run [case/options]`: the installed `warpsph-run` CLI.
- `distributed [torchrun options/script]`: Torch's distributed launcher; the
  shipped script is `/opt/warpsph/scripts/run_distributed.py`.
- `companion [--origin URL] [--max-snapshot-mib N]`: optional local viewer service.
- `manifest`: source/build provenance; `exec COMMAND [args]`: explicit commands.

For example, use `warpsph:local run --help` in place of `companion` to inspect
case options. Eager distributed execution continues to use the existing JSON
configuration, NCCL backend and one process per GPU. The companion's preset
launcher remains the ordinary single-process runner. **Distributed live viewing
and collective pause/resume/stop are not implemented by this packaging.**

Apptainer can consume the target-architecture OCI image, but an Isambard job must
still supply the site-specific runtime and launch profile. Do not use this local
Docker proxy as an already-qualified Isambard network solution. Keep producer
access private and host driver/network dependencies outside the generic image.

References: [Docker architectures](https://docs.docker.com/build/building/multi-platform/),
[Docker port publishing](https://docs.docker.com/engine/network/port-publishing/),
[Isambard NCCL containers](https://docs.isambard.ac.uk/user-documentation/guides/nccl/#build-nccl-in-a-container),
[companion connection contract](https://github.com/wi-re/warpSPHRemote).
