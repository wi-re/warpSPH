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
