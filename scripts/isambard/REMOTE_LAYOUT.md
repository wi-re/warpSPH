# Remote Isambard layout

The idle Isambard account was reorganized on 2026-11-03 under
`/home/b6ez/wire.b6ez/warpsph`. This is a placement change for the previously
qualified deployment, not a new numerical or fabric qualification. No jobs were
submitted. The smaller 0.7.1 runtime remains local and pending site qualification.

```text
~/warpsph/
├── releases/accepted-2ecb382/
│   ├── images/warpsph-arm64-io-repair-2ecb382.sif
│   ├── communication/ofi-7999853/
│   ├── harness/                 # Original profile, helpers and provenance
│   ├── warpsph-isambard/        # Original source bundle
│   └── environment.sh          # Explicit physical deployment paths
├── current -> releases/accepted-2ecb382
├── environments/{current,accepted}.sh
├── tools/
│   ├── miniforge3/              # Preserved installation and warp_env
│   └── host-python/             # Fresh Python 3.11 maintenance venv
├── inputs/
├── qualification/accepted-2ecb382/
├── runs/
├── staging/
├── cache/apptainer-legacy/
└── archive/
    ├── images/
    ├── builds/
    ├── communication-sources/
    ├── installers/
    ├── submission-logs/
    └── migration-20261103/      # Move map and before/after verification
```

## Select a deployment

Interactive login shells automatically source `environments/current.sh` from
`~/.bashrc`. The block is guarded by the interactive shell flag, so noninteractive
shells retain exported deployment paths. A backup and startup-check receipt are
in `archive/migration-20261103/`. To refresh an existing shell or select explicitly:

```bash
source "$HOME/warpsph/environments/current.sh"
# Or select the qualified release explicitly:
source "$HOME/warpsph/environments/accepted.sh"
```

The selector resolves the release physically and explicitly sets `QA_ROOT`,
`QA_BUNDLE`, `QA_IMAGE`, `QA_COMM_ROOT`, `QA_NETWORK_SETUP`, `QA_LIBFABRIC_HOST`,
`QA_RESULTS_ROOT`, `QA_PYTHON`, `QA_HARNESS_ROOT`, `QA_DEPLOYMENT_PROFILE` and the
accepted account before sourcing the original profile. Exported job paths
therefore stay pinned if `current` changes later. It clears `QA_PREVIOUS`; set
that and run-specific resource overrides **after** selecting the environment.

Batch orchestration retains the original Conda Python interpreter through
`tools/miniforge3/bin/python`; the fresh Python3.11 venv is for maintenance.
Changing host interpreters on compute nodes is not part of this migration.
The frozen harness/profile, SIF, communication payload and historical receipts
remain unchanged. Do not source the original harness profile alone in a fresh
shell: it retains its historical `$HOME` path defaults.

## Data and historical paths

Use `runs/<case>/<timestamp>-<jobid>/` for fresh production output, including a
copied configuration, deployment receipt, logs and native checkpoints. Inputs,
results and checkpoints remain on the existing shared home filesystem; no new
scratch filesystem or retention policy is assumed. Historical absolute paths in
sealed reports are provenance and must not be rewritten. The migration's
`moves.json` maps their old roots to the new locations. The four accepted stages
remain together under `qualification/accepted-2ecb382`, alongside older failed
and passing stages and their logs.

`~/dev` remains unchanged to preserve editable installs; the existing Conda/mamba
initialization in `.bashrc` is retained before the deployment selector. The only
home-level compatibility link is `~/miniforge3 -> warpsph/tools/miniforge3`.
It preserves embedded Conda prefixes and shell activation; do not remove it
without recreating the environments. The installation was moved on the same
filesystem and was not repackaged. Explicit base and warp_env Conda package
specifications are retained with the migration records. The old source bundle's
Apptainer cache links to `cache/apptainer-legacy` and is not an application runtime
dependency.

## Verification records

`archive/migration-20261103/` contains before/after deployment receipts, the
inode-preserving move map, environment checks and a submission dry-run record.
The dry run replaces `sbatch` with a stub and checks physical image/harness paths
and the preceding restart report without submitting a job. The accepted image,
profile and communication library hashes must match their original pins; all
113 sealed accepted artifacts/checkpoints are verified from their new locations.
Original images, installers, wrappers, failed runs and submission logs are
retained; no historical simulation data is deleted.

## Shared scratch for larger tests

The 2026-11-03 128³ turbulence test uses personal Lustre scratch at
`/scratch/b6ez/wire.b6ez/warpsph/`, with `releases/0.7.1-runtime`,
`qualification/0.7.1-runtime`, `inputs` and `runs` kept separate. Its explicit
selector is `~/warpsph/environments/0.7.1-runtime.sh`; interactive `current`
continues to select the earlier qualified home deployment.

The published Isambard-AI Phase2 limits are home100GiB, personal scratch5TiB,
project200TiB and compute-local48GiB. Actual personal scratch quota was checked:
5TiB and1,024,000 inodes. Home is intended for small configuration/software;
use shared scratch for active simulation output/checkpoints and transfer results
promptly. No site storage is backed up. Compute-local storage is wiped when a
job ends. The AI retention table says scratch lasts to project expiry, while the
page's generic scratch text also mentions a60-day cleanup; do not depend on
scratch for archival. Export important results before project expiry.

Sources: [storage policy](https://docs.isambard.ac.uk/user-documentation/information/system-storage/),
[file transfer](https://docs.isambard.ac.uk/user-documentation/guides/file_transfer/local/).
