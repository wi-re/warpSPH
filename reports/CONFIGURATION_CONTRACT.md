# Configuration contract: first migration

Follow-up: [configuration catalogue and recipe bridge](CONFIGURATION_CATALOGUE.md).

Implemented 2026-10-18 in warpSPH. The canonical runtime contract lives beside the
runner and cases; warpSPHRemote and warpSPHCases can consume it without copying
solver defaults or deriving numerical semantics from field names.

## Delivered

- Versioned YAML/JSON run documents and the equivalent RunConfiguration Python API.
- Shared common simulation descriptors derived from CaseSpec types, its help and
  enum registries. Shared compressible/WCSPH case parameter descriptions.
- Complete parameter contracts for 1D Sod and 2D WCSPH Taylor–Green. Named physical
  or initialization variants and curated solver-specific numerical overrides.
- Strict document keys, scalar types, finite values, enums, bounds and relevant
  parameter dependencies. Duplicate JSON/YAML keys fail rather than last-wins.
- A single setup sequence: build context, configure case, apply numerical
  overrides, validate solver, then construct particles. No numerical kernel edits.
- Schema and Markdown generation from the same descriptors and configured defaults.
- A check-only CLI path, pre-import file precision selection and nullable CLI
  parameter typing supplied by the migrated case descriptors.
- Requested/resolved configuration manifests with actual simulation/solver values,
  initialization stage, installed package versions and available source revisions
  and dirty flags. Manifests are automatically added to new-format recorded runs.

## Validation

98 tests passed across `test_configuration.py`, `test_caseSpec.py`,
`test_runner.py` and `test_computeConfig.py`. Coverage includes YAML/JSON/Python
resolution parity, schema validity and actual solver defaults, all seven Sod
solver setups, WCSPH's separate diffusion block, explicit override precedence,
invalid/inactive settings, duplicate keys, the float64 CLI bootstrap, no-sampling
checks, initialized recording manifests and two-step CUDA numerical comparisons
against the legacy Python calls for Sod and TGV.

All 457 package modules imported successfully. The final source-only static
check scanned 978 source files and checked 4,762 first-party imports, with 11
optional-package imports skipped. The wider static scan of the user's checkout
found three pre-existing stale imports in Git-ignored `.tmp/disk_ab.py`,
`.tmp/ic_audit.py` and `.tmp/probe_acsphWallDrift.py`. Those private scratch files
were left untouched; the full local scan therefore remains nonzero.
The existing CLI smoke sweep passed Sod and tgv-wc, five steps each at nx=64.
Both shipped YAML examples also completed check-only setup, and their parameter
references and JSON schemas were regenerated from runtime descriptors.

Evidence (logs, schemas, setup snapshots and smoke outputs) is stored outside
Git at `../warpSPHDevelopment/core/configuration-contract-20261018/`.

## Reproduction

Use the normal Warp/CUDA environment with the editable sibling packages. From
warpSPH's checkout:

```bash
python -m pytest -o addopts= tests/test_configuration.py tests/test_caseSpec.py \
  tests/test_runner.py tests/test_computeConfig.py -q
python scripts/check_imports.py
python scripts/run_sweep.py --cases sod tgv-wc --nSteps 5 \
  --outRoot /tmp/new-configuration-smoke -- --nx 64 --no-plot --no-store
python -m warpSPHRun --config examples/configuration/sod.yaml --checkConfig \
  --saveResolvedConfig /tmp/sod-setup.json
python -m warpSPHRun --config examples/configuration/tgv-wc.yaml --checkConfig
python -m warpSPHRun sod --parameterReference docs/configuration/sod.md
python -m warpSPHRun tgv-wc --parameterReference docs/configuration/tgv-wc.md
```

## Remaining migration

- Complete parameter contracts for the remaining cases and solver families.
  Legacy Python calls and flat CaseSpec files remain available in the meantime.
- Extend the curated numerical catalogue as each formulation's consumed options
  and dependencies are verified. Experimental settings are not all exposed.
- Generalize the Sod sampler and its reference evaluation before offering arbitrary
  initial discontinuity placement or domain layout. The existing fixed periodic
  layout is explicitly constrained in the new contract.
- Bridge warpSPHCases' physical recipes/run matrices to versioned documents. Its
  existing RunEntry.spec can already pass `schemeOverrides` through CaseSpec;
  its `demo`/`reference` labels are not reinterpreted as physical case variants.
- Expose discovery/validation through the companion and build browser controls.
  This pass provides the underlying contract and does not change the web viewer.

Schema structural validation cannot certify numerical stability. Cross-field and
solver validation in resolve/prepare remain authoritative. Check-only manifests
precede particle sampling; final derived timestep/sound-speed values appear in
initialized manifests. A manifest records a dirty source tree but does not save
its uncommitted changes; preserve the patch separately for exact reproduction.
