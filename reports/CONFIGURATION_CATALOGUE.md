# Configuration catalogue and recipe bridge

Implemented 2026-10-18, following the [first contract migration](CONFIGURATION_CONTRACT.md).

## Delivered

The version-1 YAML/JSON/Python contract now covers all 16 compressible-family
cases and the periodic `tgv-wc` and `randomFlow` cases. The runtime catalogue
reports complete, invalid and unmigrated contracts explicitly. Case-specific
defaults still come from the existing case definitions; shared descriptors
supply types, meaning, units, bounds and applicability without copying defaults.

Structured controls include Woodward–Colella region objects and random-flow
obstacle vectors. Derived controls include Kidder's geometry index/gamma and
Noh's reference shock speed. Legacy unused controls are read-only/deprecated;
meshless restrictions and fixed geometry are checked before particle setup.
ACSPH exposes its own pseudo-time and convergence controls for TGV/random flow.
No numerical kernels or initial-condition algorithms were changed.

`RunConfiguration.fromOverrides` accepts sparse runner arguments. The optional
`examples/configuration/case_recipes.py` bridge exports existing warpSPHCases
recipes through this contract and audits public recipes without editing that
checkout. Recipe labels such as `demo` or `reference` are not reinterpreted as
physical variants. The regular warpSPHCases runtime and remote GUI are not yet
switched to the new contract.

See the [configuration guide](../docs/CONFIGURATION.md) and
[case reference index](../docs/configuration/README.md).

## Validation

151 tests passed across configuration, catalogue, CaseSpec, runner and compute
configuration tests. They cover structured validation, derived values,
applicability, metadata completeness, CLI discovery/checks, schema validity,
recipe conversion and two-step CUDA parity with legacy Python runs. Numerical
comparisons include Sod, TGV, Gresho, KHH, RT, Woodward–Colella, Kidder, random
flow and ACSPH TGV.

All 18 ready cases also completed five steps at nx=64 through generated
versioned YAML inputs, in separate sequential processes. This exercises default
case/solver combinations; it is not an exhaustive run of every solver/variant
combination or a physical convergence study.

The local warpSPHCases audit resolved 48 public recipes, skipped 26 whose case
contracts are not migrated, and reported no failures. One existing Sod recipe
was exported and checked through the normal CLI. These are configuration
resolution checks, not 48 benchmark simulations.

The source-only import check imported all 457 package modules, scanned 981 files
and checked 4,805 first-party imports successfully. The user's wider checkout
contains three pre-existing stale imports in ignored `.tmp` scripts, documented
in the first milestone report; those files are outside this change.

## Evidence and regeneration

Generated evidence is stored outside source under
`../warpSPHDevelopment/core/configuration-catalogue-20261018/`: test/import logs,
recipe audit/export, generated JSON schemas and catalogue, the 18-case smoke
inputs/logs/resolved manifests, and an incremental source patch. The preceding
`configuration-contract-20261018` evidence is preserved separately.

In the normal Warp environment, from the warpSPH checkout:

```bash
WARPSPH_COMPILE_WALLS=1 python -m pytest -o addopts= \
  tests/test_configuration.py tests/test_configuration_catalogue.py \
  tests/test_caseSpec.py tests/test_runner.py tests/test_computeConfig.py -q
python scripts/check_imports.py
python -m warpSPHRun --configurationCatalogue /tmp/warpsph-catalogue.json
python examples/configuration/check_catalogue.py --steps 5 --nx 64 \
  --outputRoot ../warpSPHDevelopment/core/a-new-catalogue-smoke
python examples/configuration/case_recipes.py --root ../warpSPHCases \
  --audit --output /tmp/warpsph-recipe-audit.json
```

The recipe example requires the optional warpSPHCases package installed in this
environment. Choose a fresh evidence output directory. See the reference index
for Markdown regeneration commands.

## Remaining scope

Free-surface, wall/boundary, incompressible and other non-migrated families still
use their existing interfaces. Catalogue status makes this visible. Additional
solver controls need individual consumption/applicability audits before being
advertised. Arbitrary Sod diaphragm placement still requires sampler/reference
changes. Neither the daemon schema endpoint nor the hierarchical GUI editor is
implemented in this milestone; they should consume this runtime contract once
the desired case families are migrated.
