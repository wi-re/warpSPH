# Configuring simulations

The version-1 configuration contract is shared by Python, YAML/JSON and CLI
validation. Parameter metadata supplies types, documentation, units, constraints,
groups and dependencies for future clients. It lives with the solver and cases;
the browser and warpSPHCases do not maintain another copy of solver defaults.

The strict contract covers all 16 cases in the compressible family, plus
`tgv-wc` and `randomFlow`. See the [case catalogue](configuration/README.md) for
case references, physical variants and supported solver combinations. TGV and
random flow include deltaSPH, sun2017DeltaSPH and artificialCompressible (ACSPH).
Other families retain their Python/flat-CaseSpec interfaces pending migration;
unsupported versioned case/solver combinations fail explicitly.

## Run, inspect and validate

From the repository root, in the normal warpSPH environment:

```bash
# Configure and validate: no particle sampling or timesteps.
python -m warpSPHRun --config examples/configuration/sod.yaml --checkConfig \
  --saveResolvedConfig /tmp/sod-setup.json

# CLI values override file values. This actually runs a short case.
python -m warpSPHRun --config examples/configuration/sod.yaml --nx 80 --nSteps 2

# The analogous weakly compressible configuration.
python -m warpSPHRun --config examples/configuration/tgv-wc.yaml --checkConfig

# Discover parameters for the chosen solver, and generate reference Markdown.
python -m warpSPHRun sod --scheme CRKSPH --describeConfig /tmp/sod-schema.json
python -m warpSPHRun tgv-wc --parameterReference /tmp/tgv-parameters.md

# Discover ready and unmigrated cases without sampling particles.
python -m warpSPHRun --configurationCatalogue /tmp/case-catalogue.json
```

Checking imports the runtime and configures the domain and solver. It is not a
pure file lint: it may initialize Warp/CUDA, but does not build a particle system,
compile a timestep, launch a simulation or write a trajectory. Some settings,
including the final timestep and sound speed, are derived during particle setup;
the check output identifies its stage so those preliminary values are not
mistaken for the final initialized configuration.

The CLI reads precision from the configuration before importing warpSPHCore.
An explicit `--precision` overrides it. JSON and YAML use the same strict types:
booleans are not integers, strings are not automatically converted to numbers,
non-finite numbers and unknown fields are rejected. With PyYAML, write scientific
notation as `1.0e-3`, not `1e-3` (which YAML 1.1 reads as a string).

## Document structure

```yaml
schemaVersion: 1
case: sod
variant: classic
simulation:
  nx: 400
  precision: float32
  tLimit: 0.15
params:
  left_rho: 1.0
  right_pressure: 0.1
solver:
  scheme: CompSPH
  diffusionParams:
    K: 0.5
```

- `simulation`: the common CaseSpec controls: sampling/discretization, integration,
  timestep, runtime and output. Existing field spellings are retained. Parameters
  with different semantics are not collapsed into a universal `alpha`.
- `params`: physical, geometry and initialization parameters declared by the case,
  plus its existing convenience controls. Descriptions and groups make their
  role explicit. Nullable values mean inheritance only where the descriptor says
  so. For WCSPH, `inviscid` chooses artificial (`alpha`) versus physical (`nu`)
  viscosity; non-default inactive controls in legacy files are diagnosed by
  `--checkConfig`, and explicitly inactive controls in versioned documents fail.
- `solver`: a scheme name and a curated hierarchy of numerical overrides using
  actual solver paths. CompSPH/CRKSPH/Monaghan expose artificial-viscosity and
  conductivity settings; WCSPH exposes its density-diffusion controls; Godunov
  and meshless solvers expose their own reconstruction/Riemann/time controls.
  ACSPH exposes a separate `acParams` block for pseudo-time stepping and
  convergence; it does not inherit WCSPH's density-diffusion block.
- `variant`: a named physical or initialization recipe declared by the case.
  Sod provides `default`, `classic` and `smoothed`; TGV provides `default`,
  `analytic-pressure` and `no-shifting`. These do not reinterpret the existing
  warpSPHCases run labels such as `demo` or `reference`.

Defaults are sourced from CaseSpec, the case's existing defaults/params and the
solver config builders. Descriptors contain no independent copies of those
values. Precedence is common defaults → case defaults → named variant → document
values → CLI flags. Numerical overrides are applied **after** `configureScheme`
and before solver validation and particle construction. A case hook cannot
silently replace an explicit numerical override.

The initial Sod layout is still the original mirrored periodic box. Its `L=2`,
`dim=1` and periodicity are constrained accordingly; arbitrary diaphragm positions
are not advertised as editable until sampling and reference evaluation support
them. The `sod2d` and `sod3d` contracts describe the existing fixed-length
periodic slabs, including transverse sampling and equal-mass controls.

## Structured and derived case parameters

Descriptors support typed vectors and arrays of objects. For example,
[Woodward–Colella](../examples/configuration/woodward.yaml) declares contiguous
`regions` with `begin`, `end`, `pressure` and `density`. Nested validation reports
the failing path, such as `params.regions[0].pressure`. Random-flow obstacle
placement uses a two-component `obstacleOffset` vector, with shape names sourced
from the existing geometry presets.

Some existing parameters are derived or legacy placeholders rather than useful
editable controls. Kidder derives its geometry index `nu=dim` and
`gamma=1+2/dim`; conflicting explicit values fail. Its integration duration and
Sedov's duration are derived during initialization. Unused compatibility fields
are documented as read-only/deprecated. Defaults keep their original behavior;
the contract does not silently offer a slider for an ignored quantity.

## Python

```python
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.runner import RunConfiguration, resolvedConfiguration

config = RunConfiguration.load('examples/configuration/sod.yaml')
ctx = config.prepare()                 # configuration only, no particle sampling
setup = resolvedConfiguration(ctx)
result = config.run()                  # normal runner, no visualization dependency
actual = resolvedConfiguration(result.ctx)
```

Python construction has the same checks as file loading:

```python
config = RunConfiguration('sod', variant='classic',
    simulation={'nx': 80, 'nSteps': 2},
    solver={'scheme': 'CRKSPH', 'diffusionParams': {'K': 0.5}})
config.save('/tmp/my-sod.yaml')
```

Bootstrap remains explicit in Python because scalar precision is locked at the
first heavy import. The existing `run(case, ...)`, CaseSpec, case hooks and flat
configuration files remain usable. Existing arbitrary case-specific experiments
are not silently subjected to a partly migrated schema; opt into the strict
contract through RunConfiguration or a versioned file.

## Reproducibility and discovery

`--saveResolvedConfig` writes the requested versioned document, effective
CaseSpec, actual simulation/solver configurations, package versions, source Git
revisions and dirty flags when available. The manifest states whether it came
from the check stage or a fully initialized run. Recorded/plot-producing runs
using the new contract also write `resolvedConfiguration.json` alongside the
existing `caseSpec.json` and `config.json`. A manifest is evidence, not an input
document: reload its `requested` object, or use the saved CaseSpec for legacy
replay. Uncommitted source changes require their own saved patch; a dirty flag
alone does not preserve those changes.

`configurationSchema(case, scheme)` returns JSON Schema with `x-group`, `x-units`,
`x-applies-when`, `x-applies-when-any`, `x-derived-from` and
`x-supported-schemes` annotations, plus standard `readOnly`/`deprecated` flags. Structural/type/bounds
checks are represented in the schema. Cross-parameter and actual solver checks
are authoritative in RunConfiguration.resolve/prepare; JSON Schema alone is
not a guarantee that a numerical configuration is suitable or stable.

The [generated case references](configuration/README.md) cover each migrated
case's default solver, with a separate ACSPH TGV reference. Regenerate a reference
using `--parameterReference` after editing descriptors.

To migrate another case, add complete `parameterDefinitions`, constrain any
fixed `simulationParameters`, declare `supportedSchemes` and optional `variants`
and `validateConfiguration`. Reuse descriptors from `cases/common/parameters.py`
when meaning and units agree. Add formulation-specific controls to
`solverParameters` only when that solver consumes them. Compare defaults and a
short configured run against the existing Python path before enabling the case.

## Using existing warpSPHCases recipes

`RunConfiguration.fromOverrides(case, overrides)` converts sparse runner keyword
arguments through the same strict validation. It accepts the existing
`RunEntry.caseSpecOverrides()` representation. Recipe labels such as `reference`
are kept separate from the core case's physical `variant`; no recipe catalogue
or numerical default is copied into the solver.

The optional bridge example requires an installed `warpSPHCases` package:

```bash
python examples/configuration/case_recipes.py --root ../warpSPHCases \
  --page sod --run CompSPH__default__nx400 --output /tmp/sod-recipe.yaml
python -m warpSPHRun --config /tmp/sod-recipe.yaml --checkConfig

# Audit every public recipe; unmigrated cases are reported as skipped.
python examples/configuration/case_recipes.py --root ../warpSPHCases \
  --audit --output /tmp/recipe-audit.json
```

These commands export/validate recipes without changing warpSPHCases. Its regular
runner has not been switched over, and the daemon/browser do not yet consume the
schema. This contract is the prerequisite for those integrations.

To exercise all ready case defaults through versioned files in separate processes:

```bash
python examples/configuration/check_catalogue.py --steps 5 --nx 64 \
  --outputRoot ../warpSPHDevelopment/core/my-new-config-smoke
```

Use a fresh output directory. Omitting `--steps` checks setup without constructing
particles; positive values run the normal solver. These are smoke checks, not
resolution/convergence or physical benchmark validation.
