# Case configuration catalogue

Generated case references use the default solver unless the link names a solver.
The runtime catalogue also reports unmigrated cases explicitly.

| Case | Supported solvers | Physical / initialization variants |
|---|---|---|
| [gresho](gresho.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `smoothed` |
| [hydrostatic](hydrostatic.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH` | `default` |
| [kelvinHelmholtz](kelvinHelmholtz.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `smooth-density` |
| [kidder](kidder.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH` | `default` |
| [linearWave](linearWave.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |
| [noh](noh.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |
| [randomFlow](randomFlow.md) | `deltaSPH`, `sun2017DeltaSPH`, `artificialCompressible` | `default`, `bounded`, `obstacle` |
| [rayleighTaylor](rayleighTaylor.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH` | `default` |
| [sedov](sedov.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `singular`, `quadrant` |
| [shearBox](shearBox.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |
| [shearingNoh](shearingNoh.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |
| [sod](sod.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `classic`, `smoothed` |
| [sod2d](sod2d.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `classic` |
| [sod3d](sod3d.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default`, `classic` |
| [tgv-wc](tgv-wc.md) | `deltaSPH`, `sun2017DeltaSPH`, `artificialCompressible` | `default`, `analytic-pressure`, `no-shifting` |
| [triplePoint](triplePoint.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |
| [woodwardColella](woodwardColella.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH` | `default` |
| [yee](yee.md) | `CompSPH`, `Monaghan`, `CRKSPH`, `GSPH`, `InutsukaGSPH`, `MFM`, `MFV` | `default` |

See the [ACSPH Taylor–Green controls](tgv-wc-artificialCompressible.md) for its separate pseudo-time and convergence block.

Generate current coverage with `python -m warpSPHRun --configurationCatalogue /tmp/cases.json`.
Regenerate a case reference with `python -m warpSPHRun CASE --parameterReference PATH`.
Add `--scheme artificialCompressible` to regenerate the ACSPH reference.

See [Configuring simulations](../CONFIGURATION.md) for precedence, validation and recipe exports.
