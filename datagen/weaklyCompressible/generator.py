#!/usr/bin/env python
"""Dam-break data generator -- the runner, plus this directory's archival step.

The simulation itself is `warpSPH.cases.freeSurface.dambreak`; what stays here is the bit
that is specific to generating a dataset: stamping the run directory with a
timestamp and the obstacle description, and collecting the trajectory, video and
end frames into `compressed/`.

    python datagen/weaklyCompressible/generator.py --nx 128 --plot --store
    python datagen/weaklyCompressible/generator.py --config sweeps/dambreak/some_config.json

This only runs `dambreakCase`. For a config from a different sweep family
(e.g. `sweeps/impact/*.json`, written by `obstacle_init.ipynb`), use
`run_sweep.py` instead -- it dispatches on the family and also archives.
"""

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from utils import archive                                   # noqa: E402

from warpSPH.cases.freeSurface.dambreak import dambreakCase             # noqa: E402
from warpSPH.runner import buildArgumentParser, run, specFromArgs  # noqa: E402
from warpSPH.utils import getCurrentTimestamp               # noqa: E402


def main():
    parser = buildArgumentParser(description=dambreakCase.description,
                                 caseParams=dambreakCase.params)
    args = parser.parse_args()

    defaults = dict(dambreakCase.defaults)
    defaults.update(caseName='3-dambreak', store=True, video=True)
    spec = specFromArgs(args, caseParams=dambreakCase.params, defaults=defaults)

    # The dataset layout wants one directory per run, self-describing.
    obstacleText = (f"obstacle_{spec.param('maxExtent'):.4g}_{spec.param('aoa'):.4g}"
                    f"_{spec.param('offsetX'):.4g}" if spec.param('obstacleActive')
                    else 'no_obstacle')
    baseName = spec.caseName
    spec = spec.merged(caseName=(
        f'{baseName}/{getCurrentTimestamp()}_{spec.nx}_{spec.n_h}_'
        f"{spec.L}_{spec.param('W')}_{obstacleText}"))

    result = run(dambreakCase, spec)
    archive(result, baseName)
    return result


if __name__ == '__main__':
    main()
