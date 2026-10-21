"""Two-gas Richtmyer–Meshkov example; see examples/configuration/richtmyerMeshkov.yaml."""
from warpSPHBootstrap import bootstrap
import sys

if __name__ == '__main__':
    bootstrap()
    from warpSPH.cases.compressible.richtmyerMeshkov import richtmyerMeshkovCase
    from warpSPH.runner import caseMain
    caseMain(richtmyerMeshkovCase, ['--plot', '--video', '--progress'] + sys.argv[1:])
