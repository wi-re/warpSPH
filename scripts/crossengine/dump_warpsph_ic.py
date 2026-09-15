import sys, json
sys.path.insert(0, '/home/lu26029/dev/warpSPH/examples/weaklyCompressible')

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import caseMain

args = [
    '--nx', '64',
    '--L', '2.0',
    '--n_h', '4.0',
    '--kernel', 'Wendland4',
    '--integrationScheme', 'symplecticEuler',
    '--supportMode', 'SuperSymmetric',
    '--tLimit', '4.0',
    '--targetDt', '0.0005',
    '--band', '7',
    '--fillRatio', '0.5',
    '--fluidWidth', str(5.0/12.0),
    '--gravityMagnitude', '10.0',
    '--wallBC', 'constant',
    '--nSteps', '0',
    '--no-plot', '--no-store', '--no-video',
    '--quiet',
    '--precision', 'float32',
]

result = caseMain(dambreakCase, args)
ctx = result.ctx
state = result.state.state

import torch
kinds = state.kinds.detach().cpu()
pos = state.positions.detach().cpu()
dens = state.densities.detach().cpu()
mass = state.masses.detach().cpu()
supp = state.supports.detach().cpu() if getattr(state, 'supports', None) is not None else None

out = {}
out['nx'] = ctx.spec.nx
out['dx'] = ctx.config.dx
out['band'] = ctx.param('band')
out['targetNeighbors'] = float(ctx.config.targetNeighbors)
out['c_s'] = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
out['dt'] = float(ctx.config.dt)
dom = ctx.config.domain
out['domain_min'] = dom.min.detach().cpu().tolist()
out['domain_max'] = dom.max.detach().cpu().tolist()
interior = ctx.scratch['interiorDomain']
out['interior_min'] = interior.min.detach().cpu().tolist()
out['interior_max'] = interior.max.detach().cpu().tolist()
out['n_total'] = int(kinds.shape[0])
for k in sorted(set(kinds.tolist())):
    m = kinds == k
    out[f'n_kind{int(k)}'] = int(m.sum())
fluidMask = kinds == 0
out['fluid_pos_min'] = pos[fluidMask].min(dim=0).values.tolist()
out['fluid_pos_max'] = pos[fluidMask].max(dim=0).values.tolist()
out['fluid_density_min'] = float(dens[fluidMask].min())
out['fluid_density_mean'] = float(dens[fluidMask].mean())
out['fluid_density_max'] = float(dens[fluidMask].max())
out['fluid_mass_mean'] = float(mass[fluidMask].mean())
if supp is not None:
    out['fluid_support_mean'] = float(supp[fluidMask].mean())
out['pressure_term'] = ctx.schemeConfig.pressureForceTerm.name
out['shifting_active'] = ctx.schemeConfig.shiftProperties.active
out['gravity_magnitude'] = ctx.schemeConfig.gravityConfig.magnitude
out['kernel'] = ctx.config.kernel.name
out['integrationScheme'] = ctx.spec.integrationScheme
out['supportMode'] = ctx.spec.supportMode

print(json.dumps(out, indent=2))
with open('/tmp/claude-598314/-home-lu26029-dev-warpSPH/8a063ee7-6d7c-4204-b6a2-872345cf3993/scratchpad/warpsph_ic.json', 'w') as f:
    json.dump(out, f, indent=2)
