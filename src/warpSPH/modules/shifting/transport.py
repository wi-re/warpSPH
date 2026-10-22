"""Post-integration shifting transport, preserving its evaluation/application order."""
import torch
from warpSPHCore import OperationProperties, WarpOperation, SupportScheme, OperationDirection, GradientScheme, warpOperation
from warpSPHCore.profiling import record_function
from .wrapper import solveShifting
from ..boundaryConditions.pinned import pinnedKeepWeight


def shiftTransport(self, dt, config, schemeConfig):
    dx = drhodt_shift = dudt = duCross = None
    if schemeConfig.shiftProperties.active:
        with record_function("[warpSPH] - [deltaSPH] - solve shifting"):
            dx = solveShifting(
                systemState = self.state,
                config = config,
                schemeConfig = schemeConfig,
                adjacency = self.adjacency,
                dt = dt,
            )
            # print(f"Applied shifting update with max shift magnitude: {dx.norm(dim=1).max().item()}")
            keep = pinnedKeepWeight(self.state, config, schemeConfig)        # a pinned band is not shifted
            if keep is not None:
                dx = dx * keep

            du = dx / dt
            rho = self.state.densities
            u = self.state.velocities

            if schemeConfig.shiftProperties.correctdrhodt or schemeConfig.shiftProperties.correctdvdt:
                divDu = warpOperation(self.state, operationProperties=OperationProperties(
                    operation=WarpOperation.Divergence, kernel=config.kernel,
                    supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll,
                    gradientMode=GradientScheme.Difference), queryValues=du,
                    domain=config.domain, adjacency=self.adjacency)
            if schemeConfig.shiftProperties.correctdrhodt:
                with record_function("[warpSPH] - [deltaSPH] - compute drhodt_shift"):
                    drhodt_shift = warpOperation(
                        self.state,
                        operationProperties = OperationProperties(
                            operation=WarpOperation.Divergence,
                            kernel = config.kernel,
                            supportMode = SupportScheme.Gather,
                            operationMode = OperationDirection.AllToAll,
                            gradientMode = GradientScheme.Summation
                        ),
                        queryValues = rho.view(-1,1) * du,
                        domain = config.domain,
                        adjacency = self.adjacency
                    ) - rho * divDu
            if schemeConfig.shiftProperties.correctdvdt:
                with record_function("[warpSPH] - [deltaSPH] - compute dudt shift"):
                    dudt = -u * divDu.view(-1,1)

                    duCross = warpOperation(
                        self.state,
                        operationProperties = OperationProperties(
                            operation=WarpOperation.Divergence,
                            kernel = config.kernel,
                            supportMode = SupportScheme.Gather,
                            operationMode = OperationDirection.AllToAll,
                            gradientMode = GradientScheme.Summation
                        ),
                        queryValues =  torch.einsum('ij,ik->ijk', u, du),
                        domain = config.domain,
                        adjacency = self.adjacency
                    )




    return dx, drhodt_shift, dudt, duCross
