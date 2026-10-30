"""Geometry certificates for the static, frozen-cell reference solve.

These bounds certify a candidate list; they do not impose a physical timestep
or alter any SPH term. R is the uniform physical support radius. The native
surface cone only tests distances<R/4 in a one-cell stencil, and the selected
warpSPH bounce only tests distances<1.155dx against stationary wall rows.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReferenceMotionBudget:
    pair: float
    cone: float
    bounce: float

    @property
    def limit(self):
        return min(self.pair,self.cone,self.bounce)


def referenceMotionBudget(support,candidateCutoff,*,bounceCutoff=None,dx=None):
    """Sufficient bounds, applied to EVERY RK geometry and final iterate.

    If each physical row moves at most D, an interacting pair at distance<R
    started at distance<R+2D. Its old candidate radius must exceed that.
    A cone pair at distance<R/4 is guaranteed to occupy the native one-cell
    stencil if its reference row moved less than3R/4. For static bounce walls,
    an eventual 1.155dx contact started within D+1.155dx of the query.
    bounceCutoff must already exclude motion since the bounce list was built.
    """
    R=float(support);cut=float(candidateCutoff)
    if not math.isfinite(R) or R<=0 or not math.isfinite(cut) or cut<=R:
        raise ValueError('Reference candidates need a finite radius larger than physical support')
    wall=math.inf
    if bounceCutoff is not None:
        if dx is None or not math.isfinite(float(dx)) or float(dx)<=0:
            raise ValueError('Static bounce certificate requires positive finite dx')
        wall=float(bounceCutoff)-1.155*float(dx)
        if not math.isfinite(wall) or wall<=0:
            raise ValueError('Bounce candidates do not cover the contact radius')
    return ReferenceMotionBudget(pair=.5*(cut-R),cone=.75*R,bounce=wall)
