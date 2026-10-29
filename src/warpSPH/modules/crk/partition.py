"""Native CRK entropy partition (Frontiere2017 Eq63), shared by storage paths."""
import warp as wp
from warpSPHCore import scalar_t


@wp.func
def crkEntropyFraction(Pi:scalar_t,rhoi:scalar_t,gi:scalar_t,
                       Pj:scalar_t,rhoj:scalar_t,gj:scalar_t,work:scalar_t):
    if gi != gj:
        return scalar_t(0.5)
    si=Pi/wp.pow(rhoi,gi)
    sj=Pj/wp.pow(rhoj,gj)
    smin=wp.min(wp.abs(si),wp.abs(sj))
    smax=wp.max(wp.abs(si),wp.abs(sj))
    same=wp.abs(si-sj)==scalar_t(0.0)
    low=(work>=scalar_t(0.0) and si>=sj) or (work<scalar_t(0.0) and si<sj)
    high=(work>=scalar_t(0.0) and si<sj) or (work<scalar_t(0.0) and si>=sj)
    f=scalar_t(0.5)
    if not same and low:f=smin/(smin+smax+scalar_t(1.0e-14))
    if not same and not low and high:f=smax/(smin+smax+scalar_t(1.0e-14))
    return wp.max(scalar_t(0.0),wp.min(scalar_t(1.0),f))
