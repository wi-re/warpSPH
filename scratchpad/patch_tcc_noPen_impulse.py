# exec()'d by ceiling_forensics.py --patch: timeCentredContinuity on + mdbcNoPenShiftMode='impulse'
# (CEILING_STICKING_PLAN.md §7 control: is the noPen velocity replacement the pump?)
exec(open('scratchpad/patch_timeCentredContinuity.py').read())
_prevCfgNP = dambreakCase.configureScheme


def _cfgNP(ctx, _p=_prevCfgNP):
    _p(ctx)
    ctx.schemeConfig.mdbcNoPenShiftMode = 'impulse'
    print('[patch] mdbcNoPenShiftMode = impulse', flush=True)


dambreakCase.configureScheme = _cfgNP
