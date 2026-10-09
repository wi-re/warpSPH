# exec()'d by ceiling_forensics.py --patch / ceiling_run.py --patch: turn on
# schemeConfig.timeCentredContinuity (CEILING_STICKING_PLAN.md §3)
_prevCfgTCC = dambreakCase.configureScheme


def _cfgTCC(ctx, _p=_prevCfgTCC):
    _p(ctx)
    ctx.schemeConfig.timeCentredContinuity = True
    print('[patch] timeCentredContinuity = True', flush=True)


dambreakCase.configureScheme = _cfgTCC
