"""Michel A/B controls; no dynGPUSPH shift/remap/operator/stopping switches."""
import importlib


def rows():
    result=[]
    for index,(name,remap,bdf) in enumerate((('baseline',False,False),('transport',True,False),('bdf',True,True))):
        result.append(dict(id=f'{index:02d}-michel-{name}',index=index,
            change=('full-field-remap' if index==1 else 'bdf-shift' if index==2 else None),
            description=('Corrected static/direct/warmup Michel baseline' if index==0 else
                'Full Taylor velocity and pressure transport' if index==1 else
                'External-shift BDF history correction'),
            options=dict(static_geometry=True,accepted_fields=True,warmup=True,
                source_shift=False,source_operators=False,full_remap=remap,bdf_shift=bdf),
            reuse=False,coupledGroup=index==1))
    return result


def select(name):
    if name=='oracle':
        row=dict(rows()[0]);row['id']='oracle-accepted-fields';row['options']=dict(row['options'],warmup=False,bdf_shift=True)
        return row
    for row in rows():
        if name in (row['id'],row['id'].split('-')[-1]):return row
    raise ValueError(name)


def trial_parameters(scheme):
    """Use the configuration's supported JSON schema, including enum names."""
    from warpSPH.configurations.artificialCompressible import _acParamsToDict
    return _acParamsToDict(scheme.acParams)


def install(row):
    from ablation_controls import install as install_base,legacy_warm_solve
    from ablation_matrix import Options
    from warpSPH.modules.artificialCompressible import dualTime as dual
    from warpSPH.systems import artificialCompressible as system
    from warpSPH.modules.shifting.transport import taylorShiftFields
    from warpSPH.configurations.compute import neighborSearchOptions
    from warpSPHCore import buildVerletList,SupportScheme
    options=row['options'];receipt=install_base(Options(static_geometry=True,accepted_fields=True))
    step=importlib.import_module('warpSPH.schemes.incompressible.artificialCompressible')
    if options['warmup']:
        dual.solveDualTime=legacy_warm_solve;step.solveDualTime=legacy_warm_solve
    if options['full_remap']:
        original=system.solveShifting
        def shifted(*args,**kwargs):
            state=kwargs['systemState'];config=kwargs['config']
            dx=original(*args,**kwargs)
            # The displacement solver returns restored accepted positions.
            # Rebuild/validate the list here rather than reuse a solve-entry
            # neighborhood after physical motion. Same physical density weights.
            adjacency=buildVerletList(state,config.domain,verletScale=config.verletScale,
                supportMode=SupportScheme.SuperSymmetric,priorNeighborhood=kwargs['adjacency'],
                verbose=False,**neighborSearchOptions(config))
            state.pressures,state.velocities=taylorShiftFields(state,dx,config,adjacency)
            return dx
        system.solveShifting=shifted
    receipt['michelTrial']=row
    receipt['transport']='shared difference-gradient full Taylor; physical density volume weights; fluid outputs only'
    return receipt
