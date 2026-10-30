"""Native companion publication around an already-selected runner callable.

Do not call run_live here: that imports runner.run, which the fixture replaces.
Publication wraps completed-step hooks, without changing the numerical map.
"""
import json
from pathlib import Path
import threading


def daemon_enabled(steps, requested=None):
    return steps >= 100 if requested is None else requested


def run_published(run, case, *, name, out, publisher_factory=None, **kwargs):
    if publisher_factory is None:
        from warpsph_remote.publisher import SimulationPublisher
        publisher_factory=SimulationPublisher
    publisher=publisher_factory(name,measurements=False)
    out=Path(out)
    (out/'daemon-run.json').write_text(json.dumps(dict(id=publisher.id,name=name,
        endpoint=publisher.info['endpoint'],pid=__import__('os').getpid(),
        adapter='native completed-step publisher'),indent=2)+'\n')
    done=threading.Event()
    final={'status':None}
    def heartbeat():
        while not done.is_set():
            with publisher.lock:
                meta={k:publisher.meta.get(k) for k in ('status','step','time')}
            meta['id']=publisher.id
            path=out/'daemon-status.json';pending=path.with_suffix('.pending')
            pending.write_text(json.dumps(meta)+'\n');pending.replace(path)
            done.wait(.5)
    thread=threading.Thread(target=heartbeat,name='daemon-status-receipt',daemon=True);thread.start()
    try:
        result=run(publisher.bind_case(case),**kwargs)
        publisher.recording(result.ctx)
        publisher.finish(result.ctx,result.state,result.nSteps)
        publisher.update(step=result.nSteps,time=float(result.state.t),
            dt=float(result.ctx.config.dt),particles=int(result.state.state.positions.shape[0]))
        final['status']='failed' if result.diverged or result.stopReason else 'completed'
        publisher.close(final['status'],result.stopReason or '')
        return result
    except BaseException as error:
        # Keep the original exception/stop visible to the campaign supervisor.
        final['status']='stopped' if type(error).__name__=='SimulationStopped' else 'failed'
        publisher.close(final['status'],str(error));raise
    finally:
        done.set();thread.join(timeout=2)
        pending=out/'daemon-status.pending'
        pending.write_text(json.dumps(dict(id=publisher.id,status=final['status'],closed=True))+'\n')
        pending.replace(out/'daemon-status.json')
