"""Progress watchdog for an already-owned native blob process; no scheduler."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time


def watch(pid,log,output):
    identity=Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'run_spheralBlobExample.py' in identity and b'run\x00' in identity
    begin=time.monotonic();anchor=begin;startTime=None;latest=None
    def save(status,**kw):
        output.write_text(json.dumps(dict(status=status,pid=pid,log=str(log),lastSimulationTime=latest,
            wallSeconds=time.monotonic()-begin,**kw),indent=2)+'\n')
    while True:
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or proc.read_bytes()!=identity:
            save('processExited');return
        now=time.monotonic()
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-32768));lines=stream.read().decode(errors='replace')
        samples=re.findall(r'BLOB cycle=\d+ t=([\deE.+-]+)',lines)
        if samples:latest=float(samples[-1])
        if startTime is None and latest is not None:startTime=latest;anchor=now
        reason=None
        if now-begin>120 and time.time()-log.stat().st_mtime>120:reason='No log output for 120 seconds'
        if latest is not None and now-anchor>=60:
            if latest-startTime<1e-3:reason='Simulated time advanced <1e-3 in 60 wall seconds'
            startTime=latest;anchor=now
        if reason:
            save('interrupting',reason=reason)
            print('BLOB WATCHDOG:',reason,'sending SIGINT to owned process',pid,flush=True)
            os.kill(pid,signal.SIGINT);return
        save('watching');time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();watch(a.pid,a.log,a.output)
