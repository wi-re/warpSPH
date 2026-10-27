"""Build an isolated opt-in 2D paper-policy DSO against the pinned Spheral ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/materialEOS/spheral-paper-policies'))
    p.add_argument('--source',type=Path,default=Path('benchmarks/eos/paperSpheral/paperPolicies.cc'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    base=Path('results/materialEOS/spheral-build').resolve()
    flags=(base/'compiled/build/src/CRKSPH/CMakeFiles/Spheral_CRKSPH.dir/flags.make').read_text().splitlines()
    options={line.split(' = ',1)[0]:shlex.split(line.split(' = ',1)[1]) for line in flags if ' = ' in line}
    source=a.source.resolve()
    sourceHash=hashlib.sha256(source.read_bytes()).hexdigest()
    library=(a.output/('libSpheralPaperPolicies_'+sourceHash[:12]+'.so')).resolve()
    command=['/usr/bin/g++',*options['CXX_DEFINES'],*options['CXX_INCLUDES'],*options['CXX_FLAGS'],'-shared',str(source),
        '-L'+str(base/'install/lib'),'-lSpheral_CXX','-Wl,-rpath,'+str(base/'install/lib'),'-o',str(library)]
    record=dict(source=str(source),sourceSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),command=command,
        baseLibrary=str(base/'install/lib/libSpheral_CXX.so'),baseSHA256=hashlib.sha256((base/'install/lib/libSpheral_CXX.so').read_bytes()).hexdigest(),
        status='building',scope='Opt-in LD_PRELOAD overlay; original source, objects and installed libraries untouched.')
    immutable=a.output/('build_'+sourceHash[:12]+'.json')
    if library.exists():
        previous=json.loads(immutable.read_text())
        assert all(previous[k]==record[k] for k in ('sourceSHA256','baseSHA256','command')), 'Existing immutable build has different inputs; select a fresh --output directory'
        assert previous['librarySHA256']==hashlib.sha256(library.read_bytes()).hexdigest(), 'Existing immutable library hash mismatch'
        record=previous
        print('Reusing immutable library',library,flush=True)
    manifest=a.output/'build.json';manifest.write_text(json.dumps(record,indent=2)+'\n')
    code=0 if record.get('status')=='completed' else subprocess.run(command).returncode
    record['status']='completed' if code==0 else 'failed'
    if code==0:record['librarySHA256']=hashlib.sha256(library.read_bytes()).hexdigest()
    if code==0:
        alias=a.output/'libSpheralPaperPolicies.so'
        if alias.exists() and not alias.is_symlink():alias.rename(a.output/'libSpheralPaperPolicies_initial.so')
        if alias.is_symlink():alias.unlink()
        alias.symlink_to(library.name)
        record['libraryPath']=str(library)
        (a.output/('build_'+sourceHash[:12]+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    manifest.write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(code)
