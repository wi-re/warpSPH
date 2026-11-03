#!/usr/bin/env python3
"""Verify a pinned Isambard deployment and save its exact setup for later runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def capture(profile_path, *, image=None, image_sha256=None, environment=None):
    env = os.environ if environment is None else environment
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    if profile.get('formatVersion') != 1:
        raise ValueError('Unsupported deployment profile')
    root = Path(env.get('QA_ROOT', str(Path.home()))).resolve()
    image = Path(image or env.get('QA_IMAGE') or root/profile['image']['relativePath']).resolve()
    comm = Path(env.get('QA_COMM_ROOT') or root/profile['communication']['relativePath']).resolve()
    setup = Path(env.get('QA_NETWORK_SETUP') or comm/'network-setup.sh').resolve()
    fabric = Path(env.get('QA_LIBFABRIC_HOST') or profile['communication']['libfabricHost']).resolve()
    if env.get('QA_NETWORK') != 'custom':
        raise ValueError('Pinned deployment requires QA_NETWORK=custom')
    if 'brics/apptainer-multi-node' in env.get('LOADEDMODULES',''):
        raise ValueError('Unload the incompatible host injection module')
    if env.get('APPTAINER_BINDPATH') or env.get('SINGULARITY_BINDPATH'):
        raise ValueError('Clear implicit container bind paths; this profile uses explicit binds')
    if env.get('QA_SLURM_NETWORK') != profile['defaults']['slurmNetwork']:
        raise ValueError('Pinned Slurm network option differs')
    image_sha256 = image_sha256 or digest(image)
    if image_sha256 != profile['image']['sha256']:
        raise ValueError('SIF differs from the pinned accepted deployment')
    if digest(setup) != profile['communication']['setupSha256']:
        raise ValueError('Network setup script differs from accepted profile')
    for name, expected in profile['communication']['librarySha256'].items():
        if digest(comm/name) != expected:
            raise ValueError('Communication library differs: '+name)
    native = json.loads((comm/'native-build.json').read_text())
    if (native['ncclRuntime'] != profile['image']['ncclRuntime'] or
            native['sources'] != profile['communication']['sources'] or
            native['acceptedImage']['imageSha256'] != profile['communication'].get('buildImageSha256',image_sha256)):
        raise ValueError('Communication build provenance differs')
    if not (fabric/'lib64/libfabric.so').is_file() or not (fabric/'include/rdma/fabric.h').is_file():
        raise ValueError('Host Libfabric installation missing')
    # Record host-provider binaries too, so a site upgrade is visible in future receipts.
    providers = sorted(Path('/usr/lib64').glob('libcxi.so.*'))
    provider_hashes = {str(p.resolve()): digest(p) for p in providers if p.is_file()}
    return dict(formatVersion=1, profile=profile, profileSha256=digest(profile_path),
        image=dict(path=str(image),sha256=image_sha256),
        communicationRoot=str(comm), networkSetup=dict(path=str(setup),sha256=digest(setup),text=setup.read_text()),
        hostLibfabric=dict(path=str(fabric),librarySha256=digest(fabric/'lib64/libfabric.so')),
        hostCxiLibraries=provider_hashes,
        launchEnvironment={key:value for key,value in env.items() if key in
            ('QA_ROOT','QA_IMAGE','QA_BUNDLE','QA_RESULTS_ROOT','QA_COMM_ROOT','QA_NETWORK_SETUP',
             'QA_LIBFABRIC_HOST','QA_NETWORK','QA_ACCOUNT','QA_PARTITION','QA_SLURM_NETWORK','QA_PYTHON',
             'QA_HARNESS_ROOT','QA_DEPLOYMENT_PROFILE','QA_NODES','QA_GPUS_PER_NODE','QA_CPUS_PER_NODE',
             'QA_TIME','QA_PRECISION','QA_PREVIOUS','QA_MASTER_ADDR','QA_MASTER_PORT','QA_RAS_PORT') or key in
            ('SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_NNODES','SLURM_CPUS_PER_TASK','SLURM_JOB_GPUS','SLURM_NETWORK','LOADEDMODULES')},
        bootstrap=dict(apptainer=['--nv','--cleanenv','--no-home'],
            cudaDevices='Literal Slurm CUDA_VISIBLE_DEVICES forwarded via APPTAINERENV_CUDA_VISIBLE_DEVICES',
            slingshot='Per-node srun SLINGSHOT_VNIS/DEVICES/SVC_IDS and FI_CXI_DEFAULT_VNI forwarded literally via APPTAINERENV; per-rank probe records actual values',
            rendezvous='first allocated host, port 20000 + jobId % 35000',
            ras='loopback, port 10000 + jobId % 10000; QA_RAS_PORT overrides',
            processes='one torchrun parent per node, one worker per physical GPU'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    record = capture(args.profile)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        parser.error('Deployment output must be fresh')
    args.output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print('Deployment verified:',args.output,flush=True)


if __name__ == '__main__':
    main()
