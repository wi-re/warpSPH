#!/usr/bin/env bash
# Source on Isambard. Defaults describe the accepted NGC26.01/OFI1.18.0 deployment.
# Shared root may move if the pinned image/build are copied intact.
export QA_ROOT=${QA_ROOT:-$HOME}
export QA_BUNDLE=${QA_BUNDLE:-$QA_ROOT/warpsph-isambard}
export QA_IMAGE=${QA_IMAGE:-$QA_ROOT/images/warpsph-arm64-io-repair-2ecb382.sif}
export QA_COMM_ROOT=${QA_COMM_ROOT:-$QA_ROOT/communication/ofi-7999853}
export QA_LIBFABRIC_HOST=${QA_LIBFABRIC_HOST:-/opt/cray/libfabric/2.3.1}
export QA_NETWORK_SETUP=${QA_NETWORK_SETUP:-$QA_COMM_ROOT/network-setup.sh}
export QA_RESULTS_ROOT=${QA_RESULTS_ROOT:-$QA_ROOT/qualification}
export QA_PYTHON=${QA_PYTHON:-$QA_ROOT/miniforge3/bin/python}
export QA_ACCOUNT=${QA_ACCOUNT:-brics.b6ez}
export QA_NETWORK=custom QA_PREFLIGHT_NETWORK=custom QA_SLURM_NETWORK=disable_rdzv_get
# Use the directory containing this profile, rather than Slurm's spool directory.
export QA_HARNESS_ROOT
QA_HARNESS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export QA_DEPLOYMENT_PROFILE=${QA_DEPLOYMENT_PROFILE:-$QA_HARNESS_ROOT/site-profile.json}
