# Source inside the container AFTER binding your site-compatible communication
# builds to /opt/slingshot, host libfabric to /host/libfabric and libcxi support
# to /host/usr/lib64. Build against this image's CUDA/NCCL per the Isambard guide.
# Do not point this at x86-64 libraries or the local shared-GPU debugging overlay.
export LD_LIBRARY_PATH="/opt/slingshot/nccl/lib:/opt/slingshot/aws-ofi-nccl/lib:/opt/slingshot/hwloc/lib:/host/libfabric/lib64:/host/usr/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_NET='AWS Libfabric' NCCL_SOCKET_IFNAME=hsn FI_PROVIDER=cxi
export FI_PROVIDER_PATH="${FI_PROVIDER_PATH:-/host/libfabric/lib64/libfabric}"
export NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 NCCL_MIN_NCHANNELS=4
export NCCL_GDRCOPY_ENABLE=1 NCCL_NET_FORCE_FLUSH=0
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1 FI_HMEM_CUDA_USE_GDRCOPY=1 FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RX_MATCH_MODE=hybrid
