# syntax=docker/dockerfile:1
# Choose a CUDA/PyTorch base supporting the requested Linux architecture.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS simulation
ARG WARP_VERSION=1.17.0
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    HOME=/work XDG_STATE_HOME=/work/state XDG_CACHE_HOME=/work/cache \
    WARPSPH_REMOTE_DATA=/work/remote-library \
    WARP_CACHE_PATH=/work/cache/warp NUMBA_CACHE_DIR=/work/cache/numba \
    MPLCONFIGDIR=/work/cache/matplotlib
WORKDIR /opt/warpsph
COPY . /opt/warpsph/
# Keep the base's CUDA-enabled Torch build; the resolver must not replace it.
RUN python -c "import importlib.metadata as m; open('base-constraints.txt','w').write('torch=='+m.version('torch')+'\n')" \
    && python -m pip install -c base-constraints.txt -c constraints.txt "warp-lang==${WARP_VERSION}" setuptools wheel \
    && python containers/install_stack.py simulation \
    && python -m pip check \
    && python -m pip freeze > /opt/warpsph/installed-simulation.txt
# Existing frontend imports include visualization dependencies even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 ffmpeg xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /work && chmod 1777 /work
WORKDIR /work
ENTRYPOINT ["python", "/opt/warpsph/containers/entrypoint.py"]
CMD ["help"]

FROM simulation AS with-remote
WORKDIR /opt/warpsph
RUN python containers/install_stack.py remote \
    && python -m pip check \
    && python -m pip freeze > /opt/warpsph/installed-remote.txt \
    && apt-get update && apt-get install -y --no-install-recommends nginx-light \
    && rm -rf /var/lib/apt/lists/*
# Viewer proxy only. The producer endpoint remains on 127.0.0.1:8766.
EXPOSE 8767
WORKDIR /work
CMD ["companion"]
