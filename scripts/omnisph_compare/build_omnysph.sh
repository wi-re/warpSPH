#!/usr/bin/env bash
# Build omniSPH's `omnySPH` pybind module (~/dev/omniSPH/omnySPH) against the
# warp conda env's Python 3.13, reusing the already-built static libs in
# ~/dev/omniSPH/build/lib (libsimulation.a, libtools.a, libimgui.a). The
# repo ships a _core.*.so for Python 3.14 only, which will not import in the
# warp env.
set -euo pipefail
O=~/dev/omniSPH
PB=$(conda run -n warp python -m pybind11 --includes)
EXT=$(conda run -n warp python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
cd "$O"
/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -fopenmp -fPIC -shared -fvisibility=hidden \
  -DNOMINMAX -D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DYAML_CPP_STATIC_DEFINE -DVERSION_INFO=0.0.2 \
  -I build -I . \
  -isystem build/vcpkg_installed/x64-linux/include -isystem build/vcpkg_installed/x64-linux/include/eigen3 \
  $PB omnySPH/src/main.cpp \
  -Wl,--start-group \
    build/lib/libsimulation.a build/lib/libtools.a build/lib/libimgui.a \
    build/vcpkg_installed/x64-linux/lib/libyaml-cpp.a build/vcpkg_installed/x64-linux/lib/libboost_atomic.a \
    build/vcpkg_installed/x64-linux/lib/libglad.a build/vcpkg_installed/x64-linux/lib/libglfw3.a build/vcpkg_installed/x64-linux/lib/libglm.a \
  -Wl,--end-group \
  -ldl -lm -lrt -lgomp -lstdc++ -lpthread \
  -o "omnySPH/src/omnySPH/_core${EXT}"
echo "built omnySPH/src/omnySPH/_core${EXT}"
conda run -n warp python -c "import sys; sys.path.insert(0,'$O/omnySPH/src'); import omnySPH; print('import OK', omnySPH.__version__)"
