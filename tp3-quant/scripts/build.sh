#!/usr/bin/env bash
# Configure and build the tp3-compression tree (same flags as cremi.sh), in its own build dir.
set -euo pipefail
W=/net/cremi/raosseiran/espaces/travail/tp3-compression
export PATH=/usr/local/cuda-13.4/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.4
[ -e /tmp/raoss_bench_running ] && { echo "benchmark lock present: not building"; exit 1; }
if [ ! -f "$W/build/CMakeCache.txt" ]; then
  cmake -S "$W/src" -B "$W/build" -DENABLE_CUDA=ON -DUSE_KOKKOS=ON -DUSE_VECTOR=OFF \
    -DCMAKE_BUILD_TYPE=Release > "$W/cmake.log" 2>&1 || { tail -30 "$W/cmake.log"; exit 1; }
fi
cmake --build "$W/build" -j 24 > "$W/make.log" 2>&1 || { grep -n "error" -A6 "$W/make.log" | head -80; exit 1; }
grep -n "warning" "$W/make.log" | grep -i "sem_proxy" | head -20 || true
ls -la "$W/build/bin/semproxy"
