#!/usr/bin/env bash
# run_one.sh OUTDIR SIZE [semproxy options...]: one semproxy run with outputs under OUTDIR
set -euo pipefail
[ -e /tmp/raoss_bench_running ] && { echo "benchmark lock present: not running"; exit 1; }
BIN=/net/cremi/raosseiran/espaces/travail/tp3-compression/build/bin/semproxy
D=$1; N=$2; shift 2
mkdir -p "$D/snapshots"
cd "$D"
"$BIN" --ex $N --ey $N --ez $N -o 2 --dt 0.001 --timemax ${TIMEMAX:-1.5} --snapshots-interval 100 \
  --snapshots-folder "$D/snapshots" --quant-folder "$D/quant" --slices-folder "$D/slices" \
  --seismograms-folder "$D/seismos" --metrics-file "$D/metrics.json" "$@" < /dev/null > "$D/log.txt" 2>&1
