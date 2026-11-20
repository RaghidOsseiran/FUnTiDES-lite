#!/usr/bin/env bash
# Indicative timings (not a campaign): one output type per run, REPS repetitions.
# usage: test_timing.sh SIZE REPS OUTROOT [TIMEMAX]
set -uo pipefail
W=/net/cremi/raosseiran/espaces/travail/tp3-compression
N=$1; REPS=$2; T=$3; export TIMEMAX=${4:-1.5}
R=$W/tests/timing_$N
mkdir -p $T $R
for rep in $(seq 1 $REPS); do
  for cfg in none raw "float32 gpu" "float32 host" "float16 gpu" "float16 host" "uint16 gpu" "uint16 host" \
             "uint12 gpu" "uint12 host" "uint8 gpu" "uint8 host"; do
    set -- $cfg
    name=$(echo $cfg | tr ' ' '_')
    D=$T/${name}
    rm -rf $D
    case $1 in
      none) opts="" ;;
      raw) opts="--use-snapshots --snapshots-use-headers" ;;
      *) opts="--snapshots-quant $1 --quant-where $2" ;;
    esac
    $W/run_one.sh $D $N $opts || { echo "$name FAILED"; tail -5 $D/log.txt; continue; }
    cp $D/metrics.json $R/metrics_${name}_r$rep.json
    rm -rf $D
  done
done
