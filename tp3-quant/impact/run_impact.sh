#!/usr/bin/env bash
# Part B: one raw bin-header run per reference config (NFS, temporary), then the
# offline quantization impact study. Raw fields are deleted afterwards.
set -uo pipefail
W=/net/cremi/raosseiran/espaces/travail/tp3-compression
PY=$HOME/espaces/travail/venvs/funtides-visu/bin/python
mkdir -p $W/results $W/tests
for cfg in "S 50 12" "M 100 10" "L 200 6"; do
  set -- $cfg; C=$1; N=$2; J=$3
  D=$W/tests/raw_$C
  rm -rf $D; mkdir -p $D/snapshots
  ( cd $D && $W/build/bin/semproxy --ex $N --ey $N --ez $N -o 2 --dt 0.001 --timemax 1.5 \
      --snapshots-interval 100 --use-snapshots --snapshots-use-headers --snapshots-folder $D/snapshots \
      --receiver-positions-file $W/src/receiver_positions.txt \
      --seismograms-folder $D/seismos --metrics-file $D/metrics.json < /dev/null > $D/log.txt 2>&1 ) \
    || { echo "$C: semproxy FAILED"; continue; }
  echo "$C: raw run done, $(du -sh $D/snapshots | cut -f1)"
  $PY $W/src/quant_impact.py run $D/snapshots --config $C --out $W/results/impact_$C.csv --jobs $J \
    || echo "$C: impact FAILED"
  cp $D/metrics.json $W/results/raw_run_metrics_$C.json
  rm -rf $D/snapshots
done
$PY $W/src/quant_impact.py summary $W/results/impact_S.csv $W/results/impact_M.csv $W/results/impact_L.csv --outdir $W/results
echo ALL_DONE
