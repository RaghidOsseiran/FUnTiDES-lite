#!/usr/bin/env bash
# Functional tests on S: each run writes the raw and the quantized snapshots of
# the same field; check = byte-identical to the numpy reference encoder,
# eval = reconstruction errors.
set -uo pipefail
W=/net/cremi/raosseiran/espaces/travail/tp3-compression
T=/tmp/raoss_q/func
source ~/espaces/travail/venvs/funtides-visu/bin/activate
QT="python $W/src/quant_tools.py"
mkdir -p $T $W/tests/func
: > $W/tests/func/summary.txt
for cfg in "uint8 snapshot" "uint12 snapshot" "uint16 snapshot" "float16 snapshot" "float32 snapshot" \
           "uint8 symmetric" "uint16 symmetric" "float16 symmetric" "uint8 fixed" "uint16 fixed" "uint12 symmetric"; do
  set -- $cfg; kind=$1; bounds=$2
  for where in gpu host; do
    name=${kind}_${bounds}_${where}
    D=$T/$name
    rm -rf $D
    extra=""
    [ $bounds = fixed ] && extra="--quant-amax ${AMAX:-1}"
    $W/run_one.sh $D 50 --use-snapshots --snapshots-use-headers --snapshots-quant $kind \
       --quant-where $where --quant-bounds $bounds $extra || { echo "$name: RUN FAILED"; tail -5 $D/log.txt; continue; }
    echo "== $name" | tee -a $W/tests/func/summary.txt
    $QT check $D/snapshots $D/quant | tail -1 | tee -a $W/tests/func/summary.txt
    $QT eval $D/snapshots $D/quant --csv $W/tests/func/eval_$name.csv > $W/tests/func/eval_$name.txt
    grep -A3 "^$" $W/tests/func/eval_$name.txt | grep snapshots | tee -a $W/tests/func/summary.txt
    cp $D/metrics.json $W/tests/func/metrics_$name.json
    cp $D/quant/quant-header.json $W/tests/func/quant-header_$name.json
    [ "$name" = "${KEEP:-uint8_snapshot_gpu}" ] || rm -rf $D
  done
done
