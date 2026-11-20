#!/usr/bin/env python3
"""Table of the per-output times from the metrics files of test_timing.sh (median over repetitions)."""
import json, sys, glob, re, statistics as st
from collections import defaultdict
runs = defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] + "/metrics_*_r*.json")):
    name = re.match(r".*/metrics_(.*)_r\d+\.json", f).group(1)
    runs[name].append(json.load(open(f)))
med = lambda xs: st.median(xs)
base = med([m["time_loop_s"] for m in runs["none"]]) if "none" in runs else float("nan")
print(f"{'config':<14} {'reps':>4} {'loop_s':>8} {'overhead_s':>10} {'out/ms':>8} {'bounds/ms':>9} {'encode/ms':>9} {'write/ms':>9} {'MB/out':>8} {'after_out step/ms':>17} {'other step/ms':>13}")
for name, ms in runs.items():
    def per(m):
        if m["quant_count"]:
            c = m["quant_count"]
            return (m["quant_bounds_s"] / c, m["quant_encode_s"] / c, m["quant_write_s"] / c, m["quant_bytes"] / c)
        if m["snapshot_count"]:
            c = m["snapshot_count"]
            return (0, 0, m["snapshot_write_s"] / c, m["snapshot_bytes"] / c)
        return (0, 0, 0, 0)
    b, e, w, by = (med([per(m)[i] for m in ms]) for i in range(4))
    loop = med([m["time_loop_s"] for m in ms])
    after = med([m["compute_after_output_s"] / max(1, m["compute_after_output_steps"]) for m in ms])
    other = med([(m["compute_s"] - m["compute_after_output_s"]) / (m["compute_steps"] - m["compute_after_output_steps"]) for m in ms])
    print(f"{name:<14} {len(ms):>4} {loop:>8.3f} {loop - base:>10.3f} {(b + e + w) * 1e3:>8.3f} {b * 1e3:>9.3f} {e * 1e3:>9.3f} {w * 1e3:>9.3f} {by / 1e6:>8.3f} {after * 1e3:>17.3f} {other * 1e3:>13.3f}")
