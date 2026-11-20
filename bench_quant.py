#!/usr/bin/env python3
"""Measure the cost of quantized snapshots in FUnTiDES-lite (lossy compression).

Same protocol as bench_slices.py (same reference configurations, repetitions
with the variant order rotated, untimed warm-up, files deleted after each run,
runs.csv rewritten after each run, env/ record), so the numbers compare with
the slices campaigns. ONE output type per run:
  none            no output                       (baseline)
  raw             full snapshots, float32 pressure only (--use-snapshots
                  --snapshots-use-headers): 4 bytes per node, the reference
  <kind>_gpu      --snapshots-quant <kind> --quant-where gpu
  <kind>_host     --snapshots-quant <kind> --quant-where host
with <kind> in --kinds (default uint8 uint12 uint16 float16) and the bounds of
--bounds (default snapshot).

For every run it reads the proxy's metrics.json (timers and bytes) and the
per-output times printed on stdout (first output apart: for raw it also writes
headers.dat, 12 bytes per node, inside the timed call).

Results go to OUTDIR: runs.csv (one row per run), summary.csv / summary.md
(medians over repetitions, one row per size and variant), env/, and
runs/<size>_<variant>_r<rep>/ (stdout, stderr, metrics, seismograms).

Consistency check of the summary: the overhead on the time loop (loop - loop
of "none") should be the time spent in the outputs (sum of the output timers)
plus the page-migration penalty paid by the solver step that follows an output
(compute_after_output_s - steps * normal step). What is left is "unexplained".

Examples
  python bench_quant.py --outdir ~/espaces/travail/tp3-compression/bench_quant_SML --dry-run
  python bench_quant.py --outdir ~/espaces/travail/tp3-compression/bench_quant_SML \\
      --receivers receiver_positions.txt
"""

import argparse
import csv
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quant_tools import parse_kind, read_qsnap  # noqa: E402

METRIC_KEYS = ["time_loop_s", "compute_s", "compute_steps", "compute_after_output_s",
               "compute_after_output_steps", "snapshot_write_s", "snapshot_count",
               "snapshot_bytes", "header_bytes", "seismo_bytes", "quant_bounds_s",
               "quant_encode_s", "quant_write_s", "quant_count", "quant_bytes",
               "quant_payload_bytes"]
CONFIGS = {"S": 50, "M": 100, "L": 200}  # see bench_slices.py
QSNAP_HEADER_BYTES = 72
# loop time of one run, measured on holtzman (s): only used for the estimate
# printed before starting (none, raw, quant gpu, quant host)
LOOP_ESTIMATE = {50: (0.4, 1.5, 0.5, 0.7), 100: (3, 9, 3.5, 5), 200: (23, 70, 25, 28)}


def num_steps(timemax, dt):
    """Same as the proxy: int(timemax / dt) computed in float32."""
    return int(np.float32(timemax) / np.float32(dt))


def kind_of(variant):
    return variant.rsplit("_", 1)[0] if variant not in ("none", "raw") else None


def where_of(variant):
    return variant.rsplit("_", 1)[1] if variant not in ("none", "raw") else None


def predict(size, args):
    """Expected nodes, outputs and bytes written by one run of each variant."""
    nodes = (size * args.order + 1) ** 3
    outputs = len(range(0, num_steps(args.timemax, args.dt), args.interval))
    b = {"none": 0, "raw": outputs * nodes * 4 + nodes * 12}
    for v in args.variants:
        if kind_of(v):
            bits = parse_kind(kind_of(v))[1]
            b[v] = outputs * (QSNAP_HEADER_BYTES + (nodes * bits + 7) // 8)
    return {"nodes": nodes, "outputs": outputs, "bytes": b}


def plan_runs(args):
    """Runs in execution order; inside each repetition the variants are rotated
    (rep 1: base order, rep 2: shifted by one, ...), so no variant always runs
    at the same place, e.g. right after the long I/O-bound raw run."""
    base = args.variants
    runs = []
    for size in args.sizes:
        for rep in range(1, args.reps + 1):
            shift = (rep - 1) % len(base)
            for position, variant in enumerate(base[shift:] + base[:shift], 1):
                runs.append({"config": args.labels[size], "size": size, "variant": variant,
                             "kind": kind_of(variant) or "", "where": where_of(variant) or "",
                             "rep": rep, "position": position,
                             "name": f"{size}_{variant}_r{rep}"})
    return runs


def proxy_command(run, rundir, args):
    s = str(run["size"])
    cmd = [str(args.semproxy), "--ex", s, "--ey", s, "--ez", s, "-o", str(args.order),
           "--timemax", str(args.timemax), "--dt", str(args.dt),
           "--snapshots-interval", str(args.interval),
           "--seismograms-folder", str(rundir / "seismos"),
           "--metrics-file", str(rundir / "metrics.json")]
    if args.receivers:
        cmd += ["--receiver-positions-file", str(args.receivers)]
    if run["variant"] == "raw":
        cmd += ["--use-snapshots", "--snapshots-use-headers",
                "--snapshots-folder", str(rundir / "snapshots")]
    elif run["kind"]:
        cmd += ["--snapshots-quant", run["kind"], "--quant-where", run["where"],
                "--quant-bounds", args.bounds, "--quant-folder", str(rundir / "quant")]
        if args.bounds == "fixed":
            cmd += ["--quant-amax", str(args.amax)]
    return cmd + args.extra


# ----------------------------------------------------------------------------
# Running and reading back
# ----------------------------------------------------------------------------
def sh(cmd):
    """Output of a shell command, or a note if it fails (environment record)."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{cmd!r} failed: {e}>"


def record_env(args, plan_info):
    env = args.outdir / "env"
    env.mkdir(parents=True, exist_ok=True)
    (env / "command.txt").write_text(" ".join(map(str, sys.argv)) + "\n")
    (env / "plan.json").write_text(json.dumps(plan_info, indent=2))
    build = args.semproxy.resolve().parent.parent
    (env / "build.txt").write_text(
        f"semproxy: {args.semproxy}\n"
        f"binary: {sh(f'ls -l --time-style=full-iso {args.semproxy}')}\n"
        f"sha256: {sh(f'sha256sum {args.semproxy}')}\n"
        f"cmake cache:\n"
        + sh(f"grep -E 'CMAKE_BUILD_TYPE|ENABLE_CUDA|USE_KOKKOS|USE_VECTOR|CMAKE_CUDA_COMPILER:|"
             f"Kokkos_ENABLE_(CUDA|OPENMP|SERIAL|CUDA_UVM):|Kokkos_ARCH_.*=ON' "
             f"{build / 'CMakeCache.txt'}") + "\n"
        f"kokkos: {sh(f'git -C {HERE}/external/kokkos describe --tags --always')}\n"
        f"nvcc: {sh('/usr/local/cuda-13.4/bin/nvcc --version | tail -n 1')}\n")
    (env / "machine.txt").write_text(
        f"host: {platform.node()}\n"
        f"date: {sh('date -Iseconds')}\n"
        f"python: {sys.version.split()[0]}, numpy {np.__version__}\n"
        f"gpu:\n{sh('nvidia-smi -L')}\n"
        f"{sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv')}\n"
        f"cpu:\n{sh('lscpu | head -20')}\n"
        f"memory:\n{sh('free -h')}\n"
        f"load: {sh('uptime')}\n")
    (env / "filesystem.txt").write_text(sh(f"df -hT {args.outdir}") + "\n"
                                        + sh(f"grep ' /autofs\\|nfs' /proc/mounts | head -5") + "\n")
    return sh(f"df -T {args.outdir} | tail -1 | awk '{{print $2}}'")


def allocated_bytes(path):
    """Space really taken on the filesystem (blocks), not the logical size."""
    return sum(p.stat().st_blocks * 512 for p in path.rglob("*") if p.is_file())


def step_of(p):
    return int(re.findall(r"\d+", p.stem)[-1])


def last_output_finite(rundir, run):
    """NaN/inf check of the last output: a diverging run still writes files of
    the right size, so sizes alone do not catch it."""
    if run["variant"] == "raw":
        files = sorted((rundir / "snapshots").glob("snap-*.dat"), key=step_of)
        if files:
            return bool(np.isfinite(np.fromfile(files[-1], dtype="<f4")).all())
    elif run["kind"]:
        files = sorted((rundir / "quant").glob("qsnap-*.bin"), key=step_of)
        if files:
            h, p = read_qsnap(files[-1])
            return bool(np.isfinite(p).all() and np.isfinite([h["data_min"], h["data_max"]]).all())
    return ""


def do_run(run, args):
    rundir = args.outdir / "runs" / run["name"]
    if rundir.exists():
        shutil.rmtree(rundir)
    (rundir / "snapshots").mkdir(parents=True)  # the proxy refuses a missing folder
    cmd = proxy_command(run, rundir, args)
    (rundir / "command.txt").write_text(" ".join(cmd) + "\n")
    t0 = time.perf_counter()
    with open(rundir / "stdout.txt", "w") as out, open(rundir / "stderr.txt", "w") as err:
        rc = subprocess.run(cmd, stdout=out, stderr=err, cwd=rundir,
                            stdin=subprocess.DEVNULL).returncode
    row = dict(run, wall_s=time.perf_counter() - t0, status="ok" if rc == 0 else f"exit {rc}")
    if rc != 0:
        return row

    m = json.loads((rundir / "metrics.json").read_text())
    row.update({k: m.get(k, "") for k in METRIC_KEYS})
    row["nodes"] = m.get("nodes", "")
    stdout = (rundir / "stdout.txt").read_text()
    for key, label in (("init_s", "Elapsed Initial Time"), ("total_exe_s", "Elapsed TotalExe Time")):
        hit = re.search(label + r" : ([0-9.eE+-]+)", stdout)
        row[key] = float(hit.group(1)) if hit else ""
    # per-output times as printed by the proxy: the first output apart
    label = {"raw": "Elapsed Saving Snapshot Time"}.get(
        run["variant"], "Elapsed Saving Quantized Snapshot Time")
    per_out = [float(x) for x in re.findall(label + r" : ([0-9.eE+-]+)", stdout)]
    if per_out and run["variant"] != "none":
        row["out_first_s"] = per_out[0]
        row["out_rest_median_s"] = statistics.median(per_out[1:]) if len(per_out) > 1 else ""
        row["out_rest_max_s"] = max(per_out[1:]) if len(per_out) > 1 else ""
    row["disk_alloc_bytes"] = allocated_bytes(rundir)
    row["finite"] = last_output_finite(rundir, run)
    if not args.keep_files:
        for sub in ("snapshots", "quant"):
            shutil.rmtree(rundir / sub, ignore_errors=True)
    return row


# ----------------------------------------------------------------------------
# Summary: medians over repetitions, one row per size and variant
# ----------------------------------------------------------------------------
def vals(rows, f):
    out = []
    for r in rows:
        try:
            v = f(r)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
        if v is not None:
            out.append(v)
    return out


def med(rows, f):
    v = vals(rows, f)
    return statistics.median(v) if v else None


def spread_pct(rows, f):
    v = vals(rows, f)
    if len(v) < 2 or statistics.median(v) == 0:
        return None
    return 100 * (max(v) - min(v)) / statistics.median(v)


def num(r, k):
    return float(r[k]) if r.get(k) not in ("", None) else None


def out_count(r):
    return (num(r, "quant_count") or 0) + (num(r, "snapshot_count") or 0)


def out_total_s(r):
    """Time spent in the output code of one run (sum of its timers)."""
    return sum(num(r, k) or 0 for k in ("snapshot_write_s", "quant_bounds_s", "quant_encode_s",
                                        "quant_write_s"))


def per_output(key):
    return lambda r: (num(r, key) or 0) / out_count(r)


def summarise(rows, args):
    out = []
    ok = [r for r in rows if r["status"] == "ok"]
    for size in args.sizes:
        g = {v: [r for r in ok if r["size"] == size and r["variant"] == v] for v in args.variants}
        none, raw = g.get("none", []), g.get("raw", [])
        step_none = med(none, lambda r: num(r, "compute_s") / num(r, "compute_steps"))
        loop_none = med(none, lambda r: num(r, "time_loop_s"))
        loop_raw = med(raw, lambda r: num(r, "time_loop_s"))
        raw_bytes = med(raw, lambda r: num(r, "snapshot_bytes") / out_count(r))
        if raw_bytes is None:  # raw variant not run: 4 bytes per node
            raw_bytes = 4 * (size * args.order + 1) ** 3
        for v in args.variants:
            grp = g[v]
            if not grp:
                continue
            s = {"config": args.labels[size], "size": size, "variant": v,
                 "nodes": med(grp, lambda r: num(r, "nodes")), "reps": len(grp),
                 "outputs": med(grp, out_count)}
            is_out = v != "none"
            # disk
            s["bytes_per_output"] = med(grp, lambda r: (
                (num(r, "quant_bytes") or 0) + (num(r, "snapshot_bytes") or 0)) / out_count(r)) \
                if is_out else None
            s["header_bytes"] = med(grp, lambda r: num(r, "header_bytes"))
            s["ratio_vs_raw"] = (raw_bytes / s["bytes_per_output"]
                                 if s["bytes_per_output"] else None)
            # time per output (run totals / outputs), and as printed (first output apart)
            for key in ("quant_bounds_s", "quant_encode_s"):
                s[key.replace("_s", "_per_output_s")] = (
                    med(grp, per_output(key)) if kind_of(v) else None)
            s["write_per_output_s"] = (
                med(grp, per_output("quant_write_s" if kind_of(v) else "snapshot_write_s"))
                if is_out else None)
            s["total_per_output_s"] = med(grp, lambda r: out_total_s(r) / out_count(r)) \
                if is_out else None
            s["total_per_output_spread_pct"] = spread_pct(
                grp, lambda r: out_total_s(r) / out_count(r)) if is_out else None
            s["out_first_s"] = med(grp, lambda r: num(r, "out_first_s"))
            s["out_rest_median_s"] = med(grp, lambda r: num(r, "out_rest_median_s"))
            s["mb_per_s"] = (s["bytes_per_output"] / 1e6 / s["total_per_output_s"]
                             if is_out and s["total_per_output_s"] else None)
            # solver step: right after an output vs the others
            s["step_after_output_s"] = med(grp, lambda r: num(r, "compute_after_output_s")
                                           / num(r, "compute_after_output_steps")) if is_out else None
            s["step_other_s"] = med(grp, lambda r: (
                num(r, "compute_s") - num(r, "compute_after_output_s"))
                / (num(r, "compute_steps") - num(r, "compute_after_output_steps")))
            s["step_none_s"] = step_none
            # time loop and overhead
            s["loop_s"] = med(grp, lambda r: num(r, "time_loop_s"))
            s["loop_spread_pct"] = spread_pct(grp, lambda r: num(r, "time_loop_s"))
            if loop_none is not None:
                s["overhead_s"] = s["loop_s"] - loop_none
                s["overhead_pct"] = 100 * s["overhead_s"] / loop_none
            else:
                s["overhead_s"] = s["overhead_pct"] = None
            if loop_raw is not None and is_out:
                s["gain_vs_raw_s"] = loop_raw - s["loop_s"]
                s["gain_vs_raw_pct"] = 100 * s["gain_vs_raw_s"] / loop_raw
            else:
                s["gain_vs_raw_s"] = s["gain_vs_raw_pct"] = None
            # consistency: overhead = outputs + penalty of the steps after an output + rest
            if is_out and step_none is not None and loop_none is not None:
                s["check_outputs_s"] = med(grp, out_total_s)
                s["check_after_penalty_s"] = med(grp, lambda r: (
                    num(r, "compute_after_output_s")
                    - num(r, "compute_after_output_steps") * step_none))
                s["check_other_steps_s"] = med(grp, lambda r: (
                    num(r, "compute_s") - num(r, "compute_after_output_s")
                    - (num(r, "compute_steps") - num(r, "compute_after_output_steps")) * step_none))
                s["check_explained_s"] = s["check_outputs_s"] + s["check_after_penalty_s"]
                s["check_unexplained_s"] = s["overhead_s"] - s["check_explained_s"]
            else:
                for k in ("check_outputs_s", "check_after_penalty_s", "check_other_steps_s",
                          "check_explained_s", "check_unexplained_s"):
                    s[k] = None
            out.append(s)
    return out


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    if unit == "B":
        for u in ("B", "kB", "MB", "GB"):
            if abs(v) < 1000 or u == "GB":
                return f"{v:.4g} {u}"
            v /= 1000
    if unit == "ms":
        return f"{v * 1e3:.3g}"
    if unit == "s":
        return f"{v:.4g}"
    if unit == "%":
        return f"{v:.2g}"
    return f"{v:.4g}"


def write_summary(summary, args, fs):
    if not summary:
        return
    with open(args.outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    lines = ["# Quantized snapshots: none vs raw float32 vs quantized (gpu / host)", "",
             f"Medians over the repetitions ({args.reps} asked). Bounds: {args.bounds}. "
             f"Output filesystem: {fs}. No --sync-writes. Times per output are run totals "
             "divided by the number of outputs; spread = (max - min) / median over repetitions.",
             ""]
    cols = [
        ("variant", lambda s: s["variant"]),
        ("MB/output", lambda s: fmt(s["bytes_per_output"] / 1e6) if s["bytes_per_output"] else "n/a"),
        ("ratio vs raw", lambda s: fmt(s["ratio_vs_raw"])),
        ("bounds ms", lambda s: fmt(s["quant_bounds_per_output_s"], "ms")),
        ("encode ms", lambda s: fmt(s["quant_encode_per_output_s"], "ms")),
        ("write ms", lambda s: fmt(s["write_per_output_s"], "ms")),
        ("total ms/output", lambda s: fmt(s["total_per_output_s"], "ms")),
        ("spread %", lambda s: fmt(s["total_per_output_spread_pct"], "%")),
        ("first output ms", lambda s: fmt(s["out_first_s"], "ms")),
        ("median other outputs ms", lambda s: fmt(s["out_rest_median_s"], "ms")),
        ("MB/s", lambda s: fmt(s["mb_per_s"])),
        ("step after output ms", lambda s: fmt(s["step_after_output_s"], "ms")),
        ("other steps ms", lambda s: fmt(s["step_other_s"], "ms")),
        ("loop s", lambda s: fmt(s["loop_s"], "s")),
        ("loop spread %", lambda s: fmt(s["loop_spread_pct"], "%")),
        ("overhead s", lambda s: fmt(s["overhead_s"], "s")),
        ("overhead %", lambda s: fmt(s["overhead_pct"], "%")),
        ("gain vs raw s", lambda s: fmt(s["gain_vs_raw_s"], "s")),
        ("gain vs raw %", lambda s: fmt(s["gain_vs_raw_pct"], "%")),
        ("check: outputs s", lambda s: fmt(s["check_outputs_s"], "s")),
        ("check: after-output penalty s", lambda s: fmt(s["check_after_penalty_s"], "s")),
        ("check: unexplained s", lambda s: fmt(s["check_unexplained_s"], "s")),
        ("(other steps drift s)", lambda s: fmt(s["check_other_steps_s"], "s")),
    ]
    for size in args.sizes:
        sel = [s for s in summary if s["size"] == size]
        if not sel:
            continue
        lines += [f"## {sel[0]['config']} ({size} elements, {fmt(sel[0]['nodes'])} nodes, "
                  f"{fmt(next((s['outputs'] for s in sel if s['outputs']), None))} outputs)", "",
                  "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(c[1](s) for c in cols) + " |" for s in sel]
        lines.append("")
    text = "\n".join(lines) + "\n"
    (args.outdir / "summary.md").write_text(text)
    print("\n" + text)


def read_runs(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["size"], r["rep"] = int(r["size"]), int(r["rep"])
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True,
                    help="where runs and results go; its filesystem is what the write "
                         "times measure (NFS, local disk, tmpfs = RAM)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--sizes", type=int, nargs="+",
                    help="other sizes instead of --configs: elements per direction")
    ap.add_argument("--reps", type=int, default=3, help="repetitions (default: %(default)s)")
    ap.add_argument("--kinds", nargs="+", default=["uint8", "uint12", "uint16", "float16"])
    ap.add_argument("--where", nargs="+", default=["gpu", "host"], choices=["gpu", "host"])
    ap.add_argument("--no-none", action="store_true", help="skip the run without output")
    ap.add_argument("--no-raw", action="store_true", help="skip the raw snapshot run")
    ap.add_argument("--bounds", default="snapshot", choices=["snapshot", "symmetric", "fixed"])
    ap.add_argument("--amax", type=float, help="--quant-amax for --bounds fixed")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--timemax", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--interval", type=int, default=100,
                    help="steps between outputs (default: %(default)s, i.e. 15 outputs)")
    ap.add_argument("--receivers", type=Path, help="--receiver-positions-file for the proxy")
    ap.add_argument("--keep-files", action="store_true",
                    help="keep the output files (default: deleted after each run)")
    ap.add_argument("--max-gb", type=float, default=20.0,
                    help="refuse to start if the predicted disk peak exceeds this")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up run")
    ap.add_argument("--semproxy", type=Path, default=HERE.parent / "build" / "bin" / "semproxy")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the commands only")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild summary.csv / summary.md from OUTDIR/runs.csv")
    ap.add_argument("extra", nargs="*", help="extra proxy options, after --")
    args = ap.parse_args()
    args.outdir = args.outdir.expanduser().resolve()
    args.semproxy = args.semproxy.expanduser().resolve()
    if args.receivers:
        args.receivers = args.receivers.expanduser().resolve()
        if not args.receivers.is_file():
            sys.exit(f"no receiver file at {args.receivers}")
    if args.bounds == "fixed" and not (args.amax and args.amax > 0):
        ap.error("--bounds fixed requires --amax > 0")
    for k in args.kinds:
        parse_kind(k)
    args.variants = ([] if args.no_none else ["none"]) + ([] if args.no_raw else ["raw"]) \
        + [f"{k}_{w}" for k in args.kinds for w in args.where]

    if args.sizes is None:
        args.sizes = [CONFIGS[c] for c in dict.fromkeys(args.configs)]
    by_size = {v: k for k, v in CONFIGS.items()}
    reference = (args.order, args.timemax, args.dt, args.interval) == (2, 1.5, 0.001, 100)
    args.labels = {s: by_size[s] if s in by_size and reference else str(s) for s in args.sizes}

    if args.summary_only:
        fs = sh(f"df -T {args.outdir} | tail -1 | awk '{{print $2}}'")
        write_summary(summarise(read_runs(args.outdir / "runs.csv"), args), args, fs)
        return

    runs = plan_runs(args)
    pred = {size: predict(size, args) for size in args.sizes}
    per_run = [pred[r["size"]]["bytes"][r["variant"]] for r in runs]
    peak = sum(per_run) if args.keep_files else max(per_run)
    est = 0.0
    for r in runs:
        e = LOOP_ESTIMATE.get(r["size"])
        if e is None:
            est = None
            break
        est += e[0 if r["variant"] == "none" else 1 if r["variant"] == "raw"
                 else 2 if r["where"] == "gpu" else 3]
    print(f"{len(runs)} runs ({len(args.variants)} variants x {args.reps} reps x "
          f"{len(args.sizes)} sizes); outputs per run: "
          + ", ".join(f"size {s}: {p['outputs']}" for s, p in pred.items()))
    for s, p in pred.items():
        print(f"  {args.labels[s]}: {p['nodes']:,} nodes; bytes per run: "
              + ", ".join(f"{v} {fmt(b, 'B')}" for v, b in p["bytes"].items() if v != "none"))
    print(f"predicted peak disk use: {fmt(peak, 'B')} "
          f"({'all runs kept' if args.keep_files else 'largest single run, files deleted after each'})"
          f"; total written over the campaign: {fmt(sum(per_run), 'B')}")
    if est is not None:
        print(f"estimated time loops: {est / 60:.0f} min (plus initialisation and seismogram "
              "output of each run, not included)")

    if args.dry_run:
        for r in runs:
            print(" ".join(proxy_command(r, args.outdir / "runs" / r["name"], args)))
        return
    if peak > args.max_gb * 1e9:
        sys.exit(f"predicted {fmt(peak, 'B')} > --max-gb {args.max_gb}")
    if not args.semproxy.exists():
        sys.exit(f"no proxy binary at {args.semproxy} (use --semproxy)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fs = record_env(args, {"runs": runs, "predicted": pred, "args": {
        k: str(v) for k, v in vars(args).items()}})
    print(f"output filesystem: {fs or 'unknown'}"
          + ("  WARNING: tmpfs lives in RAM, write times will not reflect a disk"
             if fs == "tmpfs" else ""))

    if not args.no_warmup:  # first GPU run of a session pays one-off costs
        warm = args.outdir / "warmup"
        warm.mkdir(exist_ok=True)
        subprocess.run([str(args.semproxy), "--ex", "10", "--ey", "10", "--ez", "10",
                        "--timemax", "0.01", "--seismograms-folder", str(warm / "seismos"),
                        "--metrics-file", str(warm / "metrics.json")],
                       cwd=warm, capture_output=True, stdin=subprocess.DEVNULL)

    rows, fieldnames = [], None
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['name']} ...", end=" ", flush=True)
        row = do_run(run, args)
        rows.append(row)
        print(f"{row['status']}, loop {fmt(row.get('time_loop_s') or None, 's')} s"
              + ("" if row.get("finite") in ("", True) else "  WARNING: NaN/inf in last output"),
              flush=True)
        fieldnames = list(dict.fromkeys((fieldnames or []) + list(row)))
        with open(args.outdir / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_summary(summarise(rows, args), args, fs)
    print(f"results in {args.outdir}")


if __name__ == "__main__":
    main()
