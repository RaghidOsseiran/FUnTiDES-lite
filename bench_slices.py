#!/usr/bin/env python3
"""Measure the cost of outputs in FUnTiDES-lite: none vs full snapshots vs z-slices.

For every size and repetition it runs the proxy in each variant
  none    no snapshot, no slice           (baseline: what outputs cost on top)
  snap    full 3D snapshots               (--use-snapshots)
  slice   z-slices only                   (--use-slices)
  both    snapshots and slices            (optional)
reads the run's metrics.json (timers + bytes written by the proxy), then times
the post-processing of the snapshot and slice runs with the viewer
(visualise_snapshots.py --benchmark), and deletes the heavy files unless
--keep-files. It compares, per size:
  disk space    bytes per output, snapshot vs slice
  I/O           write time per output in the simulation; read time afterwards
  computation   solver time per step (after an output vs normal), slice
                extraction, plane cut and statistics in post-processing
and the gain of slices over full outputs, for the simulation alone and for the
whole workflow (simulation + post-processing).

Results go to OUTDIR: runs.csv (one row per run), summary.csv / summary.md
(medians over repetitions), env/ (machine, build, filesystem, versions), and
runs/<size>_<variant>[_sync]_r<rep>/ (stdout, stderr, metrics, seismograms).

The sizes are the project's reference configurations S, M and L (see CONFIGS),
the same for every benchmark of the project; --sizes runs other sizes instead.

Examples
  python bench_slices.py --outdir ~/espaces/travail/bench --dry-run
  python bench_slices.py --outdir ~/espaces/travail/bench --receivers receiver_positions.txt
  python bench_slices.py --outdir ~/espaces/travail/bench --configs S M --reps 5
  python bench_slices.py --outdir ~/espaces/travail/bench --sizes 80 150
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

try:
    import numpy as np
except ImportError:  # only needed for the NaN check of the last output
    np = None

HERE = Path(__file__).resolve().parent
METRIC_KEYS = ["time_loop_s", "compute_s", "compute_steps", "compute_after_output_s",
               "compute_after_output_steps", "snapshot_write_s", "snapshot_count",
               "slice_extract_s", "slice_write_s", "slice_count", "snapshot_bytes",
               "slice_bytes", "header_bytes", "seismo_bytes"]
SNAP_FORMATS = {"bin": [], "bin-header": ["--snapshots-use-headers"],
                "csv": ["--snapshots-use-csv"],
                "csv-header": ["--snapshots-use-csv", "--snapshots-use-headers"]}
# bytes per node of one snapshot (csv: rough estimate, text length varies)
SNAP_BYTES_PER_NODE = {"bin": 16, "bin-header": 4, "csv": 40, "csv-header": 12}

# Reference configurations of the project: elements per direction of the cubic
# 2000 m domain, at order 2, dt = 0.001 s, timemax = 1.5 s (1499 steps), one
# output every 100 steps (15 outputs). Each size doubles the previous one, so
# the node count grows 8x per step: 101^3, 201^3, 401^3 nodes.
# Chosen from measurements on holtzman (RTX 4000 Ada 20 GB, outputs on NFS):
#        nodes    GPU memory   solver step   one bin snapshot   15 snapshots
#   S    1.03 M     ~0.03 GB      0.22 ms          16.5 MB          0.25 GB
#   M    8.12 M     ~0.4  GB      1.9  ms         130   MB          1.9  GB
#   L   64.5  M     ~1.5  GB     15    ms           1.03 GB        15.5  GB
# What limits L is the I/O, not the GPU memory (~21 bytes per node): at 300
# elements the run still fits the GPU (4.8 GB) but its 15 snapshots take 52 GB
# of the shared NFS space and about 9 minutes to write, per run. dt = 0.001 s
# is stable at all three sizes (checked up to 300 elements).
CONFIGS = {"S": 50, "M": 100, "L": 200}


# ----------------------------------------------------------------------------
# Plan: which runs, and how big they will be
# ----------------------------------------------------------------------------
def num_steps(timemax, dt):
    """Same as the proxy: int(timemax / dt) computed in float32."""
    if np is not None:
        return int(np.float32(timemax) / np.float32(dt))
    return int(timemax / dt)


def predict(size, args):
    """Expected nodes, outputs and bytes of one run of each variant."""
    n = size * args.order + 1
    nodes = n ** 3
    outputs = len(range(0, num_steps(args.timemax, args.dt), args.interval))
    snap = outputs * nodes * SNAP_BYTES_PER_NODE[args.snap_format]
    if args.snap_format.endswith("header"):
        snap += nodes * 12  # coordinates written once
    sl = outputs * n * n * 4
    return {"nodes": nodes, "outputs": outputs, "snap_bytes": snap, "slice_bytes": sl,
            "bytes": {"none": 0, "snap": snap, "slice": sl, "both": snap + sl}}


def plan_runs(args):
    """Runs in execution order. Inside each repetition the variants are rotated
    (rep 1: base order, rep 2: shifted by one, ...), so no variant always runs
    first, e.g. right after a long I/O-bound run while the GPU sat idle. The base
    order puts slice runs before snap runs, so in rep 1 the snap run's
    post-processing can already cut at the slice's z."""
    syncs = {"off": [False], "on": [True], "both": [False, True]}[args.sync]
    base = [(v, sync) for v in sorted(args.variants, key=["none", "slice", "both", "snap"].index)
            for sync in ([False] if v == "none" else syncs)]
    runs = []
    for size in args.sizes:
        for rep in range(1, args.reps + 1):
            shift = (rep - 1) % len(base)
            for position, (variant, sync) in enumerate(base[shift:] + base[:shift], 1):
                name = f"{size}_{variant}{'_sync' if sync else ''}_r{rep}"
                runs.append({"config": args.labels[size], "size": size,
                             "variant": variant, "sync": sync,
                             "rep": rep, "position": position, "name": name})
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
    if run["variant"] in ("snap", "both"):
        cmd += ["--use-snapshots", "--snapshots-folder", str(rundir / "snapshots")]
        cmd += SNAP_FORMATS[args.snap_format]
    if run["variant"] in ("slice", "both"):
        cmd += ["--use-slices", "--slices-folder", str(rundir / "slices")]
        if args.slice_z is not None:
            cmd += ["--slice-z", str(args.slice_z)]
    if run["sync"]:
        cmd += ["--sync-writes"]
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
    build_info = args.semproxy.resolve().parent.parent / "build_info.txt"
    for cand in (build_info, HERE / "install" / "build_info.txt"):
        if cand.is_file():
            shutil.copy(cand, env / "build_info.txt")
            break
    (env / "machine.txt").write_text(
        f"host: {platform.node()}\n"
        f"python: {sys.version.split()[0]}\n"
        f"gpu:\n{sh('nvidia-smi -L')}\n"
        f"cpu:\n{sh('lscpu | head -20')}\n")
    (env / "filesystem.txt").write_text(sh(f"df -hT {args.outdir}") + "\n")
    (env / "pip_freeze.txt").write_text(sh(f"{sys.executable} -m pip freeze") + "\n")
    return sh(f"df -T {args.outdir} | tail -1 | awk '{{print $2}}'")


def allocated_bytes(path):
    """Space really taken on the filesystem (blocks), not the logical size."""
    return sum(p.stat().st_blocks * 512 for p in path.rglob("*") if p.is_file())


def last_output_finite(rundir, variant, snap_format):
    """NaN/inf check of the last output: a diverging run (dt too large for the
    mesh) still writes files of the right size, so sizes alone do not catch it."""
    if np is None:
        return ""
    if variant in ("slice", "both"):
        files = sorted((rundir / "slices").glob("slice-*.dat"),
                       key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if files:
            return bool(np.isfinite(np.fromfile(files[-1], dtype="<f4")).all())
    if variant == "snap" and snap_format.startswith("bin"):
        files = sorted((rundir / "snapshots").glob("snap-*.dat"),
                       key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if files:
            a = np.fromfile(files[-1], dtype="<f4")
            if snap_format == "bin":
                a = a.reshape(-1, 4)[:, 3]
            return bool(np.isfinite(a).all())
    return ""


def visu_benchmark(rundir, mode, z, args):
    """Viewer --benchmark on this run; returns per-pass totals and per-file means."""
    cmd = [args.python, str(args.viewer), str(rundir), "--benchmark",
           "--repeat", str(args.visu_repeat), "--seismos", str(rundir / "seismos")]
    if mode == "slice":
        cmd.append("--slices")
    elif z is not None:
        cmd += ["--z", str(z)]
    with open(rundir / f"visu_{mode}.out", "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    csv_path = rundir / f"visu_bench_{mode}.csv"
    if rc != 0 or not csv_path.is_file():
        return {"visu_status": f"failed ({rc})"}
    rows = list(csv.DictReader(csv_path.open()))
    passes = sorted({int(r["pass"]) for r in rows})
    num = lambda r, k: float(r[k]) if r[k] not in ("", None) else 0.0  # noqa: E731
    total = {p: sum(num(r, "t_read_s") + num(r, "t_cut_s") + num(r, "t_stats_s")
                    for r in rows if int(r["pass"]) == p) for p in passes}
    # warm = passes 2.. (files already in the OS cache); pass 1 alone if only one
    warm = [r for r in rows if int(r["pass"]) > 1] or rows
    mean = lambda k: statistics.mean(num(r, k) for r in warm)  # noqa: E731
    return {"visu_status": "ok",
            "visu_files": len([r for r in rows if int(r["pass"]) == 1]),
            "visu_pass1_total_s": total[passes[0]],
            "visu_warm_total_s": statistics.median([total[p] for p in passes[1:]] or
                                                   [total[passes[0]]]),
            "visu_read_per_file_s": mean("t_read_s"),
            "visu_cut_per_file_s": mean("t_cut_s"),
            "visu_stats_per_file_s": mean("t_stats_s"),
            "visu_bytes_per_file": statistics.mean(num(r, "bytes_read") for r in rows)}


def do_run(run, args, slice_z_by_size):
    rundir = args.outdir / "runs" / run["name"]
    if rundir.exists():
        shutil.rmtree(rundir)
    (rundir / "snapshots").mkdir(parents=True)  # the proxy refuses a missing folder
    cmd = proxy_command(run, rundir, args)
    (rundir / "command.txt").write_text(" ".join(cmd) + "\n")
    t0 = time.perf_counter()
    with open(rundir / "stdout.txt", "w") as out, open(rundir / "stderr.txt", "w") as err:
        rc = subprocess.run(cmd, stdout=out, stderr=err, cwd=rundir).returncode
    row = dict(run, wall_s=time.perf_counter() - t0, status="ok" if rc == 0 else f"exit {rc}")
    if rc != 0:
        return row

    m = json.loads((rundir / "metrics.json").read_text())
    row.update({k: m.get(k, "") for k in METRIC_KEYS})
    row["nodes"] = m.get("nodes", "")
    row["slice_z"] = m.get("options", {}).get("slice_z", "")
    stdout = (rundir / "stdout.txt").read_text()
    for key, label in (("init_s", "Elapsed Initial Time"), ("total_exe_s", "Elapsed TotalExe Time")):
        hit = re.search(label + r" : ([0-9.eE+-]+)", stdout)
        row[key] = float(hit.group(1)) if hit else ""
    row["disk_alloc_bytes"] = allocated_bytes(rundir)
    row["finite"] = last_output_finite(rundir, run["variant"], args.snap_format)
    if row["slice_z"] != "" and run["variant"] in ("slice", "both"):
        slice_z_by_size[run["size"]] = row["slice_z"]

    # post-processing, on files that were just written (see summary note)
    if run["variant"] == "snap" and not args.no_visu:
        row.update(visu_benchmark(rundir, "volume", slice_z_by_size.get(run["size"]), args))
    elif run["variant"] == "slice" and not args.no_visu:
        row.update(visu_benchmark(rundir, "slice", None, args))

    if not args.keep_files:
        for sub in ("snapshots", "slices"):
            shutil.rmtree(rundir / sub, ignore_errors=True)
    return row


# ----------------------------------------------------------------------------
# Summary: medians over repetitions, then the comparisons
# ----------------------------------------------------------------------------
def med(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return statistics.median(vals) if vals else None


def spread_pct(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    if len(vals) < 2 or statistics.median(vals) == 0:
        return None
    return 100 * (max(vals) - min(vals)) / statistics.median(vals)


def div(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def sub(a, b):
    return a - b if a is not None and b is not None else None


def summarise(rows, args):
    out = []
    ok = [r for r in rows if r["status"] == "ok"]
    for size in args.sizes:
        for sync in sorted({r["sync"] for r in ok if r["variant"] != "none"}):
            g = {v: [r for r in ok if r["size"] == size and r["variant"] == v
                     and (v == "none" or r["sync"] == sync)] for v in args.variants}
            snap, sl, none = g.get("snap", []), g.get("slice", []), g.get("none", [])
            n_out = med(snap, "snapshot_count") or med(sl, "slice_count")
            s = {"config": args.labels[size], "size": size, "sync": sync,
                 "nodes": med(snap or sl or none, "nodes"),
                 "outputs": n_out, "reps": args.reps}
            # disk space
            s["snap_bytes_per_output"] = div(med(snap, "snapshot_bytes"), n_out)
            s["slice_bytes_per_output"] = div(med(sl, "slice_bytes"), n_out)
            s["disk_ratio"] = div(s["snap_bytes_per_output"], s["slice_bytes_per_output"])
            s["snap_header_bytes"] = med(snap, "header_bytes")
            # I/O and work in the simulation, per output
            s["snap_write_per_output_s"] = div(med(snap, "snapshot_write_s"), n_out)
            s["slice_extract_per_output_s"] = div(med(sl, "slice_extract_s"), n_out)
            s["slice_write_per_output_s"] = div(med(sl, "slice_write_s"), n_out)
            s["slice_total_per_output_s"] = (
                None if s["slice_extract_per_output_s"] is None
                else s["slice_extract_per_output_s"] + (s["slice_write_per_output_s"] or 0))
            s["write_time_ratio"] = div(s["snap_write_per_output_s"],
                                        s["slice_total_per_output_s"])
            s["snap_write_spread_pct"] = spread_pct(snap, "snapshot_write_s")
            s["slice_write_spread_pct"] = spread_pct(sl, "slice_write_s")
            # computation: solver time per step
            s["step_none_s"] = div(med(none, "compute_s"), med(none, "compute_steps"))
            for name, grp in (("snap", snap), ("slice", sl)):
                after, n_after = med(grp, "compute_after_output_s"), med(grp, "compute_after_output_steps")
                tot, n_tot = med(grp, "compute_s"), med(grp, "compute_steps")
                s[f"step_after_output_{name}_s"] = div(after, n_after)
                s[f"step_other_{name}_s"] = div(sub(tot, after),
                                                sub(n_tot, n_after) if n_tot else None)
            # whole time loop
            for name, grp in (("none", none), ("snap", snap), ("slice", sl)):
                s[f"loop_{name}_s"] = med(grp, "time_loop_s")
                s[f"loop_{name}_spread_pct"] = spread_pct(grp, "time_loop_s")
            s["overhead_snap_s"] = sub(s["loop_snap_s"], s["loop_none_s"])
            s["overhead_slice_s"] = sub(s["loop_slice_s"], s["loop_none_s"])
            s["sim_gain_s"] = sub(s["loop_snap_s"], s["loop_slice_s"])
            s["sim_gain_pct"] = (None if s["sim_gain_s"] is None
                                 else 100 * s["sim_gain_s"] / s["loop_snap_s"])
            # post-processing (all files, warm passes)
            s["post_volume_s"] = med(snap, "visu_warm_total_s")
            s["post_slice_s"] = med(sl, "visu_warm_total_s")
            s["post_volume_pass1_s"] = med(snap, "visu_pass1_total_s")
            s["post_slice_pass1_s"] = med(sl, "visu_pass1_total_s")
            for k in ("read", "cut", "stats"):
                s[f"post_volume_{k}_per_file_s"] = med(snap, f"visu_{k}_per_file_s")
                s[f"post_slice_{k}_per_file_s"] = med(sl, f"visu_{k}_per_file_s")
            # whole workflow: simulation loop + post-processing of every output
            s["workflow_snap_s"] = (None if None in (s["loop_snap_s"], s["post_volume_s"])
                                    else s["loop_snap_s"] + s["post_volume_s"])
            s["workflow_slice_s"] = (None if None in (s["loop_slice_s"], s["post_slice_s"])
                                     else s["loop_slice_s"] + s["post_slice_s"])
            s["workflow_gain_s"] = sub(s["workflow_snap_s"], s["workflow_slice_s"])
            out.append(s)
    return out


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    if unit == "B":
        for u in ("B", "kB", "MB", "GB"):
            if abs(v) < 1000 or u == "GB":
                return f"{v:.3g} {u}"
            v /= 1000
    if unit == "s":
        return f"{v * 1e3:.3g} ms" if abs(v) < 1 else f"{v:.3g} s"
    if unit == "%":
        return f"{v:.2g} %"
    return f"{v:.3g}"


def write_summary(summary, args):
    if not summary:
        return
    keys = list(summary[0])
    with open(args.outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    lines = ["# Outputs: none vs full snapshots vs z-slices", "",
             f"Medians over {args.reps} repetition(s). Snapshot format: {args.snap_format}. "
             f"Post-processing timed on files just written (warm OS cache, passes >= 2).", ""]
    rows = [
        ("nodes / outputs per run", lambda s: f"{fmt(s['nodes'])} / {fmt(s['outputs'])}"),
        ("DISK bytes per output: snapshot | slice",
         lambda s: f"{fmt(s['snap_bytes_per_output'], 'B')} | {fmt(s['slice_bytes_per_output'], 'B')}"),
        ("DISK ratio snapshot / slice", lambda s: fmt(s["disk_ratio"])),
        ("I/O write per output: snapshot | slice (extract + write)",
         lambda s: f"{fmt(s['snap_write_per_output_s'], 's')} | "
                   f"{fmt(s['slice_extract_per_output_s'], 's')} + {fmt(s['slice_write_per_output_s'], 's')}"),
        ("I/O write-time ratio snapshot / slice", lambda s: fmt(s["write_time_ratio"])),
        ("I/O write time spread (max-min)/median: snapshot | slice",
         lambda s: f"{fmt(s['snap_write_spread_pct'], '%')} | {fmt(s['slice_write_spread_pct'], '%')}"),
        ("COMPUTE step, no output run", lambda s: fmt(s["step_none_s"], "s")),
        ("COMPUTE step after an output: snap | slice",
         lambda s: f"{fmt(s['step_after_output_snap_s'], 's')} | {fmt(s['step_after_output_slice_s'], 's')}"),
        ("COMPUTE other steps: snap | slice",
         lambda s: f"{fmt(s['step_other_snap_s'], 's')} | {fmt(s['step_other_slice_s'], 's')}"),
        ("time loop: none | snap | slice",
         lambda s: f"{fmt(s['loop_none_s'], 's')} | {fmt(s['loop_snap_s'], 's')} | {fmt(s['loop_slice_s'], 's')}"),
        ("time loop spread (max-min)/median: none | snap | slice",
         lambda s: " | ".join(fmt(s[f"loop_{v}_spread_pct"], "%") for v in ("none", "snap", "slice"))),
        ("output overhead on the loop: snap | slice",
         lambda s: f"{fmt(s['overhead_snap_s'], 's')} | {fmt(s['overhead_slice_s'], 's')}"),
        ("SIMULATION gain of slices", lambda s: f"{fmt(s['sim_gain_s'], 's')} ({fmt(s['sim_gain_pct'], '%')})"),
        ("POST per file read / cut / stats: volume",
         lambda s: " / ".join(fmt(s[f"post_volume_{k}_per_file_s"], "s") for k in ("read", "cut", "stats"))),
        ("POST per file read / stats: slice",
         lambda s: " / ".join(fmt(s[f"post_slice_{k}_per_file_s"], "s") for k in ("read", "stats"))),
        ("POST all files: volume | slice",
         lambda s: f"{fmt(s['post_volume_s'], 's')} | {fmt(s['post_slice_s'], 's')}"),
        ("WORKFLOW loop + post: snap | slice",
         lambda s: f"{fmt(s['workflow_snap_s'], 's')} | {fmt(s['workflow_slice_s'], 's')}"),
        ("WORKFLOW gain of slices", lambda s: fmt(s["workflow_gain_s"], "s")),
    ]
    head = ["quantity"] + [(f"size {s['size']}" if s["config"] == str(s["size"])
                            else f"{s['config']} ({s['size']})")
                           + (" sync" if s["sync"] else "") for s in summary]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for label, f in rows:
        cells = [label] + [f(s) for s in summary]
        lines.append("| " + " | ".join(c.replace(" | ", " / ") for c in cells) + " |")
    text = "\n".join(lines) + "\n"
    (args.outdir / "summary.md").write_text(text)
    print("\n" + text)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True,
                    help="where runs and results go; its filesystem is what the write "
                         "times measure (NFS, local disk, tmpfs = RAM)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                    help="reference configurations to run: "
                         + ", ".join(f"{k} = {v} elements per direction" for k, v in CONFIGS.items())
                         + " (default: all)")
    ap.add_argument("--sizes", type=int, nargs="+",
                    help="other sizes instead of --configs: elements per direction, cubic domain")
    ap.add_argument("--reps", type=int, default=3, help="repetitions (default: %(default)s)")
    ap.add_argument("--variants", nargs="+", default=["none", "snap", "slice"],
                    choices=["none", "snap", "slice", "both"])
    ap.add_argument("--snap-format", default="bin", choices=list(SNAP_FORMATS))
    ap.add_argument("--sync", default="off", choices=["off", "on", "both"],
                    help="--sync-writes for the output variants (both: run each twice)")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--timemax", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--interval", type=int, default=100,
                    help="steps between outputs (default: %(default)s, i.e. 15 outputs)")
    ap.add_argument("--slice-z", type=float, help="slice z (default: the proxy's, source z)")
    ap.add_argument("--receivers", type=Path, help="--receiver-positions-file for the proxy")
    ap.add_argument("--visu-repeat", type=int, default=3,
                    help="viewer --benchmark passes per run (default: %(default)s)")
    ap.add_argument("--no-visu", action="store_true", help="skip the post-processing timing")
    ap.add_argument("--keep-files", action="store_true",
                    help="keep snapshot and slice files (default: deleted after each run)")
    ap.add_argument("--max-gb", type=float, default=20.0,
                    help="refuse to start if the predicted disk use exceeds this (default: %(default)s)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up run")
    ap.add_argument("--semproxy", type=Path, default=HERE / "build" / "bin" / "semproxy")
    ap.add_argument("--viewer", type=Path, default=HERE / "visualise_snapshots.py")
    ap.add_argument("--python", default=sys.executable, help="python running the viewer")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the commands only")
    ap.add_argument("extra", nargs="*", help="extra proxy options, after --")
    args = ap.parse_args()
    # absolute paths: the proxy runs with its run folder as working directory,
    # so a relative path given here would no longer point to the right file
    args.outdir = args.outdir.expanduser().resolve()
    args.semproxy = args.semproxy.expanduser().resolve()
    args.viewer = args.viewer.expanduser().resolve()
    if args.receivers:
        args.receivers = args.receivers.expanduser().resolve()
        if not args.receivers.is_file():
            sys.exit(f"no receiver file at {args.receivers}")

    if args.sizes is None:
        args.sizes = [CONFIGS[c] for c in dict.fromkeys(args.configs)]
    by_size = {v: k for k, v in CONFIGS.items()}
    reference = (args.order, args.timemax, args.dt, args.interval) == (2, 1.5, 0.001, 100)
    # a size is only called S/M/L if the rest of the configuration is the reference one
    args.labels = {s: by_size[s] if s in by_size and reference else str(s) for s in args.sizes}

    runs = plan_runs(args)
    pred = {size: predict(size, args) for size in args.sizes}
    per_run = [pred[r["size"]]["bytes"][r["variant"]] for r in runs]
    peak = sum(per_run) if args.keep_files else max(per_run)
    print(f"{len(runs)} runs; outputs per run: "
          + ", ".join(f"size {s}: {p['outputs']}" for s, p in pred.items()))
    if not reference:
        print("NOTE: order/timemax/dt/interval differ from the reference configuration, "
              "so the sizes are not labelled S/M/L")
    for s, p in pred.items():
        print(f"  {args.labels[s] + ' = ' if args.labels[s] != str(s) else ''}"
              f"size {s}: {p['nodes']:,} nodes, per run snapshots {fmt(p['snap_bytes'], 'B')}"
              f"{' (estimate)' if args.snap_format.startswith('csv') else ''}, "
              f"slices {fmt(p['slice_bytes'], 'B')}")
    print(f"predicted peak disk use: {fmt(peak, 'B')} "
          f"({'all runs kept' if args.keep_files else 'largest single run, files deleted after each'})")

    if args.dry_run:
        for r in runs:
            print(" ".join(proxy_command(r, args.outdir / "runs" / r["name"], args)))
        return
    if peak > args.max_gb * 1e9:
        sys.exit(f"predicted {fmt(peak, 'B')} > --max-gb {args.max_gb}: reduce sizes, raise "
                 "--interval, or raise --max-gb if you checked your quota")
    if not args.semproxy.exists():
        sys.exit(f"no proxy binary at {args.semproxy} (use --semproxy)")
    if not args.no_visu:
        if not args.viewer.exists():
            sys.exit(f"no viewer at {args.viewer} (use --viewer or --no-visu)")
        # the viewer runs with args.python: check it has numpy and VTK before
        # spending the whole campaign on runs whose post-processing then fails
        probe = subprocess.run([args.python, "-c", "import numpy, vtkmodules.vtkFiltersCore"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            sys.exit(f"{args.python} cannot import numpy and vtk, so the viewer would fail:\n"
                     f"{probe.stderr.strip().splitlines()[-1]}\n"
                     "activate the venv (source ~/espaces/travail/venvs/funtides-visu/bin/activate) and run "
                     "this script with its python, or pass --python /path/to/venv/bin/python")
    if np is None:
        print("WARNING: numpy not importable here, the NaN/inf check of outputs is skipped")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fs = record_env(args, {"runs": runs, "predicted": pred, "args": {
        k: str(v) for k, v in vars(args).items()}})
    print(f"output filesystem: {fs or 'unknown'}"
          + ("  WARNING: tmpfs lives in RAM, write times will not reflect a disk" if fs == "tmpfs" else ""))

    if not args.no_warmup:  # first GPU run of a session pays one-off costs
        warm = args.outdir / "warmup"
        warm.mkdir(exist_ok=True)
        subprocess.run([str(args.semproxy), "--ex", "10", "--ey", "10", "--ez", "10",
                        "--timemax", "0.01", "--seismograms-folder", str(warm / "seismos"),
                        "--metrics-file", str(warm / "metrics.json")],
                       cwd=warm, capture_output=True)

    rows, slice_z_by_size = [], {}
    fieldnames = None
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['name']} ...", end=" ", flush=True)
        row = do_run(run, args, slice_z_by_size)
        rows.append(row)
        print(f"{row['status']}, loop {fmt(row.get('time_loop_s') or None, 's')}"
              + ("" if row.get("finite") in ("", True) else "  WARNING: NaN/inf in last output"))
        # rewrite runs.csv after every run, so an interrupted campaign keeps its data
        fieldnames = list(dict.fromkeys((fieldnames or []) + list(row)))
        with open(args.outdir / "runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_summary(summarise(rows, args), args)
    print(f"results in {args.outdir}")


if __name__ == "__main__":
    main()