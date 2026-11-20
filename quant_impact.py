#!/usr/bin/env python3
"""Error of the quantization and its impact on the snapshot analyses, offline.

Works on the raw snapshots of one run (snap-<step>.dat + headers.dat, written
with --use-snapshots --snapshots-use-headers): every snapshot is quantized and
reconstructed with the reference encoder of quant_tools.py (same arithmetic as
the proxy, checked byte for byte with "quant_tools.py check"), for every kind
(uint2 .. uint16, float16) and every bounds mode (snapshot, symmetric, fixed
with A = max|p| over the snapshots of the run). Deterministic: the original
and the reconstruction are the same field of the same run.

  run      RAW_DIR --config S --out impact_S.csv
           one row per (step, bounds, kind) with, for the original (o_*) and
           the reconstruction (r_*):
             volume   min, max, mean, std, q1, q2, q3 (sorted values at the
                      indices (n-1)/4 * 1, 2, 3, as the in-situ quartiles of
                      TP2), absmax = max|p|, energy = sum p^2, node of max|p|
             layer    the same on the z node layer nearest to the source z
                      (lowest on a tie: the layer of the z-slices), plus the
                      wavefront radius: largest distance to the source, in the
                      layer, of a node with |p| >= 1 % of the layer's max|p|
           and the errors rmse, maxerr (volume and layer).
  summary  CSV... --outdir DIR
           impact_worst.csv       normalised deviations, worst over the
                                  snapshots (step 0 apart), per config / bounds / kind
           impact_thresholds.csv  smallest N from which each analysis stays
                                  within 1 %, 0.1 %, and exact
           impact_summary.md      the same as tables

Normalisation of the deviations (no point-wise relative error):
  min, max, mean, quartiles, absmax   |r - o| / max|p| of the snapshot (or layer)
  std, energy                         |r - o| / o
  rmse, maxerr                        / max|p| of the snapshot (or layer)
  node of max|p|                      distance in m between the two nodes
  wavefront radius                    |r - o| / o  (and in m)
"""

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_tools as qt  # noqa: E402

STATS = ["min", "max", "mean", "std", "q1", "q2", "q3", "absmax", "energy"]
FRONT_FRACTION = 0.01


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def quartile_indices(n):
    key = (n - 1) // 4  # as SEMproxy::computeInSituQuart
    return [key, 2 * key, 3 * key]


def direct_stats(a):
    """Statistics of a float32 array, sums in float64."""
    a = a.ravel()
    a64 = a.astype(np.float64)
    part = np.partition(a, quartile_indices(a.size))
    q = [float(part[i]) for i in quartile_indices(a.size)]
    mn, mx = float(a.min()), float(a.max())
    return {"min": mn, "max": mx, "mean": float(a64.mean()), "std": float(a64.std()),
            "q1": q[0], "q2": q[1], "q3": q[2], "absmax": max(abs(mn), abs(mx)),
            "energy": float(np.dot(a64, a64))}


def hist_stats(levels_q, table):
    """Same statistics for a reconstruction given as levels (uint) and the
    table level -> float32 value: exact, from the histogram of the levels."""
    cnt = np.bincount(levels_q, minlength=table.size).astype(np.float64)
    n = levels_q.size
    used = np.flatnonzero(cnt)
    t = table.astype(np.float64)
    mn, mx = float(t[used].min()), float(t[used].max())
    mean = float(np.dot(cnt, t) / n)
    var = float(np.dot(cnt, (t - mean) ** 2) / n)
    # order statistics: the table is non-decreasing in the level
    cum = np.cumsum(cnt)
    q = [float(t[np.searchsorted(cum, i, side="right")]) for i in quartile_indices(n)]
    return {"min": mn, "max": mx, "mean": mean, "std": var ** 0.5,
            "q1": q[0], "q2": q[1], "q3": q[2], "absmax": max(abs(mn), abs(mx)),
            "energy": float(np.dot(cnt, t * t))}


class Grid:
    """Axes of the structured grid, from headers.dat (x fastest, then y, z)."""

    def __init__(self, raw_dir, n, source):
        nx = round(n ** (1 / 3))
        if nx ** 3 != n:
            raise SystemExit(f"{n} nodes is not a cube")
        c = np.memmap(Path(raw_dir) / "headers.dat", dtype="<f4", mode="r").reshape(-1, 3)
        if len(c) != n:
            raise SystemExit("headers.dat does not match the snapshots")
        self.n = nx
        self.x = np.array(c[:nx, 0], dtype=np.float64)
        self.y = np.array(c[::nx][:nx, 1], dtype=np.float64)
        self.z = np.array(c[::nx * nx, 2], dtype=np.float64)
        self.source = source
        # layer of the z-slices: nearest to the source z, lowest on a tie
        self.k = int(np.argmin(np.abs(self.z - float(np.float32(source[2])))))
        self.r_layer = np.hypot(self.x[None, :] - source[0], self.y[:, None] - source[1])

    def layer(self, a):
        return a.reshape(self.n, self.n, self.n)[self.k]

    def coords(self, node):
        k, j, i = np.unravel_index(node, (self.n, self.n, self.n))
        return np.array([self.x[i], self.y[j], self.z[k]])

    def source_dist(self, node):
        """Distance to the source of a node of the volume."""
        return float(np.linalg.norm(self.coords(node) - np.asarray(self.source)))

    def front_radius(self, layer):
        a = np.abs(layer)
        m = a.max()
        if m == 0:
            return 0.0
        return float(self.r_layer[a >= FRONT_FRACTION * m].max())


def errors(o, r):
    d = r.astype(np.float64) - o
    return float(np.sqrt(np.mean(d * d))), float(np.abs(d).max())


# ----------------------------------------------------------------------------
# One snapshot: every kind and bounds mode
# ----------------------------------------------------------------------------
G = {}  # worker globals


def init_worker(raw_dir, n, source, kinds, bounds, amax, config):
    G.update(grid=Grid(raw_dir, n, source), kinds=kinds, bounds=bounds, amax=amax,
             config=config, n=n)


def analyse_snapshot(job):
    step, path = job
    grid = G["grid"]
    p = qt.read_raw(path, G["n"])
    p64 = p.astype(np.float64)
    pl = np.ascontiguousarray(grid.layer(p))
    pl64 = pl.astype(np.float64)
    o = direct_stats(p)
    ol = direct_stats(pl)
    o_node = int(np.argmax(np.abs(p)))
    ol_node = int(np.argmax(np.abs(pl)))
    o_front = grid.front_radius(pl)
    rows = []
    for bounds in G["bounds"]:
        for kind in G["kinds"]:
            h, stored = qt.quantize(p, kind, bounds, G["amax"], step)
            if h["kind"] == qt.KIND_UINT:
                table = qt.dequantize(h, np.arange(h["levels"], dtype=np.uint32))
                rec = table[stored]
                r = hist_stats(stored, table)
            else:
                rec = qt.dequantize(h, stored)
                r = direct_stats(rec)
            rl = np.ascontiguousarray(grid.layer(rec))
            rls = direct_stats(rl)
            rmse, maxerr = errors(p64, rec)
            l_rmse, l_maxerr = errors(pl64, rl)
            r_node = int(np.argmax(np.abs(rec)))
            rl_node = int(np.argmax(np.abs(rl)))
            row = {"config": G["config"], "step": step, "bounds": bounds, "kind": kind,
                   "bits": h["bits"], "family": "float" if h["kind"] == qt.KIND_FLOAT else "uint",
                   "nodes": p.size, "layer_k": grid.k, "layer_z": grid.z[grid.k],
                   "vmin": float(h["vmin"]), "vmax": float(h["vmax"]), "levels": h["levels"],
                   "rmse": rmse, "maxerr": maxerr, "l_rmse": l_rmse, "l_maxerr": l_maxerr}
            for s in STATS:
                row["o_" + s], row["r_" + s] = o[s], r[s]
                row["ol_" + s], row["rl_" + s] = ol[s], rls[s]
            row["argmax_dist_m"] = float(np.linalg.norm(grid.coords(r_node) - grid.coords(o_node)))
            nn = grid.n
            row["l_argmax_dist_m"] = float(np.hypot(
                grid.x[rl_node % nn] - grid.x[ol_node % nn],
                grid.y[rl_node // nn] - grid.y[ol_node // nn]))
            # distance to the source of the max|p| node: unlike its position, it
            # does not depend on which of the (near-)equal nodes of the front wins
            row["o_argmax_r_m"], row["r_argmax_r_m"] = (grid.source_dist(o_node),
                                                        grid.source_dist(r_node))
            row["ol_argmax_r_m"] = float(grid.r_layer.ravel()[ol_node])
            row["rl_argmax_r_m"] = float(grid.r_layer.ravel()[rl_node])
            row["o_front_m"], row["r_front_m"] = o_front, grid.front_radius(rl)
            rows.append(row)
    return rows


def cmd_run(args):
    raw = qt.find_raw(args.raw_dir)
    steps = sorted(raw)
    n = raw[steps[0]].stat().st_size // 4
    # "global" bound of the fixed mode: max|p| over the snapshots of the run
    amax = 0.0
    for s in steps:
        a = np.fromfile(raw[s], dtype="<f4")
        amax = max(amax, float(np.abs(a).max()))
    kinds = [f"uint{b}" for b in range(args.min_bits, args.max_bits + 1)] + ["float16"]
    print(f"{args.config}: {len(steps)} snapshots of {n} nodes, run max|p| = {amax:.6g}, "
          f"{len(kinds)} kinds x {len(args.bounds)} bounds modes, {args.jobs} processes",
          flush=True)
    t0 = time.perf_counter()
    with Pool(args.jobs, init_worker,
              (args.raw_dir, n, args.source, kinds, args.bounds, amax, args.config)) as pool:
        rows = []
        for res in pool.imap(analyse_snapshot, [(s, raw[s]) for s in steps]):
            rows += res
            print(f"  step {res[0]['step']} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    for r in rows:
        r["run_absmax"] = amax
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {args.out} ({len(rows)} rows)")


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
# (name, description) of the normalised deviations
DEVS = [("rmse", "RMSE / max|p|"), ("maxerr", "max abs error / max|p|"),
        ("min", "min"), ("max", "max"), ("absmax", "max|p|"), ("mean", "mean"),
        ("q1", "1st quartile"), ("q2", "median"), ("q3", "3rd quartile"),
        ("std", "std (relative)"), ("energy", "energy (relative)"),
        ("argmax", "node of max|p| (distance / 2000 m)"),
        ("argmax_r", "radius of the max|p| node (/ 2000 m)"),
        ("l_rmse", "layer: RMSE / max|p|"), ("l_maxerr", "layer: max abs error / max|p|"),
        ("l_min", "layer: min"), ("l_max", "layer: max"), ("l_absmax", "layer: max|p|"),
        ("l_mean", "layer: mean"), ("l_q1", "layer: 1st quartile"), ("l_q2", "layer: median"),
        ("l_q3", "layer: 3rd quartile"), ("l_std", "layer: std (relative)"),
        ("l_energy", "layer: energy (relative)"),
        ("l_argmax", "layer: node of max|p| (distance / 2000 m)"),
        ("l_argmax_r", "layer: radius of the max|p| node (/ 2000 m)"),
        ("front", "layer: wavefront radius (relative)")]
DOMAIN = 2000.0


def deviations(r):
    """Normalised deviations of one CSV row."""
    f = lambda k: float(r[k])  # noqa: E731
    d = {}
    for pre, scope in (("", ""), ("l", "l_")):
        o, rr = "o" + pre + "_", "r" + pre + "_"
        amax = f(o + "absmax")
        norm = amax if amax > 0 else np.nan
        d[scope + "rmse"] = f(scope + "rmse") / norm
        d[scope + "maxerr"] = f(scope + "maxerr") / norm
        for s in ("min", "max", "absmax", "mean", "q1", "q2", "q3"):
            d[scope + s] = abs(f(rr + s) - f(o + s)) / norm
        for s in ("std", "energy"):
            d[scope + s] = abs(f(rr + s) - f(o + s)) / f(o + s) if f(o + s) > 0 else np.nan
        d[scope + "argmax"] = f(scope + "argmax_dist_m") / DOMAIN
        d[scope + "argmax_r"] = abs(f(rr + "argmax_r_m") - f(o + "argmax_r_m")) / DOMAIN
        d[scope + "argmax_r_m"] = abs(f(rr + "argmax_r_m") - f(o + "argmax_r_m"))
    d["front"] = abs(f("r_front_m") - f("o_front_m")) / f("o_front_m") if f("o_front_m") > 0 else np.nan
    d["front_m"] = abs(f("r_front_m") - f("o_front_m"))
    d["argmax_m"] = f("argmax_dist_m")
    d["l_argmax_m"] = f("l_argmax_dist_m")
    return d


def nanmax(v):
    v = [x for x in v if not np.isnan(x)]
    return max(v) if v else float("nan")


def threshold(by_bits, tol):
    """Smallest N such that every N' >= N is within tol; '>max' if none."""
    bits = sorted(by_bits)
    ok_from = None
    for b in reversed(bits):
        if by_bits[b] <= tol:
            ok_from = b
        else:
            break
    if ok_from is None:
        return f">{bits[-1]}"
    return f"<={ok_from}" if ok_from == bits[0] else str(ok_from)


def g3(v):
    return "nan" if np.isnan(v) else ("0" if v == 0 else f"{v:.2e}")


def cmd_summary(args):
    rows = []
    for path in args.csv:
        rows += list(csv.DictReader(open(path)))
    for r in rows:
        r["dev"] = deviations(r)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    configs = list(dict.fromkeys(r["config"] for r in rows))
    bounds_modes = list(dict.fromkeys(r["bounds"] for r in rows))
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    names = [n for n, _ in DEVS] + ["front_m", "argmax_m", "l_argmax_m", "argmax_r_m",
                                     "l_argmax_r_m"]

    worst = []
    for c in configs:
        for b in bounds_modes:
            for k in kinds:
                sel = [r for r in rows if (r["config"], r["bounds"], r["kind"]) == (c, b, k)]
                for label, part in (("steps>0", [r for r in sel if int(r["step"]) > 0]),
                                    ("step0", [r for r in sel if int(r["step"]) == 0])):
                    if not part:
                        continue
                    w = {"config": c, "bounds": b, "kind": k, "bits": int(sel[0]["bits"]),
                         "family": sel[0]["family"], "snapshots": label, "count": len(part)}
                    for n in names:
                        w[n] = nanmax([r["dev"][n] for r in part])
                        if n in ("rmse", "maxerr", "front", "l_maxerr", "energy"):
                            vals = [(r["dev"][n], int(r["step"])) for r in part
                                    if not np.isnan(r["dev"][n])]
                            w[n + "_step"] = max(vals)[1] if vals else ""
                    worst.append(w)
    with open(outdir / "impact_worst.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(worst[0]))
        w.writeheader()
        w.writerows(worst)

    thr = []
    for c in configs:
        for b in bounds_modes:
            for label in ("steps>0", "step0"):
                sel = [w for w in worst if (w["config"], w["bounds"], w["snapshots"]) == (c, b, label)]
                if not sel:
                    continue
                ui = {w["bits"]: w for w in sel if w["family"] == "uint"}
                fl = [w for w in sel if w["family"] == "float"]
                for n, desc in DEVS:
                    t = {"config": c, "bounds": b, "snapshots": label, "analysis": n,
                         "description": desc,
                         "bits_1pct": threshold({k: v[n] for k, v in ui.items()}, 1e-2),
                         "bits_0.1pct": threshold({k: v[n] for k, v in ui.items()}, 1e-3),
                         "bits_exact": threshold({k: v[n] for k, v in ui.items()}, 0.0),
                         "uint8": ui[8][n] if 8 in ui else "", "uint12": ui[12][n] if 12 in ui else "",
                         "uint16": ui[16][n] if 16 in ui else "",
                         "float16": fl[0][n] if fl else ""}
                    thr.append(t)
    with open(outdir / "impact_thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(thr[0]))
        w.writeheader()
        w.writerows(thr)

    md = ["# Impact of the quantization on the snapshot analyses", "",
          "Deviation of each analysis between the original and the reconstructed snapshot, "
          "worst over the snapshots of the run. `bits` columns: smallest N such that uint<N'> "
          "stays within the tolerance for every N' >= N (N in "
          f"{min(int(r['bits']) for r in rows if r['family'] == 'uint')}.."
          f"{max(int(r['bits']) for r in rows if r['family'] == 'uint')}).", ""]
    for label in ("steps>0", "step0"):
        md += [f"## Bit thresholds, {label}", ""]
        for tolname, key in (("1 %", "bits_1pct"), ("0.1 %", "bits_0.1pct"), ("exact", "bits_exact")):
            head = ["analysis"] + [f"{c} {b}" for c in configs for b in bounds_modes]
            md += [f"### within {tolname}", "", "| " + " | ".join(head) + " |",
                   "|" + "---|" * len(head)]
            for n, desc in DEVS:
                cells = [desc]
                for c in configs:
                    for b in bounds_modes:
                        hit = [t for t in thr if (t["config"], t["bounds"], t["snapshots"],
                                                  t["analysis"]) == (c, b, label, n)]
                        cells.append(hit[0][key] if hit else "n/a")
                md.append("| " + " | ".join(cells) + " |")
            md.append("")
    for c in configs:
        for b in bounds_modes:
            md += [f"## Errors and deviations per kind: {c}, bounds {b}, worst over steps > 0", "",
                   "| kind | RMSE/max | maxerr/max | min | max | mean | Q1 | median | Q3 | std rel | "
                   "energy rel | argmax m | argmax radius m | layer RMSE/max | layer maxerr/max | "
                   "layer std rel | layer energy rel | layer argmax m | layer argmax radius m | "
                   "front rel | front m |",
                   "|" + "---|" * 21]
            for w in worst:
                if (w["config"], w["bounds"], w["snapshots"]) != (c, b, "steps>0"):
                    continue
                md.append("| " + " | ".join([w["kind"]] + [g3(w[n]) for n in (
                    "rmse", "maxerr", "min", "max", "mean", "q1", "q2", "q3", "std", "energy",
                    "argmax_m", "argmax_r_m", "l_rmse", "l_maxerr", "l_std", "l_energy",
                    "l_argmax_m", "l_argmax_r_m", "front", "front_m")]) + " |")
            md.append("")
    (outdir / "impact_summary.md").write_text("\n".join(md) + "\n")
    print(f"written {outdir}/impact_worst.csv, impact_thresholds.csv, impact_summary.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run", help="quantize / reconstruct / analyse the raw snapshots of a run")
    s.add_argument("raw_dir")
    s.add_argument("--config", required=True, help="label of the configuration (S, M, L)")
    s.add_argument("--out", required=True, help="CSV file to write")
    s.add_argument("--min-bits", type=int, default=2)
    s.add_argument("--max-bits", type=int, default=16)
    s.add_argument("--bounds", nargs="+", default=["snapshot", "symmetric", "fixed"])
    s.add_argument("--source", type=float, nargs=3, default=[1010.0, 1010.0, 1010.0])
    s.add_argument("--jobs", type=int, default=8)
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("summary", help="worst deviations and bit thresholds")
    s.add_argument("csv", nargs="+")
    s.add_argument("--outdir", required=True)
    s.set_defaults(func=cmd_summary)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
