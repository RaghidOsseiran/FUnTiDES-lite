#!/usr/bin/env python3
"""Compact tables for the report from results/impact_worst.csv and impact_thresholds.csv."""
import csv, sys
from pathlib import Path
d = Path(sys.argv[1])
worst = list(csv.DictReader(open(d / "impact_worst.csv")))
thr = list(csv.DictReader(open(d / "impact_thresholds.csv")))
g = lambda v: "nan" if v in ("", "nan") else ("0" if float(v) == 0 else f"{float(v):.1e}")
out = []
out.append("## Thresholds (smallest N from which uint<N'> stays within the tolerance for all N' >= N), steps > 0\n")
anal = list(dict.fromkeys(t["analysis"] for t in thr))
cols = [(c, b) for c in "SML" for b in ("snapshot", "symmetric", "fixed")]
for key, name in (("bits_1pct", "1 %"), ("bits_0.1pct", "0.1 %")):
    out.append(f"### within {name}\n")
    out.append("| analysis | " + " | ".join(f"{c} {b[:4]}" for c, b in cols) + " |")
    out.append("|" + "---|" * (len(cols) + 1))
    for a in anal:
        cells = []
        for c, b in cols:
            hit = [t for t in thr if (t["config"], t["bounds"], t["snapshots"], t["analysis"]) == (c, b, "steps>0", a)]
            cells.append(hit[0][key] if hit else "n/a")
        desc = [t["description"] for t in thr if t["analysis"] == a][0]
        out.append(f"| {desc} | " + " | ".join(cells) + " |")
    out.append("")
out.append("## float16 worst deviation (steps > 0), per config (bounds snapshot = plain conversion)\n")
out.append("| analysis | S | M | L |")
out.append("|---|---|---|---|")
for a in anal:
    cells = []
    for c in "SML":
        hit = [t for t in thr if (t["config"], t["bounds"], t["snapshots"], t["analysis"]) == (c, "snapshot", "steps>0", a)]
        cells.append(g(hit[0]["float16"]) if hit else "n/a")
    desc = [t["description"] for t in thr if t["analysis"] == a][0]
    out.append(f"| {desc} | " + " | ".join(cells) + " |")
out.append("")
out.append("## Error per N (worst over steps > 0): RMSE/max|p| and max abs error/max|p|\n")
out.append("| kind | " + " | ".join(f"{c} {b[:4]} RMSE | {c} {b[:4]} maxerr" for c, b in cols) + " |")
out.append("|" + "---|" * (2 * len(cols) + 1))
kinds = list(dict.fromkeys(w["kind"] for w in worst))
for k in kinds:
    cells = []
    for c, b in cols:
        hit = [w for w in worst if (w["config"], w["bounds"], w["kind"], w["snapshots"]) == (c, b, k, "steps>0")]
        cells += [g(hit[0]["rmse"]), g(hit[0]["maxerr"])] if hit else ["n/a", "n/a"]
    out.append(f"| {k} | " + " | ".join(cells) + " |")
out.append("")
out.append("## Step 0 only: RMSE/max|p| and max abs error/max|p|\n")
out.append("| kind | " + " | ".join(f"{c} {b[:4]} RMSE | {c} {b[:4]} maxerr" for c, b in cols) + " |")
out.append("|" + "---|" * (2 * len(cols) + 1))
for k in kinds:
    cells = []
    for c, b in cols:
        hit = [w for w in worst if (w["config"], w["bounds"], w["kind"], w["snapshots"]) == (c, b, k, "step0")]
        cells += [g(hit[0]["rmse"]), g(hit[0]["maxerr"])] if hit else ["n/a", "n/a"]
    out.append(f"| {k} | " + " | ".join(cells) + " |")
out.append("")
sel = ["uint6", "uint8", "uint10", "uint12", "uint14", "uint16", "float16"]
names = [("mean", "mean"), ("q2", "median"), ("std", "std rel"), ("energy", "energy rel"),
         ("argmax_m", "argmax m"), ("argmax_r_m", "argmax radius m"), ("l_absmax", "layer max|p|"),
         ("l_std", "layer std rel"), ("l_energy", "layer energy rel"), ("l_argmax_r_m", "layer argmax radius m"),
         ("front_m", "front m")]
for c in "SML":
    for b in ("snapshot", "symmetric"):
        out.append(f"## Analyses, {c}, bounds {b}, worst over steps > 0\n")
        out.append("| kind | " + " | ".join(n for _, n in names) + " |")
        out.append("|" + "---|" * (len(names) + 1))
        for k in sel:
            hit = [w for w in worst if (w["config"], w["bounds"], w["kind"], w["snapshots"]) == (c, b, k, "steps>0")]
            if hit:
                out.append(f"| {k} | " + " | ".join(g(hit[0][n]) for n, _ in names) + " |")
        out.append("")
(d / "impact_report_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
