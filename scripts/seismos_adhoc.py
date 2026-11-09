#!/usr/bin/env python3

from pathlib import Path

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

parser = argparse.ArgumentParser(description= "Adhoc analysis of seismograms recording")
parser.add_argument("-if", "--input-file", type=str, required=True, help="Recording file to analyse")
parser.add_argument("--header-file", default=False, action='store_true', required=False, help="If the recording use a separate header file for the coordinates")
parser.add_argument("--metrics-file", type=str, required=True, help="Parameters used in the simulation.")
args = parser.parse_args()

HEADER_FILE = "headers.csv"
SOURCE = np.array([1010.0, 1010.0, 1010.0]) 
C = 1500.0
F0 = 5.0

def load_parameters(metrics_file):
    params = {}
    with open(metrics_file) as metrics:
            sim = json.load(metrics)["options"]
    params["domain"] = np.array([sim["lx"], sim["ly"], sim["lz"]])
    params["dt"] = sim["dt"]
    params["num_sample"] = sim["num_sample"]
    params["timemax"] = sim["timemax"]
    params["n_receivers"] = sim["receivers"]
    return params

def timed(timings, name, func, *args):
     start = time.perf_counter()
     out = func(*args)
     timings[name] = time.perf_counter() - start
     return out

def write_outputs(out_dir, params, results, timings):
    report = {
        "simulation": {
            "source": SOURCE.tolist(),
            "dt": params["dt"],
            "num_sample": params["num_sample"],
            "domain": params["domain"].tolist(),
        },
        # "global": speeds,
        "receivers": json.loads(results.reset_index(names="rcv")
                                       .to_json(orient="records", double_precision=15)),
    }
    with open(out_dir / "adhoc_results.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "adhoc_metrics.json", "w") as f:
        json.dump(timings, f, indent=2)

def compute_stats(pressures):
     return pd.DataFrame({
        "min": pressures.min(),
        "max": pressures.max(),
        "mean": pressures.mean(),
        "var": pressures.var(ddof=0)    
     })

def compute_peaks(pressures):
     abs_p = pressures.abs()
     return pd.DataFrame({
          "peak_abs": abs_p.max(),
          "t_peak" : abs_p.idxmax(),
     })

# def compute_energy(pressures, dt):
#      return pd.DataFrame({"energy": (pressures ** 2).sum() * dt})

# def compute_distances(coords, domain, margin):
#      xyz = coords[["x","y","z"]].to_numpy(float)
#      distance = np.linalg.norm(xyz-SOURCE, axis=1)
#      dist_border = np.minimum(xyz, domain - xyz).min(axis = 1)
#      return pd.DataFrame({
#           "x": xyz[:, 0], "y": xyz[:,1], "z": xyz[:,2],
#           "distance": distance,
#           "dist_border": dist_border,
#           "used": (dist_border > margin) & (distance > 0),
#      }, index=coords.index)

# def compute_speeds(results):
#      kept = results[results["used"]]
#      (slope, t0), cov = np.polyfit(kept["distance"], kept["t_peak"], 1, cov = True)
#      slope_err, t0_err = np.sqrt(np.diag(cov))
#      residuals = kept["t_peak"] - (t0 + slope * kept["distance"])
#      return {
#           "speed": float(1 / slope),
#           "speed_err": float(slope_err / slope**2),
#           "t0": float(t0),
#           "t0_err": float(t0_err),
#           "residual_rms": float(np.sqrt((residuals**2).mean())),
#           "n_rcv_used": int(len(kept)),
#      }

# def compute_ft(pressures, distance, dt):
#     t = pressures.index.to_numpy()
#     fc = {}
#     for rcv, r in distance.items():
#         s = pressures[rcv].to_numpy()[t <= r / C + 2 / F0]
#         power = np.abs(np.fft.rfft(s))**2
#         freqs = np.fft.rfftfreq(len(s), dt)
#         fc[rcv] = np.average(freqs, weights=power)
#     return pd.DataFrame({"f_central": pd.Series(fc)})

def main(): 
    timings = {}
    csv = args.input_file
    header_file = args.header_file
    start = time.perf_counter()
    params = load_parameters(Path(csv).parent / args.metrics_file)
    data = pd.read_csv(csv)
    if (header_file):
        coords = pd.read_csv(Path(csv).parent / HEADER_FILE)
        coords = coords.set_index("rcv_" + coords["id"].astype(str))
    else:
        xyz = data.filter(regex=r"^rcv_\d+_[xyz]$").iloc[0].to_numpy().reshape(-1, 3)
        coords = pd.DataFrame(xyz, columns=["x", "y", "z"],
                              index=data.filter(regex=r"^rcv_\d+$").columns)
    pressures = data.set_index("time").filter(regex=r"^rcv_\d+$")
    timings["read"] = time.perf_counter() - start

    stats = timed(timings, "stats", compute_stats, pressures)
    peaks = timed(timings, "peaks", compute_peaks, pressures)
    # energies = timed(timings, "energies", compute_energy, pressures, params["dt"])
    margin = C / F0 / 2  
    # distances = timed(timings, "distances", compute_distances, coords, params["domain"], margin)
    # freq_centrals = timed(timings, "freq_centrals", compute_ft, pressures, distances["distance"], params["dt"])
    results = stats.join(peaks)
    # results = results.join(energies)
    # results = results.join(distances)
    # results = results.join(freq_centrals)
    # speeds = timed(timings, "speeds", compute_speeds, results)
    # speeds["margin"] = margin

    write_outputs(Path(csv).parent, params, results, timings)

if __name__ == "__main__":
    main()