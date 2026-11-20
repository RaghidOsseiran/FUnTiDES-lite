#!/usr/bin/env python3
"""Quantized snapshots of the SEM proxy (--snapshots-quant): reader, inverse
function (reconstruction) and error evaluation. Needs numpy only.

The proxy writes, in <quant-folder> (default "quant"):
  quant-header.json   kind, bits, bounds mode, grid axes x / y / z (written once)
  qsnap-<step>.bin    72-byte little-endian header + payload:
       0 char[4] "QSNP"      4 u32 version          8 u32 header_bytes
      12 u32 kind (0 float, 1 linear uint)          16 u32 bits per value
      20 u32 encoding (0 raw; reserved: delta, RLE) 24 u32 step
      28 u32 levels          32 u64 n_values        40 u64 payload_bytes
      48 f32 vmin  52 f32 vmax    bounds of the uint mapping
      56 f32 data_min  60 f32 data_max    min / max of the original field
      64 f32 scale (float kinds)          68 u32 flags (bit 0: fixed bounds)
    payload, float kinds: one IEEE value per node (float32 / float16)
    payload, uint kind:   LSB-first bit stream, node i in bits [i*bits, (i+1)*bits)

Reconstruction
  float kinds   p' = stored * scale
  uint kind     p' = vmin + (vmax - vmin) * q / (levels - 1)

Commands
  info    FILE...                    print the header of qsnap files
  eval    RAW_DIR QUANT_DIR [--csv]  error of the reconstruction against the raw
                                     snapshots written by the same run
  decode  QUANT_DIR OUT_DIR          write the reconstruction as raw snapshots
                                     ("bin-header": snap-<step>.dat + headers.dat),
                                     to run the usual analyses / viewer on it
  check   RAW_DIR QUANT_DIR          the files equal the numpy reference encoder
                                     applied to the raw snapshots, byte for byte
  sweep   RAW_DIR --kinds K [K...]   offline: quantize the raw snapshots with the
                                     reference encoder and report the errors

RAW_DIR holds snap-<step>.dat written with --snapshots-use-headers (float32
pressure per node) or without (float32 x, y, z, p per node).

Errors (per snapshot, p original, p' reconstructed):
  rmse_norm    sqrt(mean((p' - p)^2)) / max|p|
  maxerr_norm  max|p' - p| / max|p|
No point-wise relative error: the field is zero or nearly zero on most nodes.
The ratio is 4 bytes per node (raw float32 pressure) over the qsnap file size.
"""

import argparse
import csv
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<4sIIIIIIIQQfffffI")  # 72 bytes
FIELDS = ("magic", "version", "header_bytes", "kind", "bits", "encoding", "step", "levels",
          "n_values", "payload_bytes", "vmin", "vmax", "data_min", "data_max", "scale", "flags")
KIND_FLOAT, KIND_UINT = 0, 1
ENCODING_RAW = 0
FLAG_FIXED_BOUNDS = 1


# ----------------------------------------------------------------------------
# Reading and reconstruction
# ----------------------------------------------------------------------------
def step_of(path):
    """qsnap-120.bin / snap-120.dat -> 120."""
    return int(re.findall(r"\d+", Path(path).stem)[-1])


def read_header(path):
    """Header of a qsnap file as a dict (floats kept as np.float32)."""
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
    if len(raw) < HEADER.size or raw[:4] != b"QSNP":
        raise ValueError(f"{path}: not a qsnap file")
    h = dict(zip(FIELDS, HEADER.unpack(raw)))
    if h["version"] != 1:
        raise ValueError(f"{path}: unsupported qsnap version {h['version']}")
    for k in ("vmin", "vmax", "data_min", "data_max", "scale"):
        h[k] = np.float32(h[k])
    return h


def kind_name(h):
    return f"float{h['bits']}" if h["kind"] == KIND_FLOAT else f"uint{h['bits']}"


def unpack_bits(buf, n, bits):
    """LSB-first bit stream -> n unsigned integers of `bits` bits."""
    if bits == 8:
        return buf[:n].astype(np.uint32)
    if bits == 16:
        return buf[:2 * n].view("<u2").astype(np.uint32)
    stream = np.unpackbits(buf, bitorder="little")[:n * bits].reshape(n, bits)
    q = np.zeros(n, dtype=np.uint32)
    for b in range(bits):
        q |= stream[:, b].astype(np.uint32) << np.uint32(b)
    return q


def pack_bits(q, bits):
    """Inverse of unpack_bits (zero-padded to a whole byte)."""
    if bits == 8:
        return q.astype(np.uint8)
    if bits == 16:
        return q.astype("<u2").view(np.uint8)
    stream = np.empty((q.size, bits), dtype=np.uint8)
    for b in range(bits):
        stream[:, b] = (q >> np.uint32(b)) & np.uint32(1)
    return np.packbits(stream.reshape(-1), bitorder="little")


def read_stored(path):
    """(header, stored values): float32 / float16 array, or uint32 levels."""
    h = read_header(path)
    if h["encoding"] != ENCODING_RAW:
        raise ValueError(f"{path}: payload encoding {h['encoding']} is not supported")
    buf = np.fromfile(path, dtype=np.uint8, offset=h["header_bytes"])
    if buf.size != h["payload_bytes"]:
        raise ValueError(f"{path}: {buf.size} payload bytes, header says {h['payload_bytes']} "
                         "(truncated file?)")
    n, bits = h["n_values"], h["bits"]
    if h["kind"] == KIND_FLOAT:
        return h, buf.view("<f4" if bits == 32 else "<f2")[:n]
    return h, unpack_bits(buf, n, bits)


def dequantize(h, stored):
    """The inverse function: stored values -> float32 pressure."""
    if h["kind"] == KIND_FLOAT:
        return stored.astype(np.float32) * h["scale"]  # power of two: exact
    vmin, vmax = np.float64(h["vmin"]), np.float64(h["vmax"])
    # q / (levels - 1) first: with symmetric bounds the middle level gives 0.5
    # exactly, hence an exact 0
    return (vmin + (vmax - vmin) * (stored / np.float64(h["levels"] - 1))).astype(np.float32)


def read_qsnap(path):
    """(header, reconstructed float32 pressure) of one qsnap file."""
    h, stored = read_stored(path)
    return h, dequantize(h, stored)


# ----------------------------------------------------------------------------
# Reference encoder (same arithmetic as the proxy), for checks and offline sweeps
# ----------------------------------------------------------------------------
def parse_kind(kind):
    """'float16' -> (KIND_FLOAT, 16), 'uint12' -> (KIND_UINT, 12)."""
    m = re.fullmatch(r"(float|uint)(\d+)", kind)
    if not m:
        raise ValueError(f"unknown kind {kind!r}")
    bits = int(m.group(2))
    if m.group(1) == "float":
        if bits not in (16, 32):
            raise ValueError("float kinds are float16 and float32")
        return KIND_FLOAT, bits
    if not 2 <= bits <= 16:
        raise ValueError("uint<N>: N must be in 2..16")
    return KIND_UINT, bits


def quantize(p, kind, bounds="snapshot", amax=None, step=0):
    """(header, stored values) for the float32 field p, as the proxy computes them."""
    p = np.asarray(p, dtype=np.float32)
    k, bits = parse_kind(kind)
    dmin, dmax = p.min(), p.max()
    a = np.float32(amax) if bounds == "fixed" else max(abs(dmin), abs(dmax))
    vmin, vmax = (dmin, dmax) if bounds == "snapshot" else (-a, a)
    h = dict(magic=b"QSNP", version=1, header_bytes=HEADER.size, kind=k, bits=bits,
             encoding=ENCODING_RAW, step=step, levels=0, n_values=p.size, payload_bytes=0,
             vmin=np.float32(vmin), vmax=np.float32(vmax), data_min=dmin, data_max=dmax,
             scale=np.float32(1), flags=FLAG_FIXED_BOUNDS if bounds == "fixed" else 0)
    if k == KIND_FLOAT:
        if bits == 16 and bounds != "snapshot" and a > 0 and np.isfinite(a):
            h["scale"] = np.float32(np.ldexp(1.0, int(np.frexp(a)[1]) - 1 - 14))
        h["payload_bytes"] = p.size * bits // 8
        if bits == 32:
            return h, p
        with np.errstate(over="ignore"):
            return h, (p * (np.float32(1) / h["scale"])).astype("<f2")
    h["levels"] = (1 << bits) - (0 if bounds == "snapshot" else 1)
    h["payload_bytes"] = (p.size * bits + 7) // 8
    maxq = np.float64(h["levels"] - 1)
    rng = np.float64(h["vmax"]) - np.float64(h["vmin"])
    inv = maxq / rng if rng > 0 else 0.0
    t = np.clip((p.astype(np.float64) - np.float64(h["vmin"])) * inv, 0.0, maxq)
    return h, np.floor(t + 0.5).astype(np.uint32)


def payload_of(h, stored):
    """Bytes of the raw payload for stored values."""
    if h["kind"] == KIND_FLOAT:
        return np.ascontiguousarray(stored).view(np.uint8)
    return pack_bits(stored, h["bits"])


def write_qsnap(path, h, stored):
    with open(path, "wb") as f:
        f.write(HEADER.pack(*(h[k] for k in FIELDS)))
        payload_of(h, stored).tofile(f)


# ----------------------------------------------------------------------------
# Raw snapshots and errors
# ----------------------------------------------------------------------------
def find_raw(raw_dir):
    """{step: path} of the raw snap-<step>.dat files."""
    files = {step_of(f): f for f in Path(raw_dir).glob("snap-*.dat")}
    if not files:
        raise SystemExit(f"no snap-<step>.dat in {raw_dir}")
    return files


def find_quant(quant_dir):
    files = {step_of(f): f for f in Path(quant_dir).glob("qsnap-*.bin")}
    if not files:
        raise SystemExit(f"no qsnap-<step>.bin in {quant_dir}")
    return files


def read_raw(path, n=None):
    """Pressure of a raw binary snapshot: float32 p per node (bin-header), or
    x, y, z, p per node (bin) when the size says so."""
    data = np.fromfile(path, dtype="<f4")
    if n is not None and data.size == 4 * n:
        return np.ascontiguousarray(data.reshape(-1, 4)[:, 3])
    if n is not None and data.size != n:
        raise SystemExit(f"{path}: {data.size} float32 values for {n} nodes")
    return data


def errors(p, rec):
    """max|p|, RMSE and max abs error of rec against p (float64 arithmetic)."""
    d = rec.astype(np.float64) - p.astype(np.float64)
    return float(np.abs(p).max()), float(np.sqrt(np.mean(d * d))), float(np.abs(d).max())


COLUMNS = ["kind", "bounds", "step", "nodes", "max_abs_p", "rmse", "max_err", "rmse_norm",
           "maxerr_norm", "raw_bytes", "quant_bytes", "ratio", "outside_bounds"]


def error_row(kind, bounds, step, p, h, rec, quant_bytes):
    amax, rmse, maxerr = errors(p, rec)
    outside = 0
    if h["kind"] == KIND_UINT:
        outside = int(np.count_nonzero((p < h["vmin"]) | (p > h["vmax"])))
    norm = amax if amax > 0 else np.nan
    raw_bytes = 4 * p.size
    return [kind, bounds, step, p.size, amax, rmse, maxerr, rmse / norm, maxerr / norm,
            raw_bytes, quant_bytes, raw_bytes / quant_bytes, outside]


def report(rows, csv_path=None):
    """Print the per-snapshot table and one summary line per (kind, bounds)."""
    print(f"{'kind':<8} {'bounds':<9} {'step':>6} {'max|p|':>11} {'RMSE/max|p|':>12} "
          f"{'maxerr/max|p|':>14} {'ratio':>7} {'outside':>8}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<9} {r[2]:>6d} {r[4]:>11.4e} {r[7]:>12.4e} {r[8]:>14.4e} "
              f"{r[11]:>7.3f} {r[12]:>8d}")
    print()
    for key in dict.fromkeys((r[0], r[1]) for r in rows):
        sel = [r for r in rows if (r[0], r[1]) == key]
        gmax = max(r[4] for r in sel)  # max|p| over the run
        n = sum(r[3] for r in sel)
        grmse = np.sqrt(sum(r[5] ** 2 * r[3] for r in sel) / n)
        gerr = max(r[6] for r in sel)
        ratio = sum(r[9] for r in sel) / sum(r[10] for r in sel)
        print(f"{key[0]} / {key[1]}: {len(sel)} snapshots, ratio {ratio:.3f} "
              f"({sum(r[9] for r in sel)} -> {sum(r[10] for r in sel)} bytes); "
              f"per snapshot: worst RMSE/max|p| {np.nanmax([r[7] for r in sel]):.3e}, "
              f"worst maxerr/max|p| {np.nanmax([r[8] for r in sel]):.3e}; "
              f"whole run (normalised by the run's max|p| = {gmax:.4e}): "
              f"RMSE {grmse / gmax:.3e}, maxerr {gerr / gmax:.3e}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)
        print(f"written {csv_path}")


def common_steps(raw, quant):
    steps = sorted(set(raw) & set(quant))
    only = sorted(set(raw) ^ set(quant))
    if only:
        print(f"WARNING: steps with only a raw or only a quantized snapshot: {only}")
    if not steps:
        raise SystemExit("no common step between the raw and the quantized snapshots")
    return steps


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_info(args):
    for path in args.files:
        h = read_header(path)
        size = Path(path).stat().st_size
        print(f"{path}: {kind_name(h)}, step {h['step']}, {h['n_values']} values, "
              f"{h['payload_bytes']} payload bytes (file {size}), levels {h['levels']}, "
              f"bounds [{h['vmin']:g}, {h['vmax']:g}]{' fixed' if h['flags'] & 1 else ''}, "
              f"data [{h['data_min']:g}, {h['data_max']:g}], scale {h['scale']:g}, "
              f"encoding {h['encoding']}")


def quant_bounds_mode(quant_dir):
    header = Path(quant_dir) / "quant-header.json"
    return json.loads(header.read_text()).get("bounds", "?") if header.is_file() else "?"


def cmd_eval(args):
    raw, quant = find_raw(args.raw_dir), find_quant(args.quant_dir)
    bounds = quant_bounds_mode(args.quant_dir)
    rows = []
    for step in common_steps(raw, quant):
        h, rec = read_qsnap(quant[step])
        p = read_raw(raw[step], h["n_values"])
        rows.append(error_row(kind_name(h), bounds, step, p, h, rec,
                              quant[step].stat().st_size))
    report(rows, args.csv)


def cmd_decode(args):
    quant = find_quant(args.quant_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = Path(args.quant_dir) / "quant-header.json"
    if header.is_file():  # node coordinates, as the proxy's headers.dat
        g = json.loads(header.read_text())
        x, y, z = (np.asarray(g[k], dtype="<f4") for k in "xyz")
        coords = np.empty((len(z), len(y), len(x), 3), dtype="<f4")
        coords[..., 0] = x[None, None, :]
        coords[..., 1] = y[None, :, None]
        coords[..., 2] = z[:, None, None]
        coords.tofile(out / "headers.dat")
    else:
        print(f"WARNING: no {header}: headers.dat not written")
    for step in sorted(quant):
        _, rec = read_qsnap(quant[step])
        rec.astype("<f4").tofile(out / f"snap-{step}.dat")
    print(f"{len(quant)} reconstructed snapshots written in {out} (format bin-header)")


def cmd_check(args):
    raw, quant = find_raw(args.raw_dir), find_quant(args.quant_dir)
    g = json.loads((Path(args.quant_dir) / "quant-header.json").read_text())
    bad = 0
    for step in common_steps(raw, quant):
        h = read_header(quant[step])
        p = read_raw(raw[step], h["n_values"])
        ref_h, ref = quantize(p, g["kind"], g["bounds"], g.get("amax"), step)
        got = np.fromfile(quant[step], dtype=np.uint8, offset=h["header_bytes"])
        exp = payload_of(ref_h, ref)
        same_h = all(h[k] == ref_h[k] for k in FIELDS)
        ndiff = -1 if got.size != exp.size else int(np.count_nonzero(got != exp))
        ok = same_h and ndiff == 0
        bad += not ok
        print(f"step {step:>6d}: header {'same' if same_h else 'DIFFERENT'}, "
              f"{ndiff} different payload bytes of {exp.size} -> {'OK' if ok else 'FAILED'}")
        if not same_h:
            print("   file     ", {k: h[k] for k in FIELDS if h[k] != ref_h[k]})
            print("   reference", {k: ref_h[k] for k in FIELDS if h[k] != ref_h[k]})
    print(f"{g['kind']} / {g['bounds']} / computed on {g.get('where')}: "
          f"{'identical to the reference encoder' if not bad else f'{bad} step(s) differ'}")
    return 1 if bad else 0


def cmd_sweep(args):
    raw = find_raw(args.raw_dir)
    n = args.nodes
    rows = []
    for kind in args.kinds:
        for step in sorted(raw):
            p = read_raw(raw[step], n)
            h, stored = quantize(p, kind, args.bounds, args.amax, step)
            # through the bit packing too, as a reader would
            if h["kind"] == KIND_UINT:
                stored = unpack_bits(pack_bits(stored, h["bits"]), p.size, h["bits"])
            rows.append(error_row(kind, args.bounds, step, p, h, dequantize(h, stored),
                                  HEADER.size + h["payload_bytes"]))
    report(rows, args.csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="print qsnap headers")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_info)
    s = sub.add_parser("eval", help="reconstruction error against the raw snapshots")
    s.add_argument("raw_dir")
    s.add_argument("quant_dir")
    s.add_argument("--csv", help="also write the per-snapshot table to this CSV file")
    s.set_defaults(func=cmd_eval)
    s = sub.add_parser("decode", help="write the reconstruction as raw bin-header snapshots")
    s.add_argument("quant_dir")
    s.add_argument("out_dir")
    s.set_defaults(func=cmd_decode)
    s = sub.add_parser("check", help="compare the files with the numpy reference encoder")
    s.add_argument("raw_dir")
    s.add_argument("quant_dir")
    s.set_defaults(func=cmd_check)
    s = sub.add_parser("sweep", help="offline quantization of raw snapshots, several kinds")
    s.add_argument("raw_dir")
    s.add_argument("--kinds", nargs="+", required=True, help="e.g. float16 uint8 uint12 uint16")
    s.add_argument("--bounds", default="snapshot", choices=["snapshot", "symmetric", "fixed"])
    s.add_argument("--amax", type=float, help="A of the fixed bounds [-A, A]")
    s.add_argument("--nodes", type=int,
                   help="node count, needed for raw snapshots that also hold x, y, z")
    s.add_argument("--csv", help="also write the per-snapshot table to this CSV file")
    s.set_defaults(func=cmd_sweep)
    args = ap.parse_args()
    if args.cmd == "sweep" and args.bounds == "fixed" and not (args.amax and args.amax > 0):
        ap.error("--bounds fixed requires --amax > 0")
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
