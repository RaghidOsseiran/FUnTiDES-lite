#!/usr/bin/env python3
"""Interactive 2D viewer for SEM proxy wavefield snapshots (VTK).

Shows a z-slice of the pressure field through the source, snapshot by snapshot.

Controls
  click / hold  <  >      previous / next snapshot (holding plays continuously)
  click  play             autoplay (loops)
  click  "See more"       show / hide the seismograms panel
  Left / Right            previous / next snapshot (hold the key to play)
  Up / Down               move the slice plane along z (volume mode only)
  m                       back to the middle slice (through the source; volume mode only)
  space                   play / pause
  s                       show / hide seismograms
  c                       colour scale: per-snapshot <-> fixed for the whole run
  mouse wheel / right drag: zoom, middle drag (or shift + left drag): pan

Usage
  python visualise_snapshots.py [DIR] [--format FMT] [--seismos DIR] [--z Z] [options]
  python visualise_snapshots.py [DIR] --slices
  python visualise_snapshots.py [DIR] [--slices] --benchmark [--repeat N]
  python visualise_snapshots.py [DIR] --check-slices
  python visualise_snapshots.py --list

Two modes:
  volume (default)  read the 3D snapshots and cut them with a z-plane (vtkCutter);
                    --z sets the plane (default: source z)
  slice (--slices)  read the 2D z-slices written during the run by the proxy
                    (--use-slices): DIR/slices/slice-header.json + slice-<step>.dat
                    (raw float32, nx*ny values, x fastest); no 3D grid, no cutter

--benchmark opens no window: for every output file it times the read (file ->
numpy), the cut (volume mode) and the statistics shown by the viewer, --repeat
times, and writes DIR/visu_bench_<mode>.csv. In volume mode the plane defaults
to the z of the slices when the run has some, so both modes cut the same plane.
--check-slices (no window) compares, for every step, the volume cut at the
slice z with the slice file.

Each run of the SEM proxy writes <output-dir>/<run-name>/{snapshots,seismos}/
(<output-dir> defaults to output/ at the root of the repo). DIR is either one
run folder, or a folder of runs, in which case the most recent run with
snapshots is shown. Default DIR: the repo's output/ folder. --list prints the
runs found there.

Snapshots are read from DIR/snapshots (or DIR itself). All the save formats
of the proxy are understood (FMT, auto-detected by default):

  bin          snap-<step>.dat  float32 x,y,z,p per node
  bin-header   snap-<step>.dat  float32 p per node   + headers.dat (x,y,z)
  csv          snap-<step>.csv  "x;y;z;p" per node
  csv-header   snap-<step>.csv  one p per line       + headers.csv ("x;y;z;")
  fake         snapshots/pnGlobal_step*.csv + metadata.json (generate_fake_data.py)

Seismograms are looked for in DIR, DIR/seismos and DIR/seismograms_data, as
written by the proxy (seismogram_recording.csv: traces + receiver coordinates)
or by generate_fake_data.py (seismograms.csv + receivers.csv). They are optional.

The proxy writes no metadata, so dt comes from the seismograms (or --dt) and
the source / vp / f0 used for the expected wavefront default to the proxy's
defaults (override with --source, --vp, --f0).
"""

import argparse
import csv
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Only the VTK modules used ("import vtk" loads all of them: slow first start).
# The backend modules register the concrete classes behind VTK's object
# factory; without them vtkRenderWindow / vtkRenderWindowInteractor are the
# generic base classes (no window, no events) and the chart has no 2D device.
import vtkmodules.vtkInteractionStyle  # noqa: F401  interactor styles
import vtkmodules.vtkRenderingContextOpenGL2  # noqa: F401  2D device of the chart
import vtkmodules.vtkRenderingFreeType  # noqa: F401  text rendering
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  window, interactor, mappers
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkChartsCore import vtkAxis, vtkChart, vtkChartXY
from vtkmodules.vtkCommonCore import (vtkDoubleArray, vtkLookupTable, vtkPoints,
                                      vtkUnsignedCharArray)
from vtkmodules.vtkCommonDataModel import vtkPlane, vtkPolyData, vtkRectilinearGrid, vtkTable
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingContext2D import vtkContextActor, vtkContextScene
from vtkmodules.vtkRenderingCore import (vtkActor, vtkColorTransferFunction, vtkDataSetMapper,
                                         vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,
                                         vtkRenderWindowInteractor, vtkTextActor)

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "output"  # PROXY_OUTPUT_DIR
HOLD_INTERVAL_MS = 120  # snapshot change rate while an arrow is held / playing
RCV_FADE_DIST = 200.0  # receivers fade out over this distance (m) from the slice
RCV_MIN_OPACITY = 0.12  # ...but stay faintly visible so you know where they are
CACHE_SIZE = 32  # snapshots kept in memory (a 101^3 float32 field is 4 MB)

# defaults of the proxy (sem_proxy_options.h / sem_proxy.h)
PROXY_DEFAULTS = {"source": (1010.0, 1010.0, 1010.0), "vp": 1500.0, "f0": 5.0, "dt": 0.001}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_semicolon_csv(path):
    """Fast reader for the proxy's CSV ("a;b;c;" rows, optional text header).

    Returns a 2D array (nodes x columns). np.loadtxt is far too slow for 1M
    lines, so the text is turned into a flat array in one go.
    """
    text = path.read_text()
    # the header ("x;y;z", maybe without its newline) is letters and ';' only
    header = re.match(r"[A-Za-z_;]*[A-Za-z]", text)
    if header:
        text = text[header.end():]
    text = text.lstrip(";\n\r ")
    first_line = text.split("\n", 1)[0]
    ncols = len([f for f in first_line.split(";") if f.strip()])
    flat = np.fromstring(text.replace(";", " "), sep=" ")
    if ncols == 0 or flat.size % ncols:
        raise ValueError(f"{path}: {flat.size} values is not a multiple of {ncols} columns")
    return flat.reshape(-1, ncols)


def step_of(path):
    """snap-120.dat -> 120, pnGlobal_step00120.csv -> 120."""
    return int(re.findall(r"\d+", path.stem)[-1])


def find_snapshots(root, fmt):
    """Pick the snapshot format present in root/snapshots or root (or check
    the requested one). Returns (format, files, folder holding the files)."""
    for folder in (root / "snapshots", root):
        candidates = {
            "fake": ("pnGlobal_step*.csv", root / "metadata.json"),
            "bin-header": ("snap-*.dat", folder / "headers.dat"),
            "bin": ("snap-*.dat", None),
            "csv-header": ("snap-*.csv", folder / "headers.csv"),
            "csv": ("snap-*.csv", None),
        }
        for name in [fmt] if fmt != "auto" else list(candidates):
            pattern, needed = candidates[name]
            files = sorted(folder.glob(pattern), key=step_of)  # numeric: 2 < 10
            if files and (needed is None or needed.exists()):
                return name, files, folder
    return None


def has_slices(root):
    return (root / "slices" / "slice-header.json").is_file()


def find_runs(root, slices=False):
    """Run folders under root that contain snapshots (or slices), oldest first."""
    if not root.is_dir():
        return []
    ok = has_slices if slices else (lambda d: find_snapshots(d, "auto"))
    runs = [d for d in root.iterdir() if d.is_dir() and ok(d)]
    return sorted(runs, key=lambda d: d.stat().st_mtime)


def resolve_run(path, fmt, slices=False):
    """path is a run folder, or a folder of runs: then take the latest one."""
    if has_slices(path) if slices else find_snapshots(path, fmt):
        return path
    runs = find_runs(path, slices)
    if not runs:
        if slices:
            raise SystemExit(f"no slices/slice-header.json in {path} nor in its sub-folders; "
                             "run the proxy with --use-slices")
        raise SystemExit(f"no snapshot (format '{fmt}') in {path} nor in its sub-folders; "
                         "run the proxy with --use-snapshots")
    print(f"latest run in {path}: {runs[-1].name}")
    return runs[-1]


class Snapshots:
    """Snapshot files of one run, read on demand.

    Values are returned in VTK's structured point order (x fastest, then y,
    then z), whatever the order of the nodes in the files.
    """

    def __init__(self, root, fmt="auto", dtype="auto"):
        found = find_snapshots(root, fmt)
        if found is None:
            raise SystemExit(f"no snapshot in format '{fmt}' found in {root}")
        self.format, self.files, folder = found
        self.steps = [step_of(f) for f in self.files]
        self.dtype = None
        self.meta = {}
        self._cache = OrderedDict()

        if self.format == "fake":
            # node,x,y,z,pressure
            self.meta = json.loads((root / "metadata.json").read_text())
            coords = np.loadtxt(self.files[0], delimiter=",", skiprows=1, usecols=(1, 2, 3))
        elif self.format == "bin-header":
            coords = self._read_binary(folder / "headers.dat", 3, dtype)
        elif self.format == "bin":
            coords = self._read_binary(self.files[0], 4, dtype)[:, :3]
        elif self.format == "csv-header":
            coords = read_semicolon_csv(folder / "headers.csv")[:, :3]
        else:  # csv
            table = read_semicolon_csv(self.files[0])
            if table.shape[1] < 4:
                raise SystemExit(
                    f"{self.files[0]} only has x;y;z columns, no pressure: the snapshot was "
                    "written by a version of SEMproxy::saveSnapshotCsv that drops the value.")
            coords = table[:, :3]
        self._set_grid(coords)

    # -- geometry -------------------------------------------------------------
    def _read_binary(self, path, ncols, dtype):
        """Read a float array of ncols columns; float32 vs float64 (USE_DOUBLE)
        is guessed by checking which one gives a sensible grid."""
        for dt in (["<f4", "<f8"] if dtype == "auto" else [dtype]):
            itemsize = np.dtype(dt).itemsize
            if path.stat().st_size % (ncols * itemsize):
                continue
            data = np.fromfile(path, dtype=dt).reshape(-1, ncols)
            if self._is_grid(data[:, :3]):
                self.dtype = dt
                return data
        raise SystemExit(f"{path}: cannot read it as a structured grid of {ncols} "
                         f"float columns (dtype {dtype})")

    @staticmethod
    def _is_grid(coords):
        if not np.isfinite(coords).all():
            return False
        n = [np.unique(np.round(coords[:, c], 3)).size for c in range(3)]
        return n[0] * n[1] * n[2] == len(coords)

    def _set_grid(self, coords):
        if not self._is_grid(coords):
            raise SystemExit("the snapshot nodes do not form a structured (cartesian) grid")
        rounded = np.round(coords, 3)
        self.x, self.y, self.z = (np.unique(rounded[:, c]) for c in range(3))
        self.n_nodes = len(coords)
        # permutation file order -> VTK order (None when already x fastest)
        perm = np.lexsort((rounded[:, 0], rounded[:, 1], rounded[:, 2]))
        self.perm = None if (perm == np.arange(len(perm))).all() else perm
        self.bounds = np.array([[a[0], a[-1]] for a in (self.x, self.y, self.z)])

    # -- values ---------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def field(self, i):
        """Pressure of snapshot i (cached, LRU)."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        p = self.read(i)
        self._cache[i] = p
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return p

    def read(self, i):
        """Read snapshot i from its file (no cache), in VTK order."""
        p = self._read_values(self.files[i])
        if p.size != self.n_nodes:
            raise SystemExit(f"{self.files[i]}: {p.size} values for {self.n_nodes} nodes "
                             "(truncated file?)")
        if self.perm is not None:
            p = p[self.perm]
        return p

    def _read_values(self, path):
        if self.format == "fake":
            return np.loadtxt(path, delimiter=",", skiprows=1, usecols=4)
        if self.format == "bin-header":
            return np.fromfile(path, dtype=self.dtype)
        if self.format == "bin":
            return np.fromfile(path, dtype=self.dtype).reshape(-1, 4)[:, 3]
        return read_semicolon_csv(path)[:, -1]  # csv (x;y;z;p) and csv-header (p)


class Slices:
    """z-slices of one run (SEMproxy --use-slices), read on demand.

    root/slices/slice-header.json gives the grid (nx, ny, x, y), the node layer
    (k, z) and dt; slice-<step>.dat holds nx*ny raw float32 values, x fastest
    then y, which is already VTK's point order for a (nx, ny, 1) grid.
    Same interface as Snapshots (steps, x, y, z, bounds, field, read).
    """

    format = "slice"

    def __init__(self, root):
        folder = root / "slices"
        header = folder / "slice-header.json"
        if not header.is_file():
            raise SystemExit(f"no {header}: run the proxy with --use-slices")
        h = json.loads(header.read_text())
        if h.get("dtype") != "float32":
            raise SystemExit(f"{header}: dtype {h.get('dtype')!r}, only float32 is supported")
        self.header = h
        self.dtype = "<f4"
        self.files = sorted(folder.glob("slice-*.dat"), key=step_of)
        if not self.files:
            raise SystemExit(f"no slice-<step>.dat in {folder}")
        self.steps = [step_of(f) for f in self.files]
        self.x = np.asarray(h["x"], dtype=float)
        self.y = np.asarray(h["y"], dtype=float)
        self.z = np.array([h["z"]], dtype=float)
        if (len(self.x), len(self.y)) != (h["nx"], h["ny"]):
            raise SystemExit(f"{header}: x/y lengths do not match nx/ny")
        self.k = h["k"]
        self.n_nodes = h["nx"] * h["ny"]
        self.bounds = np.array([[a[0], a[-1]] for a in (self.x, self.y, self.z)])
        self.meta = {"dt": h["dt"]}
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.files)

    def field(self, i):
        """Pressure of slice i (cached, LRU)."""
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        p = self.read(i)
        self._cache[i] = p
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return p

    def read(self, i):
        """Read slice i from its file (no cache)."""
        p = np.fromfile(self.files[i], dtype=self.dtype)
        if p.size != self.n_nodes:
            raise SystemExit(f"{self.files[i]}: {p.size} values for {self.n_nodes} nodes "
                             "(truncated file?)")
        return p


def find_seismograms(root, folder=None):
    """Return (time, traces[nt, nrcv], receivers[nrcv, 3], ids) or None."""
    dirs = [folder] if folder else [root, root / "seismos", root / "seismograms_data"]
    for d in dirs:
        # proxy format: step,time,rcv_<id>...,rcv_<id>_x,rcv_<id>_y,rcv_<id>_z...
        f = d / "seismogram_recording.csv"
        if f.exists():
            names = f.open().readline().strip().split(",")
            data = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            col = {n: k for k, n in enumerate(names)}
            ids = [m.group(1) for n in names if (m := re.fullmatch(r"rcv_([^_]+)", n))]
            traces = data[:, [col[f"rcv_{r}"] for r in ids]]
            rcv = np.array([[data[0, col[f"rcv_{r}_{c}"]] if f"rcv_{r}_{c}" in col else np.nan
                             for c in "xyz"] for r in ids])
            return data[:, col["time"]], traces, rcv, ids
        # generate_fake_data.py format: seismograms.csv + receivers.csv
        f, r = d / "seismograms.csv", d / "receivers.csv"
        if f.exists() and r.exists():
            data = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
            rcv = np.loadtxt(r, delimiter=",", skiprows=1, ndmin=2)
            return data[:, 1], data[:, 2:], rcv[:, 1:], [str(int(i)) for i in rcv[:, 0]]
    return None


class Run:
    """Snapshots (or slices) + seismograms + the few physical parameters the
    viewer needs."""

    def __init__(self, root, fmt="auto", dtype="auto", seismos=None,
                 dt=None, source=None, vp=None, f0=None, slices=False):
        self.root = root
        self.is_slice = slices
        self.snaps = Slices(root) if slices else Snapshots(root, fmt, dtype)
        meta = self.snaps.meta
        self.x, self.y, self.z = self.snaps.x, self.snaps.y, self.snaps.z
        self.bounds = self.snaps.bounds
        self.steps = self.snaps.steps

        seis = find_seismograms(root, seismos)
        if seis is None:
            self.seis_time = self.seis = None
            self.receivers = np.empty((0, 3))
            self.rcv_ids = []
        else:
            self.seis_time, self.seis, self.receivers, self.rcv_ids = seis

        # explicit option > metadata.json > seismogram time axis > proxy default
        if dt is None:
            dt = meta.get("dt")
        if dt is None and self.seis_time is not None and len(self.seis_time) > 1:
            dt = float(self.seis_time[1] - self.seis_time[0])
        self.dt = dt or PROXY_DEFAULTS["dt"]
        self.source = np.array(source or meta.get("source") or PROXY_DEFAULTS["source"], float)
        self.vp = vp or meta.get("vp") or PROXY_DEFAULTS["vp"]
        self.f0 = f0 or meta.get("f0") or PROXY_DEFAULTS["f0"]
        self.times = [s * self.dt for s in self.steps]
        self._stats = {}

        if slices:
            grid = (f"grid {len(self.x)} x {len(self.y)} at z = {self.z[0]:g} m "
                    f"(layer k = {self.snaps.k})")
        else:
            grid = f"grid {len(self.x)} x {len(self.y)} x {len(self.z)}"
        print(f"{len(self.steps)} {'slices' if slices else 'snapshots'} ({self.snaps.format}"
              f"{', ' + self.snaps.dtype if self.snaps.dtype else ''}), {grid}, "
              f"steps {self.steps[0]}..{self.steps[-1]}, dt = {self.dt:g} s, "
              f"{len(self.rcv_ids) or 'no'} receivers")

    def field(self, i):
        return self.snaps.field(i)

    def stats(self, i):
        """viewer_stats of field i, cached (the field part only: it does not
        depend on the cut plane)."""
        if i not in self._stats:
            self._stats[i] = viewer_stats(self.field(i))
        return self._stats[i]

    def robust_range(self):
        """|p| 99.5th percentile over the whole run: the 1/r spike at the
        source would otherwise wash out everything else. Reads every snapshot."""
        stride = max(1, self.snaps.n_nodes // 200_000)
        vmax = 0.0
        for i in range(len(self.steps)):
            # non-zero values only: early on, the wave covers a few nodes
            a = np.abs(self.field(i)[::stride])
            a = a[a > 0]
            if a.size:
                vmax = max(vmax, np.percentile(a, 99.5))
        return vmax or 1.0


def viewer_stats(field, slice_p=None):
    """The numbers the info panel shows for one output: min / max / mean / std
    of the field (the volume, or the slice in slice mode), max|p| for the
    per-snapshot colour scale and, in volume mode, min / max of the cut."""
    s = {"min": field.min(), "max": field.max(), "mean": field.mean(), "std": field.std(),
         "absmax": np.abs(field).max()}
    if slice_p is not None:
        s["slice_min"] = slice_p.min() if slice_p.size else np.nan
        s["slice_max"] = slice_p.max() if slice_p.size else np.nan
    return s


class VolumeCut:
    """3D rectilinear grid of the snapshots cut by a z-plane (vtkCutter).

    Used by the viewer (volume mode), --benchmark and --check-slices.
    """

    def __init__(self, run, z):
        # SOURCE: the whole 3D wavefield as a rectilinear grid (works for any
        # GLL order, even when node spacing is not uniform).
        self.grid = vtkRectilinearGrid()
        self.grid.SetDimensions(len(run.x), len(run.y), len(run.z))
        self.grid.SetXCoordinates(numpy_to_vtk(run.x, deep=True))
        self.grid.SetYCoordinates(numpy_to_vtk(run.y, deep=True))
        self.grid.SetZCoordinates(numpy_to_vtk(run.z, deep=True))
        self.pressure = numpy_to_vtk(run.field(0), deep=True)
        self.pressure.SetName("pressure")
        self.grid.GetPointData().SetScalars(self.pressure)

        # FILTER: cut the volume with a z-plane.
        self.plane = vtkPlane()
        self.plane.SetOrigin(run.source[0], run.source[1], z)
        self.plane.SetNormal(0, 0, 1)
        self.cutter = vtkCutter()
        self.cutter.SetCutFunction(self.plane)
        self.cutter.SetInputData(self.grid)

    def cut(self, field):
        """Load field into the grid, run the cutter, return the cut's values."""
        vtk_to_numpy(self.pressure)[:] = field
        self.pressure.Modified()  # tells the pipeline the data changed
        self.cutter.Update()
        scalars = self.cutter.GetOutput().GetPointData().GetScalars()
        return vtk_to_numpy(scalars) if scalars is not None else np.empty(0)

    def cut_points(self):
        """(x, y) of the points of the last cut."""
        pts = self.cutter.GetOutput().GetPoints()
        return vtk_to_numpy(pts.GetData())[:, :2] if pts is not None else np.empty((0, 2))


# ----------------------------------------------------------------------------
# Small VTK helpers
# ----------------------------------------------------------------------------
def make_text(text, size, pos, justify="left", vjustify="bottom", button=False):
    """2D text placed in normalized viewport coordinates (0..1)."""
    actor = vtkTextActor()
    actor.SetInput(text)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.GetPositionCoordinate().SetValue(*pos)
    prop = actor.GetTextProperty()
    prop.SetFontSize(size)
    prop.SetFontFamilyToCourier()
    prop.SetColor(0.95, 0.95, 0.95)
    getattr(prop, f"SetJustificationTo{justify.capitalize()}")()
    getattr(prop, f"SetVerticalJustificationTo{vjustify.capitalize()}")()
    if button:
        prop.BoldOn()
        prop.SetBackgroundColor(0.25, 0.35, 0.55)
        prop.SetBackgroundOpacity(1.0)
        prop.FrameOn()
        prop.SetFrameColor(0.8, 0.8, 0.9)
    return actor


def diverging_lut():
    """Blue - white - red colour map, the usual choice for signed fields."""
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
    ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        lut.SetTableValue(i, *ctf.GetColor(i / 255.0), 1.0)
    lut.Build()
    return lut


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------
class Viewer:
    def __init__(self, run, z=None):
        """z: cut plane of the volume mode (default: source z). In slice mode
        the plane is the slice's own z."""
        self.run = run
        self.z0 = run.source[2] if z is None else z
        self.idx = 0
        self.direction = 0  # -1 / +1 while an arrow is held or playing
        self.playing = False
        self.timer_id = None
        self.fixed_scale = False
        self.show_seismos = False
        self.fixed_range = None  # computed on first use (reads every snapshot)

        self._build_pipeline()
        self._build_overlay()
        self._build_seismo_panel()
        self._build_window()
        self.update()

    # -- 1. source -> filter -> mapper -> actor ------------------------------
    def _build_pipeline(self):
        run = self.run
        self.lut = diverging_lut()
        if run.is_slice:
            # SOURCE: the slice itself, a (nx, ny, 1) rectilinear grid: no 3D
            # grid and no cutter. Its values are already in VTK point order.
            self.vcut = None
            self.grid = vtkRectilinearGrid()
            self.grid.SetDimensions(len(run.x), len(run.y), 1)
            self.grid.SetXCoordinates(numpy_to_vtk(run.x, deep=True))
            self.grid.SetYCoordinates(numpy_to_vtk(run.y, deep=True))
            self.grid.SetZCoordinates(numpy_to_vtk(run.z, deep=True))
            self.pressure = numpy_to_vtk(run.field(0), deep=True)
            self.pressure.SetName("pressure")
            self.grid.GetPointData().SetScalars(self.pressure)
            # MAPPER: any dataset -> colours through a lookup table.
            self.mapper = vtkDataSetMapper()
            self.mapper.SetInputData(self.grid)
        else:
            # SOURCE + FILTER: 3D grid cut by a z-plane (see VolumeCut).
            self.vcut = VolumeCut(run, self.z0)
            self.plane = self.vcut.plane
            # MAPPER: polygons + scalars -> colours through a lookup table.
            self.mapper = vtkPolyDataMapper()
            self.mapper.SetInputConnection(self.vcut.cutter.GetOutputPort())
        self.mapper.SetLookupTable(self.lut)
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.SetColorModeToMapScalars()

        # ACTOR: the thing placed in the scene.
        self.slice_actor = vtkActor()
        self.slice_actor.SetMapper(self.mapper)

        # Expected direct wavefront: r = vp * (t - 1/f0) (Ricker peak delay).
        self.front = vtkRegularPolygonSource()
        self.front.SetNumberOfSides(180)
        self.front.GeneratePolygonOff()
        self.front_actor = vtkActor()
        front_mapper = vtkPolyDataMapper()
        front_mapper.SetInputConnection(self.front.GetOutputPort())
        self.front_actor.SetMapper(front_mapper)
        self.front_actor.GetProperty().SetColor(0.1, 0.1, 0.1)
        self.front_actor.GetProperty().SetLineWidth(1.5)
        self.front_actor.GetProperty().SetLineStipplePattern(0xF0F0)

        # Receivers as dots (projected on top of the slice). Each dot has its
        # own RGBA colour so it can fade with its distance to the slice.
        pts = vtkPoints()
        for x, y, _ in run.receivers:
            pts.InsertNextPoint(x, y, 0.0)
        self.rcv_rgba = vtkUnsignedCharArray()
        self.rcv_rgba.SetNumberOfComponents(4)
        self.rcv_rgba.SetNumberOfTuples(len(run.receivers))
        vtk_to_numpy(self.rcv_rgba)[:] = (25, 205, 50, 255)
        rcv = vtkPolyData()
        rcv.SetPoints(pts)
        rcv.GetPointData().SetScalars(self.rcv_rgba)
        glyphs = vtkVertexGlyphFilter()
        glyphs.SetInputData(rcv)
        rcv_mapper = vtkPolyDataMapper()
        rcv_mapper.SetInputConnection(glyphs.GetOutputPort())
        rcv_mapper.SetColorModeToDirectScalars()  # use the RGBA as is, no LUT
        self.rcv_actor = vtkActor()
        self.rcv_actor.SetMapper(rcv_mapper)
        self.rcv_actor.GetProperty().SetPointSize(7)
        self.rcv_actor.GetProperty().RenderPointsAsSpheresOn()

        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetTitle("pressure")
        self.scalar_bar.SetLabelFormat("%.1e")
        self.scalar_bar.SetNumberOfLabels(3)
        self.scalar_bar.SetOrientationToHorizontal()
        self.scalar_bar.SetPosition(0.2, 0.075)
        self.scalar_bar.SetWidth(0.6)
        self.scalar_bar.SetHeight(0.08)

        # RENDERER: holds the actors and the camera.
        self.ren = vtkRenderer()
        self.ren.SetBackground(0.12, 0.12, 0.15)
        for a in (self.slice_actor, self.front_actor, self.rcv_actor, self.scalar_bar):
            self.ren.AddViewProp(a)

    # -- 2D overlay: metadata + buttons --------------------------------------
    def _build_overlay(self):
        # left side panel (its own renderer, so it never covers the slice)
        self.info_ren = vtkRenderer()
        self.info_ren.SetBackground(0.08, 0.08, 0.10)
        self.info = make_text("", 23, (0.05, 0.97), vjustify="top")
        controls = ("Controls\n hold < >  or\n   Left/Right\n space  play\n"
                    " s      seismos\n c      colour scale")
        if not self.run.is_slice:
            controls += "\n Up/Down slice z\n m      mid slice"
        self.help = make_text(controls, 18, (0.05, 0.03))
        self.help.GetTextProperty().SetColor(0.6, 0.6, 0.6)
        self.info_ren.AddViewProp(self.info)
        self.info_ren.AddViewProp(self.help)

        self.prev_btn = make_text("  <  ", 26, (0.35, 0.02), "centered", button=True)
        self.play_btn = make_text(" play ", 20, (0.50, 0.025), "centered", button=True)
        self.next_btn = make_text("  >  ", 26, (0.65, 0.02), "centered", button=True)
        self.more_btn = make_text(" See more: seismograms ", 16, (0.5, 0.98),
                                  "centered", "top", button=True)
        self.more_btn.SetVisibility(self.run.seis is not None)
        for a in (self.prev_btn, self.play_btn, self.next_btn, self.more_btn):
            self.ren.AddViewProp(a)

    # -- seismograms: a vtkChartXY drawn in a second renderer ----------------
    def _build_seismo_panel(self):
        run = self.run
        self.chart_ren = vtkRenderer()
        if run.seis is None:
            return
        table = vtkTable()
        col = numpy_to_vtk(run.seis_time, deep=True)
        col.SetName("time")
        table.AddColumn(col)
        # Each trace normalised by its own max and offset by its receiver id
        # (amplitude decays like 1/r so a global scale hides far receivers).
        for i in range(run.seis.shape[1]):
            trace = run.seis[:, i]
            amp = np.abs(trace).max() or 1.0
            col = numpy_to_vtk(i + 0.45 * trace / amp, deep=True)
            col.SetName(f"rcv_{i}")
            table.AddColumn(col)

        self.chart = vtkChartXY()
        self.chart.SetTitle("Seismograms (per-trace normalised)")
        self.chart.GetAxis(vtkAxis.BOTTOM).SetTitle("time (s)")
        self.chart.GetAxis(vtkAxis.LEFT).SetTitle("receiver id")
        for i in range(run.seis.shape[1]):
            line = self.chart.AddPlot(vtkChart.LINE)
            line.SetInputData(table, 0, i + 1)
            line.SetColor(40, 40, 40, 255)
            line.SetWidth(1.0)

        # time cursor = current snapshot
        self.cursor = vtkTable()
        for name in ("t", "y"):
            arr = vtkDoubleArray()
            arr.SetName(name)
            arr.SetNumberOfValues(2)
            self.cursor.AddColumn(arr)
        self.cursor.SetValue(0, 1, -1.0)
        self.cursor.SetValue(1, 1, float(run.seis.shape[1]))
        line = self.chart.AddPlot(vtkChart.LINE)
        line.SetInputData(self.cursor, 0, 1)
        line.SetColor(220, 30, 30, 255)
        line.SetWidth(2.0)

        scene = vtkContextScene()
        scene.AddItem(self.chart)
        chart_actor = vtkContextActor()
        chart_actor.SetScene(scene)
        self.chart_ren.SetBackground(1, 1, 1)
        self.chart_ren.AddActor(chart_actor)
        scene.SetRenderer(self.chart_ren)

    # -- window, interactor, events -----------------------------------------
    def _build_window(self):
        self.win = vtkRenderWindow()
        self.win.SetSize(1400, 850)
        self.win.SetWindowName("FUnTiDES slices" if self.run.is_slice else "FUnTiDES snapshots")
        self.win.AddRenderer(self.info_ren)
        self.win.AddRenderer(self.ren)

        self.iren = vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.win)
        # 2D navigation (pan / zoom, no rotation)
        self.style = vtkInteractorStyleImage()
        self.iren.SetInteractorStyle(self.style)
        # Observers on the style *replace* its default handlers, so we decide
        # whether a click is for a button or for the camera.
        self.style.AddObserver("LeftButtonPressEvent", self.on_press)
        self.style.AddObserver("LeftButtonReleaseEvent", self.on_release)
        self.iren.AddObserver("KeyPressEvent", self.on_key)
        self.iren.AddObserver("TimerEvent", self.on_timer)

        self.ren.GetActiveCamera().ParallelProjectionOn()
        self._layout()

    def start(self):
        self.iren.Initialize()
        self.win.Render()
        self.iren.Start()

    # ------------------------------------------------------------------------
    # Update everything that depends on the current snapshot
    # ------------------------------------------------------------------------
    def plane_z(self):
        return self.run.z[0] if self.run.is_slice else self.plane.GetOrigin()[2]

    def update(self):
        run, i = self.run, self.idx
        field = run.field(i)
        st = run.stats(i)
        if run.is_slice:
            vtk_to_numpy(self.pressure)[:] = field
            self.pressure.Modified()  # tells the pipeline the data changed
        else:
            slice_p = self.vcut.cut(field)

        if self.fixed_scale:
            if self.fixed_range is None:
                print("computing the colour range over all the snapshots...")
                self.fixed_range = run.robust_range()
            vmax = self.fixed_range
        else:
            vmax = st["absmax"] or 1.0
        self.mapper.SetScalarRange(-vmax, vmax)  # symmetric -> 0 is white
        self.lut.SetRange(-vmax, vmax)

        t = run.times[i]
        zc = self.plane_z()
        radius = run.vp * (t - 1.0 / run.f0)
        self.front_actor.SetVisibility(radius > 0)
        src = run.source
        self.front.SetCenter(src[0], src[1], zc + 1.0)
        self.front.SetRadius(max(radius, 1e-3))
        self.rcv_actor.SetPosition(0, 0, zc + 2.0)
        rcv_dz = np.abs(run.receivers[:, 2] - zc)
        alpha = np.clip(1.0 - rcv_dz / RCV_FADE_DIST, RCV_MIN_OPACITY, 1.0)
        vtk_to_numpy(self.rcv_rgba)[:, 3] = (255 * alpha).astype(np.uint8)
        self.rcv_rgba.Modified()

        rcv_dist = f"{rcv_dz.min():>6.0f} m" if len(rcv_dz) else "   n/a"
        field_block = (f"min   {st['min']:>10.3e}\n"
                       f"max   {st['max']:>10.3e}\n"
                       f"mean  {st['mean']:>10.3e}\n"
                       f"std   {st['std']:>10.3e}\n")
        if run.is_slice:  # statistics of the slice only
            stats_text = (f"\n-- slice z = {zc:.0f} m --\n"
                          f"(layer k = {run.snaps.k})\n" + field_block)
        else:
            stats_text = (f"\n-- volume --\n" + field_block +
                          f"\n-- slice z = {zc:.0f} m --\n"
                          f"min   {slice_p.min() if slice_p.size else np.nan:>10.3e}\n"
                          f"max   {slice_p.max() if slice_p.size else np.nan:>10.3e}\n")
        self.info.SetInput(
            f"{'Slice' if run.is_slice else 'Snapshot'} {i + 1:>3d} / {len(run.steps)}\n"
            f"step  {run.steps[i]:>6d}\n"
            f"time  {t:>8.3f} s\n"
            + stats_text +
            f"rcv dist  {rcv_dist}\n"
            f"\nexpected front\n"
            f"r = vp(t-1/f0) = {max(radius, 0):.0f} m\n"
            f"\ncolour: {'fixed' if self.fixed_scale else 'per snapshot'}"
        )

        if run.seis is not None:
            self.cursor.SetValue(0, 0, t)
            self.cursor.SetValue(1, 0, t)
            self.cursor.Modified()

        if hasattr(self, "win"):
            self.win.Render()

    def go(self, delta, wrap=False):
        n = len(self.run.steps)
        new = self.idx + delta
        new = new % n if wrap else min(max(new, 0), n - 1)
        if new != self.idx:
            self.idx = new
            self.update()
            return True
        return False

    # ------------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------------
    def _hit(self, actor):
        x, y = self.iren.GetEventPosition()
        bbox = [0.0] * 4
        actor.GetBoundingBox(self.ren, bbox)  # xmin,xmax,ymin,ymax around the anchor
        ax, ay = actor.GetPositionCoordinate().GetComputedDisplayValue(self.ren)
        return ax + bbox[0] <= x <= ax + bbox[1] and ay + bbox[2] <= y <= ay + bbox[3]

    def _start_timer(self):
        if self.timer_id is None:
            self.timer_id = self.iren.CreateRepeatingTimer(HOLD_INTERVAL_MS)

    def _stop_timer(self):
        if self.timer_id is not None:
            self.iren.DestroyTimer(self.timer_id)
            self.timer_id = None

    def on_press(self, obj, event):
        if self._hit(self.prev_btn) or self._hit(self.next_btn):
            self.set_playing(False)
            self.direction = -1 if self._hit(self.prev_btn) else 1
            self.go(self.direction)
            self._start_timer()  # keeps stepping while the button is held
        elif self._hit(self.play_btn):
            self.set_playing(not self.playing)
        elif self._hit(self.more_btn):
            self.toggle_seismos()
        else:
            self.style.OnLeftButtonDown()

    def on_release(self, obj, event):
        if self.direction and not self.playing:
            self.direction = 0
            self._stop_timer()
        self.style.OnLeftButtonUp()

    def on_timer(self, obj, event):
        if self.playing:
            self.go(1, wrap=True)
        elif self.direction and not self.go(self.direction):
            self._stop_timer()  # reached the first / last snapshot

    def set_playing(self, playing):
        self.playing = playing
        self.play_btn.SetInput(" pause " if playing else " play ")
        if playing:
            self._start_timer()
        else:
            self.direction = 0
            self._stop_timer()
        self.win.Render()

    def toggle_seismos(self):
        if self.run.seis is None:
            return
        self.show_seismos = not self.show_seismos
        if self.show_seismos:
            self.win.AddRenderer(self.chart_ren)
            self.more_btn.SetInput(" Hide seismograms ")
        else:
            self.win.RemoveRenderer(self.chart_ren)
            self.more_btn.SetInput(" See more: seismograms ")
        self._layout()
        self.win.Render()

    def _layout(self):
        """Viewports: [info panel | slice view | seismograms (optional)]."""
        split = 0.62 if self.show_seismos else 1.0
        self.info_ren.SetViewport(0.0, 0.0, 0.25, 1.0)
        self.ren.SetViewport(0.25, 0.0, split, 1.0)
        self.chart_ren.SetViewport(split, 0.0, 1.0, 1.0)
        self._fit_camera()

    def _fit_camera(self):
        """Fit the whole domain (width and height) in the slice viewport."""
        self.ren.ResetCamera()
        w, h = self.win.GetSize()
        vx0, vy0, vx1, vy1 = self.ren.GetViewport()
        aspect = (vx1 - vx0) * w / ((vy1 - vy0) * h)
        (x0, x1), (y0, y1) = self.run.bounds[:2]
        lx, ly = x1 - x0, y1 - y0
        # the domain uses the band [bottom, top] of the viewport height, the
        # strip below is left to the colour bar and buttons
        bottom, top = 0.2, 0.93
        scale = max(ly / (top - bottom), lx / (0.95 * aspect)) / 2
        cam = self.ren.GetActiveCamera()
        cam.SetParallelScale(scale)
        fx, fy, fz = cam.GetFocalPoint()
        px, py, pz = cam.GetPosition()
        cx = x0 + lx / 2
        cy = y0 + ly / 2 + (1 - top - bottom) * scale  # centre so the domain fits the band
        cam.SetFocalPoint(cx, cy, fz)
        cam.SetPosition(cx, cy, pz)

    def on_key(self, obj, event):
        key = self.iren.GetKeySym()
        if key in ("Left", "Right"):  # OS key-repeat gives the "hold" effect
            if self.playing:
                self.set_playing(False)
            self.go(-1 if key == "Left" else 1)
        elif key in ("Up", "Down", "m") and self.run.is_slice:
            pass  # one fixed plane in slice mode
        elif key in ("Up", "Down"):
            z = self.run.z
            zc = self.plane.GetOrigin()[2]
            k = int(np.argmin(np.abs(z - zc))) + (1 if key == "Up" else -1)
            origin = list(self.plane.GetOrigin())
            origin[2] = z[min(max(k, 0), len(z) - 1)]
            self.plane.SetOrigin(origin)
            self.update()
        elif key == "m":
            origin = list(self.plane.GetOrigin())
            origin[2] = self.run.source[2]
            self.plane.SetOrigin(origin)
            self.update()
        elif key == "space":
            self.set_playing(not self.playing)
        elif key == "s":
            self.toggle_seismos()
        elif key == "c":
            self.fixed_scale = not self.fixed_scale
            self.update()


# ----------------------------------------------------------------------------
# No-window modes: --benchmark, --check-slices
# ----------------------------------------------------------------------------
def benchmark(run, z, repeat):
    """Time, for every output file and `repeat` passes, the read (file ->
    numpy, no cache), the cut (volume mode: vtkCutter at z) and the statistics
    the viewer shows. Writes <run>/visu_bench_<mode>.csv. No window is made."""
    mode = "slice" if run.is_slice else "volume"
    vcut = None if run.is_slice else VolumeCut(run, z)
    where = (f"slice files at z = {run.z[0]:g} m" if run.is_slice
             else f"cut at z = {z:g} m")
    print(f"benchmark: {mode} mode, {where}, {len(run.steps)} files x {repeat} passes")

    rows = []
    for p in range(1, repeat + 1):  # pass 1: first read by this process
        tot_bytes, tot = 0, np.zeros(3)
        for i, step in enumerate(run.steps):
            nbytes = run.snaps.files[i].stat().st_size
            t0 = time.perf_counter()
            field = run.snaps.read(i)
            t1 = time.perf_counter()
            slice_p = vcut.cut(field) if vcut else None
            t2 = time.perf_counter()
            viewer_stats(field, slice_p)
            t3 = time.perf_counter()
            rows.append([mode, p, step, nbytes, t1 - t0, t2 - t1 if vcut else None, t3 - t2])
            tot_bytes += nbytes
            tot += (t1 - t0, t2 - t1, t3 - t2)
        cut = f"{tot[1]:.6f} s" if vcut else "n/a"
        print(f"pass {p}: {tot_bytes / 1e6:.3f} MB read, read {tot[0]:.6f} s, cut {cut}, "
              f"stats {tot[2]:.6f} s, total {tot.sum():.6f} s")

    out = run.root / f"visu_bench_{mode}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "pass", "step", "bytes_read", "t_read_s", "t_cut_s", "t_stats_s"])
        w.writerows(rows)  # t_cut_s is empty in slice mode
    print(f"written {out}")


def nearest_index(grid, v):
    """Index of the nearest value of the sorted array grid, for each v."""
    i = np.clip(np.searchsorted(grid, v), 1, len(grid) - 1)
    return np.where(v - grid[i - 1] <= grid[i] - v, i - 1, i)


def check_slices(root, fmt, dtype):
    """Compare, for every step, the volume cut at the slice z with the slice
    file, and the volume's node layer k (read directly) with the slice file.

    The cut is not bit-exact: on a node layer vtkCutter still interpolates
    along the edge from layer k-1 with t = 1, s0 + t*(s1 - s0) in float32, so
    the rounding of s1 - s0 shows (at most ~1 ulp of the larger neighbour).
    Returns 0 when the layer is identical and the cut is within float32
    rounding (eps * max|p|) for every step, 1 otherwise."""
    vol = Run(root, fmt, dtype)
    sl = Slices(root)
    nx, ny = len(sl.x), len(sl.y)
    # cut exactly on the grid layer nearest to the slice z (the volume grid
    # coordinates are rounded to 1 mm, the slice header ones are not)
    kz = int(np.argmin(np.abs(vol.z - sl.z[0])))
    z = vol.z[kz]
    vcut = VolumeCut(vol, z)
    print(f"slices: layer k = {sl.k}, z = {sl.z[0]:g} m; volume cut at z = {z:g} m "
          f"(grid layer {kz})")
    ok = kz == sl.k and (len(vol.x), len(vol.y)) == (nx, ny)
    if not ok:
        print("MISMATCH: the slice layer or grid size differs from the volume")

    common = sorted(set(vol.steps) & set(sl.steps))
    only = sorted(set(vol.steps) ^ set(sl.steps))
    if only:
        ok = False
        print(f"steps with only a snapshot or only a slice: {only}")
    worst = worst_layer = 0.0
    for step in common:
        field = vol.snaps.read(vol.steps.index(step))
        slice_p = sl.read(sl.steps.index(step))
        cut = vcut.cut(field).astype(np.float64)
        if cut.size == 0:
            raise SystemExit(f"step {step}: the plane z = {z:g} m does not cut the volume")
        xy = vcut.cut_points()
        ix, iy = nearest_index(sl.x, xy[:, 0]), nearest_index(sl.y, xy[:, 1])
        offset = max(np.abs(sl.x[ix] - xy[:, 0]).max(), np.abs(sl.y[iy] - xy[:, 1]).max())
        ref = slice_p.reshape(ny, nx)[iy, ix].astype(np.float64)
        diff = np.abs(cut - ref).max()
        tol = np.finfo(np.float32).eps * np.abs(field).max()
        layer = field.reshape(len(vol.z), -1)[kz]  # no cutter: the nodes themselves
        diff_layer = np.abs(layer.astype(np.float64) - slice_p).max()
        covered = np.unique(iy * nx + ix).size
        ok &= covered == nx * ny and diff <= tol and diff_layer == 0
        worst, worst_layer = max(worst, diff), max(worst_layer, diff_layer)
        print(f"step {step:>6d}: {cut.size} cut points, {covered}/{nx * ny} slice nodes "
              f"covered, max |x,y offset| {offset:.1e} m, max |cut - slice| = {diff:g} "
              f"(float32 tol {tol:.1e}), max |layer k - slice| = {diff_layer:g}")
    print(f"max abs difference over {len(common)} steps: cut vs slice {worst:g}, "
          f"layer k vs slice {worst_layer:g} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA,
                    help="run folder, or folder of runs (latest is shown) "
                         "(default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="list the runs found in DATA and exit")
    ap.add_argument("--format", default="auto",
                    choices=["auto", "bin", "bin-header", "csv", "csv-header", "fake"],
                    help="snapshot format (default: auto-detect, binary first)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "<f4", "<f8"],
                    help="float type of binary files (float64 if built with USE_DOUBLE)")
    ap.add_argument("--seismos", type=Path, help="directory of the seismogram files")
    ap.add_argument("--dt", type=float, help="time step (s), to convert steps to times")
    ap.add_argument("--source", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help=f"source position (default {PROXY_DEFAULTS['source']})")
    ap.add_argument("--vp", type=float, help=f"P velocity (default {PROXY_DEFAULTS['vp']})")
    ap.add_argument("--f0", type=float, help=f"Ricker frequency (default {PROXY_DEFAULTS['f0']})")
    ap.add_argument("--slices", action="store_true",
                    help="slice mode: show DIR/slices (proxy --use-slices) instead of the snapshots")
    ap.add_argument("--z", type=float,
                    help="z of the cut plane in volume mode (default: source z; with "
                         "--benchmark: the z of the slices if the run has some)")
    ap.add_argument("--benchmark", action="store_true",
                    help="no window: time read / cut / statistics per file, "
                         "write DIR/visu_bench_<mode>.csv")
    ap.add_argument("--repeat", type=int, default=3,
                    help="number of passes over the files for --benchmark (default: %(default)s)")
    ap.add_argument("--check-slices", action="store_true",
                    help="no window: compare the volume cut at the slice z with the slice files")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    if args.slices and args.z is not None:
        ap.error("--z only applies to the volume mode (a slice has its own z)")
    if args.check_slices:
        root = resolve_run(args.data, args.format)
        raise SystemExit(check_slices(root, args.format, args.dtype))
    if args.list:
        runs = find_runs(args.data)
        for d in runs:
            fmt, files, _ = find_snapshots(d, "auto")
            print(f"{d.name:<24} {len(files):>4} snapshots ({fmt})")
        if not runs:
            print(f"no run with snapshots in {args.data}")
        return
    root = resolve_run(args.data, args.format, args.slices)
    run = Run(root, args.format, args.dtype, args.seismos,
              args.dt, args.source, args.vp, args.f0, slices=args.slices)
    if args.benchmark:
        z = args.z
        if z is None and not args.slices:
            # same plane as the slices, so both modes cut the same plane
            z = Slices(root).z[0] if has_slices(root) else run.source[2]
        benchmark(run, z, args.repeat)
        return
    Viewer(run, args.z).start()


if __name__ == "__main__":
    main()
