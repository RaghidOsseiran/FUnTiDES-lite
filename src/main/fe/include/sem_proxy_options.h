#pragma once

#include <cxxopts.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <fstream>

#include "insitu_ppm.h"

class SemProxyOptions
{
 public:
  // Defaults
  int order = 2;
  int ex = 50, ey = 50, ez = 50;
  float lx = 2000.f, ly = 2000.f, lz = 2000.f;
  float srcx = 1010.f, srcy = 1010.f, srcz = 1010.f;
  float rcvx = 1410.f, rcvy = 1010.f, rcvz = 1010.f;
  std::string implem = "makutu";  // makutu|shiva
  std::string method = "sem";     // sem|dg
  std::string mesh = "cartesian";
  std::string receiver_positions_file = "";
  float dt = 0.001;
  float timemax = 1.5;
  bool autodt = false;
  // sponge boundaries parameters
  float boundaries_size = 0;
  bool surface_sponge = false;
  float taper_delta = 0.015;
  // Boolean to tell if the model is charged on nodes or on element
  bool isModelOnNodes = false;
  bool isElastic = false;

  bool useSnapshots = false;
  int snapshotsInterval = 10;
  std::string snapshotsFolder = ".";
  bool snapshotsUseCSV = 0;
  bool snapshotsUseHeaders = 0;

  std::string seismogramsFolder = "seismograms_recording";
  bool seismoUseHeader = 0;

  // in-situ
  int inSituInterval = 1;
  bool inSituStats = false;
  bool inSituQuart = false;
  // in-situ seismograms: same metrics as scripts/seismos_adhoc.py
  bool inSituSeismosStats = false;  // min, max, mean, var per receiver
  bool inSituSeismosPeaks = false;  // peak_abs, t_peak per receiver

  // z-slices: written at every snapshot output step (same snapshotsInterval)
  bool useSlices = false;
  std::string slicesFolder = "slices";
  float sliceZ = srcz;  // default: source z

  // in-situ visualisation: PPM image of a slice of the pressure field
  bool usePpm = false;
  std::string ppmFolder = "ppm";
  int ppmInterval = 0;            // steps between two images, 0: snapshotsInterval
  std::string ppmPlane = "xy";    // xy|xz|yz
  // position of the plane along its normal axis (m); NaN: source coordinate
  float ppmPos = std::numeric_limits<float>::quiet_NaN();
  std::string ppmFormat = "p6";          // p3 (text) | p6 (binary)
  std::string ppmColormap = "coolwarm";  // coolwarm|viridis|jet
  std::string ppmNorm = "local";         // local|running|fixed
  bool ppmMinMaxBounds = false;  // bounds [min, max] instead of [-max|p|, max|p|]
  float ppmVmax = std::numeric_limits<float>::quiet_NaN();  // ppm-norm fixed
  float ppmVmin = std::numeric_limits<float>::quiet_NaN();  // NaN: -ppmVmax
  std::string ppmInterp = "linear";  // nearest|linear|gll
  std::string ppmDevice = "gpu";     // gpu|host
  int ppmWidth = 0;                  // image width in pixels, 0: automatic

  // flush each snapshot/slice file to disk (fsync) inside the timed region
  bool syncWrites = false;
  std::string metricsFile = "metrics.json";


  void validate() const
  {
    if (order < 1) throw std::runtime_error("order must be >= 1");
    if (ex <= 0 || ey <= 0 || ez <= 0)
      throw std::runtime_error("ex/ey/ez must be > 0");
    if (lx <= 0 || ly <= 0 || lz <= 0)
      throw std::runtime_error("lx/ly/lz must be > 0");
    if (!receiver_positions_file.empty() && !std::ifstream(receiver_positions_file))
      throw std::runtime_error("The file does not exist");

    if (useSnapshots) {
      if (useSnapshots &&
        !(std::filesystem::exists(snapshotsFolder) && std::filesystem::is_directory(snapshotsFolder)))
        throw std::runtime_error("snapshot folder is incorrect");
    }

    // used as a modulo in computeInSitu / computeInSituSeismos
    if (inSituInterval <= 0)
      throw std::runtime_error("ins-interval must be > 0");

    // slices rely on the structured numbering i + j*nx + k*nx*ny
    if (useSlices && mesh != "cartesian")
      throw std::runtime_error("--use-slices requires --mesh cartesian");
    if (useSlices && snapshotsInterval <= 0)
      throw std::runtime_error("snapshots-interval must be > 0");

    if (usePpm)
    {
      // same structured numbering as the slices
      if (mesh != "cartesian")
        throw std::runtime_error("--use-ppm requires --mesh cartesian");
      if (ppmInterval < 0 || (ppmInterval == 0 && snapshotsInterval <= 0))
        throw std::runtime_error(
            "ppm-interval (or snapshots-interval) must be > 0");
      // each parser throws with the list of the accepted values
      parsePpmPlane(ppmPlane);
      parsePpmFormat(ppmFormat);
      parsePpmColormap(ppmColormap);
      parsePpmInterp(ppmInterp);
      parsePpmDevice(ppmDevice);
      if (parsePpmNorm(ppmNorm) == PpmNorm::Fixed)
      {
        if (std::isnan(ppmVmax))
          throw std::runtime_error("--ppm-norm fixed requires --ppm-vmax");
        if (!(ppmVmax > (std::isnan(ppmVmin) ? -ppmVmax : ppmVmin)))
          throw std::runtime_error("ppm-vmax must be > ppm-vmin (default "
                                   "-ppm-vmax)");
      }
      if (ppmWidth != 0 && ppmWidth < 2)
        throw std::runtime_error("ppm-width must be >= 2 (0: automatic)");
    }
  }

  // Bind CLI flags to this instance (no --help here)
  static void bind_cli(cxxopts::Options& opts, SemProxyOptions& o)
  {
    opts.add_options()("o,order", "Order of approximation",
                       cxxopts::value<int>(o.order))(
        "ex", "Number of elements on X (Cartesian mesh)",
        cxxopts::value<int>(o.ex))("ey",
                                   "Number of elements on Y (Cartesian mesh)",
                                   cxxopts::value<int>(o.ey))(
        "ez", "Number of elements on Z (Cartesian mesh)",
        cxxopts::value<int>(o.ez))("lx", "Domain size X (Cartesian)",
                                   cxxopts::value<float>(o.lx))(
        "ly", "Domain size Y (Cartesian)", cxxopts::value<float>(o.ly))(
        "lz", "Domain size Z (Cartesian)", cxxopts::value<float>(o.lz))(
        "implem", "Implementation: makutu|shiva",
        cxxopts::value<std::string>(o.implem))(
        "method", "Method: sem|dg", cxxopts::value<std::string>(o.method))(
        "mesh", "Mesh: cartesian|ucartesian",
        cxxopts::value<std::string>(o.mesh))(
        "dt", "Time step selection in s (default = 0.001s)",
        cxxopts::value<float>(o.dt))(
        "timemax", "Duration of the simulation in s (default = 1.5s)",
        cxxopts::value<float>(o.timemax))(
        "auto-dt", "Select automatique dt via CFL equation.",
        cxxopts::value<bool>(o.autodt))(
        "boundaries-size", "Size of absorbing boundaries (meters)",
        cxxopts::value<float>(o.boundaries_size))(
        "sponge-surface", "Considere the surface's nodes as non sponge nodes",
        cxxopts::value<bool>(o.surface_sponge))(
        "taper-delta", "Taper delta for sponge boundaries value",
        cxxopts::value<float>(o.taper_delta))(
        "is-model-on-nodes",
        "Boolean to tell if the model is charged on nodes (true) or on element "
        "(false)",
        cxxopts::value<bool>(o.isModelOnNodes))(
        "is-elastic", "Elastic simulation", cxxopts::value<bool>(o.isElastic))(
        "receiver-positions-file", "Receivers' positions",
        cxxopts::value<std::string>(o.receiver_positions_file))(
        "seismograms-folder", "Folder where the seismograms recording are stored", cxxopts::value<std::string>(o.seismogramsFolder)
        )(
        "seismo-use-header", "Use header for the seismograms recording", cxxopts::value<bool>(o.seismoUseHeader)
        )
        ("use-snapshots", "Create snapshots of the simulation", cxxopts::value<bool>(o.useSnapshots))
        ("snapshots-interval", "Number of simulation steps between each snapshots", cxxopts::value<int>(o.snapshotsInterval))
        ("snapshots-folder", "Folder where the snapshots are stored", cxxopts::value<std::string>(o.snapshotsFolder))
        ("snapshots-use-headers", "Use headers for the snapshots points", cxxopts::value<bool>(o.snapshotsUseHeaders))
        ("snapshots-use-csv", "Use csv for the snapshots format", cxxopts::value<bool>(o.snapshotsUseCSV))
        ("use-slices", "Write a z-slice at every snapshot output step (snapshots-interval)", cxxopts::value<bool>(o.useSlices))
        ("slices-folder", "Folder where the slices are stored (created if missing)", cxxopts::value<std::string>(o.slicesFolder))
        ("slice-z", "z of the slice in m, snapped to the nearest node layer (default: source z)", cxxopts::value<float>(o.sliceZ))
        ("use-ppm", "In-situ visualisation: write a PPM image of a slice of the pressure field", cxxopts::value<bool>(o.usePpm))
        ("ppm-folder", "Folder where the PPM images are stored (created if missing)", cxxopts::value<std::string>(o.ppmFolder))
        ("ppm-interval", "Number of simulation steps between two images (default: snapshots-interval)", cxxopts::value<int>(o.ppmInterval))
        ("ppm-plane", "Cut plane: xy|xz|yz (default xy)", cxxopts::value<std::string>(o.ppmPlane))
        ("ppm-pos", "Position of the plane along its normal axis in m, snapped to the nearest node layer (default: source coordinate)", cxxopts::value<float>(o.ppmPos))
        ("ppm-format", "p3 (text) | p6 (binary, default)", cxxopts::value<std::string>(o.ppmFormat))
        ("ppm-colormap", "coolwarm (default) | viridis | jet", cxxopts::value<std::string>(o.ppmColormap))
        ("ppm-norm", "Bounds of the colour scale: local (current image, default) | running (all the images so far) | fixed (--ppm-vmax)", cxxopts::value<std::string>(o.ppmNorm))
        ("ppm-minmax-bounds", "local/running: use [min, max] instead of the symmetric [-max|p|, +max|p|]", cxxopts::value<bool>(o.ppmMinMaxBounds))
        ("ppm-vmax", "Upper bound of the colour scale for --ppm-norm fixed", cxxopts::value<float>(o.ppmVmax))
        ("ppm-vmin", "Lower bound of the colour scale for --ppm-norm fixed (default: -ppm-vmax)", cxxopts::value<float>(o.ppmVmin))
        ("ppm-interp", "Resampling of the nodes on the pixels: nearest | linear (default) | gll (Lagrange basis of the element)", cxxopts::value<std::string>(o.ppmInterp))
        ("ppm-device", "gpu (default): the image is built by Kokkos kernels, the host reads only the image | host: the slice is read and rendered on the CPU", cxxopts::value<std::string>(o.ppmDevice))
        ("ppm-width", "Image width in pixels, the height follows (default: pixel = smallest node spacing)", cxxopts::value<int>(o.ppmWidth))
        ("sync-writes", "fsync each snapshot/slice/image file after writing it (timed)", cxxopts::value<bool>(o.syncWrites))
        ("metrics-file", "JSON file where run timers and byte counts are written", cxxopts::value<std::string>(o.metricsFile))
        ("ins-interval", "In situ timestep interval (default 1)", cxxopts::value<int>(o.inSituInterval))
        ("ins-compute-stats", "Compute the stats of all values in-situ", cxxopts::value<bool>(o.inSituStats))
        ("ins-compute-quartiles", "Compute the values of quartiles (1st, 2nd, 3rd)", cxxopts::value<bool>(o.inSituQuart))
        ("ins-seismos-compute-stats", "Compute in-situ min, max, mean and variance of each seismogram", cxxopts::value<bool>(o.inSituSeismosStats))
        ("ins-seismos-compute-peaks", "Compute in-situ the absolute peak of each seismogram and its time", cxxopts::value<bool>(o.inSituSeismosPeaks));
  }
};
