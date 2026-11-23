//************************************************************************
//  In-situ visualisation (TP3, part 1)
//
//  insitu_ppm.h: renders a 2D slice of the pressure field as a PPM image
//  during the run (resampling on a uniform grid, normalisation, colormap,
//  P3/P6 encoding), either on the GPU or on the host.
//************************************************************************

#ifndef INSITU_PPM_H_
#define INSITU_PPM_H_

#include <data_type.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

enum class PpmPlane { XY, XZ, YZ };
enum class PpmFormat { P3, P6 };  // P3: text, P6: binary
enum class PpmColormap { Coolwarm, Viridis, Jet };
// Local: bounds of the current slice. Running: bounds of all the slices
// rendered so far. Fixed: bounds given by the user.
enum class PpmNorm { Local, Running, Fixed };
// Nearest: nearest node. Linear: bilinear between the 4 surrounding nodes.
// Gll: the SEM interpolation itself (Lagrange basis of the element).
enum class PpmInterp { Nearest, Linear, Gll };
// Gpu: reduction + rendering kernels on the Kokkos default execution space,
// the host only reads the final RGB image. Host: the slice is read from the
// host and rendered by a serial CPU loop.
enum class PpmDevice { Gpu, Host };

// Text <-> enum (throw std::invalid_argument on an unknown name)
PpmPlane parsePpmPlane(const std::string& s);
PpmFormat parsePpmFormat(const std::string& s);
PpmColormap parsePpmColormap(const std::string& s);
PpmNorm parsePpmNorm(const std::string& s);
PpmInterp parsePpmInterp(const std::string& s);
PpmDevice parsePpmDevice(const std::string& s);

struct PpmConfig
{
  std::string folder = "ppm";
  PpmPlane plane = PpmPlane::XY;
  float position = 0;  // along the normal axis (m), snapped to a node layer
  PpmFormat format = PpmFormat::P6;
  PpmColormap colormap = PpmColormap::Coolwarm;
  PpmNorm norm = PpmNorm::Local;
  bool symmetric = true;  // bounds [-max|p|, +max|p|] instead of [min, max]
  float vmin = 0;         // PpmNorm::Fixed only
  float vmax = 0;         // PpmNorm::Fixed only
  PpmInterp interp = PpmInterp::Linear;
  PpmDevice device = PpmDevice::Gpu;
  int width = 0;  // pixels along the first axis of the plane, 0 = automatic
  bool sync_writes = false;
};

// steady_clock seconds accumulated over all the images of the run
struct PpmMetrics
{
  double total_s = 0;     // whole output() call
  double extract_s = 0;   // host mode: copy of the slice nodes to the host
  double minmax_s = 0;    // bounds of the slice (GPU reduction or host loop)
  double render_s = 0;    // resampling + normalisation + colormap
  double transfer_s = 0;  // gpu mode: copy of the RGB image to the host
  double encode_s = 0;    // P3: binary -> text. P6: nothing to do
  double write_s = 0;     // fopen + fwrite + fflush + fclose (no fsync)
  double fsync_s = 0;     // fsync (sync_writes only)
  int count = 0;
  std::uintmax_t image_bytes = 0;  // *.ppm files
  std::uintmax_t meta_bytes = 0;   // ppm-header.json + ppm-index.csv
};

class InSituPpm
{
 public:
  /**
   * @brief Choose the node layer and the image size, build the resampling
   * tables and the colormap, create the folder and write ppm-header.json.
   * @param node_coords coordinates of the node layers along x, y and z
   * @param order polynomial order of the elements (for PpmInterp::Gll)
   */
  void init(const PpmConfig& cfg,
            const std::array<std::vector<float>, 3>& node_coords, int order,
            float dt, int interval);

  /**
   * @brief Render column `column` of `field` (node i + j*nx + k*nx*ny) and
   * write ppm-<timestep>.ppm. Every phase is timed separately in metrics().
   */
  void output(const arrayReal& field, int column, int timestep, float time);

  // Close ppm-index.csv (one line per image: bounds used for the colours)
  void finalize();

  const PpmMetrics& metrics() const { return metrics_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int layer() const { return layer_; }
  float layerPosition() const { return layer_pos_; }
  double pixelSize() const { return pixel_size_; }
  std::string imagePath(int timestep) const;

 private:
  PpmConfig cfg_;
  PpmMetrics metrics_;

  // slice: node (a, b) of the plane is the global node
  // base_ + a * stride_u_ + b * stride_v_
  int axis_u_ = 0, axis_v_ = 1, axis_n_ = 2;
  int nu_ = 0, nv_ = 0;
  std::size_t base_ = 0, stride_u_ = 0, stride_v_ = 0;
  int layer_ = 0;        // node layer along the normal axis
  float layer_pos_ = 0;  // its coordinate

  // image
  int width_ = 0, height_ = 0;
  double pixel_size_ = 0;  // m
  int stencil_ = 1;        // nodes per axis used for one pixel

  // running bounds (PpmNorm::Running), or the fixed ones
  float lo_ = 0, hi_ = 0;
  bool has_bounds_ = false;

  // host side: resampling tables, colormap, slice and image buffers
  std::vector<int> first_u_, first_v_;      // first stencil node per column/row
  std::vector<float> weight_u_, weight_v_;  // stencil_ weights per column/row
  std::vector<unsigned char> lut_;          // colormap table (RGB triplets)
  std::vector<float> slice_buf_;            // host mode: nu * nv values
  std::vector<unsigned char> rgb_;          // 3 * width * height bytes
  std::string text_;                        // P3 body, reused

  // device side (gpu mode only): same tables + the image, in MemSpace
  Kokkos::View<int*, MemSpace> d_first_u_, d_first_v_;
  Kokkos::View<float*, MemSpace> d_weight_u_, d_weight_v_;
  Kokkos::View<unsigned char*, MemSpace> d_lut_;
  Kokkos::View<unsigned char*, MemSpace> d_rgb_;

  std::ofstream index_;
  std::string index_path_;

  void writeHeader(const std::array<std::vector<float>, 3>& node_coords,
                   int order, float dt, int interval);
};

#endif  // INSITU_PPM_H_
