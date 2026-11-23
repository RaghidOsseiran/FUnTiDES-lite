//************************************************************************
//  In-situ visualisation (TP3, part 1)
//
//  insitu_ppm.cc: slice of the pressure field -> PPM image, during the run
//************************************************************************

#include "insitu_ppm.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace
{
using sclock = std::chrono::steady_clock;

double seconds(sclock::time_point t0, sclock::time_point t1)
{
  return std::chrono::duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------- names

const char* name(PpmPlane p)
{
  return p == PpmPlane::XY ? "xy" : p == PpmPlane::XZ ? "xz" : "yz";
}
const char* name(PpmFormat f) { return f == PpmFormat::P3 ? "p3" : "p6"; }
const char* name(PpmColormap c)
{
  return c == PpmColormap::Coolwarm  ? "coolwarm"
         : c == PpmColormap::Viridis ? "viridis"
                                     : "jet";
}
const char* name(PpmNorm n)
{
  return n == PpmNorm::Local     ? "local"
         : n == PpmNorm::Running ? "running"
                                 : "fixed";
}
const char* name(PpmInterp i)
{
  return i == PpmInterp::Nearest  ? "nearest"
         : i == PpmInterp::Linear ? "linear"
                                  : "gll";
}
const char* name(PpmDevice d) { return d == PpmDevice::Gpu ? "gpu" : "host"; }

// Shortest decimal text that reads back as the same float (0.001, not
// 0.00100000005), with '.' as decimal separator whatever the locale.
std::string num(float v)
{
  std::string s;
  for (int p = 6; p <= std::numeric_limits<float>::max_digits10; ++p)
  {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(p) << v;
    s = os.str();
    if (std::strtof(s.c_str(), nullptr) == v) break;
  }
  return s;
}

// ------------------------------------------------------------ colormaps

// "Smooth cool warm" of K. Moreland (diverging, same lightness at both ends,
// neutral grey in the middle): 33 samples, linearly interpolated.
const unsigned char kCoolwarm[33][3] = {
    {59, 76, 192},   {68, 90, 204},   {77, 104, 215},  {87, 117, 225},
    {98, 130, 234},  {108, 142, 241}, {119, 154, 247}, {130, 165, 251},
    {141, 176, 254}, {152, 185, 255}, {163, 194, 255}, {174, 201, 253},
    {184, 208, 249}, {194, 213, 244}, {204, 217, 238}, {213, 219, 230},
    {221, 221, 221}, {229, 216, 209}, {236, 211, 197}, {241, 204, 185},
    {245, 196, 173}, {247, 187, 160}, {247, 177, 148}, {247, 166, 135},
    {244, 154, 123}, {241, 141, 111}, {236, 127, 99},  {229, 112, 88},
    {222, 96, 77},   {213, 80, 66},   {203, 62, 56},   {192, 40, 47},
    {180, 4, 38}};

// viridis (sequential, perceptually uniform): 17 samples of the matplotlib
// table, linearly interpolated.
const unsigned char kViridis[17][3] = {
    {68, 1, 84},    {72, 24, 106},  {71, 45, 123},  {66, 64, 134},
    {59, 82, 139},  {51, 99, 141},  {44, 114, 142}, {38, 130, 142},
    {33, 145, 140}, {31, 160, 136}, {40, 174, 128}, {63, 188, 115},
    {94, 201, 98},  {132, 212, 75}, {173, 220, 48}, {216, 226, 25},
    {253, 231, 37}};

unsigned char toByte(double v)
{
  return static_cast<unsigned char>(std::lround(std::clamp(v, 0.0, 255.0)));
}

// Number of colours of the table. Odd on purpose: the normalised value 0.5
// (p = 0 with symmetric bounds) is then the centre of the middle entry, the
// exact neutral colour of a diverging palette. With 256 entries, 0.5 would
// sit on the border between two entries and the numerical noise around 0
// would flip between two colours.
constexpr int kLutSize = 255;

// kLutSize RGB triplets: entry i is the colour of the normalised value
// i / (kLutSize - 1)
std::vector<unsigned char> buildLut(PpmColormap colormap)
{
  std::vector<unsigned char> lut(kLutSize * 3);
  for (int i = 0; i < kLutSize; ++i)
  {
    const double t = i / double(kLutSize - 1);
    if (colormap == PpmColormap::Jet)
    {
      // the classic rainbow (MATLAB definition), kept for comparison
      lut[3 * i + 0] = toByte(255 * (1.5 - std::fabs(4 * t - 3)));
      lut[3 * i + 1] = toByte(255 * (1.5 - std::fabs(4 * t - 2)));
      lut[3 * i + 2] = toByte(255 * (1.5 - std::fabs(4 * t - 1)));
      continue;
    }
    const bool cool = colormap == PpmColormap::Coolwarm;
    const unsigned char(*table)[3] = cool ? kCoolwarm : kViridis;
    const int last = (cool ? 33 : 17) - 1;
    const double pos = t * last;
    const int s = std::min(static_cast<int>(pos), last - 1);
    const double w = pos - s;
    for (int c = 0; c < 3; ++c)
      lut[3 * i + c] = toByte((1 - w) * table[s][c] + w * table[s + 1][c]);
  }
  return lut;
}

// ----------------------------------------------------------- resampling

int stencilSize(PpmInterp interp, int order)
{
  return interp == PpmInterp::Nearest  ? 1
         : interp == PpmInterp::Linear ? 2
                                       : order + 1;
}

double minSpacing(const std::vector<float>& c)
{
  double d = std::numeric_limits<double>::infinity();
  for (std::size_t a = 0; a + 1 < c.size(); ++a)
    d = std::min(d, double(c[a + 1]) - double(c[a]));
  return d;
}

// One axis of the image: n_pix pixels evenly spaced from the first to the
// last node of coordinates c (non uniform for GLL nodes of order >= 3).
// For each pixel: the first node of its stencil and `stencil` weights, so
// that value(pixel) = sum_a weight[a] * value(node first + a).
void buildStencil(const std::vector<float>& c, int order, int n_pix,
                  PpmInterp interp, std::vector<int>& first,
                  std::vector<float>& weight)
{
  const int n = static_cast<int>(c.size());
  const int stencil = stencilSize(interp, order);
  first.assign(n_pix, 0);
  weight.assign(std::size_t(n_pix) * stencil, 0.f);

  const double x0 = c.front();
  const double length = double(c.back()) - x0;
  for (int p = 0; p < n_pix; ++p)
  {
    const double x = (p == n_pix - 1) ? double(c.back())
                                      : x0 + length * p / (n_pix - 1);
    // node interval [c[a], c[a + 1]] that holds x
    int a = static_cast<int>(std::upper_bound(c.begin(), c.end(), x) -
                             c.begin()) - 1;
    a = std::clamp(a, 0, n - 2);
    float* w = &weight[std::size_t(p) * stencil];

    if (interp == PpmInterp::Nearest)
    {
      first[p] = (x - c[a] <= c[a + 1] - x) ? a : a + 1;
      w[0] = 1.f;
    }
    else if (interp == PpmInterp::Linear)
    {
      const double s = (x - c[a]) / (double(c[a + 1]) - c[a]);
      first[p] = a;
      w[0] = static_cast<float>(1 - s);
      w[1] = static_cast<float>(s);
    }
    else
    {
      // element that holds x: nodes a0 .. a0 + order, Lagrange basis on them
      const int a0 = (a / order) * order;
      first[p] = a0;
      for (int m = 0; m <= order; ++m)
      {
        double l = 1;
        for (int q = 0; q <= order; ++q)
          if (q != m)
            l *= (x - c[a0 + q]) / (double(c[a0 + m]) - c[a0 + q]);
        w[m] = static_cast<float>(l);
      }
    }
  }
}

// -------------------------------------------------------------- kernels
// Plain structs with raw pointers: the same code runs in a Kokkos kernel
// (pointers into MemSpace views) and in a host loop (pointers into
// std::vector), so both paths compute the image the same way.

// Value of node (a, b) of the slice: data[origin + a * step_u + b * step_v]
struct SliceAccess
{
  const float* data;
  std::ptrdiff_t origin, step_u, step_v;
};

struct MinMaxKernel
{
  SliceAccess slice;
  int nu;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, Kokkos::MinMax<float>::value_type& mm) const
  {
    const int b = n / nu;
    const int a = n - b * nu;
    const float v =
        slice.data[slice.origin + a * slice.step_u + b * slice.step_v];
    if (v < mm.min_val) mm.min_val = v;
    if (v > mm.max_val) mm.max_val = v;
  }
};

// One pixel: resampling (separable stencil), normalisation, colormap
struct RenderKernel
{
  SliceAccess slice;
  const int* first_u;
  const int* first_v;
  const float* weight_u;
  const float* weight_v;
  int stencil;
  const unsigned char* lut;
  int lut_last;  // index of the last colour of the table
  unsigned char* rgb;
  int width, height;
  float lo, inv_range, bias;  // t = (p - lo) * inv_range + bias, in [0, 1]

  KOKKOS_INLINE_FUNCTION
  void operator()(const int pix) const
  {
    const int row = pix / width;
    const int px = pix - row * width;
    const int py = height - 1 - row;  // first row of the image = largest v

    const float* wu = weight_u + std::ptrdiff_t(px) * stencil;
    const float* wv = weight_v + std::ptrdiff_t(py) * stencil;
    const std::ptrdiff_t corner = slice.origin + first_u[px] * slice.step_u +
                                  first_v[py] * slice.step_v;
    float value = 0.f;
    for (int b = 0; b < stencil; ++b)
    {
      float line = 0.f;
      for (int a = 0; a < stencil; ++a)
        line += wu[a] * slice.data[corner + a * slice.step_u + b * slice.step_v];
      value += wv[b] * line;
    }

    float t = (value - lo) * inv_range + bias;
    t = t > 0.f ? (t < 1.f ? t : 1.f) : 0.f;  // clamp (a NaN gives 0)
    const unsigned char* colour =
        lut + 3 * static_cast<int>(t * lut_last + 0.5f);
    unsigned char* out = rgb + 3 * std::ptrdiff_t(pix);
    out[0] = colour[0];
    out[1] = colour[1];
    out[2] = colour[2];
  }
};

template <class T>
Kokkos::View<T*, MemSpace> toDevice(const std::vector<T>& v, const char* label)
{
  Kokkos::View<T*, MemSpace> d(
      Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing),
      v.size());
  Kokkos::View<const T*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h(v.data(), v.size());
  Kokkos::deep_copy(d, h);
  return d;
}

// ------------------------------------------------------------- encoding

// P3 body: decimal text, at most 5 pixels (59 characters) per line, as the
// format asks for lines of 70 characters at most.
void encodeP3(const std::vector<unsigned char>& rgb, int width, int height,
              std::string& out)
{
  static char digits[256][3];
  static unsigned char length[256];
  static bool ready = false;
  if (!ready)
  {
    for (int v = 0; v < 256; ++v)
    {
      char buf[4];
      length[v] = static_cast<unsigned char>(
          std::snprintf(buf, sizeof(buf), "%d", v));
      std::memcpy(digits[v], buf, length[v]);
    }
    ready = true;
  }

  out.resize(rgb.size() * 4);  // 3 digits + 1 separator at most per value
  char* p = out.data();
  const unsigned char* v = rgb.data();
  for (int row = 0; row < height; ++row)
  {
    for (int px = 0; px < width; ++px)
    {
      for (int c = 0; c < 3; ++c, ++v)
      {
        std::memcpy(p, digits[*v], length[*v]);
        p += length[*v];
        *p++ = ' ';
      }
      if (px % 5 == 4 || px == width - 1) p[-1] = '\n';
    }
  }
  out.resize(p - out.data());
}

[[noreturn]] void ioError(const std::string& what, const std::string& path)
{
  throw std::runtime_error(what + " " + path + ": " + std::strerror(errno));
}

template <class Enum>
Enum parse(const std::string& s, const char* option,
           std::initializer_list<Enum> values)
{
  std::string allowed;
  for (Enum v : values)
  {
    if (s == name(v)) return v;
    allowed += std::string(allowed.empty() ? "" : "|") + name(v);
  }
  throw std::invalid_argument(std::string(option) + " must be " + allowed +
                              " (got '" + s + "')");
}
}  // namespace

PpmPlane parsePpmPlane(const std::string& s)
{
  return parse(s, "ppm-plane", {PpmPlane::XY, PpmPlane::XZ, PpmPlane::YZ});
}
PpmFormat parsePpmFormat(const std::string& s)
{
  return parse(s, "ppm-format", {PpmFormat::P3, PpmFormat::P6});
}
PpmColormap parsePpmColormap(const std::string& s)
{
  return parse(s, "ppm-colormap",
               {PpmColormap::Coolwarm, PpmColormap::Viridis, PpmColormap::Jet});
}
PpmNorm parsePpmNorm(const std::string& s)
{
  return parse(s, "ppm-norm",
               {PpmNorm::Local, PpmNorm::Running, PpmNorm::Fixed});
}
PpmInterp parsePpmInterp(const std::string& s)
{
  return parse(s, "ppm-interp",
               {PpmInterp::Nearest, PpmInterp::Linear, PpmInterp::Gll});
}
PpmDevice parsePpmDevice(const std::string& s)
{
  return parse(s, "ppm-device", {PpmDevice::Gpu, PpmDevice::Host});
}

std::string InSituPpm::imagePath(int timestep) const
{
  char file[32];
  std::snprintf(file, sizeof(file), "ppm-%06d.ppm", timestep);
  return (std::filesystem::path(cfg_.folder) / file).string();
}

void InSituPpm::init(const PpmConfig& cfg,
                     const std::array<std::vector<float>, 3>& node_coords,
                     int order, float dt, int interval)
{
  cfg_ = cfg;
  const int n[3] = {static_cast<int>(node_coords[0].size()),
                    static_cast<int>(node_coords[1].size()),
                    static_cast<int>(node_coords[2].size())};
  for (int d = 0; d < 3; ++d)
    if (n[d] < 2) throw std::runtime_error("ppm: less than 2 nodes on an axis");

  // Plane -> axes. Node (i, j, k) is i + j * nx + k * nx * ny.
  const std::size_t strides[3] = {1, std::size_t(n[0]),
                                  std::size_t(n[0]) * n[1]};
  switch (cfg_.plane)
  {
    case PpmPlane::XY: axis_u_ = 0, axis_v_ = 1, axis_n_ = 2; break;
    case PpmPlane::XZ: axis_u_ = 0, axis_v_ = 2, axis_n_ = 1; break;
    case PpmPlane::YZ: axis_u_ = 1, axis_v_ = 2, axis_n_ = 0; break;
  }
  const std::vector<float>& cu = node_coords[axis_u_];
  const std::vector<float>& cv = node_coords[axis_v_];
  const std::vector<float>& cn = node_coords[axis_n_];
  nu_ = n[axis_u_];
  nv_ = n[axis_v_];
  stride_u_ = strides[axis_u_];
  stride_v_ = strides[axis_v_];

  // nearest node layer to the requested position ('<' keeps the lower layer
  // on a tie, like the z-slices)
  double best = std::numeric_limits<double>::infinity();
  for (int l = 0; l < n[axis_n_]; ++l)
  {
    const double d = std::fabs(double(cn[l]) - double(cfg_.position));
    if (d < best)
    {
      best = d;
      layer_ = l;
    }
  }
  layer_pos_ = cn[layer_];
  base_ = std::size_t(layer_) * strides[axis_n_];

  // Image size: square pixels, as small as the smallest node spacing of the
  // plane so that no node interval falls between two pixels.
  const double length_u = double(cu.back()) - cu.front();
  const double length_v = double(cv.back()) - cv.front();
  if (cfg_.width > 0)
  {
    if (cfg_.width < 2) throw std::runtime_error("ppm-width must be >= 2");
    width_ = cfg_.width;
    pixel_size_ = length_u / (width_ - 1);
  }
  else
  {
    pixel_size_ = std::min(minSpacing(cu), minSpacing(cv));
    width_ = static_cast<int>(std::lround(length_u / pixel_size_)) + 1;
  }
  height_ =
      std::max(2, static_cast<int>(std::lround(length_v / pixel_size_)) + 1);
  if (std::uintmax_t(width_) * height_ * 3 >
      std::uintmax_t(std::numeric_limits<int>::max()))
    throw std::runtime_error("ppm: image too large");

  stencil_ = stencilSize(cfg_.interp, order);
  buildStencil(cu, order, width_, cfg_.interp, first_u_, weight_u_);
  buildStencil(cv, order, height_, cfg_.interp, first_v_, weight_v_);
  lut_ = buildLut(cfg_.colormap);
  rgb_.assign(std::size_t(width_) * height_ * 3, 0);

  if (cfg_.norm == PpmNorm::Fixed)
  {
    if (!(cfg_.vmax > cfg_.vmin))
      throw std::runtime_error("ppm: fixed bounds need vmax > vmin");
    lo_ = cfg_.vmin;
    hi_ = cfg_.vmax;
    has_bounds_ = true;
  }

  if (cfg_.device == PpmDevice::Gpu)
  {
    d_first_u_ = toDevice(first_u_, "ppm_first_u");
    d_first_v_ = toDevice(first_v_, "ppm_first_v");
    d_weight_u_ = toDevice(weight_u_, "ppm_weight_u");
    d_weight_v_ = toDevice(weight_v_, "ppm_weight_v");
    d_lut_ = toDevice(lut_, "ppm_lut");
    d_rgb_ = Kokkos::View<unsigned char*, MemSpace>("ppm_rgb", rgb_.size());
  }
  else
  {
    slice_buf_.assign(std::size_t(nu_) * nv_, 0.f);
  }

  std::filesystem::create_directories(cfg_.folder);
  if (!std::filesystem::is_directory(cfg_.folder))
    throw std::runtime_error("ppm folder is not a directory: " + cfg_.folder);

  writeHeader(node_coords, order, dt, interval);

  index_path_ =
      (std::filesystem::path(cfg_.folder) / "ppm-index.csv").string();
  index_.open(index_path_);
  if (!index_) throw std::runtime_error("Cannot open " + index_path_);
  index_ << "step,time,file,vmin,vmax,data_min,data_max\n";
}

void InSituPpm::writeHeader(
    const std::array<std::vector<float>, 3>& node_coords, int order, float dt,
    int interval)
{
  const char* axis_name[3] = {"x", "y", "z"};
  const std::vector<float>& cu = node_coords[axis_u_];
  const std::vector<float>& cv = node_coords[axis_v_];
  const std::filesystem::path path =
      std::filesystem::path(cfg_.folder) / "ppm-header.json";
  std::ofstream h(path);
  if (!h) throw std::runtime_error("Cannot open " + path.string());
  h << "{\n"
    << "  \"plane\": \"" << name(cfg_.plane) << "\",\n"
    << "  \"normal_axis\": \"" << axis_name[axis_n_] << "\",\n"
    << "  \"layer\": " << layer_ << ",\n"
    << "  \"position\": " << num(layer_pos_) << ",\n"
    << "  \"position_requested\": " << num(cfg_.position) << ",\n"
    << "  \"u_axis\": \"" << axis_name[axis_u_] << "\",\n"
    << "  \"v_axis\": \"" << axis_name[axis_v_] << "\",\n"
    << "  \"nu\": " << nu_ << ",\n"
    << "  \"nv\": " << nv_ << ",\n"
    << "  \"u_range\": [" << num(cu.front()) << ", " << num(cu.back()) << "],\n"
    << "  \"v_range\": [" << num(cv.front()) << ", " << num(cv.back()) << "],\n"
    << "  \"width\": " << width_ << ",\n"
    << "  \"height\": " << height_ << ",\n"
    << "  \"pixel_size\": " << num(pixel_size_) << ",\n"
    << "  \"first_row\": \"v_max\",\n"
    << "  \"order\": " << order << ",\n"
    << "  \"interp\": \"" << name(cfg_.interp) << "\",\n"
    << "  \"colormap\": \"" << name(cfg_.colormap) << "\",\n"
    << "  \"norm\": \"" << name(cfg_.norm) << "\",\n"
    << "  \"symmetric\": " << (cfg_.symmetric ? "true" : "false") << ",\n";
  if (cfg_.norm == PpmNorm::Fixed)
    h << "  \"vmin\": " << num(cfg_.vmin) << ",\n"
      << "  \"vmax\": " << num(cfg_.vmax) << ",\n";
  h << "  \"format\": \"" << name(cfg_.format) << "\",\n"
    << "  \"maxval\": 255,\n"
    << "  \"device\": \"" << name(cfg_.device) << "\",\n"
    << "  \"interval\": " << interval << ",\n"
    << "  \"dt\": " << num(dt) << ",\n"
    << "  \"index\": \"ppm-index.csv\"\n"
    << "}\n";
  h.close();
  if (!h) throw std::runtime_error("Cannot write " + path.string());
  metrics_.meta_bytes += std::filesystem::file_size(path);
}

void InSituPpm::output(const arrayReal& field, int column, int timestep,
                       float time)
{
  const bool on_gpu = cfg_.device == PpmDevice::Gpu;
  const std::size_t n_pix = std::size_t(width_) * height_;

  Kokkos::fence();  // the time step is over on the GPU before any timer
  const sclock::time_point t_begin = sclock::now();

  // 1. Where the slice is read. gpu: in place, in the field itself.
  //    host: the nodes of the slice are first copied to a host buffer (this
  //    read is what makes the pages of the field migrate to the host).
  SliceAccess slice;
  if (on_gpu)
  {
    const std::ptrdiff_t s0 = field.stride(0);
    const std::ptrdiff_t s1 = field.stride(1);
    slice = {field.data(), std::ptrdiff_t(base_) * s0 + column * s1,
             std::ptrdiff_t(stride_u_) * s0, std::ptrdiff_t(stride_v_) * s0};
  }
  else
  {
    for (int b = 0; b < nv_; ++b)
      for (int a = 0; a < nu_; ++a)
        slice_buf_[a + std::size_t(b) * nu_] =
            field(base_ + a * stride_u_ + b * stride_v_, column);
    slice = {slice_buf_.data(), 0, 1, nu_};
  }
  const sclock::time_point t_extract = sclock::now();

  // 2. Bounds of the colour scale. Fixed bounds need no pass on the data.
  float data_min = std::numeric_limits<float>::quiet_NaN();
  float data_max = data_min;
  if (cfg_.norm != PpmNorm::Fixed)
  {
    Kokkos::MinMax<float>::value_type mm;
    mm.min_val = std::numeric_limits<float>::max();
    mm.max_val = std::numeric_limits<float>::lowest();
    const MinMaxKernel kernel{slice, nu_};
    if (on_gpu)
    {
      Kokkos::parallel_reduce(
          "ppmMinMax",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nu_ * nv_),
          kernel, Kokkos::MinMax<float>(mm));
      Kokkos::fence();
    }
    else
    {
      for (int n = 0; n < nu_ * nv_; ++n) kernel(n, mm);
    }
    if (!(mm.min_val <= mm.max_val)) mm.min_val = mm.max_val = 0.f;
    data_min = mm.min_val;
    data_max = mm.max_val;

    float lo = data_min, hi = data_max;
    if (cfg_.symmetric)
    {
      hi = std::max(std::fabs(data_min), std::fabs(data_max));
      lo = -hi;
    }
    if (cfg_.norm == PpmNorm::Local || !has_bounds_)
    {
      lo_ = lo;
      hi_ = hi;
    }
    else
    {
      lo_ = std::min(lo_, lo);
      hi_ = std::max(hi_, hi);
    }
    has_bounds_ = true;
  }
  const sclock::time_point t_minmax = sclock::now();

  // 3. Rendering. A flat field (hi == lo) gets the middle colour.
  const float range = hi_ - lo_;
  RenderKernel render;
  render.slice = slice;
  render.stencil = stencil_;
  render.lut_last = kLutSize - 1;
  render.width = width_;
  render.height = height_;
  render.lo = lo_;
  render.inv_range = range > 0.f ? 1.f / range : 0.f;
  render.bias = range > 0.f ? 0.f : 0.5f;
  if (on_gpu)
  {
    render.first_u = d_first_u_.data();
    render.first_v = d_first_v_.data();
    render.weight_u = d_weight_u_.data();
    render.weight_v = d_weight_v_.data();
    render.lut = d_lut_.data();
    render.rgb = d_rgb_.data();
    Kokkos::parallel_for(
        "ppmRender",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, int(n_pix)),
        render);
    Kokkos::fence();
  }
  else
  {
    render.first_u = first_u_.data();
    render.first_v = first_v_.data();
    render.weight_u = weight_u_.data();
    render.weight_v = weight_v_.data();
    render.lut = lut_.data();
    render.rgb = rgb_.data();
    for (int pix = 0; pix < int(n_pix); ++pix) render(pix);
  }
  const sclock::time_point t_render = sclock::now();

  // 4. gpu: the image (3 bytes per pixel) is the only thing the host reads
  if (on_gpu)
  {
    Kokkos::View<unsigned char*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_rgb(rgb_.data(), rgb_.size());
    Kokkos::deep_copy(host_rgb, d_rgb_);
  }
  const sclock::time_point t_transfer = sclock::now();

  // 5. Encoding. The comment line keeps the bounds inside the image file.
  const std::string header =
      std::string(cfg_.format == PpmFormat::P3 ? "P3" : "P6") + "\n# step " +
      std::to_string(timestep) + " t " + num(time) + " vmin " + num(lo_) +
      " vmax " + num(hi_) + "\n" + std::to_string(width_) + " " +
      std::to_string(height_) + "\n255\n";
  if (cfg_.format == PpmFormat::P3) encodeP3(rgb_, width_, height_, text_);
  const sclock::time_point t_encode = sclock::now();

  // 6. Writing (time seen by the application), then fsync (persistence)
  const std::string path = imagePath(timestep);
  const char* body = cfg_.format == PpmFormat::P3
                         ? text_.data()
                         : reinterpret_cast<const char*>(rgb_.data());
  const std::size_t body_size =
      cfg_.format == PpmFormat::P3 ? text_.size() : rgb_.size();
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) ioError("Cannot open", path);
  if (std::fwrite(header.data(), 1, header.size(), f) != header.size() ||
      std::fwrite(body, 1, body_size, f) != body_size || std::fflush(f) != 0)
    ioError("Cannot write", path);
  const sclock::time_point t_write = sclock::now();
  if (cfg_.sync_writes && ::fsync(fileno(f)) != 0) ioError("fsync", path);
  const sclock::time_point t_fsync = sclock::now();
  if (std::fclose(f) != 0) ioError("Cannot close", path);

  index_ << timestep << ',' << num(time) << ','
         << std::filesystem::path(path).filename().string() << ','
         << num(lo_) << ',' << num(hi_) << ',';
  if (cfg_.norm != PpmNorm::Fixed)
    index_ << num(data_min) << ',' << num(data_max);
  else
    index_ << ',';
  index_ << std::endl;  // flushed: the index can be read while the run goes on
  const sclock::time_point t_end = sclock::now();

  metrics_.extract_s += seconds(t_begin, t_extract);
  metrics_.minmax_s += seconds(t_extract, t_minmax);
  metrics_.render_s += seconds(t_minmax, t_render);
  metrics_.transfer_s += seconds(t_render, t_transfer);
  metrics_.encode_s += seconds(t_transfer, t_encode);
  metrics_.write_s += seconds(t_encode, t_write) + seconds(t_fsync, t_end);
  metrics_.fsync_s += seconds(t_write, t_fsync);
  metrics_.total_s += seconds(t_begin, t_end);
  metrics_.count++;
  metrics_.image_bytes += header.size() + body_size;
}

void InSituPpm::finalize()
{
  if (!index_.is_open()) return;
  index_.close();
  if (!index_) throw std::runtime_error("Cannot write " + index_path_);
  metrics_.meta_bytes += std::filesystem::file_size(index_path_);
}
