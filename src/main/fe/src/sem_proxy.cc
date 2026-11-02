//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.cpp: the main interface of  proxy application
//
//************************************************************************

#include "sem_proxy.h"

#include <cartesian_struct_builder.h>
#include <cartesian_unstruct_builder.h>
#include <sem_solver_acoustic.h>
#include <source_and_receiver_utils.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>
#include <variant>

using namespace SourceAndReceiverUtils;

namespace
{
// Shortest decimal text that reads back as the same float (e.g. 0.001, not
// 0.00100000005), with '.' as decimal separator whatever the locale.
std::string jsonFloat(float v)
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

std::string jsonDouble(double v)
{
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
  return os.str();
}

std::string jsonString(const std::string& s)
{
  std::string out = "\"";
  for (char c : s)
  {
    if (c == '"' || c == '\\')
    {
      out += '\\';
      out += c;
    }
    else if (static_cast<unsigned char>(c) < 0x20)
    {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    }
    else
      out += c;
  }
  return out + "\"";
}

const char* jsonBool(bool b) { return b ? "true" : "false"; }

double secondsBetween(std::chrono::steady_clock::time_point t0,
                      std::chrono::steady_clock::time_point t1)
{
  return std::chrono::duration<double>(t1 - t0).count();
}
}  // namespace

SEMproxy::SEMproxy(const SemProxyOptions& opt)
{
  const int order = opt.order;
  nb_elements_[0] = opt.ex;
  nb_elements_[1] = opt.ey;
  nb_elements_[2] = opt.ez;
  nb_nodes_[0] = opt.ex * order + 1;
  nb_nodes_[1] = opt.ey * order + 1;
  nb_nodes_[2] = opt.ez * order + 1;

  const float spongex = opt.boundaries_size;
  const float spongey = opt.boundaries_size;
  const float spongez = opt.boundaries_size;
  const std::array<float, 3> sponge_size = {spongex, spongey, spongez};
  src_coord_[0] = opt.srcx;
  src_coord_[1] = opt.srcy;
  src_coord_[2] = opt.srcz;

  domain_size_[0] = opt.lx;
  domain_size_[1] = opt.ly;
  domain_size_[2] = opt.lz;

  rcv_coord_[0] = opt.rcvx;
  rcv_coord_[1] = opt.rcvy;
  rcv_coord_[2] = opt.rcvz;

  bool isModelOnNodes = opt.isModelOnNodes;
  isElastic_ = opt.isElastic;
  cout << boolalpha;
  bool isElastic = isElastic_;

  loadReceiverPositions(opt.receiver_positions_file);
  seismograms_data_folder_ = opt.seismogramsFolder;
  seismo_use_header_ = opt.seismoUseHeader;
  is_snapshots_ = opt.useSnapshots;
  snap_time_interval_ = opt.snapshotsInterval;
  snap_folder_ = opt.snapshotsFolder;
  snap_use_csv_ = opt.snapshotsUseCSV;
  snap_use_header_ = opt.snapshotsUseHeaders;
  is_slices_ = opt.useSlices;
  slice_folder_ = opt.slicesFolder;
  slice_z_req_ = opt.sliceZ;
  sync_writes_ = opt.syncWrites;
  opt_ = opt;

  const SolverFactory::methodType methodType = getMethod(opt.method);
  const SolverFactory::implemType implemType = getImplem(opt.implem);
  const SolverFactory::meshType meshType = getMesh(opt.mesh);
  const SolverFactory::modelLocationType modelLocation =
      isModelOnNodes ? SolverFactory::modelLocationType::OnNodes
                     : SolverFactory::modelLocationType::OnElements;
  const SolverFactory::physicType physicType =
      SolverFactory::physicType::Acoustic;

  float lx = domain_size_[0];
  float ly = domain_size_[1];
  float lz = domain_size_[2];
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];

  if (meshType == SolverFactory::Struct)
  {
    switch (order)
    {
      case 1: {
        model::CartesianStructBuilder<float, int, 1> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 2: {
        model::CartesianStructBuilder<float, int, 2> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 3: {
        model::CartesianStructBuilder<float, int, 3> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      default:
        throw std::runtime_error(
            "Order other than 1 2 3 is not supported (semproxy)");
    }
  }
  else if (meshType == SolverFactory::Unstruct)
  {
    model::CartesianParams<float, int> param(order, ex, ey, ez, lx, ly, lz,
                                             isModelOnNodes);
    model::CartesianUnstructBuilder<float, int> builder(param);
    m_mesh = builder.getModel();
  }
  else
  {
    throw std::runtime_error("Incorrect mesh type (SEMproxy ctor.)");
  }

  // time parameters
  if (opt.autodt)
  {
    float cfl_factor = (order == 2) ? 0.5 : 0.7;
    dt_ = find_cfl_dt(cfl_factor);
  }
  else
  {
    dt_ = opt.dt;
  }
  timemax_ = opt.timemax;
  num_sample_ = timemax_ / dt_;

  m_solver = SolverFactory::createSolver(methodType, implemType, meshType,
                                         modelLocation, physicType, order);
  m_solver->computeFEInit(*m_mesh, sponge_size, opt.surface_sponge,
                          opt.taper_delta);

  initFiniteElem();

  std::cout << "Number of node is " << m_mesh->getNumberOfNodes() << std::endl;
  std::cout << "Number of element is " << m_mesh->getNumberOfElements()
            << std::endl;
  std::cout << "Launching the Method " << opt.method << ", the implementation "
            << opt.implem << " and the mesh is " << opt.mesh << std::endl;
  std::cout << "Model is on " << (isModelOnNodes ? "nodes" : "elements")
            << std::endl;
  std::cout << "Physics type is " << (isElastic ? "elastic" : "acoustic")
            << std::endl;
  std::cout << "Order of approximation will be " << order << std::endl;
  std::cout << "Time step is " << dt_ << "s" << std::endl;
  std::cout << "Simulated time is " << timemax_ << "s" << std::endl;

  if (is_slices_) initSlices();
}

void SEMproxy::run()
{
  time_point<system_clock> startComputeTime, startOutputTime, startSeismoTime, totalComputeTime,
      totalOutputTime, totalSeismoTime;

  // steady_clock timers for the metrics file (see RunMetrics)
  using sclock = std::chrono::steady_clock;
  bool previous_step_was_output = false;

  SEMsolverDataAcoustic solverData(i1, i2, myRHSTerm, pnGlobal, rhsElement,
                                   rhsWeights);

  const sclock::time_point startLoop = sclock::now();
  for (int indexTimeSample = 0; indexTimeSample < num_sample_;
       indexTimeSample++)
  {
    startComputeTime = system_clock::now();
    const sclock::time_point t_compute = sclock::now();
    m_solver->computeOneStep(dt_, indexTimeSample, solverData);
    // computeOneStep ends with a fence: the GPU work of this step is done
    const double compute_step_s = secondsBetween(t_compute, sclock::now());
    totalComputeTime += system_clock::now() - startComputeTime;

    metrics_.compute_s += compute_step_s;
    metrics_.compute_steps++;
    if (previous_step_was_output)
    {
      metrics_.compute_after_output_s += compute_step_s;
      metrics_.compute_after_output_steps++;
    }

    startOutputTime = system_clock::now();

    if (indexTimeSample % 50 == 0)
    {
        m_solver->outputSolutionValues(indexTimeSample, i1, rhsElement[0],
                                       pnGlobal, "pnGlobal");
    }

    const bool is_output_step = (is_snapshots_ || is_slices_) &&
                                indexTimeSample % snap_time_interval_ == 0;

    // Slice before snapshot: the snapshot reads every node on the host, so
    // after it the slice pages would already be host-resident and the
    // extraction would look cheaper than in a slices-only run.
    if (is_slices_ && is_output_step)
    {
      const sclock::time_point t0 = sclock::now();
      extractSlice();
      const sclock::time_point t1 = sclock::now();
      writeSlice(indexTimeSample);
      const sclock::time_point t2 = sclock::now();

      metrics_.slice_extract_s += secondsBetween(t0, t1);
      metrics_.slice_write_s += secondsBetween(t1, t2);
      metrics_.slice_count++;
      cout << "Elapsed Saving Slice Time : " << secondsBetween(t0, t2)
           << " seconds." << endl;
      metrics_.slice_bytes +=
          std::filesystem::file_size(slicePath(indexTimeSample));
    }

    if (is_snapshots_ && is_output_step)
    {
      const sclock::time_point t0 = sclock::now();
      saveSnapshot(indexTimeSample);
      const double snapshot_s = secondsBetween(t0, sclock::now());

      metrics_.snapshot_write_s += snapshot_s;
      metrics_.snapshot_count++;
      cout << "Elapsed Saving Snapshot Time : " << snapshot_s << " seconds."
           << endl;
      metrics_.snapshot_bytes +=
          std::filesystem::file_size(snapshotPath(indexTimeSample));
      if (snap_use_header_ && indexTimeSample == 0)
        metrics_.header_bytes +=
            std::filesystem::file_size(snapshotHeaderPath());
    }
    previous_step_was_output = is_output_step;

    startSeismoTime = system_clock::now();
    // Save pressure at receiver
    const int order = m_mesh->getOrder();
    for (int p = 0; p < rcv_coords_.size(); ++p)
    {
      float varnp1 = 0.0;
      for (int i = 0; i < order + 1; i++)
      {
        for (int j = 0; j < order + 1; j++)
        {
          for (int k = 0; k < order + 1; k++)
          {
            int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[p], i, j, k);
            int globalNodeOnElement =
                i + j * (order + 1) + k * (order + 1) * (order + 1);
            varnp1 +=
                pnGlobal(nodeIdx, i2) * rhsWeightsRcv(p, globalNodeOnElement);
          }
        }
      }

      pnAtReceiver(p, indexTimeSample) = varnp1;
    }
    totalSeismoTime += system_clock::now() -startSeismoTime; 

    swap(i1, i2);

    auto tmp = solverData.m_i1;
    solverData.m_i1 = solverData.m_i2;
    solverData.m_i2 = tmp;

    totalOutputTime += system_clock::now() - startOutputTime;
  }
  metrics_.time_loop_s = secondsBetween(startLoop, sclock::now());

  float kerneltime_ms = time_point_cast<microseconds>(totalComputeTime)
                            .time_since_epoch()
                            .count();
  float outputtime_ms =
      time_point_cast<microseconds>(totalOutputTime).time_since_epoch().count();
  float seismotime_ms = time_point_cast<microseconds>(totalSeismoTime)
                            .time_since_epoch()
                            .count();

  cout << "------------------------------------------------ " << endl;
  cout << "\n---- Elapsed Kernel Time : " << kerneltime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed Recording Seismos Time : " << seismotime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed Output Time : " << outputtime_ms / 1E6 << " seconds."
       << endl;
  cout << "------------------------------------------------ " << endl;
}

// Initialize arrays
void SEMproxy::init_arrays()
{
  cout << "Allocate host memory for source and pressure values ..." << endl;

  rhsElement = allocateVector<vectorInt>(myNumberOfRHS, "rhsElement");
  rhsWeights = allocateArray2D<arrayReal>(
      myNumberOfRHS, m_mesh->getNumberOfPointsPerElement(), "RHSWeight");
  myRHSTerm = allocateArray2D<arrayReal>(myNumberOfRHS, num_sample_, "RHSTerm");
  pnGlobal =
      allocateArray2D<arrayReal>(m_mesh->getNumberOfNodes(), 2, "pnGlobal");
  pnAtReceiver = allocateArray2D<arrayReal>(rcv_coords_.size(), num_sample_,
                                            "pnAtReceiver");
  // Receiver
  rhsElementRcv =
      allocateVector<vectorInt>(rcv_coords_.size(), "rhsElementRcv");
  rhsWeightsRcv = allocateArray2D<arrayReal>(
      rcv_coords_.size(), m_mesh->getNumberOfPointsPerElement(),
      "RHSWeightRcv");
}

// Initialize sources
void SEMproxy::init_source()
{
  arrayReal myRHSLocation = allocateArray2D<arrayReal>(1, 3, "RHSLocation");
  // std::cout << "All source are currently are coded on element 50." <<
  // std::endl;
  std::cout << "All source are currently are coded on middle element."
            << std::endl;
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];

  int lx = domain_size_[0];
  int ly = domain_size_[1];
  int lz = domain_size_[2];

  // Get source element index

  int source_index = floor((src_coord_[0] * ex) / lx) +
                     floor((src_coord_[1] * ey) / ly) * ex +
                     floor((src_coord_[2] * ez) / lz) * ey * ex;

  for (int i = 0; i < 1; i++)
  {
    rhsElement[i] = source_index;
  }

  // Get coordinates of the corners of the sourc element
  float cornerCoords[8][3];
  int I = 0;
  int nodes_corner[2] = {0, m_mesh->getOrder()};
  for (int k : nodes_corner)
  {
    for (int j : nodes_corner)
    {
      for (int i : nodes_corner)
      {
        int nodeIdx = m_mesh->globalNodeIndex(rhsElement[0], i, j, k);
        cornerCoords[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
        cornerCoords[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
        cornerCoords[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
        I++;
      }
    }
  }

  // initialize source term
  vector<float> sourceTerm =
      myUtils.computeSourceTerm(num_sample_, dt_, f0, sourceOrder);
  for (int j = 0; j < num_sample_; j++)
  {
    myRHSTerm(0, j) = sourceTerm[j];
    if (j % 100 == 0)
      cout << "Sample " << j << "\t: sourceTerm = " << sourceTerm[j] << endl;
  }

  // get element number of source term
  myElementSource = rhsElement[0];
  cout << "Element number for the source location: " << myElementSource << endl
       << endl;

  int order = m_mesh->getOrder();

  switch (order)
  {
    case 1:
      SourceAndReceiverUtils::ComputeRHSWeights<1>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 2:
      SourceAndReceiverUtils::ComputeRHSWeights<2>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 3:
      SourceAndReceiverUtils::ComputeRHSWeights<3>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    default:
      throw std::runtime_error("Unsupported order: " + std::to_string(order));
  }

  // Receiver computation
  for (int p = 0; p < rcv_coords_.size(); ++p)
  {
    int receiver_index = floor((rcv_coords_[p].second[0] * ex) / lx) +
                         floor((rcv_coords_[p].second[1] * ey) / ly) * ex +
                         floor((rcv_coords_[p].second[2] * ez) / lz) * ey * ex;

    rhsElementRcv[p] = receiver_index;

    // Get coordinates of the corners of the receiver element
    float cornerCoordsRcv[8][3];
    I = 0;
    for (int k : nodes_corner)
    {
      for (int j : nodes_corner)
      {
        for (int i : nodes_corner)
        {
          int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[p], i, j, k);
          cornerCoordsRcv[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
          cornerCoordsRcv[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
          cornerCoordsRcv[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
          I++;
        }
      }
    }

    switch (order)
    {
      case 1:
        SourceAndReceiverUtils::ComputeRHSWeights<1>(
            cornerCoordsRcv, rcv_coords_[p].second, rhsWeightsRcv, p);
        break;
      case 2:
        SourceAndReceiverUtils::ComputeRHSWeights<2>(
            cornerCoordsRcv, rcv_coords_[p].second, rhsWeightsRcv, p);
        break;
      case 3:
        SourceAndReceiverUtils::ComputeRHSWeights<3>(
            cornerCoordsRcv, rcv_coords_[p].second, rhsWeightsRcv, p);
        break;
      default:
        throw std::runtime_error("Unsupported order: " + std::to_string(order));
    }
  }
}

void SEMproxy::loadReceiverPositions(string filepath)
{
  if (filepath.empty())
  {
    rcv_coords_.push_back({0, rcv_coord_});
    return;
  }
  receiver_positions_stream_.open(filepath, std::ios::in);
  if (!receiver_positions_stream_) throw std::runtime_error("Cannot open file");

  std::string line;
  while (std::getline(receiver_positions_stream_, line))
  {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.rfind("id", 0) == 0 ) continue;

    int idRcv;
    float rcv_x, rcv_y, rcv_z;
    if (std::sscanf(line.c_str(), "%d,%f,%f,%f\n", &idRcv, &rcv_x, &rcv_y,
                    &rcv_z) != 4)
      throw std::runtime_error("Wrong format in file");

    if ((rcv_x < 0 || rcv_x >= domain_size_[0]) ||
        (rcv_y < 0 || rcv_y >= domain_size_[1]) || 
        (rcv_z < 0 || rcv_z >= domain_size_[2]))
      throw std::runtime_error("Receiver " + std::to_string(idRcv) + " is outside the domain");

    rcv_coords_.push_back(
        std::pair<int, std::array<float, 3>>{idRcv, {rcv_x, rcv_y, rcv_z}});
  }

  receiver_positions_stream_.close();
}

SolverFactory::implemType SEMproxy::getImplem(string implemArg)
{
  if (implemArg == "makutu") return SolverFactory::MAKUTU;
  if (implemArg == "shiva") return SolverFactory::SHIVA;

  throw std::invalid_argument(
      "Implentation type does not follow any valid type.");
}

SolverFactory::meshType SEMproxy::getMesh(string meshArg)
{
  if (meshArg == "cartesian") return SolverFactory::Struct;
  if (meshArg == "ucartesian") return SolverFactory::Unstruct;

  std::cout << "Mesh type found is " << meshArg << std::endl;
  throw std::invalid_argument("Mesh type does not follow any valid type.");
}

SolverFactory::methodType SEMproxy::getMethod(string methodArg)
{
  if (methodArg == "sem") return SolverFactory::SEM;
  if (methodArg == "dg") return SolverFactory::DG;

  throw std::invalid_argument("Method type does not follow any valid type.");
}

float SEMproxy::find_cfl_dt(float cfl_factor)
{
  float sqrtDim3 = 1.73;  // to change for 2d
  float min_spacing = m_mesh->getMinSpacing();
  float v_max = m_mesh->getMaxSpeed();

  float dt = cfl_factor * min_spacing / (sqrtDim3 * v_max);

  return dt;
}

void SEMproxy::saveSeismograms(void)
{
  std::filesystem::path output_directory(seismograms_data_folder_);
  bool create = std::filesystem::create_directories(output_directory);
  if (!std::filesystem::is_directory(seismograms_data_folder_))
    throw std::runtime_error("Path exists but it is not a directory");

  std::ofstream out(output_directory / "seismogram_recording.csv");
  if (!out) throw std::runtime_error("Cannot open the file");

  if (seismo_use_header_) {
    std::ofstream coords_file(output_directory / "headers.csv");
    if (!coords_file) throw std::runtime_error("Cannot open the file");
    coords_file << "id,x,y,z" << std::endl;
    for (int rcv = 0; rcv < rcv_coords_.size(); ++rcv)
      coords_file << rcv_coords_[rcv].first << ',' << rcv_coords_[rcv].second[0] << ',' << rcv_coords_[rcv].second[1]
          << ',' << rcv_coords_[rcv].second[2] << std::endl;
    coords_file.close();

    out << "step,time";
    for (int rcv = 0; rcv < rcv_coords_.size(); ++rcv)
      out << ",rcv_" << rcv_coords_[rcv].first;
    out << '\n';
    
    for (int indexTimeSample = 0; indexTimeSample < num_sample_;
       indexTimeSample++)
    {
      out << indexTimeSample << ',' << indexTimeSample * dt_;
      for (int p = 0; p < rcv_coords_.size(); ++p)
      {
        out << ',' << pnAtReceiver(p, indexTimeSample);
      }

      out << '\n';
    }

  } else {

    out << "step,time";

    for (int rcv = 0; rcv < rcv_coords_.size(); ++rcv)
      out << ",rcv_" << rcv_coords_[rcv].first;

    for (int rcv = 0; rcv < rcv_coords_.size(); ++rcv)
      out << ",rcv_" << rcv_coords_[rcv].first << "_x" << ",rcv_"
          << rcv_coords_[rcv].first << "_y" << ",rcv_" << rcv_coords_[rcv].first
          << "_z";

    out << '\n';

    for (int indexTimeSample = 0; indexTimeSample < num_sample_;
        indexTimeSample++)
    {
      out << indexTimeSample << ',' << indexTimeSample * dt_;
      for (int p = 0; p < rcv_coords_.size(); ++p)
      {
        out << ',' << pnAtReceiver(p, indexTimeSample);
      }

      for (int p = 0; p < rcv_coords_.size(); ++p)
      {
        out << ',' << rcv_coords_[p].second[0] << ',' << rcv_coords_[p].second[1]
            << ',' << rcv_coords_[p].second[2];
      }

      out << '\n';
    }
  }

  out.close();
  metrics_.seismo_bytes +=
      std::filesystem::file_size(output_directory / "seismogram_recording.csv");
  std::cout <<  "Sismo save successfully at : "
            << std::filesystem::absolute(output_directory / "seismogram_recording.csv")
            << std::endl;
}

std::string SEMproxy::snapshotPath(int timestep) const {
  return snap_folder_ + "/./snap-" + std::to_string(timestep) +
         (snap_use_csv_ ? ".csv" : ".dat");
}

std::string SEMproxy::snapshotHeaderPath() const {
  return snap_folder_ + (snap_use_csv_ ? "/./headers.csv" : "/./headers.dat");
}

void SEMproxy::saveSnapshotCsv(int timestep) {
  std::ofstream file(snapshotPath(timestep));

  if (snap_use_header_) {
    for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++)
      file << pnGlobal(i, i1) << '\n';

    if(timestep == 0) {
      std::ofstream header_file (snapshotHeaderPath());
      header_file << "x;y;z\n";
      for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++) {
        header_file
          << m_mesh->nodeCoord(i, 0) << ';'
          << m_mesh->nodeCoord(i, 1) << ';'
          << m_mesh->nodeCoord(i, 2) << ';' << '\n';
      }
      header_file.close();
    }
  } else {
    file << "x;y;z;v\n";
    for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++) {
      file
        << m_mesh->nodeCoord(i, 0) << ';'
        << m_mesh->nodeCoord(i, 1) << ';'
        << m_mesh->nodeCoord(i, 2) << ';' 
        << pnGlobal(i, i1)         << ';'
        << '\n';
    }
  }

  file.close();
  if (sync_writes_) syncFile(snapshotPath(timestep));
}

void SEMproxy::saveSnapshotBin(int timestep) {
  std::ofstream file(snapshotPath(timestep), ios::binary | ios::out);

  if (snap_use_header_) {
    file.write((char *)&pnGlobal(0, i1), m_mesh->getNumberOfNodes() * sizeof(real_t));

    if(timestep == 0) {
      std::ofstream header_file (snapshotHeaderPath(), ios::binary | ios::out);
      real_t val[3];
      for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++) {
        val[0] = m_mesh->nodeCoord(i, 0);
        val[1] = m_mesh->nodeCoord(i, 1);
        val[2] = m_mesh->nodeCoord(i, 2);
        header_file.write((char *)&val, 3 * sizeof(real_t));
      }
      header_file.close();
    }

  } else {
    real_t val[4];
    for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++){
      val[0] = m_mesh->nodeCoord(i, 0);
      val[1] = m_mesh->nodeCoord(i, 1);
      val[2] = m_mesh->nodeCoord(i, 2);
      val[3] = pnGlobal(i, i1);
      file.write((char *)&val, 4 * sizeof(real_t));
    }
  }

  file.close();
  if (sync_writes_) syncFile(snapshotPath(timestep));
}

void SEMproxy::saveSnapshot(int timestep) {
  if (timestep % snap_time_interval_)
    return;

  if (snap_use_csv_) {
    saveSnapshotCsv(timestep);
  } else {
    saveSnapshotBin(timestep);
  }
}

std::string SEMproxy::slicePath(int timestep) const {
  return (std::filesystem::path(slice_folder_) /
          ("slice-" + std::to_string(timestep) + ".dat"))
      .string();
}

// Choose the node layer, create the folder, allocate the buffer and write
// slice-header.json. Called once from the constructor (before the time loop).
void SEMproxy::initSlices() {
  // slices are raw float32: check the value type of pnGlobal and the byte order
  using value_t =
      std::remove_cv_t<std::remove_reference_t<decltype(pnGlobal(0, 0))>>;
  static_assert(std::is_same_v<value_t, float>,
                "slice output assumes a float pnGlobal");
  const std::uint32_t one = 1;
  if (*reinterpret_cast<const unsigned char*>(&one) != 1)
    throw std::runtime_error("slice output assumes a little-endian host");

  const int nx = nb_nodes_[0];
  const int ny = nb_nodes_[1];
  const int nz = nb_nodes_[2];
  const std::size_t plane = std::size_t(nx) * ny;
  if (std::size_t(m_mesh->getNumberOfNodes()) != plane * nz)
    throw std::runtime_error("slices: node count is not nx*ny*nz");

  // nearest layer to the requested z; strict '<' keeps the lower k on a tie
  double best_d = std::numeric_limits<double>::infinity();
  for (int k = 0; k < nz; ++k) {
    const double d = std::fabs(double(m_mesh->nodeCoord(int(k * plane), 2)) -
                               double(slice_z_req_));
    if (d < best_d) {
      best_d = d;
      slice_k_ = k;
    }
  }
  slice_z_ = m_mesh->nodeCoord(int(slice_k_ * plane), 2);
  cout << "Slice layer k = " << slice_k_ << ", z = " << slice_z_
       << " m (requested " << slice_z_req_ << " m)" << endl;

  std::filesystem::create_directories(slice_folder_);
  if (!std::filesystem::is_directory(slice_folder_))
    throw std::runtime_error("slices folder is not a directory: " +
                             slice_folder_);

  slice_buf_.assign(plane, 0.f);

  const std::filesystem::path header_path =
      std::filesystem::path(slice_folder_) / "slice-header.json";
  std::ofstream h(header_path);
  if (!h) throw std::runtime_error("Cannot open " + header_path.string());
  h << "{\n"
    << "  \"nx\": " << nx << ",\n"
    << "  \"ny\": " << ny << ",\n"
    << "  \"k\": " << slice_k_ << ",\n"
    << "  \"z\": " << jsonFloat(slice_z_) << ",\n"
    << "  \"z_requested\": " << jsonFloat(slice_z_req_) << ",\n"
    << "  \"order\": " << m_mesh->getOrder() << ",\n"
    << "  \"ex\": " << nb_elements_[0] << ",\n"
    << "  \"ey\": " << nb_elements_[1] << ",\n"
    << "  \"ez\": " << nb_elements_[2] << ",\n"
    << "  \"interval\": " << snap_time_interval_ << ",\n"
    << "  \"dt\": " << jsonFloat(dt_) << ",\n"
    << "  \"dtype\": \"float32\",\n"
    << "  \"x\": [";
  for (int i = 0; i < nx; ++i)
    h << (i ? ", " : "") << jsonFloat(m_mesh->nodeCoord(i, 0));
  h << "],\n  \"y\": [";
  for (int j = 0; j < ny; ++j)
    h << (j ? ", " : "") << jsonFloat(m_mesh->nodeCoord(j * nx, 1));
  h << "]\n}\n";
  h.close();
  if (!h) throw std::runtime_error("Cannot write " + header_path.string());
  metrics_.header_bytes += std::filesystem::file_size(header_path);
}

// Copy layer slice_k_ of the newest time column (i1) into slice_buf_.
// Layer k is the nodes [k*nx*ny, (k+1)*nx*ny). With LayoutLeft (CUDA build)
// column i1 is contiguous, so this is one sequential read of that block;
// going through the view keeps it correct for LayoutRight (CPU builds) too.
void SEMproxy::extractSlice() {
  const std::size_t plane = slice_buf_.size();
  const std::size_t first = std::size_t(slice_k_) * plane;
  for (std::size_t n = 0; n < plane; ++n)
    slice_buf_[n] = pnGlobal(first + n, i1);
}

// Write slice_buf_ as raw float32 (x fastest, then y).
void SEMproxy::writeSlice(int timestep) {
  const std::string path = slicePath(timestep);
  std::ofstream file(path, std::ios::binary | std::ios::out);
  file.write(reinterpret_cast<const char*>(slice_buf_.data()),
             slice_buf_.size() * sizeof(float));
  file.close();
  if (!file) throw std::runtime_error("Cannot write slice file " + path);
  if (sync_writes_) syncFile(path);
}

// Force the data of an already closed file to the disk (--sync-writes).
void SEMproxy::syncFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("open " + path + " for fsync: " +
                             std::strerror(errno));
  if (::fsync(fd) != 0) {
    const int err = errno;
    ::close(fd);
    throw std::runtime_error("fsync " + path + ": " + std::strerror(err));
  }
  if (::close(fd) != 0)
    throw std::runtime_error("close " + path + ": " + std::strerror(errno));
}

void SEMproxy::writeMetrics() {
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0) std::strcpy(host, "unknown");

  const std::string& path = opt_.metricsFile;
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot open the metrics file " + path);

  const auto& m = metrics_;
  out << "{\n"
      << "  \"hostname\": " << jsonString(host) << ",\n"
      << "  \"options\": {\n"
      << "    \"ex\": " << nb_elements_[0] << ",\n"
      << "    \"ey\": " << nb_elements_[1] << ",\n"
      << "    \"ez\": " << nb_elements_[2] << ",\n"
      << "    \"order\": " << m_mesh->getOrder() << ",\n"
      << "    \"lx\": " << jsonFloat(domain_size_[0]) << ",\n"
      << "    \"ly\": " << jsonFloat(domain_size_[1]) << ",\n"
      << "    \"lz\": " << jsonFloat(domain_size_[2]) << ",\n"
      << "    \"mesh\": " << jsonString(opt_.mesh) << ",\n"
      << "    \"implem\": " << jsonString(opt_.implem) << ",\n"
      << "    \"method\": " << jsonString(opt_.method) << ",\n"
      << "    \"dt\": " << jsonFloat(dt_) << ",\n"
      << "    \"auto_dt\": " << jsonBool(opt_.autodt) << ",\n"
      << "    \"timemax\": " << jsonFloat(timemax_) << ",\n"
      << "    \"num_sample\": " << num_sample_ << ",\n"
      << "    \"receivers\": " << rcv_coords_.size() << ",\n"
      << "    \"use_snapshots\": " << jsonBool(is_snapshots_) << ",\n"
      << "    \"snapshots_interval\": " << snap_time_interval_ << ",\n"
      << "    \"snapshots_folder\": " << jsonString(snap_folder_) << ",\n"
      << "    \"snapshots_format\": " << jsonString(snap_use_csv_ ? "csv" : "bin") << ",\n"
      << "    \"snapshots_use_headers\": " << jsonBool(snap_use_header_) << ",\n"
      << "    \"use_slices\": " << jsonBool(is_slices_) << ",\n"
      << "    \"slices_folder\": " << jsonString(slice_folder_) << ",\n"
      << "    \"slice_z_requested\": " << jsonFloat(slice_z_req_) << ",\n"
      << "    \"slice_k\": " << (is_slices_ ? std::to_string(slice_k_) : "null") << ",\n"
      << "    \"slice_z\": " << (is_slices_ ? jsonFloat(slice_z_) : "null") << ",\n"
      << "    \"sync_writes\": " << jsonBool(sync_writes_) << ",\n"
      << "    \"seismograms_folder\": " << jsonString(seismograms_data_folder_) << ",\n"
      << "    \"metrics_file\": " << jsonString(path) << "\n"
      << "  },\n"
      << "  \"nodes\": " << m_mesh->getNumberOfNodes() << ",\n"
      << "  \"nx\": " << nb_nodes_[0] << ",\n"
      << "  \"ny\": " << nb_nodes_[1] << ",\n"
      << "  \"nz\": " << nb_nodes_[2] << ",\n"
      << "  \"time_loop_s\": " << jsonDouble(m.time_loop_s) << ",\n"
      << "  \"compute_s\": " << jsonDouble(m.compute_s) << ",\n"
      << "  \"compute_steps\": " << m.compute_steps << ",\n"
      << "  \"compute_after_output_s\": " << jsonDouble(m.compute_after_output_s) << ",\n"
      << "  \"compute_after_output_steps\": " << m.compute_after_output_steps << ",\n"
      << "  \"snapshot_write_s\": " << jsonDouble(m.snapshot_write_s) << ",\n"
      << "  \"snapshot_count\": " << m.snapshot_count << ",\n"
      << "  \"slice_extract_s\": " << jsonDouble(m.slice_extract_s) << ",\n"
      << "  \"slice_write_s\": " << jsonDouble(m.slice_write_s) << ",\n"
      << "  \"slice_count\": " << m.slice_count << ",\n"
      << "  \"snapshot_bytes\": " << m.snapshot_bytes << ",\n"
      << "  \"slice_bytes\": " << m.slice_bytes << ",\n"
      << "  \"header_bytes\": " << m.header_bytes << ",\n"
      << "  \"seismo_bytes\": " << m.seismo_bytes << "\n"
      << "}\n";
  out.close();
  if (!out) throw std::runtime_error("Cannot write the metrics file " + path);
  std::cout << "Metrics save successfully at : "
            << std::filesystem::absolute(path) << std::endl;
}
