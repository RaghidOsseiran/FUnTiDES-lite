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

#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>
#include "Kokkos_Sort.hpp"

using namespace SourceAndReceiverUtils;

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
  is_snapshots_ = opt.useSnapshots;
  snap_time_interval_ = opt.snapshotsInterval;
  snap_folder_ = opt.snapshotsFolder;
  snap_use_csv_ = opt.snapshotsUseCSV;
  snap_use_header_ = opt.snapshotsUseHeaders;

  in_situ_interval_ = opt.inSituInterval;
  in_situ_compute_stats_ = opt.inSituStats;
  in_situ_compute_quarts_ = opt.inSituQuart;
  in_situ_active_ = in_situ_compute_stats_ | in_situ_compute_quarts_;

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
}

void SEMproxy::run()
{
  time_point<system_clock> startComputeTime, startOutputTime, startSeismoTime, totalComputeTime,
      totalOutputTime, totalSeismoTime, startInSituTime, totalInSituTime;

  SEMsolverDataAcoustic solverData(i1, i2, myRHSTerm, pnGlobal, rhsElement,
                                   rhsWeights);

  for (int indexTimeSample = 0; indexTimeSample < num_sample_;
       indexTimeSample++)
  {
    startComputeTime = system_clock::now();
    m_solver->computeOneStep(dt_, indexTimeSample, solverData);
    totalComputeTime += system_clock::now() - startComputeTime;

    startOutputTime = system_clock::now();

    if (indexTimeSample % 50 == 0)
    {
        m_solver->outputSolutionValues(indexTimeSample, i1, rhsElement[0],
                                       pnGlobal, "pnGlobal");
    }

    if (is_snapshots_)
      saveSnapshot(indexTimeSample);

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

    if (in_situ_active_) {
      startInSituTime = system_clock::now();
      computeInSitu(indexTimeSample);
      totalInSituTime += system_clock::now() - startInSituTime;
    }

    swap(i1, i2);

    auto tmp = solverData.m_i1;
    solverData.m_i1 = solverData.m_i2;
    solverData.m_i2 = tmp;

    totalOutputTime += system_clock::now() - startOutputTime;
  }

  float kerneltime_ms = time_point_cast<microseconds>(totalComputeTime)
                            .time_since_epoch()
                            .count();
  float outputtime_ms =
      time_point_cast<microseconds>(totalOutputTime).time_since_epoch().count();
  float seismotime_ms = time_point_cast<microseconds>(totalSeismoTime)
                            .time_since_epoch()
                            .count();

  float insitutime_ms =
      time_point_cast<microseconds>(totalInSituTime).time_since_epoch().count();

  cout << "------------------------------------------------ " << endl;
  cout << "\n---- Elapsed Kernel Time : " << kerneltime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed Recording Seismos Time : " << seismotime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed Output Time : " << outputtime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed In Situ Time : " << insitutime_ms / 1E6 << " seconds."
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

  if (in_situ_compute_quarts_)
    inSituArr =
      allocateVector<vectorReal>(m_mesh->getNumberOfNodes(), "inSituArr");
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
  line_of_contents_ = 0;
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
    line_of_contents_++;
  }

  out.close();
  std::cout <<  "Sismo save successfully at : "
            << std::filesystem::absolute(output_directory / "seismogram_recording.csv")
            << std::endl;
}

void SEMproxy::saveSnapshotCsv(int timestep) {
  std::ofstream file(snap_folder_ + "/./snap-" + std::to_string(timestep) + ".csv");

  if (snap_use_header_) {
    for (size_t i = 0; i < m_mesh->getNumberOfNodes(); i++)
      file << pnGlobal(i, i1) << '\n';

    if(timestep == 0) {
      std::ofstream header_file (snap_folder_ + "/./headers.csv");
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
}

void SEMproxy::saveSnapshotBin(int timestep) {
  std::ofstream file(snap_folder_ + "/./snap-" + std::to_string(timestep) + ".dat", ios::binary | ios::out);

  if (snap_use_header_) {
    file.write((char *)&pnGlobal(0, i1), m_mesh->getNumberOfNodes() * sizeof(real_t));

    if(timestep == 0) {
      std::ofstream header_file (snap_folder_ + "/./headers.dat", ios::binary | ios::out);
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

void SEMproxy::computeInSituQuart(int timestep) {
  const int size = m_mesh->getNumberOfNodes();
  auto view = Kokkos::subview(pnGlobal, Kokkos::ALL(), i1);

  Kokkos::deep_copy(inSituArr, view);
  Kokkos::sort(inSituArr);
  size_t key = (size - 1) * 1 / 4;
  in_situ_quartiles_[0] = inSituArr(key);
  in_situ_quartiles_[1] = inSituArr(key * 2);
  in_situ_quartiles_[2] = inSituArr(key * 3);

  std::cout <<
    "[local]1st: " << inSituArr(key) << ", [local]2nd: " << inSituArr(key * 2) << ", [local]3rd: " << inSituArr(key*3) << std::endl;
    // " [global]mean: " << in_situ_mean_ << ", [global]min: " << in_situ_min_ << ", [global]max: " << in_situ_max_ << std::endl;
}

void SEMproxy::computeInSituStats(int timestep) {
  const int size = m_mesh->getNumberOfNodes();
    auto lambda = KOKKOS_LAMBDA(const int i, real_t& min, real_t & max, real_t& mean) {
      auto p = pnGlobal;
      auto c = i1;
      const real_t val = pnGlobal(i, i1);
      min = std::min(min, val);
      max = std::max(max, val);
      mean += val / size;
    };

    real_t mean = 0, ma = pnGlobal(0,i1), mi = pnGlobal(0,i1);
    Kokkos::parallel_reduce("inSitu", size, lambda, Kokkos::Min<real_t>(mi), Kokkos::Max<real_t>(ma), mean);
    Kokkos::fence();

    auto K = (timestep - 1) * in_situ_k_;
    if (in_situ_k_ == 1) {
      in_situ_mean_= mean;
      in_situ_min_ = mi;
      in_situ_max_ = ma;
    } else {
      /* update old mean and add new elements */
      in_situ_mean_= in_situ_mean_ * ((real_t)K / (K + size)) + (mean * ((real_t)size / (K + size)));
      in_situ_min_ = min(mi, in_situ_min_);
      in_situ_max_ = max(ma, in_situ_max_);
    }

    std::cout <<
      "[local]mean: " << mean << ", [local]min: " << mi << ", [local]max: " << ma <<
      " [global]mean: " << in_situ_mean_ << ", [global]min: " << in_situ_min_ << ", [global]max: " << in_situ_max_ << std::endl;
}

void SEMproxy::computeInSitu(int timestep) {
  if (timestep % in_situ_interval_ != 0)
    return;
  in_situ_k_++;

  if (in_situ_compute_stats_)
    computeInSituStats(timestep);
  if(in_situ_compute_quarts_)
    computeInSituQuart(timestep);
  }
