//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.hpp: the main interface of SEM proxy application
//
//************************************************************************

#ifndef SEMPROXY_HPP_
#define SEMPROXY_HPP_

#include <data_type.h>
#include <model_struct.h>
#include <model_unstruct.h>
#include <solver_factory.h>
#include <utils.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "insitu_ppm.h"
#include "sem_proxy_options.h"

/**
 * @class SEMproxy
 */

class SEMproxy
{
 public:
  /**
   * @brief Constructor of the SEMproxy class
   */
  SEMproxy(const SemProxyOptions& cfg);

  /**
   * @brief Destructor of the SEMproxy class
   */
  ~SEMproxy() {};

  /**
   * @brief Initialize the simulation.
   * @post run()
   */
  void initFiniteElem()
  {
    init_arrays();
    init_source();
  };

  /**
   * @brief Run the simulation.
   * @pre This must be called after init()
   * @post Optional printout performance resutls
   */
  void run();

  /**
   * @brief Computes optimal time step using CFL stability condition for seismic
   * wave propagation
   *
   * Calculates the maximum stable time step for explicit finite difference
   * schemes using the CFL condition: dt ≤ CFL_factor × min_spacing / (√D ×
   * v_max) where D is the number of dimension.
   *
   * @param cfl_factor Stability factor (0.5-0.7 for 2nd-order, 0.3-0.4 for
   * higher-order schemes)
   * @return dt the max timestep.
   *
   * @note Typical values: 0.5-0.7 for 2nd-order, 0.3-0.4 for higher-order
   * schemes
   * @warning Must be called before time-stepping loop to ensure numerical
   * stability
   *
   */
  float find_cfl_dt(float cfl_factor);

  void saveSnapshotCsv(int timestep);
  void saveSnapshotBin(int timestep);
  void saveSnapshot(int timestep);
  void computeInSituStats(int timestep);
  void computeInSituQuart(int timestep);
  void computeInSitu(int timestep);
  void computeInSituSeismosStats(int timestep);
  void computeInSituSeismosPeaks(int timestep);
  void computeInSituSeismos(int timestep);
  // Write insitu_results.json / insitu_metrics.json in the seismograms folder
  // (same layout as adhoc_results.json / adhoc_metrics.json)
  void saveInSituSeismos(void);

  // z-slice output, two separately timed steps: copy layer k of column i1
  // into slice_buf_, then write slice_buf_ to slice-<timestep>.dat
  void extractSlice();
  void writeSlice(int timestep);

  void saveSeismograms(void);
  int getSeismosLoC(void);

  // Write timers, counts and byte totals of the run to --metrics-file
  void writeMetrics();

 private:
  int i1 = 0;
  int i2 = 1;

  // proper to cartesian mesh
  // or any structured mesh
  int nb_elements_[3] = {0};
  int nb_nodes_[3] = {0};
  std::array<float, 3> src_coord_ = {0};
  std::array<float, 3> rcv_coord_ = {0};
  float domain_size_[3] = {0};

  // snapshots
  bool is_snapshots_;
  int snap_time_interval_;
  std::string snap_folder_;
  bool snap_use_csv_ = 0;
  bool snap_use_header_ = 0;

  // in-situ
  size_t in_situ_interval_ = 0;
  size_t in_situ_k_ = 0; /* Number of iterations passed in in-situ */
  bool in_situ_active_ = false;
  bool in_situ_compute_stats_ = false;
  bool in_situ_compute_quarts_ = false;

  real_t in_situ_min_ = 0;
  real_t in_situ_max_ = 0;
  real_t in_situ_mean_ = 0;
  real_t in_situ_quartiles_[3] = {0, 0, 0};

  // in-situ seismograms: one value per receiver, updated at each in-situ step
  // (accumulated in double, like the float64 of pandas in the adhoc script)
  size_t in_situ_seismos_k_ = 0; /* Number of samples passed in in-situ */
  bool in_situ_seismos_active_ = false;
  bool in_situ_seismos_compute_stats_ = false;
  bool in_situ_seismos_compute_peaks_ = false;

  std::vector<double> in_situ_seismos_min_;
  std::vector<double> in_situ_seismos_max_;
  std::vector<double> in_situ_seismos_mean_;
  std::vector<double> in_situ_seismos_m2_; /* sum of squared deviations (Welford) */
  std::vector<double> in_situ_seismos_peak_abs_;
  std::vector<double> in_situ_seismos_t_peak_;


  // slices
  bool is_slices_ = false;
  std::string slice_folder_;
  int slice_k_ = 0;           // node layer written
  float slice_z_ = 0;         // z of that layer
  float slice_z_req_ = 0;     // z asked with --slice-z
  std::vector<float> slice_buf_;  // nx*ny values, allocated once

  // in-situ visualisation (PPM images of a slice), see insitu_ppm.h
  bool is_ppm_ = false;
  int ppm_interval_ = 0;  // steps between two images
  InSituPpm ppm_;

  bool sync_writes_ = false;
  SemProxyOptions opt_;  // kept for the metrics file

  // run metrics (steady_clock seconds, counts, bytes on disk)
  struct RunMetrics
  {
    double time_loop_s = 0;
    double compute_s = 0;
    int compute_steps = 0;
    double compute_after_output_s = 0;
    int compute_after_output_steps = 0;
    double snapshot_write_s = 0;
    int snapshot_count = 0;
    double slice_extract_s = 0;
    double slice_write_s = 0;
    int slice_count = 0;
    std::uintmax_t snapshot_bytes = 0;
    std::uintmax_t slice_bytes = 0;
    std::uintmax_t header_bytes = 0;
    std::uintmax_t seismo_bytes = 0;
    double insitu_seismos_stats_s = 0;
    double insitu_seismos_peaks_s = 0;
    int insitu_seismos_count = 0;
  } metrics_;

  // Physical time of output step `step`. computeOneStep(step) writes p at
  // (step + 1) * dt into column i1, and every output (seismograms, snapshots,
  // slices, in-situ) reads i1 after it: they all describe this time.
  float sampleTime(int step) const { return (step + 1) * dt_; }

  std::string snapshotPath(int timestep) const;
  std::string snapshotHeaderPath() const;
  std::string slicePath(int timestep) const;
  void initSlices();
  void initPpm();
  static void syncFile(const std::string& path);

  // sismos
  std::ifstream receiver_positions_stream_;
  std::string seismograms_data_folder_;
  std::vector<std::pair<int, std::array<float, 3>>> rcv_coords_ = {};
  bool seismo_use_header_ = 0;

  // physics
  bool isElastic_;

  // time parameters
  float dt_;
  float timemax_;
  int num_sample_;
  // source parameters
  const int myNumberOfRHS = 1;
  const float f0 = 5.;
  const int sourceOrder = 2;
  int myElementSource = 0;

  std::shared_ptr<model::ModelApi<float, int>> m_mesh;
  std::unique_ptr<SEMSolverBase> m_solver;
  SolverUtils myUtils;

  // arrays
  arrayReal myRHSTerm;
  arrayReal pnGlobal;
  vectorInt rhsElement;
  vectorInt rhsElementRcv;
  arrayReal rhsWeights;
  arrayReal rhsWeightsRcv;
  arrayReal pnAtReceiver;
  vectorReal inSituArr;

  // initialize source and RHS
  void init_source();

  // allocate arrays and vectors
  void init_arrays();

  // private methods to pars argv options
  int getPhysic(string physicArg);
  void loadReceiverPositions(string filepath);
  SolverFactory::implemType getImplem(string implemArg);
  SolverFactory::methodType getMethod(string methodArg);
  SolverFactory::meshType getMesh(string meshArg);

};

#endif /* SEMPROXY_HPP_ */
