#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cxxopts.hpp>


namespace Piloting {
  const std::string default_file = ".pilote";

  class PilotingOptions {
    public:
    bool stop;
    size_t snap_now, frequency, z_slice;
  };

  class PilotingMap {
    public:

    void create_map(std::string file) {
      filename_ = file;
      is_file_owner_ = true;
      fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
      ftruncate(fd_, sizeof(PilotingOptions));
      opts = (PilotingOptions *)mmap(NULL, sizeof(PilotingOptions), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_LOCKED, fd_, 0);
    }

    void overwrite_map(std::string file) {
      filename_ = file;
      fd_ = open(file.c_str(), O_RDWR);
      opts = (PilotingOptions *)mmap(NULL, sizeof(PilotingOptions), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_LOCKED, fd_, 0);

      std::cout << opts->snap_now << " " << snap_now_ << std::endl;
      opts->stop |= stop_;
      if (opts->z_slice != z_slice_) opts->z_slice = z_slice_;
      if (opts->frequency != frequency_) opts->frequency = frequency_;
      if (snap_now_) opts->snap_now++;

      munmap(opts, sizeof(PilotingOptions));
      close(fd_);
    }

    ~PilotingMap() {
      if (!is_file_owner_)
        return;

      munmap(opts, sizeof(PilotingOptions));
      close(fd_);
      while(remove(filename_.c_str()));
    }

    /* I dont want to think about this */
    PilotingOptions mocked;
    PilotingOptions* opts = &mocked;

    static void bind_cli(cxxopts::Options& opts, PilotingMap& map)
    {
      opts.add_options()
          ("k,kill", "Stop the simulation immediatly", cxxopts::value<bool>(map.stop_))
          ("s,snap", "Take an immediate snap of the simulation", cxxopts::value<bool>(map.snap_now_))
          ("f,freq", "Change the frequency of the simulation", cxxopts::value<size_t>(map.frequency_))
          ("z,z-slice", "Change the z-slice taken", cxxopts::value<size_t>(map.z_slice_))
        ;
    }

    private:
    size_t fd_;
    bool is_file_owner_ = false, snap_now_ = false, stop_ = false;
    size_t frequency_ = 1, z_slice_ = 0;
    std::string filename_;
  };

}
