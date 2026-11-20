#include "piloting.h"

#include <cxxopts.hpp>
#include <iostream>


int main(int argc, char **argv) {
  std::string file = Piloting::default_file;
  cxxopts::Options options("Piloting", "Pilote the simulation.");
  options.add_options()("h,help", "Print help message");
  options.add_options()("file", "Path to the piloting file (default: .pilote)", cxxopts::value<std::string>(file));
  options.allow_unrecognised_options();

  Piloting::PilotingMap changed;
  Piloting::PilotingMap::bind_cli(options, changed);
  auto result = options.parse(argc, argv);

  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  changed.overwrite_map(file);

  // Piloting::PilotingMap map(file);
  // map.overwrite();
}
