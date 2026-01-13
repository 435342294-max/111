#include <fstream>
#include <regex>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "utils/Logger.h"
#include "utils/Mesh_3.h"

int main(int argc, char *argv[]) {
  // initialize log
  cm::initialize_logger();

  // -D output/ -I [0-9]{6}_b[0-9]+_lod0.off -O dump0.off
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description desc("Allowed options");
  try {
    desc.add_options()
      ("directory,D", po::value<std::string>()->required(), "file directory")
      ("input_file,I", po::value<std::string>()->required(), "input file regex")
      ("output_file,O", po::value<std::string>()->required(), "output file name");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
    LOG_ERROR << desc;
    return 1;
  }

  using namespace boost::filesystem;
  path p(vm["directory"].as<std::string>());
  if (!exists(p) || !is_directory(p)) {
    LOG_ERROR << "Invalid directory: " << p;
    return -1;
  }

  // input file regex
  const std::regex name_regex(vm["input_file"].as<std::string>());

  Mesh_3 sm;
  sm.add_property_map<face_descriptor, CGAL::Color>("f:color");
  for (auto fitr = directory_iterator(p); fitr != directory_iterator(); ++fitr) {
    if (fitr->path().filename().extension().string() == ".off") {
      const std::string fname = fitr->path().filename().string();
      if (std::regex_match(fname, name_regex)) {
        std::ifstream ifs(fitr->path().string());
        if (!ifs.is_open()) {
          LOG_ERROR << "Failed to read " << fitr->path().string();
          return 1;
        }
        Mesh_3 tmp;
        ifs >> tmp;
        ifs.close();
        sm += tmp;
        LOG_INFO << "Merging file " << fitr->path().string();
      }
    }
  }

  const std::string fname(p.string() + vm["output_file"].as<std::string>());
  std::ofstream ofs(fname);
  ofs << sm;
  ofs.close();
  LOG_INFO << "Write to " << fname;
  LOG_INFO << "Done.";

  return 0;
}
