#include <boost/timer/timer.hpp>
#include <ctime>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

#include "utils/Logger.h"
#include "utils/Config.h"

#include "segment_modeling/UBlock.h"
#include "segment_modeling/UBlock_building.h"

namespace fs = boost::filesystem;
int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG_INFO << "No configuration file.";
    return 1;
  }
  boost::timer::auto_cpu_timer t("%w s\n");

  // working directory
  const std::string wdir(argv[1]);

  // initialize log
  cm::initialize_logger(cm::severity_level::debug);
  // load configuration
  cm::read_config(wdir + "/config_modeling.xml");
  LOG_INFO << "Configuration loaded.";
  
  try {
    auto& config = cm::get_config();  // 修改为非const引用
    const std::string input = config.get<std::string>("input_dir");
    const std::string output = config.get<std::string>("output_dir");
    const int rec_thread_num = std::stoi(config.get<std::string>("rec_thread_num"));
    const int cal_thread_num = std::stoi(config.get<std::string>("cal_thread_num"));
    
    // 查找所有以 _building.ply 结尾的文件
    std::vector<fs::path> ply_files;
    for (const auto &entry : fs::directory_iterator(input)) {
      if (entry.path().extension() == ".ply" && entry.path().stem().string().find("_building") != std::string::npos) {
        ply_files.push_back(entry.path());
      }
    }

    // 如果没有找到任何文件，打印错误并退出
    if (ply_files.empty()) {
      LOG_ERROR << "No '_building.ply' files found in the input directory.";
      return 1;
    }

    // 遍历所有符合条件的ply文件
    for (const auto &file : ply_files) {
      const std::string prefix = file.stem().string();  // 获取文件名（不含扩展名）
      
      // 更新配置中的prefix
      config.put("prefix", prefix);  // 修改此行
      LOG_INFO << "Processing file: " << file.string() << " with prefix: " << prefix;

      // generate RGB and mask images by orthogonally sampling
      double bbox_xmin, bbox_ymin, step;
      bbox_xmin, bbox_ymin, step = orthogonal(input, output);

      // empty block
      LOG_INFO << "Load data...";
      UBlock bk(output + prefix);  // 使用当前的prefix作为文件路径
      bk.load_data();

      // for block mesh and orthophoto not same bbox
      // e.g. cut mesh is off set to bigger range,
      // we have to set the sampling upper left corner manually
      const auto ulx = config.get_optional<double>("block.ulx");
      const auto uly = config.get_optional<double>("block.uly");

      if (ulx && uly)
        bk.compute_height_and_normal_map(*ulx, *uly);
      else
        bk.compute_height_and_normal_map();
      LOG_INFO << "Height and Normal map finished.";

      // tree-classifier + height MRF
      //bk.boost_tree_MRF_labeling();

      auto buildings = bk.retrieve_buildings(input, output, bbox_xmin, bbox_ymin, step);
      const int nb_buildings = (int)buildings.size(); 
      clock_t start = clock();
      omp_set_num_threads(rec_thread_num);
      #pragma omp parallel for
      for (int i = 0; i < nb_buildings; ++i) {
        LOG_INFO << "Process #" << i << " building:";
        buildings[i]->segment_arrangement_modeling();
      }
      clock_t end = clock();
      double BuildingTime = ((double)end - start) / CLOCKS_PER_SEC;
      LOG_INFO << "Building Time: " << BuildingTime << " s.";
      write_block_lod2(nb_buildings, output + prefix);
      write_block_lod1(nb_buildings, output + prefix);
      write_block_lod0(nb_buildings, output + prefix);
      // generate texture
      // LOG_INFO << input;
      // calculate_error(nb_buildings,  input, output, cal_thread_num);
      // generate_texture(input, output);
    }
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
  }
  return 0;
}
