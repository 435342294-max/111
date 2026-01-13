#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/GL_renderer.h"
#include "utils/Logger.h"

int main(int argc, char *argv[]) {
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description desc("Allowed options");
  try {
    desc.add_options()
      ("model", po::value<std::string>()->required(), "input model file with texture")
      ("step", po::value<float>()->required(), "sampling step")
      ("out_dir", po::value<std::string>()->required(), "output directory")
      ("backface_cull", po::bool_switch(), "enable backface culling");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  }
  catch (std::exception &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
    std::cerr << desc;
    return 1;
  }

  cm::initialize_logger();

  const float step = vm["step"].as<float>();
  try {
    cv::Mat color_frame, normal_frame, depth_frame;
    {
      cm::GL_renderer glr;
      glr.orthogonal_sample(vm["model"].as<std::string>(), step);
      color_frame = glr.grab_color_frame();
      LOG_INFO << "Color buffer copied.";
      normal_frame = glr.grab_normal_frame();
      LOG_INFO << "Normal buffer copied.";
      depth_frame = glr.grab_depth_frame();
      LOG_INFO << "Depth buffer copied.";
    }

    // save to disk
    std::ostringstream ss;
    ss << vm["out_dir"].as<std::string>() << "/step" << std::setprecision(3) << step << '/';
    const std::string out_dir(ss.str());
    if (!boost::filesystem::exists(out_dir))
      boost::filesystem::create_directory(out_dir);
    cv::imwrite(out_dir + "color.png", color_frame);
    {
      // XYZ (RGB) to BGR
      cv::cvtColor(normal_frame, normal_frame, CV_RGB2BGR);
      // [-1, 1] to [255, 0]
      cv::Mat temp;
      normal_frame.convertTo(temp, CV_8U, 255.0f / 2.0f, 255.0f / 2.0f);
      cv::imwrite(out_dir + "normal.png", temp);
    }
    {
      cv::Mat temp;
      // [0, 1] to [65535, 0]
      depth_frame.convertTo(temp, CV_16U, -65535.0f, 65535.0f);
      cv::imwrite(out_dir + "depth.png", temp);
    }
  }
  catch (std::exception &e) {
    std::cerr << "ERROR::" << e.what() << std::endl;
    return 1;
  }

  return 0;
}
