#include <vector>
#include <string>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef __cplusplus
extern "C" {
#endif
#include "lsd_1.6/lsd.h"
#ifdef __cplusplus
}
#endif

#include "utils/Logger.h"

int main(int argc, char *argv[]) {
  // initialize log
  cm::initialize_logger(cm::severity_level::debug);

  // parse command options
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description desc("Allowed options");
  try {
    desc.add_options()
    ("image", po::value<std::string>()->required(),
      "Input image.")
    ("scale", po::value<double>()->default_value(0.8),
      "Scale image by Gaussian filter before processing.")
    ("sigma_scale", po::value<double>()->default_value(0.6),
      "Sigma for Gaussian filter is computed as sigma_scale/scale.")
    ("quant", po::value<double>()->default_value(2.0),
      "Bound to quantization error on the gradient norm.")
    ("ang_th", po::value<double>()->default_value(22.5),
      "Gradient angle tolerance in degrees.")
    ("log_eps", po::value<double>()->default_value(0.0),
      "Detection threshold, The larger the value, the less detections.")
    ("density_th", po::value<double>()->default_value(0.7),
      "Minimal density of region points in a rectangle to be accepted.")
    ("n_bins", po::value<int>()->default_value(1024),
      "Number of bins in 'ordering' of gradient modulus.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
    LOG_INFO << desc;
    return 1;
  }

  // read image and converto single channle gray image
  cv::Mat image = cv::imread(vm["image"].as<std::string>(), cv::IMREAD_GRAYSCALE);
  std::vector<double> img(image.rows * image.cols, 0.0);
  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j)
      img[i * image.cols + j] = double(image.at<unsigned char>(i, j));

  // number of lines detected
  std::vector<cv::Vec4d> segments;
  int n_out = 0;
  double *segs = ::LineSegmentDetection(&n_out,
    img.data(), image.cols, image.rows,
    vm["scale"].as<double>(),
    vm["sigma_scale"].as<double>(),
    vm["quant"].as<double>(),
    vm["ang_th"].as<double>(),
    vm["log_eps"].as<double>(),
    vm["density_th"].as<double>(),
    vm["n_bins"].as<int>(),
    nullptr, nullptr, nullptr);
  for (int i = 0; i < n_out; ++i)
    segments.push_back(cv::Vec4d(
      segs[7 * i], segs[7 * i + 1],
      segs[7 * i + 2], segs[7 * i + 3]));
  free(segs);
  LOG_INFO << "#nsegments " << segments.size();

  // write image
  for (const auto &s : segments) {
    const cv::Point2i ps((int)s[0], (int)s[1]);
    const cv::Point2i pt((int)s[2], (int)s[3]);
    cv::line(image, ps, pt, cv::Scalar(0));
  }
  cv::imwrite("dump_lsd.png", image);

  return 0;
}
