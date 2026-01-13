#include <vector>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "libnpy/npy.hpp"
#include "gco-v3.0/GCoptimization.h"

#include "utils/Logger.h"
#include "utils/Config.h"
#include "utils/Mesh_3.h"
#include "utils/GL_renderer.h"

/*
 * \brief Deeplab prediction and height MRF for semantic prediction.
 * \param probability_map input Deeplab prediction,
 * a CV_64FC3 Mat of ground, building and vegetation probability
 * \param height_map input height map of CV_64FC1 mat, same resolution as probability_map
 * \param ground_height ground height
 * \return semantic label map, 0 - 3 representing ground, building, grass and tree respectively
 */
cv::Mat deeplab_height_mrf(
  const cv::Mat &probability_map,
  const cv::Mat &height_map,
  const double ground_height) {
  class Smooth_functor : public GCoptimization::SmoothCostFunctor {
  public:
    Smooth_functor(const cv::Mat &hmap, const double step, const double balance) :
      m_hmap(hmap),
      m_step(step),
      m_cols(hmap.cols),
      m_balance(balance) {}

    // TODO: try cross entropy smooth or earth moving distance
    // smooth here does not seem to help too much
    double compute(int s1, int s2, int l1, int l2) {
      if (l1 == l2)
        return 0.0;
      const double h1 = m_hmap.at<double>(s1 / m_cols, s1 % m_cols);
      const double h2 = m_hmap.at<double>(s2 / m_cols, s2 % m_cols);
      // modulate height difference to [0, 1] with logistic function:
      // 1 / (1 + e^(-2 * (hd - m_step)))
      // modulate mid point is approximately 45 degree
      const double hd = 1.0 / (1.0 + std::exp(-2.0 * (std::abs(h1 - h2) - m_step)));
      // greater height difference -> less smooth -> cut here
      // TODO: thre reverse does not necessarily hold:
      // smaller height difference -> more smooth -> do not cut here
      // e.g. ground and grass, building and trees with same height
      return (1.0 - hd) * m_balance;
    }

  private:
    // height map
    const cv::Mat m_hmap;
    const double m_step;
    const int m_cols;
    const double m_balance;
  };

  if (probability_map.rows != height_map.rows
    || probability_map.cols != height_map.cols)
    throw std::runtime_error("Inconsistent probability and height map.");

  const int rows = probability_map.rows;
  const int cols = probability_map.cols;
  const int num_sites = rows * cols;
  const int num_labels = 4;

  const auto &config = cm::get_config();
  LOG_INFO << "MRF params: " <<
    config.get<bool>("block.semantic.mrf.use_swap") << ' ' <<
    config.get<double>("block.semantic.mrf.balance") << ' ' <<
    config.get<int>("block.semantic.mrf.iterations");

  const double mid_height = config.get<double>("block.semantic.mid_height");
  std::vector<double> data(num_sites * num_labels, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // modulate height to [0, 1] with logistic function:
      // 1 / (1 + e^(-2 * (h - mid_height)))
      double h = std::max(height_map.at<double>(i, j) - ground_height, 0.0);
      h = 1.0 / (1.0 + std::exp(-2.0 * (h - mid_height)));
      const cv::Vec3d p = probability_map.at<cv::Vec3d>(i, j);
      const double pground = p[0];
      const double pbuilding = p[1];
      const double pvegetation = p[2];
      const int offset = (i * cols + j) * num_labels;
      // set data for ground, building, grass and tree
      // data[offset] = (1.0 - pground) * pbuilding * pvegetation * h;
      // data[offset + 1] = pground * (1.0 - pbuilding) * pvegetation * (1.0 - h);
      // data[offset + 2] = pground * pbuilding * (1.0 - pvegetation) * h;
      // data[offset + 3] = pground * pbuilding * (1.0 - pvegetation) * (1.0 - h);
      data[offset] = 1.0 - pground * (1.0 - h);
      data[offset + 1] = 1.0 - pbuilding * h;
      data[offset + 2] = 1.0 - pvegetation * (1.0 - h);
      data[offset + 3] = 1.0 - pvegetation * h;
    }
  }
  Smooth_functor smooth(height_map,
    config.get<double>("scene.step"),
    config.get<double>("block.semantic.mrf.balance"));

  cv::Mat lable_map(rows, cols, CV_32SC1, cv::Scalar(0));
  try {
    // grid graph
    GCoptimizationGridGraph gc(cols, rows, num_labels);
    gc.setDataCost(data.data());
    gc.setSmoothCostFunctor(&smooth);

    LOG_INFO << "Before optimization energy is: " << gc.compute_energy();
    if (config.get<bool>("block.semantic.mrf.use_swap")) {
      LOG_INFO << "Alpha-beta swap algorithm.";
      gc.swap(config.get<int>("block.semantic.mrf.iterations"));
    }
    else {
      LOG_INFO << "Alpha expansion algorithm.";
      gc.expansion(config.get<int>("block.semantic.mrf.iterations"));
    }
    LOG_INFO << "After optimization energy is: " << gc.compute_energy();

    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        lable_map.at<int>(i, j) = gc.whatLabel(i * cols + j);
  }
  catch (GCException e) {
    throw std::runtime_error(e.message);
  }

  return lable_map;
}

/*
 * \brief Generate mask from Deeplab probability map and height map.
 */
int main(int argc, char *argv[]) {
  // working directory
  const std::string wdir(argv[1]);
  // initialize log
  cm::initialize_logger(cm::severity_level::debug);
  // load configuration
  cm::read_config(wdir + "/config_modeling.xml");
  LOG_INFO << "Configuration loaded.";

  const auto &config = cm::get_config();
  const int nb_rows = config.get<int>("scene.cut.rows");
  const int nb_cols = config.get<int>("scene.cut.cols");
  const int subimg_rows = config.get<int>("scene.sub_image.rows");
  const int subimg_cols = config.get<int>("scene.sub_image.cols");
  const CGAL::Bbox_3 bbox(
    config.get<double>("scene.bbox.xmin"),
    config.get<double>("scene.bbox.ymin"),
    config.get<double>("scene.bbox.zmin"),
    config.get<double>("scene.bbox.xmax"),
    config.get<double>("scene.bbox.ymax"),
    config.get<double>("scene.bbox.zmax")
  );

  LOG_INFO << "#nb_rows: " << nb_rows << ", #nb_cols: " << nb_cols;
  LOG_INFO << "#subimg_rows: " << subimg_rows << ", #subimg_cols: " << subimg_cols;
  LOG_INFO << "#bbox: " << bbox;

  const double step = config.get<double>("scene.step");
  const double bbox_xmin = bbox.xmin();
  const double bbox_ymax = bbox.ymax();
  const double bbox_zmin = bbox.zmin(), bbox_zmax = bbox.zmax();
  const double ystep = step * double(subimg_rows);
  const double xstep = step * double(subimg_cols);

  try {
#pragma omp parallel for
    for (int r = 0; r < nb_rows; ++r) {
      for (int c = 0; c < nb_cols; ++c) {
        std::stringstream ss;
        ss << wdir << std::setfill('0')
          << std::setw(3) << r
          << std::setw(3) << c;
        const std::string prefix = ss.str();

        // load mesh
        Mesh_3 mesh;
        const std::string file_mesh = prefix + ".off";
        std::ifstream ifs(file_mesh);
        if (!ifs.is_open())
          throw std::runtime_error("Failed to read " + file_mesh);
        ifs >> mesh;
        ifs.close();

        // sampling
        const double ulx = bbox_xmin + c * xstep;
        const double uly = bbox_ymax - r * ystep;
        cm::GL_renderer glr;
        glr.orthogonal_sample(mesh,
          subimg_cols,
          subimg_rows,
          { ulx, uly - ystep, bbox_zmin },
          { ulx + xstep, uly, bbox_zmax });
        // height map, CV_32F to CV_64F, [0, 1] to real height value
        cv::Mat height_map = glr.grab_depth_frame().clone();
        height_map.convertTo(height_map, CV_64F);
        height_map = (1.0 - height_map) * (bbox_zmax - bbox_zmin) + bbox_zmin;
        // valid map
        cv::Mat valid_map = glr.grab_color_frame().clone();

        // load Deeplab prediction: ground, building, vegetation
        const std::string npy_file = prefix + ".npy";
        std::vector<unsigned long> shape;
        std::vector<float> data;
        npy::LoadArrayFromNumpy(npy_file, shape, data);
        if (shape.size() != 3
          || shape[0] != subimg_rows
          || shape[1] != subimg_cols
          || shape[2] != 3)
          throw std::runtime_error("Failed to read " + npy_file);
        cv::Mat probability = cv::Mat(subimg_rows, subimg_cols, CV_32FC3, data.data()).clone();
        probability.convertTo(probability, CV_64F);

        // initial Deeplab label map
        cv::Mat label_map(subimg_rows, subimg_cols, CV_32SC1);
        for (int i = 0; i < subimg_rows; ++i) {
          for (int j = 0; j < subimg_cols; ++j) {
            const auto p = probability.at<cv::Vec3d>(i, j);
            double max_val = p[0];
            int max_pos = 0;
            if (max_val < p[1]) {
              max_val = p[1];
              max_pos = 1;
            }
            if (max_val < p[2]) {
              max_val = p[2];
              max_pos = 2;
            }
            label_map.at<int>(i, j) = max_pos;
          }
        }

        // compute averaged ground height
        int count = 0;
        double sum = 0.0;
        for (int i = 0; i < subimg_rows; ++i) {
          for (int j = 0; j < subimg_cols; ++j) {
            if (label_map.at<int>(i, j) == 0
              && valid_map.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
              ++count;
              sum += height_map.at<double>(i, j);
            }
          }
        }
        const double ground_height = count == 0 ? bbox_zmin : sum / count;

        // Deeplab prediction and height MRF
        label_map = deeplab_height_mrf(probability, height_map, ground_height);
        const cv::Vec3b color_map[4] = {
          { 51, 105, 153 }, { 255, 153, 102 }, { 0, 150, 0 }, { 0, 250, 0 }};
        cv::Mat dump(subimg_rows, subimg_cols, CV_8UC3);
        for (int i = 0; i < subimg_rows; ++i)
          for (int j = 0; j < subimg_cols; ++j)
            dump.at<cv::Vec3b>(i, j) = color_map[label_map.at<int>(i, j)];
        const std::string fname = prefix + "_mask.png";
        if (!cv::imwrite(fname, dump))
          throw std::runtime_error("Failed to write " + fname);

        LOG_WARNING << "#block <" << r << ", " << c << "> done.";
      }
    }
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
  }

  return 0;
}
