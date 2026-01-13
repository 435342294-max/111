#include <fstream>

// #include <CGAL/Point_set_3.h>
// #include <CGAL/Point_set_3/IO.h>
// #include <CGAL/Shape_detection_3.h>
// #include <CGAL/regularize_planes.h>
// #include <CGAL/Polygon_2.h>
// #include <CGAL/Surface_mesh_approximation/approximate_triangle_mesh.h>
// #include <CGAL/Aff_transformation_2.h>
// #include <CGAL/Weighted_point_3.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "gco-v3.0/GCoptimization.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "lsd_1.6/lsd.h"
#ifdef __cplusplus
}
#endif

#include "utils/Logger.h"
#include "utils/Config.h"

#include "UBlock_building.h"
#include "regularize_segments_graph.h"
#include "regularize_segments.h"



/*!
 * \brief Compute centroid of a polygon.
 */
Kernel::Point_2 polygon_centroid(const std::vector<Kernel::Point_2> &plg);

UBlock_building::UBlock_building(
  const std::string &prefix,
  const cv::Mat &building_map,
  const cv::Mat &color_map,
  const cv::Mat &position_map,
  const cv::Mat &normal_map,
  const double &ground_level,
  const double &max_height,
  const Mesh_3 &mesh) :
  m_prefix(prefix),
  m_bmap(building_map),
  m_cmap(color_map),
  m_pmap(position_map),
  m_nmap(normal_map),
  m_ground_level(ground_level),
  m_max_height(max_height),
  m_mesh(mesh),
  m_roof_level(0.0),
  m_rows(m_bmap.rows),
  m_cols(m_bmap.cols),
  m_step(cm::get_config().get<double>("scene.step")),
  m_xmin(m_pmap.at<cv::Vec3d>(0, 0)[0] - m_step / 2.0),
  m_ymax(m_pmap.at<cv::Vec3d>(0, 0)[1] + m_step / 2.0) {
  // compute average roof level
  int count = 0;
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      if (m_bmap.at<unsigned char>(i, j) > 0) {
        m_roof_level += m_pmap.at<cv::Vec3d>(i, j)[2];
        ++count;
      }
    }
  }
  m_roof_level /= (double)count;
}

void UBlock_building::segment_arrangement_modeling() {
  const auto &config = cm::get_config();

  // detect roof planes
  detect_roof_planes();
  
  // detect line segments
  if (config.get<bool>("building.image.enabled", true))
    color_space_segment_detection();
  height_space_segment_detection();
  // if (m_hsegments.size() == 0)
  //   return ;

  // if (config.get<bool>("building.facade_segment.enabled", true))
  //   facade_segment_detection();
  if (config.get<bool>("building.normal.enabled", true))
    normal_space_segment_detection();

  // regularize segments
  std::vector<Kernel::Segment_2> segments;
  for (const auto &s : m_fsegments)
    segments.push_back({{s[0], s[1]}, {s[2], s[3]}});
  for (const auto &s : m_csegments)
    segments.push_back({{s[0], s[1]}, {s[2], s[3]}});
  for (const auto &s : m_hsegments)
    segments.push_back({{s[0], s[1]}, {s[2], s[3]}});
  for (const auto &dnv : m_nsegments)
    for (const auto &s : dnv)
      segments.push_back({{s[0], s[1]}, {s[2], s[3]}});
  LOG_INFO << "#segments: " << segments.size();


/* 
  // find facade main direction (eccv 2018)
  std::vector<Kernel::Vector_2> main_dir;
  main_dir = find_mainly_dir(segments, config.get<double>(
    "building.regularization.parallel_threshold"));

  // projection four main directions
    m_segments = regularize_segments<Kernel_epec>(
    segments,
    main_dir,
    config.get<double>(
      "building.regularization.collinear_dis_threshold"));
*/
  
  
  // local regularize
  m_segments = regularize_segments<Kernel_epec>(
    segments,
    config.get<double>(
      "building.regularization.parallel_threshold"),
    config.get<double>(
      "building.regularization.collinear_threshold"),
    config.get<double>(
      "building.regularization.collinear_dis_threshold"));


  LOG_INFO << "#m_segments: " << m_segments.size();

  // detected line segments to arrangement
  construct_segment_arrangement();
  LOG_INFO << "#data " << m_arr_data.size();

  // MRF labeling
  segment_arrangement_MRF_labeling();

  // extrude arrangement contour
  segment_arrangement_extrusion();
  LOG_INFO << "model extrusion done";

  // write data
  if (config.get("building.write.color_segments", false))
    write_color_segments();
  if (config.get("building.save.mesh", false))
    save_mesh();
  if (config.get("building.save.point_cloud", false))
    save_point_cloud();
  // segments
  if (config.get("building.write.height_segments", false))
    write_height_segments();
  if (config.get("building.write.normal_segments", false))
    write_normal_segments();
  if (config.get("building.write.facade_segments", false))
    write_facade_segments();
  if (config.get("building.write.all_segments", false))
    write_all_segments();
  // arrangement
  if (config.get("building.write.regularized_segments", false))
    write_regularized_segments();
  if (config.get("building.write.arrangement", false))
    write_arrangement();
  if (config.get("building.write.arrangement_labeling", false))
    write_arrangement_labeling();
  if (config.get("building.write.local_ground", false))
    write_local_ground();
  if (config.get("building.write.levels", false))
    write_levels();
    
}

void UBlock_building::facade_segment_detection() {
  // add face proxy id map
  bool created = false;
  Mesh_3::Property_map<face_descriptor, std::size_t> fpx_map;
  boost::tie(fpx_map, created) =
    m_mesh.add_property_map<face_descriptor, std::size_t>("f:px_map", 0);
  assert(created);

  // add face area map
  Mesh_3::Property_map<face_descriptor, double> farea_map;
  boost::tie(farea_map, created) =
    m_mesh.add_property_map<face_descriptor, double>("f:area_map", 0.0);
  assert(created);
  double sum_area = 0.0;
  for (const face_descriptor f : m_mesh.faces()) {
    const auto he = m_mesh.halfedge(f);
    const double area = std::sqrt(CGAL::squared_area(
      m_mesh.point(m_mesh.source(he)),
      m_mesh.point(m_mesh.target(he)),
      m_mesh.point(m_mesh.target(m_mesh.next(he)))));
    farea_map[f] = area;
    sum_area += area;
  }

  // VSA
  namespace VSA = CGAL::Surface_mesh_approximation;
  const double seed_area = cm::get_config().get<double>(
    "building.facade_segment.vsa.seed_area");
  const std::size_t iterations = cm::get_config().get<std::size_t>(
    "building.facade_segment.vsa.iterations");


  std::vector<Vector_3> px_normals;
  VSA::approximate_triangle_mesh(m_mesh,
    CGAL::parameters::verbose_level(VSA::SILENT).
    seeding_method(VSA::RANDOM).
    max_number_of_proxies(std::max(int(sum_area / seed_area), 2)).
    number_of_iterations(iterations).
    face_proxy_map(fpx_map).
    proxies(std::back_inserter(px_normals)));
  LOG_INFO << "#px " << px_normals.size();

  std::vector<double> px_areas(px_normals.size(), 0.0);
  std::vector<Vector_3> px_centers(px_normals.size(), Vector_3(0.0, 0.0, 0.0));
  for (const face_descriptor f : m_mesh.faces()) {
    const auto he = m_mesh.halfedge(f);
    px_areas[fpx_map[f]] += farea_map[f];
    px_centers[fpx_map[f]] += farea_map[f] * (
      CGAL::centroid(m_mesh.point(m_mesh.source(he)),
        m_mesh.point(m_mesh.target(he)),
        m_mesh.point(m_mesh.target(m_mesh.next(he)))) - CGAL::ORIGIN);
  }
  for (std::size_t i = 0; i < px_normals.size(); ++i)
    px_centers[i] /= px_areas[i];

  // project vertical proxies as line segments
  std::vector<CGAL::Bbox_3> proxy_bbox(px_normals.size());
  for (const face_descriptor f : m_mesh.faces()) {
    const auto he = m_mesh.halfedge(f);
    proxy_bbox[fpx_map[f]] += (m_mesh.point(m_mesh.source(he)).bbox() + 
      m_mesh.point(m_mesh.target(he)).bbox() + 
      m_mesh.point(m_mesh.target(m_mesh.next(he))).bbox());
  }

  const double min_area = cm::get_config().get<double>(
    "building.facade_segment.min_area");
  const double normal_threshold = cm::get_config().get<double>(
    "building.facade_segment.normal_threshold");
  m_fsegments.clear();
  for (std::size_t i = 0; i < px_normals.size(); ++i) {
    if (px_areas[i] > min_area && std::abs(px_normals[i].z()) < normal_threshold) {
      const Kernel::Point_2 mid_pt(px_centers[i].x(), px_centers[i].y());
      const auto &b = proxy_bbox[i];
      const double len = std::sqrt((b.xmax() - b.xmin()) * (b.xmax() - b.xmin())
        + (b.ymax() - b.ymin()) * (b.ymax() - b.ymin())) / 2.0;
      Kernel::Vector_2 dir(px_normals[i].x(), px_normals[i].y());
      dir /= std::sqrt(dir.squared_length());
      const Kernel::Vector_2 dir_ortho(dir.y(), -dir.x());
      const Kernel::Point_2 p0 = mid_pt + dir_ortho * len;
      const Kernel::Point_2 p1 = mid_pt - dir_ortho * len;
      m_fsegments.push_back({p0.x(), p0.y(), p1.x(), p1.y()});
    }
  }

  LOG_INFO << "#fsegments " << m_fsegments.size();
}




void UBlock_building::color_space_segment_detection() {
  // brg to gracy scale
  cv::Mat img_32f;
  m_cmap.convertTo(img_32f, CV_32F);
  img_32f *= 1.0f / 255.0f;
  cv::cvtColor(img_32f, img_32f, cv::COLOR_BGR2GRAY);

  std::vector<double> img(img_32f.rows * img_32f.cols, 0.0);
 for (int i = 0; i < img_32f.rows; ++i)
    for (int j = 0; j < img_32f.cols; ++j)
      img[i * img_32f.cols + j] = img_32f.at<float>(i, j) * 255.0;//将二维的灰度图像 img_32f 转换为一维数组 img。

  // number of lines detected
  int n_out = 0;
  const auto &config = cm::get_config();
  double *segs = ::LineSegmentDetection(&n_out,
    img.data(), img_32f.cols, img_32f.rows,
    config.get<double>("building.image.lsd.scale"),
    config.get<double>("building.image.lsd.sigma_scale"),
    config.get<double>("building.image.lsd.quant"),
    config.get<double>("building.image.lsd.ang_th"),
    config.get<double>("building.image.lsd.log_eps"),
    config.get<double>("building.image.lsd.density_th"),
    config.get<int>("building.image.lsd.n_bins"),
    nullptr, nullptr, nullptr);

  m_csegments.clear();
  for (int i = 0; i < n_out; ++i)
    m_csegments.push_back(cv::Vec4d(
      itw_x(segs[7 * i]), itw_y(segs[7 * i + 1]),
      itw_x(segs[7 * i + 2]), itw_y(segs[7 * i + 3])));
  free(segs);

  LOG_INFO << "#csegments " << m_csegments.size();
}

void UBlock_building::color_space_segment_detection_hawp() {
    try {
        
        if (!Py_IsInitialized()) {
            throw std::runtime_error("Python interpreter is not initialized.");
        }
        
        // Initialize Python interpreter and acquire GIL
        py::gil_scoped_acquire acquire;

        // Import HAWP interface module
        py::module_ hawp_interface = py::module_::import("hawp.interface.hawp_interface");

        // Initialize HAWPv3 model
        auto hawp_model = hawp_interface.attr("HAWPv3Wrapper")(
            "/root/work/code_lod/external/hawp/checkpoints/hawpv3-fdc5487a.pth", // Model weights
            "/root/work/code_lod/external/hawp/hawp/ssl/config/hawpv3.yaml",                  // Model config
            512,                                                                    // Input width
            512,                                                                    // Input height
            0.05,                                                                   // Confidence threshold
            "cuda"                                                                  // Device
        );

        // Convert OpenCV color image to grayscale
        cv::Mat img_gray;
        cv::cvtColor(m_cmap, img_gray, cv::COLOR_BGR2GRAY);

        // Convert OpenCV grayscale image to numpy array
        py::array_t<uint8_t> img_np = py::array_t<uint8_t>(
            {img_gray.rows, img_gray.cols},       // Shape
            img_gray.data                         // Data pointer
        );

        // Call the detect method from HAWPv3 model
        py::object lines_pred_obj = hawp_model.attr("detect")(img_np);

        // Convert returned numpy array to C++ data structure
        auto lines_pred = lines_pred_obj.cast<py::array_t<float>>();
        auto lines_data = lines_pred.unchecked<2>();

        // Clear and populate m_csegments with detected lines
        m_csegments.clear();
        for (ssize_t i = 0; i < lines_data.shape(0); ++i) {
            m_csegments.push_back(cv::Vec4d(
                itw_x(lines_data(i, 0)), itw_y(lines_data(i, 1)),
                itw_x(lines_data(i, 2)), itw_y(lines_data(i, 3))
            ));
        }

        LOG_INFO << "#csegments " << m_csegments.size();

    } catch (const py::error_already_set& e) {
        LOG_ERROR << "Python error in color_space_segment_detection_hawp: " << e.what();
    } catch (const std::exception& e) {
        LOG_ERROR << "Error in color_space_segment_detection_hawp: " << e.what();
    }
}

void UBlock_building::height_space_segment_detection() {
  // normalize height map to 0 ~ 255 double intensity image
  // TODO: will this lose details of small height variation?
  double hmin = std::numeric_limits<double>::max();
  double hmax = std::numeric_limits<double>::lowest();
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      const double h = m_pmap.at<cv::Vec3d>(i, j)[2];
      hmin = h < hmin ? h : hmin;
      hmax = h > hmax ? h : hmax;
    }
  }
  const double range = hmax - hmin;
  std::vector<double> img(m_rows * m_cols, 0.0);
  for (int x = 0; x < m_cols; ++x)
    for (int y = 0; y < m_rows; ++y)
      img[x + y * m_cols] = (m_pmap.at<cv::Vec3d>(y, x)[2] - hmin) / range * 255.0;

  // number of lines detected
  int n_out = 0;
  const auto &config = cm::get_config();
  double *segs = ::LineSegmentDetection(&n_out,
    img.data(), m_cols, m_rows,
    config.get<double>("building.depth.lsd.scale"),
    config.get<double>("building.depth.lsd.sigma_scale"),
    config.get<double>("building.depth.lsd.quant"),
    config.get<double>("building.depth.lsd.ang_th"),
    config.get<double>("building.depth.lsd.log_eps"),
    config.get<double>("building.depth.lsd.density_th"),
    config.get<int>("building.depth.lsd.n_bins"),
    nullptr, nullptr, nullptr);

  m_hsegments.clear();
  for (int i = 0; i < n_out; ++i)
    m_hsegments.push_back(cv::Vec4d(
      itw_x(segs[7 * i]), itw_y(segs[7 * i + 1]),
      itw_x(segs[7 * i + 2]), itw_y(segs[7 * i + 3])));
  free(segs);

  LOG_INFO << "#hsegments " << m_hsegments.size();
}
void UBlock_building::height_space_segment_detection_hawp() {
    try {
        // Initialize Python interpreter and acquire GIL
        py::gil_scoped_acquire acquire;

        // Import HAWP interface module
        py::module_ hawp_interface = py::module_::import("hawp.interface.hawp_interface");

        // Initialize HAWPv3 model
        auto hawp_model = hawp_interface.attr("HAWPv3Wrapper")(
            "/root/work/code_lod/external/hawp/checkpoints/hawpv3-fdc5487a.pth", // Model weights
            "/root/work/code_lod/external/hawp/hawp/ssl/config/hawpv3.yaml",          // Model config
            512,                                                                      // Input width
            512,                                                                      // Input height
            0.05,                                                                     // Confidence threshold
            "cuda"                                                                    // Device
        );

        // Normalize height map to 0 ~ 255 and convert to a grayscale image
        // double hmin = std::numeric_limits<double>::max();
        // double hmax = std::numeric_limits<double>::lowest();
        // for (int i = 0; i < m_rows; ++i) {
        //     for (int j = 0; j < m_cols; ++j) {
        //         const double h = m_pmap.at<cv::Vec3d>(i, j)[2];
        //         hmin = std::min(h, hmin);
        //         hmax = std::max(h, hmax);
        //     }
        // }
        // const double range = hmax - hmin;

        cv::Mat img_gray(m_rows, m_cols, CV_8UC1);
        for (int y = 0; y < m_rows; ++y) {
            for (int x = 0; x < m_cols; ++x) {
                img_gray.at<uint8_t>(y, x) = static_cast<uint8_t>(
                    m_pmap.at<cv::Vec3d>(y, x)[2]);
                //if(m_pmap.at<cv::Vec3d>(y, x)[2]!=0) LOG_INFO << m_pmap.at<cv::Vec3d>(y, x)[2]<<" ";
            }
        }

        // Convert OpenCV grayscale image to numpy array
        py::array_t<uint8_t> img_np = py::array_t<uint8_t>(
            {img_gray.rows, img_gray.cols},       // Shape
            img_gray.data                         // Data pointer
        );

        // Call the detect method from HAWPv3 model
        py::object lines_pred_obj = hawp_model.attr("detect")(img_np);

        // Convert returned numpy array to C++ data structure
        auto lines_pred = lines_pred_obj.cast<py::array_t<float>>();
        auto lines_data = lines_pred.unchecked<2>();

        // Clear and populate m_hsegments with detected lines
        m_hsegments.clear();
        for (ssize_t i = 0; i < lines_data.shape(0); ++i) {
            m_hsegments.push_back(cv::Vec4d(
                itw_x(lines_data(i, 0)), itw_y(lines_data(i, 1)),
                itw_x(lines_data(i, 2)), itw_y(lines_data(i, 3))
            ));
        }

        LOG_INFO << "#hsegments " << m_hsegments.size();

    } catch (const py::error_already_set& e) {
        LOG_ERROR << "Python error in height_space_segment_detection_hawp: " << e.what();
    } catch (const std::exception& e) {
        LOG_ERROR << "Error in height_space_segment_detection_hawp: " << e.what();
    }
}
void UBlock_building::normal_space_segment_detection() {
  m_nsegments.clear();

  int count = 0;
  for (const auto &p : m_planes) {
    // roof plane projection direction
    const auto ov = p.orthogonal_vector();
    cv::Vec3d proj_dir(ov.x(), ov.y(), 0);
    if (ov.x() == 0 && ov.y() == 0)      // edit
      proj_dir[2] = ov.z();
    cv::normalize(proj_dir, proj_dir, 1.0, 0.0, cv::NORM_L2);
    cv::Mat img_di = cv::Mat::zeros(m_rows, m_cols, CV_64F);
    for (int i = 0; i < m_rows; ++i)
      for (int j = 0; j < m_cols; ++j)
        img_di.at<double>(i, j) =
          (m_nmap.at<cv::Vec3d>(i, j).dot(proj_dir) + 1.0) * 127.5;

    int n_out = 0;
    const auto &config = cm::get_config();
    double *segs = ::LineSegmentDetection(&n_out,
      img_di.ptr<double>(), m_cols, m_rows,
      config.get<double>("building.normal.lsd.scale"),
      config.get<double>("building.normal.lsd.sigma_scale"),
      config.get<double>("building.normal.lsd.quant"),
      config.get<double>("building.normal.lsd.ang_th"),
      config.get<double>("building.normal.lsd.log_eps"),
      config.get<double>("building.normal.lsd.density_th"),
      config.get<int>("building.normal.lsd.n_bins"),
      nullptr, nullptr, nullptr);
  
    std::vector<cv::Vec4d> segments;
    for (int i = 0; i < n_out; ++i)
      segments.push_back(cv::Vec4d(
        itw_x(segs[7 * i]), itw_y(segs[7 * i + 1]),
        itw_x(segs[7 * i + 2]), itw_y(segs[7 * i + 3])));
    free(segs);
    count += segments.size();

    m_nsegments.push_back(segments);
  }

  LOG_INFO << "#nsegments " << count;
}

void UBlock_building::normal_space_segment_detection_hawp() {
    try {
        // Initialize Python interpreter and acquire GIL
        py::gil_scoped_acquire acquire;

        // Import HAWP interface module
        py::module_ hawp_interface = py::module_::import("hawp.interface.hawp_interface");

        // Initialize HAWPv3 model
        auto hawp_model = hawp_interface.attr("HAWPv3Wrapper")(
            "/root/work/code_lod/external/hawp/checkpoints/hawpv3-fdc5487a.pth", // Model weights
            "/root/work/code_lod/external/hawp/hawp/ssl/config/hawpv3.yaml",          // Model config
            512,                                                                      // Input width
            512,                                                                      // Input height
            0.05,                                                                     // Confidence threshold
            "cuda"                                                                    // Device
        );

        m_nsegments.clear();
        int count = 0;

        for (const auto &p : m_planes) {
            // Calculate roof plane projection direction
            const auto ov = p.orthogonal_vector();
            cv::Vec3d proj_dir(ov.x(), ov.y(), 0);
            if (ov.x() == 0 && ov.y() == 0) // Adjust projection direction
                proj_dir[2] = ov.z();
            cv::normalize(proj_dir, proj_dir, 1.0, 0.0, cv::NORM_L2);

            // Generate intensity image for the current normal map
            cv::Mat img_di = cv::Mat::zeros(m_rows, m_cols, CV_8UC1);
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_cols; ++j) {
                    double intensity = (m_nmap.at<cv::Vec3d>(i, j).dot(proj_dir) + 1.0) * 127.5;
                    img_di.at<uint8_t>(i, j) = static_cast<uint8_t>(intensity);
                }
            }

            // Convert OpenCV image to numpy array
            py::array_t<uint8_t> img_np = py::array_t<uint8_t>(
                {img_di.rows, img_di.cols}, img_di.data);

            // Call the detect method from HAWPv3 model
            py::object lines_pred_obj = hawp_model.attr("detect")(img_np);

            // Convert returned numpy array to C++ data structure
            auto lines_pred = lines_pred_obj.cast<py::array_t<float>>();
            auto lines_data = lines_pred.unchecked<2>();

            // Store detected lines for the current plane
            std::vector<cv::Vec4d> segments;
            for (ssize_t i = 0; i < lines_data.shape(0); ++i) {
                segments.push_back(cv::Vec4d(
                    itw_x(lines_data(i, 0)), itw_y(lines_data(i, 1)),
                    itw_x(lines_data(i, 2)), itw_y(lines_data(i, 3))
                ));
            }

            count += segments.size();
            m_nsegments.push_back(segments);
        }

        LOG_INFO << "#nsegments " << count;

    } catch (const py::error_already_set &e) {
        LOG_ERROR << "Python error in normal_space_segment_detection_hawp: " << e.what();
    } catch (const std::exception &e) {
        LOG_ERROR << "Error in normal_space_segment_detection_hawp: " << e.what();
    }
}

void UBlock_building::detect_roof_planes() {
  typedef CGAL::Shape_detection_3::Plane<Traits> Plane;

  LOG_INFO << "detect planes...";
// 添加法线平滑预处理
  cv::Mat smoothed_normal_map;
  cv::GaussianBlur(m_nmap, smoothed_normal_map, cv::Size(5,5), 3.0);
  // construct building roof top point cloud with normal.
  Point_set points(true);
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      
      if (m_bmap.at<unsigned char>(i, j) > 0) {
        const cv::Vec3d p = m_pmap.at<cv::Vec3d>(i, j);
        const cv::Vec3d n = m_nmap.at<cv::Vec3d>(i, j);
        // 添加高度权重
        const double weight = 1.0 + 0.1*(p[2] - m_ground_level);
        points.insert(
          Point_3(p[0], p[1], p[2]),
          Vector_3(n[0]*weight, n[1]*weight, n[2]*weight));
      }//遍历位置图和法线图，将属于建筑区域（掩码值 > 0）的点插入到点云集合 points 中。
//点云集合包含每个点的位置和法线信息。
    }
  }

  const auto &config = cm::get_config();
  Efficient_RANSAC::Parameters parameters;
  parameters.probability = config.get<double>(
    "building.plane_detection.ransac.probability");
  parameters.min_points = std::size_t(config.get<double>(
    "building.plane_detection.ransac.min_area") / (m_step * m_step));
  parameters.epsilon = config.get<double>(
    "building.plane_detection.ransac.epsilon");
  parameters.cluster_epsilon = config.get<double>(
    "building.plane_detection.ransac.cluster_epsilon");
  parameters.normal_threshold = config.get<double>(
    "building.plane_detection.ransac.normal_threshold");

  Efficient_RANSAC ransac;
  ransac.set_input(points, points.point_map(), points.normal_map());
  ransac.add_shape_factory<Plane>();
  ransac.detect(parameters);
  const Efficient_RANSAC::Plane_range shapes = ransac.planes();
  write_planes_to_ply(points, shapes, m_prefix);
  // CGAL::regularize_planes(points,
  //   points.point_map(),
  //   shapes,
  //   CGAL::Shape_detection_3::Plane_map<Traits>(),
  //   CGAL::Shape_detection_3::Point_to_shape_index_map<Traits>(points, shapes),
  //   true,
  //   true,
  //   true,
  //   true,
  //   config.get<double>("building.plane_detection.ransac.parallelism"),
  //   config.get<double>("building.plane_detection.ransac.coplanar"));

  // Prints number of assigned shapes and unassigned points.
  const double coverage =
    double(points.size() - ransac.number_of_unassigned_points()) /
    double(points.size());
  // delete 
  LOG_INFO << "#primitives: " << shapes.size() << ", #coverage: " << coverage;
  for (const auto s : shapes)
    LOG_DEBUG << s->info();

  // include ground plane for labeling
  m_planes.clear();
  m_planes.push_back({
    Point_3(0.0, 0.0, m_ground_level),
    Vector_3(0.0, 0.0, 1.0)});

  if (shapes.size() < 1) {
    LOG_INFO << "No roof plane detected, use default.";
    // average height
    int count = 0;
    double avg_height = 0.0;
    for (int i = 0; i < m_rows; ++i) {
      for (int j = 0; j < m_cols; ++j) {
        if (m_bmap.at<unsigned char>(i, j) > 0) {
          avg_height += m_pmap.at<cv::Vec3d>(i, j)[2];
          ++count;
        }
      }
    }
    avg_height /= double(count);
    m_planes.push_back({
      Point_3(0.0, 0.0, avg_height),
      Vector_3(0.0, 0.0, 1.0)});

    return;
  }

  // TODO: fitting all inliers
  // collect detected planes
  // sort shapes in decreasing order of assigned percentage
  const std::size_t total_assigned = ransac.number_of_unassigned_points();
  struct Shape_sorter {
    boost::shared_ptr<CGAL::Shape_detection_3::Shape_base<Traits>> ptr;
    double pct;
  };
  std::vector<Shape_sorter> sorted_shapes;
  for (const auto s : ransac.shapes()) {
    if (std::abs((dynamic_cast<Plane *>(s.get()))->plane_normal().z()) < 0.5) // 70 degree
      continue;
    sorted_shapes.push_back({ s,
      double(s->indices_of_assigned_points().size()) / double(total_assigned) });
    std::sort(sorted_shapes.begin(), sorted_shapes.end(),
      [](const Shape_sorter &a, const Shape_sorter &b) {
      return a.pct > b.pct;
    });
  }
   
  for (const auto ss : sorted_shapes)
    m_planes.push_back(
      static_cast<Plane_3>(*dynamic_cast<Plane *>(ss.ptr.get())));
 
}


void UBlock_building::write_planes_to_ply(const Point_set& points, const Efficient_RANSAC::Plane_range& shapes, const std::string& output_file) {

    // 为每个形状分配唯一颜色
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < shapes.size(); ++i) {
        colors.emplace_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
    }

    size_t shape_index = 0;
    for (const auto& shape : shapes) {
        const auto& indices = shape->indices_of_assigned_points();
        
        if (indices.empty()) {
            ++shape_index;
            continue;
        }

        std::string output_file2 = output_file + "_shape_" + std::to_string(shape_index) + ".ply";

        std::ofstream out(output_file2);
        if (!out.is_open()) {
            throw std::runtime_error("无法打开输出文件: " + output_file2);
        }

        // PLY 文件头部
        out << "ply\nformat ascii 1.0\nelement vertex " << indices.size()
            << "\nproperty float x\nproperty float y\nproperty float z\n"
            << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

        cv::Vec3b color = colors[shape_index];

        // 写入点和颜色信息
        for (const auto& idx : indices) {
            const auto& point = points.point(idx);
            //LOG_INFO<<point.x() << " " << point.y() << " " << point.z() << " ";
            out << point.hx() << " " << point.hy() << " " << point.hz() << " "
                << static_cast<int>(color[2]) << " "  // R
                << static_cast<int>(color[1]) << " "  // G
                << static_cast<int>(color[0]) << "\n"; // B
        }

        out.close();
        ++shape_index;
    }
}

// void UBlock_building::detect_roof_planes_without_normals(){
//   typedef CGAL::Shape_detection_3::Plane<Traits> Plane;

//   LOG_INFO << "detect planes...";
//   // construct building roof top point cloud with normal.
//   Point_set points(false);
//   for (int i = 0; i < m_rows; ++i) {
//     for (int j = 0; j < m_cols; ++j) {
//       if (m_bmap.at<unsigned char>(i, j) > 0) {
//         const cv::Vec3d p = m_pmap.at<cv::Vec3d>(i, j);
//         points.insert(
//           Point_3(p[0], p[1], p[2]));
//       }//遍历位置图和法线图，将属于建筑区域（掩码值 > 0）的点插入到点云集合 points 中。
// //点云集合包含每个点的位置和法线信息。
//     }
//   }

//   const auto &config = cm::get_config();
//   Efficient_RANSAC::Parameters parameters;
//   parameters.probability = config.get<double>(
//     "building.plane_detection.ransac.probability");
//   parameters.min_points = std::size_t(config.get<double>(
//     "building.plane_detection.ransac.min_area") / (m_step * m_step));
//   parameters.epsilon = config.get<double>(
//     "building.plane_detection.ransac.epsilon");
//   parameters.cluster_epsilon = config.get<double>(
//     "building.plane_detection.ransac.cluster_epsilon");

//   Efficient_RANSAC ransac;
//   ransac.set_input(points);
//   ransac.add_shape_factory<Plane>();
//   ransac.detect(parameters);
//   const Efficient_RANSAC::Plane_range shapes = ransac.planes();
//   // Prints number of assigned shapes and unassigned points.
//   const double coverage =
//     double(points.size() - ransac.number_of_unassigned_points()) /
//     double(points.size());
//   // delete 
//   LOG_INFO << "#primitives: " << shapes.size() << ", #coverage: " << coverage;
//   for (const auto s : shapes)
//     LOG_DEBUG << s->info();

//   // include ground plane for labeling
//   m_planes.clear();
//   m_planes.push_back({
//     Point_3(0.0, 0.0, m_ground_level),
//     Vector_3(0.0, 0.0, 1.0)});

//   if (shapes.size() < 1) {
//     LOG_INFO << "No roof plane detected, use default.";
//     // average height
//     int count = 0;
//     double avg_height = 0.0;
//     for (int i = 0; i < m_rows; ++i) {
//       for (int j = 0; j < m_cols; ++j) {
//         if (m_bmap.at<unsigned char>(i, j) > 0) {
//           avg_height += m_pmap.at<cv::Vec3d>(i, j)[2];
//           ++count;
//         }
//       }
//     }
//     avg_height /= double(count);
//     m_planes.push_back({
//       Point_3(0.0, 0.0, avg_height),
//       Vector_3(0.0, 0.0, 1.0)});

//     return;
//   }

//   // TODO: fitting all inliers
//   // collect detected planes
//   // sort shapes in decreasing order of assigned percentage
//   const std::size_t total_assigned = ransac.number_of_unassigned_points();
//   struct Shape_sorter {
//     boost::shared_ptr<CGAL::Shape_detection_3::Shape_base<Traits>> ptr;
//     double pct;
//   };
//   std::vector<Shape_sorter> sorted_shapes;
//   for (const auto s : ransac.shapes()) {
//     if (std::abs((dynamic_cast<Plane *>(s.get()))->plane_normal().z()) < 0.5) // 70 degree
//       continue;
//     sorted_shapes.push_back({ s,
//       double(s->indices_of_assigned_points().size()) / double(total_assigned) });
//     std::sort(sorted_shapes.begin(), sorted_shapes.end(),
//       [](const Shape_sorter &a, const Shape_sorter &b) {
//       return a.pct > b.pct;
//     });
//   }
   
//   for (const auto ss : sorted_shapes)
//     m_planes.push_back(
//       static_cast<Plane_3>(*dynamic_cast<Plane *>(ss.ptr.get())));
// }

void UBlock_building::construct_segment_arrangement() {
  const double extend_ratio = cm::get_config().get<double>(
    "building.arr.extend_ratio");
  // overlay grid arrangement and compute data
  struct Overlay_data {
    Overlay_data() : fidx(0), pixel(0, 0) {}
    Overlay_data(const int &f, const cv::Vec2i &p) : fidx(f), pixel(p) {}
    Overlay_data(const Overlay_data &obj) : fidx(obj.fidx), pixel(obj.pixel) {}
    int fidx;
    cv::Vec2i pixel;
  };
  struct Overlay_label {
    Overlay_data operator() (const int &f, const cv::Vec2i &p) const {
      return {f, p};
    }
  };
  typedef CGAL::Arr_face_extended_dcel<Arr_traits, cv::Vec2i> Dcel_grid;
  typedef CGAL::Arrangement_2<Arr_traits, Dcel_grid> Arr_grid;
  typedef CGAL::Arr_face_extended_dcel<Arr_traits, Overlay_data> Dcel_overlay;
  typedef CGAL::Arrangement_2<Arr_traits, Dcel_overlay> Arr_overlay;
  typedef CGAL::Arr_face_overlay_traits<
    Arr_with_index, Arr_grid, Arr_overlay, Overlay_label> Overlay_traits;

  // grid segments, in world coordinate
  std::vector<Kernel_epec::Segment_2> grid_segs;
  for (int i = 0; i < m_rows + 1; ++i)
    grid_segs.push_back({
      Kernel_epec::Point_2(itw_x(-1), itw_y(i)),
      Kernel_epec::Point_2(itw_x(m_cols + 1), itw_y(i))});
  for (int j = 0; j < m_cols + 1; ++j)
    grid_segs.push_back({
      Kernel_epec::Point_2(itw_x(j), itw_y(-1)),
      Kernel_epec::Point_2(itw_x(j), itw_y(m_rows + 1))});

  // construct sampling grid arrangement
  LOG_INFO << "Construct sampling grid arrangement";
  LOG_INFO << "#grid_segs: " << grid_segs.size();
  Arr_grid grid_arr;
  CGAL::insert(grid_arr, grid_segs.begin(), grid_segs.end());
  LOG_INFO << "#f: " << grid_arr.number_of_faces()
    << ", #v: " << grid_arr.number_of_vertices();
  for (auto fitr = grid_arr.faces_begin(); fitr != grid_arr.faces_end(); ++fitr) {
    if (fitr->is_unbounded()) {
      fitr->set_data(cv::Vec2i(-1, -1));
      continue;
    }
    // find lower left corner point in world coordinate
    // it is the upper left corner in image coordinate
    auto curr = fitr->outer_ccb();
    Kernel_epec::Point_2 ll_pt = curr->target()->point();
    do {
      if (curr->target()->point() < ll_pt)
        ll_pt = curr->target()->point();
      ++curr;
    } while (curr != fitr->outer_ccb());
    // set face row / column in image: world x, y -> image x, y -> image i, j
    const int ridx = int(std::round(wti_y(CGAL::to_double(ll_pt.y())))) - 1;
    const int cidx = int(std::round(wti_x(CGAL::to_double(ll_pt.x()))));
    fitr->set_data(cv::Vec2i(ridx, cidx));
  }

  // extend segments to close possible gaps
  typedef CGAL::Cartesian_converter<Kernel_epec, Kernel> To_geom;
  To_geom to_geom;
  std::vector<Kernel::Segment_2> segments;
  
  for (const auto &seg : m_segments) {
    const auto midpt = to_geom(CGAL::midpoint(seg.source(), seg.target()));
    const auto dir = to_geom(seg.to_vector());
    segments.push_back({
      midpt + (extend_ratio / 2.0) * dir,
      midpt - (extend_ratio / 2.0) * dir
    });
  }
 
  // extend segments local merge
  std::vector<Kernel_epec::Segment_2> segments_re;
  segments_re = regularize_segments_graph<Kernel_epec>(
    segments,
    cm::get_config().get<double>(
      "building.regularization.parallel_threshold_re"),
    cm::get_config().get<double>(
      "building.regularization.collinear_threshold")
    );
  // add bounding segments to close the arrangement
  segments_re.push_back(grid_segs[0]);
  segments_re.push_back(grid_segs[m_rows]);
  segments_re.push_back(grid_segs[m_rows + 1]);
  segments_re.push_back(grid_segs[m_rows + m_cols + 1]);

  // line segment arrangement
  LOG_INFO << "Construct line segment arrangement";
  LOG_INFO << "#segments: " << segments_re.size();
  m_arr.clear();
  CGAL::insert(m_arr, segments_re.begin(), segments_re.end()); 
  LOG_INFO << "#f: " << m_arr.number_of_faces()
    << ", #v: " << m_arr.number_of_vertices();

  int idx = 0;
  for (auto fitr = m_arr.faces_begin(); fitr != m_arr.faces_end(); ++fitr) {
    if (fitr->is_unbounded())
      fitr->set_data(-1);
    else
      fitr->set_data(idx++);
  }

  // trim arrangement
  LOG_INFO << "Trim arrangement";
  for (auto eitr = m_arr.edges_begin(); eitr != m_arr.edges_end(); ++eitr) {
    if (eitr->face()->data() == eitr->twin()->face()->data()) {
      m_arr.remove_edge(eitr);
      eitr = m_arr.edges_begin();
    }
  }
  LOG_INFO << "#f: " << m_arr.number_of_faces()
    << ", #v: " << m_arr.number_of_vertices();
  // merge collinear edges
  LOG_INFO << "Merge collinear edges";
  for (auto fitr = m_arr.faces_begin(); fitr != m_arr.faces_end(); ++fitr) {
    if (fitr->is_unbounded())
      continue;
    auto curr = fitr->outer_ccb();
    bool merged = false;
    do {
      merged = false;
      const Arr_with_index::Halfedge_handle h0 = curr;
      const Arr_with_index::Halfedge_handle h1 = curr->next();
      if (h0->target()->degree() == 2 &&
        h0->curve().line().has_on(h1->target()->point())) {
        curr = m_arr.merge_edge(h0, h1,
          Kernel_epec::Segment_2(h0->source()->point(), h1->target()->point()));
        merged = true;
      }
      else
        ++curr;
    } while (merged || curr != fitr->outer_ccb());
  }
  LOG_INFO << "#f: " << m_arr.number_of_faces()
    << ", #v: " << m_arr.number_of_vertices();

  // overlay
  LOG_INFO << "Overlaying arrangement";
  Arr_overlay overlay_arr;
  Overlay_traits overlay_traits;
  CGAL::overlay(m_arr, grid_arr, overlay_arr, overlay_traits);
  LOG_INFO << "#f: " << overlay_arr.number_of_faces()
    << ", #v: " << overlay_arr.number_of_vertices();

  // compute pixel distances to planes
  std::vector<cv::Mat> pixel_distance_to_planes;
  for (const auto &s : m_planes) {
    cv::Mat dis_map(m_rows, m_cols, CV_64F, cv::Scalar(0.0));
    for (int i = 0; i < m_rows; ++i) {
      for (int j = 0; j < m_cols; ++j) {
        const cv::Vec3d pos = m_pmap.at<cv::Vec3d>(i, j);
        auto result = CGAL::intersection(
          s,
          Kernel::Line_3(Point_3(pos[0], pos[1], pos[2]), Vector_3(0.0, 0.0, 1.0)));
        const Point_3 *intersect_pt = nullptr;
        if (result && (intersect_pt = boost::get<Point_3>(&(*result)))) {
          Point_3 p = *intersect_pt;
          dis_map.at<double>(i, j) = std::abs( p.z() - pos[2] );
            //std::sqrt(CGAL::squared_distance(p, Point_3(pos[0], pos[1], pos[2])));
        }

        else
          LOG_WARNING << "no intersection";
      }
    }
    pixel_distance_to_planes.push_back(dis_map);
  }

  // set segment arrangement face data
  m_arr_data = std::vector<Arr_face_data>(
    m_arr.number_of_faces() - m_arr.number_of_unbounded_faces(),
    Arr_face_data(m_planes.size()));
  for (auto fitr = overlay_arr.faces_begin(); fitr != overlay_arr.faces_end(); ++fitr) {
    const int fidx = fitr->data().fidx;
    const cv::Vec2i pixel = fitr->data().pixel;
    if (fitr->is_unbounded() || fidx < 0)
      continue;
    Arr_face_data &arrd = m_arr_data[fidx];

    CGAL::Polygon_2<Kernel_epec> plg;
    auto curr = fitr->outer_ccb();
    do {
      plg.push_back(curr->target()->point());
      ++curr;
    } while (curr != fitr->outer_ccb());
    const double area = CGAL::to_double(plg.area());
    assert(area > 0.0);
    arrd.area += area;

    if (pixel[0] < 0 || pixel[1] < 0) {
      // faces out side sampling grid is assigned with ground-like data
      for (auto &d : arrd.distances)
        d = 1e20;
      arrd.distances.front() = 0.0;
      arrd.normal = cv::Vec3d(0.0, 0.0, 1.0);
    }
    else {
      for (std::size_t i = 0; i < pixel_distance_to_planes.size(); ++i)
        arrd.distances[i] += area * pixel_distance_to_planes[i].at<double>(pixel);
      arrd.normal += area * m_nmap.at<cv::Vec3d>(pixel);
    }
  }
  for (auto &arrd : m_arr_data) {
    for (auto &d : arrd.distances)
      d /= arrd.area;
    arrd.normal /= cv::norm(arrd.normal);
  }
}

void UBlock_building::segment_arrangement_MRF_labeling() {
  class SmoothFn : public GCoptimization::SmoothCostFunctor {
  public:
    SmoothFn(const std::vector<Arr_face_data> &fdata_, const double &height_) : fdata(fdata_) ,  height(height_) {}

    double compute(int s1, int s2, int l1, int l2) {
      if (l1 == l2)
        return 0.0;
      // mutual distance
      // if s1 and s2 are quite different, low smooth value is set
      // if s1 and s2 are similar, stronger connection is set
      double mutu_dis = (fdata[s1].distances[l2] + fdata[s2].distances[l1]) / 2;
      return mutu_dis > height ? 1 : mutu_dis / height;
    }

  private:
    const std::vector<Arr_face_data> &fdata;
    const double height; // mesh box max height
  };

  // two faces may be separated by collinear segments, we need mapping
  typedef std::pair<int, int> Adj_facets;
  typedef std::map<Adj_facets, double> Adj_map;

  const auto &config = cm::get_config();
  LOG_INFO << "MRF params: " <<
    config.get<bool>("building.arr.mrf.use_swap") << ' ' <<
    config.get<double>("building.arr.mrf.balance") << ' ' <<
    config.get<int>("building.arr.mrf.iterations");
  if (m_planes.size() < 2) {
    LOG_ERROR << "Insufficient shape.";
    return;
  }

  // set up neighboring
  Adj_map adj_map;
  for (auto eitr = m_arr.edges_begin(); eitr != m_arr.edges_end(); ++eitr) {
    if (eitr->face()->is_unbounded() || eitr->twin()->face()->is_unbounded())
      continue;
    int si = eitr->face()->data();
    int sj = eitr->twin()->face()->data();
    if (si == sj)
      continue;
    else if (si > sj)
      std::swap(si, sj);
    const double elen = std::sqrt(CGAL::to_double(
      CGAL::squared_distance(eitr->target()->point(), eitr->source()->point())));
    auto find_itr = adj_map.find({ si, sj });
    if (find_itr == adj_map.end())
      adj_map.insert({{si, sj}, elen});
    else
      find_itr->second += elen;
  }

  // set up arrangement graph
  const int nb_sites = int(m_arr_data.size());
  const int nb_labels = int(m_planes.size());

  // setup data term
  std::vector<double> data(nb_sites * nb_labels, 0.0);
  for (int i = 0; i < nb_sites; ++i)
    for (int j = 0; j < nb_labels; ++j) {
      data[i * nb_labels + j] =
        m_arr_data[i].distances[j] * m_arr_data[i].area;
    }
     


  // setup smooth
  std::vector<double> smooth(nb_labels * nb_labels, 1.0);
  for (int i = 0; i < nb_labels; ++i)
    smooth[i * nb_labels + i] = 0.0;

  // MRF labeling
  try {
    GCoptimizationGeneralGraph gc(nb_sites, nb_labels);
    gc.setDataCost(data.data());
    SmoothFn smoothfn(m_arr_data, m_max_height);
    //gc.setSmoothCostFunctor(&smoothfn);
     gc.setSmoothCost(smooth.data());

    // set neighboring
    // arrangement face area and pixel length are both in image coordinate
    // const double pixel_length = 0.2;
    for (const auto &adj_pair : adj_map)
      gc.setNeighbors(
        adj_pair.first.first,
        adj_pair.first.second,
        adj_pair.second * config.get<double>("building.arr.mrf.balance"));

    LOG_INFO << "Before optimization energy is: " << gc.compute_energy();
    if (config.get<bool>("building.arr.mrf.use_swap")) {
      LOG_INFO << "Alpha-beta swap algorithm.";
      gc.swap(config.get<int>("building.arr.mrf.iterations"));
    }
    else {
      LOG_INFO << "Alpha expansion algorithm.";
      gc.expansion(config.get<int>("building.arr.mrf.iterations"));
    }
    LOG_INFO << "After optimization energy is: " << gc.compute_energy();

    for (auto fitr = m_arr.faces_begin(); fitr != m_arr.faces_end(); ++fitr) {
      if (fitr->data() >= 0)
        fitr->set_data(gc.whatLabel(fitr->data()));
      //if (fitr->is_unbounded()) continue;
      //std::cout << fitr->data() << " ";
    }
    std::cout << "\n";
  }
  catch (GCException e) {
    e.Report();
  }
 // for (auto &arrd : m_arr_data) {
   // std::cout << arrd.area << " " << arrd.distances[0] << " " << arrd.distances[1] << " " << std::endl;
 // }
} 


void UBlock_building::segment_arrangement_extrusion() {
  // operate on the copy
  // hopefully extra data is also copied
  auto arr(m_arr);

  // trim arrangement by merging faces of same shape label
  
    LOG_INFO << "Trim arrangement";
  for (auto eitr = arr.edges_begin(); eitr != arr.edges_end(); ++eitr) {
    if (eitr->face()->data() == eitr->twin()->face()->data()) {
      // should set data for the returned face
      arr.remove_edge(eitr);
      eitr = arr.edges_begin();
    }
  }
  

  // merge collinear segments
  LOG_INFO << "#v: " << arr.number_of_vertices();
  for (auto fitr = arr.faces_begin(); fitr != arr.faces_end(); ++fitr) {
    if (fitr->is_unbounded())
      continue;
    auto curr = fitr->outer_ccb();
    bool merged = false;
    do {
      merged = false;
      const Arr_with_index::Halfedge_handle h0 = curr;
      const Arr_with_index::Halfedge_handle h1 = curr->next();
      if (h0->target()->degree() == 2 &&
        h0->curve().line().has_on(h1->target()->point())) {
        curr = arr.merge_edge(h0, h1,
          Kernel_epec::Segment_2(h0->source()->point(), h1->target()->point()));
        merged = true;
      }
      else
        ++curr;
    } while (merged || curr != fitr->outer_ccb());
  }
  LOG_INFO << "#v: " << arr.number_of_vertices();
  LOG_INFO << "#f: " << arr.number_of_faces();

  // lod2 extrusion, with color
  {
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;
    int nn = 0;
    for (auto fitr = arr.faces_begin(); fitr != arr.faces_end(); ++fitr) {
      const int label = fitr->data();
      if (label <= 0 || label > m_planes.size())
        continue;
      nn++;
      // points
      const std::size_t offset = points.size();
      std::size_t nb_vertices = 0;
      auto curr = fitr->outer_ccb();
      do {
        const Point_3 p(
          CGAL::to_double(curr->target()->point().x()),
          CGAL::to_double(curr->target()->point().y()),
          m_ground_level);
        points.push_back(p);
        ++curr;
        ++nb_vertices;
      } while (curr != fitr->outer_ccb());
      for (std::size_t i = 0; i < nb_vertices; ++i) {
        Point_3 p(0.0, 0.0, 0.0);
        auto result = CGAL::intersection(
          m_planes[label],
          Kernel::Line_3(points[offset + i], Vector_3(0.0, 0.0, 1.0)));
        const Point_3 *intersect_pt = nullptr;
        if (result && (intersect_pt = boost::get<Point_3>(&(*result))))
          p = *intersect_pt;
        else
          LOG_WARNING << "no intersection";
        if (p.z() < m_ground_level)
          points.push_back(Point_3(p.x(), p.y(), m_ground_level));
        else
        points.push_back(p);
      }
     

      // polygons
      std::vector<std::size_t> roof_plg;
      for (std::size_t i = 0; i < nb_vertices; ++i)
        roof_plg.push_back(offset + nb_vertices + i);
      // push color
      roof_plg.push_back(102); roof_plg.push_back(153); roof_plg.push_back(255);
      polygons.push_back(roof_plg);

      for (std::size_t i = 0; i < nb_vertices; ++i) {
        std::vector<std::size_t> side_plg{
          offset + i,
          offset + (i + 1) % nb_vertices,
          offset + nb_vertices + (i + 1) % nb_vertices,
          offset + nb_vertices + i,
          // color
          255, 255, 153};
        polygons.push_back(side_plg);
      }
    }
    //std::cout << nn << std::endl;
    std::ofstream ofs(m_prefix + "_lod2.off");
    ofs << "OFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
    for (const auto &p : points)
      ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (const auto &plg : polygons) {
      ofs << (plg.size() - 3);
      for (const auto &p : plg)
        ofs << ' ' << p;
      ofs << '\n';
    }
  }

  // lod1 extrusion, with color
  {
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;

    for (auto fitr = arr.faces_begin(); fitr != arr.faces_end(); ++fitr) {
      const int label = fitr->data();
      if (label <= 0)
        continue;

      const std::size_t offset = points.size();
      // points
      std::vector<Kernel::Point_2> boundary;
      auto curr = fitr->outer_ccb();
      do {
        boundary.push_back({
          CGAL::to_double(curr->target()->point().x()),
          CGAL::to_double(curr->target()->point().y())});
        points.push_back({
          CGAL::to_double(curr->target()->point().x()),
          CGAL::to_double(curr->target()->point().y()),
          m_ground_level});
        ++curr;
      } while (curr != fitr->outer_ccb());

      // lod1 extrusion to the averaged height of current roof patch
      // i.e z component of the centroid of the 3D polygon
      const Kernel::Point_2 centroid = polygon_centroid(boundary);
      Point_3 centroid_pt(0.0, 0.0, 0.0);
      auto result = CGAL::intersection(
        m_planes[label],
        Kernel::Line_3(Point_3(centroid.x(), centroid.y(), m_ground_level),
          Vector_3(0.0, 0.0, 1.0)));
      const Point_3 *intersect_pt = nullptr;
      if (result && (intersect_pt = boost::get<Point_3>(&(*result))))
        centroid_pt = *intersect_pt;
      else
        LOG_WARNING << "no intersection";

      const std::size_t nb_vertices = boundary.size();
      for (std::size_t i = 0; i < nb_vertices; ++i) {
        const Point_3 &p = points[offset + i];
        points.push_back({p.x(), p.y(), centroid_pt.z()});
      }

      // polygons
      std::vector<std::size_t> roof_plg;
      for (std::size_t i = 0; i < nb_vertices; ++i)
        roof_plg.push_back(offset + nb_vertices + i);
      // push color
      roof_plg.push_back(102); roof_plg.push_back(153); roof_plg.push_back(255);
      polygons.push_back(roof_plg);

      for (std::size_t i = 0; i < nb_vertices; ++i) {
        std::vector<std::size_t> side_plg{
          offset + i,
          offset + (i + 1) % nb_vertices,
          offset + nb_vertices + (i + 1) % nb_vertices,
          offset + nb_vertices + i,
          // color
          255, 255, 153};
        polygons.push_back(side_plg);
      }
    }

    std::ofstream ofs(m_prefix + "_lod1.off");
    ofs << "OFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
    for (const auto &p : points)
      ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (const auto &plg : polygons) {
      ofs << (plg.size() - 3);
      for (const auto &p : plg)
        ofs << ' ' << p;
      ofs << '\n';
    }
  }

  // lod0 extrusion
  {
    // construct the outer boundary arrangement
    std::vector<Kernel_epec::Segment_2> segs;
    for (auto eitr = arr.edges_begin(); eitr != arr.edges_end(); ++eitr) {
      if (eitr->face()->is_unbounded()) {
        if (eitr->twin()->face()->data() > 0)
          segs.push_back({eitr->source()->point(), eitr->target()->point()});
      }
      // edges of same label has been removed
      else if (eitr->face()->data() == 0) {
        if (eitr->twin()->face()->data() > 0)
          segs.push_back({eitr->source()->point(), eitr->target()->point()});
      }
      else if (eitr->face()->data() > 0) {
        if (eitr->twin()->face()->data() <= 0)
          segs.push_back({eitr->source()->point(), eitr->target()->point()});
      }
    }
    Arr_with_index tmp_arr;
    CGAL::insert(tmp_arr, segs.begin(), segs.end());

    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;
    // holes of the only unbounded face, clock wise
    const auto uf = tmp_arr.unbounded_face();
    for (Arr_with_index::Hole_iterator hi = uf->holes_begin(); hi != uf->holes_end(); ++hi) {
      std::vector<std::size_t> foot_print_plg;
      const std::size_t offset = points.size();
      std::size_t nb_vertices = 0;
      Arr_with_index::Ccb_halfedge_circulator curr = *hi;
      do {
        const Point_3 p(
          CGAL::to_double(curr->target()->point().x()),
          CGAL::to_double(curr->target()->point().y()),
          m_ground_level);
        points.push_back(p);
        ++nb_vertices;
        ++curr;
      } while (curr != *hi);
      for (std::size_t i = nb_vertices; i > 0; --i)
        foot_print_plg.push_back(offset + i - 1);
      // push color
      foot_print_plg.push_back(102); foot_print_plg.push_back(153); foot_print_plg.push_back(255);
      polygons.push_back(foot_print_plg);
    }
    std::ofstream ofs(m_prefix + "_lod0.off");
    ofs << "OFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
    for (const auto &p : points)
      ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    for (const auto &plg : polygons) {
      ofs << (plg.size() - 3);
      for (const auto &p : plg)
        ofs << ' ' << p;
      ofs << '\n';
    }
  }
}

#define pi 3.1415926
std::vector<Kernel::Vector_2> UBlock_building::find_mainly_dir(std::vector<Kernel::Segment_2>& segments, double ang) {
  double thre = ang * pi / 180;
  std::vector<std::pair<Kernel::Vector_2, int>> dir;
  for (auto &seg : segments) {
    Kernel::Vector_2 seg_dir = seg.to_vector() / CGAL::sqrt(seg.to_vector().squared_length());
    int tag = 0;
    for (auto &sub_dir : dir) {
      if (std::abs(sub_dir.first * seg_dir) > cos(thre) || std::abs(sub_dir.first.perpendicular(CGAL::POSITIVE) * seg_dir) > cos(thre)) {
        sub_dir.second++;
        sub_dir.first = sub_dir.first + (seg_dir * sub_dir.first > 0 ? seg_dir : (-seg_dir));
        sub_dir.first /= CGAL::sqrt(sub_dir.first.squared_length());
        tag == 1;
        break;
      }
    }
    if (tag == 0) {
      dir.push_back(std::make_pair(seg_dir,1));
    }
  }
  Kernel::Vector_2 main1;
  int max_num = 0;
  for (auto &d : dir) {
    if (d.second > max_num) {
      max_num = d.second;
      main1 = d.first;
    }
  }
  std::vector<Kernel::Vector_2> res;
  res.push_back(main1);
  res.push_back(main1.perpendicular(CGAL::POSITIVE));

  typedef CGAL::Aff_transformation_2<Kernel> Transformation;
  Transformation rotate(CGAL::ROTATION, sin(pi/4), cos(pi/4)); 
  Kernel::Vector_2 ro_main1(rotate(main1));
  res.push_back(ro_main1);
  res.push_back(ro_main1.perpendicular(CGAL::POSITIVE));
  return res;
}

void UBlock_building::save_mesh() {
  const std::string fname = m_prefix + "_mesh.off";
  std::ofstream ofs(fname);
  ofs << m_mesh;
}

void UBlock_building::save_point_cloud() {
  Point_set points(true);
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      if (m_bmap.at<unsigned char>(i, j) == 0)
        continue;
      const cv::Vec3d &nom = m_nmap.at<cv::Vec3d>(i, j);
      const cv::Vec3d &pos = m_pmap.at<cv::Vec3d>(i, j);
      points.insert(
        Point_3(pos[0], pos[1], pos[2]),
        Vector_3(nom[0], nom[1], nom[2]));
    }
  }
  std::ofstream ofs(m_prefix + ".ply");
  ofs << points;
}

void UBlock_building::write_color_segments() {
  cv::Mat img_out = m_cmap.clone();
  cv::cvtColor(img_out, img_out, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2BGR);
  for (const auto &s : m_csegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    cv::line(img_out, ps, pt, cv::Scalar(0, 255, 255));
  }
  const std::string fname = m_prefix + "_segc.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_color_segments_pnt() {
    cv::Mat img_out = m_cmap.clone();
    cv::cvtColor(img_out, img_out, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2BGR);

    for (const auto &s : m_csegments) {
        // 计算线段的起点和终点
        const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
        const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));

        // 绘制线段
        cv::line(img_out, ps, pt, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        // 绘制起点和终点的小圆点
        cv::circle(img_out, ps, 3, cv::Scalar(255, 0, 0), -1); // 起点，红色小圆
        cv::circle(img_out, pt, 3, cv::Scalar(255, 0, 0), -1); 
    }

    // 保存结果图像
    const std::string fname = m_prefix + "_segc.png";
    if (!cv::imwrite(fname, img_out)) {
        throw std::runtime_error("Failed to write file " + fname);
    }
}


void UBlock_building::write_height_segments() {
  // height map to gray scale
  cv::Mat img_out(m_rows, m_cols, CV_64F);
  int from_to[] = {2, 0};
  cv::mixChannels(&m_pmap, 1, &img_out, 1, from_to, 1);

  double min = 0.0, max = 0.0;
  cv::minMaxLoc(img_out, &min, &max);
  for (int i = 0; i < m_rows; ++i)
    for (int j = 0; j < m_cols; ++j)
      img_out.at<double>(i, j) =
        (img_out.at<double>(i, j) - min) / (max -min) * 255.0;
  img_out.convertTo(img_out, CV_8U);
  cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2BGR);

  for (const auto &s : m_hsegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    // color image
    // cv::line(img_out, ps, pt, cv::Scalar(0, 255, 0));
    // gray scale image
    cv::line(img_out, ps, pt, cv::Scalar(255, 135, 0));
  }
  const std::string fname = m_prefix + "_segh.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_normal_segments() {
  for (int i = 0; i < m_planes.size(); ++i) {
    // roof plane projection direction
    const auto ov = m_planes[i].orthogonal_vector();
    cv::Vec3d proj_dir(ov.x(), ov.y(), 0.0);
    cv::normalize(proj_dir, proj_dir, 1.0, 0.0, cv::NORM_L2);
    cv::Mat img_di = cv::Mat::zeros(m_rows, m_cols, CV_64F);
    for (int i = 0; i < m_rows; ++i)
      for (int j = 0; j < m_cols; ++j)
        img_di.at<double>(i, j) =
          (m_nmap.at<cv::Vec3d>(i, j).dot(proj_dir) + 1.0) * 127.5;
    img_di.convertTo(img_di, CV_8U);

    cv::cvtColor(img_di, img_di, cv::COLOR_GRAY2BGR);
    for (const auto &s : m_nsegments[i]) {
      const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
      const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
      cv::line(img_di, ps, pt, cv::Scalar(0, 0, 255));
    }

    std::string fname = m_prefix + "_ndi" + std::to_string(i) + ".png";
    if (!cv::imwrite(fname, img_di))
      throw std::runtime_error("Failed to write file " + fname);
  }

  cv::Mat img = cv::Mat(m_rows, m_cols, CV_8UC3, cv::Scalar(255, 255, 255));
  for (const auto &dnv : m_nsegments) {
    for (const auto &s : dnv) {
      const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
      const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
      cv::line(img, ps, pt, cv::Scalar(0, 0, 255));
    }
  }
  cv::imwrite(m_prefix + "_ndi.png", img);
}

void UBlock_building::write_facade_segments() {
  cv::Mat img_out(m_rows, m_cols, CV_8UC3, cv::Scalar(255, 255, 255));
  for (const auto &s : m_fsegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    cv::line(img_out, ps, pt, cv::Scalar(0, 255, 0));
  }
  const std::string fname = m_prefix + "_segf.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_all_segments() {
  // write on white canvas
  cv::Mat img_out(m_rows, m_cols, CV_8UC3, cv::Scalar(255, 255, 255));
  for (const auto &s : m_csegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    cv::line(img_out, ps, pt, cv::Scalar(0, 215, 215));
  }
  for (const auto &s : m_hsegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    cv::line(img_out, ps, pt, cv::Scalar(255, 135, 0));
  }
  for (const auto &dnv : m_nsegments) {
    for (const auto &s : dnv) {
      const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
      const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
      cv::line(img_out, ps, pt, cv::Scalar(0, 0, 255));
    }
  }
  for (const auto &s : m_fsegments) {
    const cv::Point2i ps((int)(wti_x(s[0])), (int)(wti_y(s[1])));
    const cv::Point2i pt((int)(wti_x(s[2])), (int)(wti_y(s[3])));
    cv::line(img_out, ps, pt, cv::Scalar(0, 255, 0));
  }
  const std::string fname = m_prefix + "_seg.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_regularized_segments() {
  cv::Mat img_out(m_rows, m_cols, CV_8UC3, cv::Scalar(255.0, 255.0, 255.0));
  // const cv::Point2i pc(img_out.cols / 2, img_out.rows / 2);
  // for (const auto &fd : m_fdir) {
  //   const cv::Point2i pt((int)(fd.x() * 100) + pc.x, (int)(-fd.y() * 100) + pc.y);
  //   cv::line(img_out, pc, pt, cv::Scalar(255.0, 255.0, 0.0));
  // }

  for (const auto &s : m_segments) {
    const cv::Point2i ps(
      int(wti_x(CGAL::to_double(s.source().x()))),
      int(wti_y(CGAL::to_double(s.source().y()))));
    const cv::Point2i pt(
      int(wti_x(CGAL::to_double(s.target().x()))),
      int(wti_y(CGAL::to_double(s.target().y()))));
    cv::line(img_out, ps, pt, cv::Scalar(0.0, 0.0, 0.0));
  }
  const std::string fname = m_prefix + "_seg_reg.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}
/*
void UBlock_building::write_arrangement() {
  cv::Mat img_out(m_rows, m_cols, CV_8UC3, cv::Scalar(255.0, 255.0, 255.0));
  for (auto eitr = m_arr.edges_begin(); eitr != m_arr.edges_end(); ++eitr) {
    const cv::Point2i ps(
      int(wti_x(CGAL::to_double(eitr->source()->point().x()))),
      int(wti_y(CGAL::to_double(eitr->source()->point().y()))));
    const cv::Point2i pt(
      int(wti_x(CGAL::to_double(eitr->target()->point().x()))),
      int(wti_y(CGAL::to_double(eitr->target()->point().y()))));
    cv::line(img_out, ps, pt, cv::Scalar(0.0, 0.0, 0.0));
  }
  const std::string fname = m_prefix + "_arr.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}
*/
void UBlock_building::write_arrangement() {
  cv::Mat img_out(m_rows, m_cols, CV_8UC3, cv::Scalar(255.0, 255.0, 255.0));
  for (auto eitr = m_arr.edges_begin(); eitr != m_arr.edges_end(); ++eitr) {
    const cv::Point2i ps(
      int(wti_x(CGAL::to_double(eitr->source()->point().x()))),
      int(wti_y(CGAL::to_double(eitr->source()->point().y()))));
    const cv::Point2i pt(
      int(wti_x(CGAL::to_double(eitr->target()->point().x()))),
      int(wti_y(CGAL::to_double(eitr->target()->point().y()))));
    cv::line(img_out, ps, pt, cv::Scalar(0.0, 0.0, 0.0));
  }
  const std::string fname = m_prefix + "_arr.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_arrangement_labeling() {
  cv::Mat img_out = cv::Mat::zeros(m_rows, m_cols, CV_8U);
  for (auto fitr = m_arr.faces_begin(); fitr != m_arr.faces_end(); ++fitr) {
    // do not draw ground polygon, it can have huge holes
    // treat it as huge background
    if (fitr->data() <= 0)
      continue;
    std::vector<cv::Point2i> plg;
    auto curr = fitr->outer_ccb();
    do {
      plg.push_back({
        int(wti_x(CGAL::to_double(curr->target()->point().x()))),
        int(wti_y(CGAL::to_double(curr->target()->point().y())))});
      ++curr;
    } while (curr != fitr->outer_ccb());
    const cv::Point2i *pts[] = {plg.data()};
    const int npts[] = {int(plg.size())};
    cv::fillPoly(img_out, pts, npts, 1, cv::Scalar(255.0 * fitr->data() / m_planes.size()));
  }
  cv::applyColorMap(img_out, img_out, cv::COLORMAP_JET);
  const std::string fname = m_prefix + "_arr_mrf.png";
  if (!cv::imwrite(fname, img_out))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock_building::write_local_ground() {
  // off set down little bit
  const double offset = 0.1;
  std::ofstream ofs(m_prefix + "_lod_ground.off");
  ofs << "OFF\n4 1 0\n";
  cv::Vec3d p = m_pmap.at<cv::Vec3d>(0, 0);
  ofs << p[0] << ' ' << p[1] << ' ' << m_ground_level - offset << '\n';
  p = m_pmap.at<cv::Vec3d>(m_rows - 1, 0);
  ofs << p[0] << ' ' << p[1] << ' ' << m_ground_level - offset << '\n';
  p = m_pmap.at<cv::Vec3d>(m_rows - 1, m_cols - 1);
  ofs << p[0] << ' ' << p[1] << ' ' << m_ground_level - offset << '\n';
  p = m_pmap.at<cv::Vec3d>(0, m_cols - 1);
  ofs << p[0] << ' ' << p[1] << ' ' << m_ground_level - offset << '\n';
  ofs << "4 0 1 2 3 153 102 51\n";
}

void UBlock_building::write_levels() {
  std::ofstream ofs(m_prefix + ".levels");
  ofs << m_ground_level << ' ' << m_roof_level << std::endl;
}

Kernel::Point_2 polygon_centroid(const std::vector<Kernel::Point_2> &plg) {
  // https://stackoverflow.com/questions/2792443/finding-the-centroid-of-a-polygon

  double centroid_x = 0.0;
  double centroid_y = 0.0;
  double signedArea = 0.0;
  for (std::size_t i = 0; i < plg.size(); ++i) {
    const double x0 = plg[i].x();
    const double y0 = plg[i].y();
    const double x1 = plg[(i + 1) % plg.size()].x();
    const double y1 = plg[(i + 1) % plg.size()].y();
    const double a = x0 * y1 - x1 * y0;
    signedArea += a;
    centroid_x += (x0 + x1) * a;
    centroid_y += (y0 + y1) * a;
  }

  signedArea *= 0.5;
  centroid_x /= (6.0 * signedArea);
  centroid_y /= (6.0 * signedArea);

  return Kernel::Point_2(centroid_x, centroid_y);
}
