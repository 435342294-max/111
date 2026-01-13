#ifndef UBLOCK_BUILDING_H
#define UBLOCK_BUILDING_H

#include <vector>

#include <opencv2/core.hpp>

#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_extended_dcel.h>
#include <CGAL/Arr_overlay_2.h>
#include <CGAL/Arr_default_overlay_traits.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Shape_detection_3.h>
#include <CGAL/regularize_planes.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Surface_mesh_approximation/approximate_triangle_mesh.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Weighted_point_3.h>
#include "utils/Kernel_epec.h"
#include "utils/Mesh_3.h"

/*!
 * \brief Urban block building.
 */
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef CGAL::Shape_detection_3::Shape_detection_traits<
  Kernel, Point_set, Point_set::Point_map, Point_set::Vector_map> Traits;
typedef CGAL::Shape_detection_3::Efficient_RANSAC<Traits> Efficient_RANSAC;
class UBlock_building {
  typedef CGAL::Arr_segment_traits_2<Kernel_epec> Arr_traits;
  typedef CGAL::Arr_face_extended_dcel<Arr_traits, int> Dcel_with_index;
  typedef CGAL::Arrangement_2<Arr_traits, Dcel_with_index> Arr_with_index;

  // segment arrangement attached data
  struct Arr_face_data {
    Arr_face_data(const std::size_t &n) :
      distances(n, 0.0),
      area(0.0),
      normal(0.0, 0.0, 0.0) {}
    // averaged face distance to detected planes
    std::vector<double> distances;
    // area
    double area;
    // averaged normal
    cv::Vec3d normal;
    // possible future data
  };

public:
  /*!
   * \brief Constructor
   * \param prefix building data naming convention prefix
   * \param building_map building pixel map
   * \param color_map bgr color image
   * \param position_map pixel position map
   * \param normal_map pixel normal map
   * \param ground_level ground level of the building
   * \param mesh building mesh, deep copy construct
   */
  UBlock_building(
    const std::string &prefix,
    const cv::Mat &building_map,
    const cv::Mat &color_map,
    const cv::Mat &position_map,
    const cv::Mat &normal_map,
    const double &ground_level,
    const double &max_height,
    const Mesh_3 &mesh);

  /*!
   * \brief Line segment arrangement modeling.
   */
  void segment_arrangement_modeling();
  void write_planes_to_ply(
        const Point_set& points, 
        const Efficient_RANSAC::Plane_range& shapes, 
        const std::string& output_file
    );
private:

  /*!
   * \brief Detect roof planes with RANSAC.
   */
  void detect_roof_planes();

  /*!
   * \brief Facade segment detection using VSA on the building mesh.
   */
  void facade_segment_detection();

  /*!
   * \brief Color space segment detection.
   */
  void color_space_segment_detection();

  /*!
   * \brief Height space segment detection.
   */
  void height_space_segment_detection();

  /*!
   * \brief Normal space directional intensity segment detection.
   * Different intensity map are calculated from the normal map along the detected roof normals.
   */
  void normal_space_segment_detection();

  /*!
   * \brief Construct regularized segment arrangement.
   * Overlay grid arrangement onto it to compute data.
   */
  void construct_segment_arrangement();

  /*!
   * \brief Segment arrangement labeling.
   * Label set: detected planes.
   * Graph: segment arrangement faces.
   */
  void segment_arrangement_MRF_labeling();

  /*!
   * \brief Extrude labeled arrangement faces to their respective planes.
   */
  void segment_arrangement_extrusion();

  /*!
  * \brief Find mainly direction.
  */
  std::vector<Kernel::Vector_2> find_mainly_dir(std::vector<Kernel::Segment_2>& segments, double thre);


  /*!
   * \brief Image to world coordinate x.
   */
  double itw_x(const double &x) { return m_xmin + x * m_step; }

  /*!
   * \brief Image to world coordinate y.
   */
  double itw_y(const double &y) { return m_ymax - y * m_step; }

  /*!
   * \brief World to image coordinate x.
   */
  double wti_x(const double &x) { return (x - m_xmin) / m_step; }

  /*!
   * \brief World to image coordinate y.
   */
  double wti_y(const double &y) { return (m_ymax - y) / m_step; }

  /*!
   * \brief Write building mesh to file.
   */
  void save_mesh();

  /*!
   * \brief Write building (roof top) point cloud with normal to file.
   */
  void save_point_cloud();

  /*!
   * \brief Write color segments.
   */
  void write_color_segments();

  /*!
   * \brief Write height segments.
   */
  void write_height_segments();

  /*!
   * \brief Write normal segments.
   */
  void write_normal_segments();

  /*!
   * \brief Write facade segments.
   */
  void write_facade_segments();

  /*!
   * \brief Write all segments.
   */
  void write_all_segments();

  /*!
   * \brief Write regularized segments.
   */
  void write_regularized_segments();

  /*!
   * \brief Write arrangement.
   */
  void write_arrangement();

  /*!
   * \brief Write arrangement MRF labeling.
   */
  void write_arrangement_labeling();

  /*!
   * \brief Write local ground.
   */
  void write_local_ground();

  /*!
   * \brief Write ground and roof levels.
   */
  void write_levels();

private:
  // building data naming convention prefix
  const std::string m_prefix;

  // building pixel map
  const cv::Mat m_bmap;
  // building color image
  const cv::Mat m_cmap;
  // building pixel position map
  const cv::Mat m_pmap;
  // building pixel normal map
  const cv::Mat m_nmap;

  // ground height
  const double m_ground_level;

  // max height
  const double m_max_height;

  // average roof height
  double m_roof_level;

  const int m_rows;
  const int m_cols;

  const double m_step;

  const double m_xmin;
  const double m_ymax;

  // building mesh
  Mesh_3 m_mesh;

  // detected segments
  std::vector<cv::Vec4d> m_csegments;
  std::vector<cv::Vec4d> m_hsegments;
  std::vector<std::vector<cv::Vec4d>> m_nsegments;
  std::vector<cv::Vec4d> m_fsegments;
  std::vector<cv::Vec4d> m_rsegments;

  // facade directions detected by RANSAC
  std::vector<Vector_3> m_fdir;

  // detected roof shapes
  std::vector<Plane_3> m_planes;

  // regularized segments
  std::vector<Kernel_epec::Segment_2> m_segments;

  // segment arrangement
  Arr_with_index m_arr;

  // segment arrangement data
  std::vector<Arr_face_data> m_arr_data;
};

#endif // UBLOCK_BUILDING_H
