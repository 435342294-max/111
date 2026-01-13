#ifndef UBLOCK_H
#define UBLOCK_H

#include <vector>
#include <string>
#include <map>
#include <memory>

#include <boost/optional.hpp>

#include <opencv2/core.hpp>

#include "utils/Kernel_epec.h"
#include "utils/Mesh_3.h"


class UBlock_building;

/*!
 * \brief Urban block.
 */
class UBlock {
public:
  /*!
   * \brief Constructor.
   * \note All data e.g. image / mesh / mask should follow the prefix naming convention.
   * \param prefix working directory prefix
   */
  UBlock(const std::string &prefix) : m_prefix(prefix) {}

  /*!
   * \brief Load block data: ortho photo, segmentation mask and mesh.
   * \note Assuming they share the same prefix: xxx.png, xxx_mask.png, xxx.off.
   */
  void load_data();

  /*!
   * \brief Compute block height and normal map from mesh.
   * \param ulx optional block upper left corner position in world coordinate
   * \param uly optional block upper left corner position in world coordinate
   */
  void compute_height_and_normal_map(
    const boost::optional<double> ulx = boost::none,
    const boost::optional<double> uly = boost::none);

  /*!
   * \brief Retrieve all buildings from the orthophoto.
   * \return Vector of block buildings.
   */
  std::vector<std::unique_ptr<UBlock_building>> retrieve_buildings(const std::string &in_dir, const std::string &out_dir, double bbox_xmin, double bbox_ymin, double step);

  /*!
  * \brief tree boost result and height info labeling. eccv 2018 method
  * Label set: tree, glass, building, ground.
  */
  void boost_tree_MRF_labeling();

private:
  /*!
   * \brief Retrieve building mesh from surrounding mesh.
   * \param bmap building mask
   * \param pmap building position map
   */
  Mesh_3 retrieve_building_mesh(const cv::Mat bmap, const cv::Mat pmap);
Mesh_3 single_building_mesh(
    double minx, double maxx, double miny, double maxy, 
    cv::Mat& dilation_map, short current_building_index);
  /*!
   * \brief Write height map to image.
   */
  void write_height_map();

  /*!
   * \brief Write normal map to image.
   */
  void write_normal_map();

  /*!
  * \brief Write color map to image.
  */
  void write_color_map();

private:
  // data prefix, all data e.g. image / mesh / mask
  // should follow the same prefix naming convention
  const std::string m_prefix;

  // global sampling step size
  double m_step = 0.0;

  // block orthophoto
  cv::Mat m_img;
  int m_rows = -1;
  int m_cols = -1;

  // building binary mask
  cv::Mat m_bmap;

  // block mesh
  Mesh_3 m_mesh;

  // mesh 2d projection to vertex mapping
  std::map<Kernel_epec::Point_2, vertex_descriptor> m_pv_map;

  // block upper left corner position in world coordinate
  double m_ulx = 0.0;
  double m_uly = 0.0;

  // estimated block ground level
  double m_ground_level = 0.0;

  // sampling data
  // block projection valid mask
  cv::Mat m_height_map;
  cv::Mat m_normal_map;
  cv::Mat m_color_map;
};

#endif // UBLOCK_H


/*!
* \brief Write block lod buildings.
*/
void write_block_lod2(int nb_buildings, std::string m_prefix);

void write_block_lod1(int nb_buildings, std::string m_prefix);

void write_block_lod0(int nb_buildings, std::string m_prefix);

/*!
* \brief Write scene lod buildings.
*/
void write_scene_lod2(int nb_rows, int nb_cols, std::string m_prefix);

void write_scene_lod1(int nb_rows, int nb_cols, std::string m_prefix);

void write_scene_lod0(int nb_rows, int nb_cols, std::string m_prefix);

/*!
* \brief Orthogonally sample RGB and mask images.
*/
double orthogonal(const std::string& input_dir, const std::string& output_dir);

/*!
* \brief Generate textures of lod2 model from dense texture mesh.
*/
void generate_texture(const std::string& input_dir, const std::string& output_dir, double length, int n_iter);

void calculate_error(int nb_buildings, const std::string& input_dir, const std::string& output_dir, int cal_thread_num);
