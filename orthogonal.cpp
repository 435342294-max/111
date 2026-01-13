#include <fstream>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>
#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "utils/Logger.h"
#include "utils/Config.h"
#include "utils/Rply_loader.h"
#include "utils/GL_renderer.h"
#include "segment_modeling/UBlock.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Triangle_3   Triangle_3;
typedef K::Ray_3        Ray_3;
typedef std::vector<Triangle_3>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef boost::optional<Tree::Intersection_and_primitive_id<Ray_3>::Type> Ray_intersection;


/*!
 * \brief Compute bounding box for a ply model.
 * \param ply input ply model
 * \return bouding box of the model
 */
CGAL::Bbox_3 compute_bounding_box(const cm::Rply_loader &ply) {
  // compute bounding box
  double xmax = std::numeric_limits<double>::lowest();
  double ymax = std::numeric_limits<double>::lowest();
  double zmax = std::numeric_limits<double>::lowest();
  double xmin = std::numeric_limits<double>::max();
  double ymin = std::numeric_limits<double>::max();
  double zmin = std::numeric_limits<double>::max();
  const auto &vertices = ply.vertices;
  LOG_INFO << vertices.size();
  for (std::size_t i = 0; i < vertices.size(); i += 3) {
    const double vx = vertices[i];
    const double vy = vertices[i + 1];
    const double vz = vertices[i + 2];
    xmin = vx < xmin ? vx : xmin;
    xmax = vx > xmax ? vx : xmax;
    ymin = vy < ymin ? vy : ymin;
    ymax = vy > ymax ? vy : ymax;
    zmin = vz < zmin ? vz : zmin;
    zmax = vz > zmax ? vz : zmax;
  }

  return { xmin, ymin, zmin, xmax, ymax, zmax };
}


/*!
* \brief Generate manifold mesh with .off format.
* \param ply input ply model
* \param out_dir output directory
*/
void generate_manifold_mesh(
  const cm::Rply_loader &ply,
  const std::string &out_dir) {
  // stat
  const unsigned int ntriangles = ply.num_faces;
  const auto &vertices = ply.vertices;
  const auto &triangles = ply.faces;

  // write file using memory friendly Surface_mesh
  Mesh_3 m;
  std::map<std::size_t, vertex_descriptor> vtx_map;
  for (std::size_t i = 0; i < ntriangles; ++i) {
     const unsigned int v0 = triangles[i * 3];
     const unsigned int v1 = triangles[i * 3 + 1];
     const unsigned int v2 = triangles[i * 3 + 2];
     if (vtx_map.find(v0) == vtx_map.end())
         vtx_map.insert({ v0, m.add_vertex(Point_3(vertices[v0 * 3], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2])) });
     if (vtx_map.find(v1) == vtx_map.end())
         vtx_map.insert({ v1, m.add_vertex(Point_3(vertices[v1 * 3], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2])) });
     if (vtx_map.find(v2) == vtx_map.end())
         vtx_map.insert({ v2, m.add_vertex(Point_3(vertices[v2 * 3], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2])) });
     m.add_face(vtx_map[v0], vtx_map[v1], vtx_map[v2]);
      }

      std::stringstream ss;
      ss << out_dir << cm::get_config().get<std::string>("prefix") << ".off";
      std::ofstream ofs(ss.str());
      ofs << m;

      LOG_INFO << "Write mesh " << ss.str();

}

/*!
* \brief Generate 2D mask image.
* \param ply input model with 'building' semantic
* \param bbox boundary boxes of the whole scene
* \param width sample width
* \param height sample height
* \param step sample size
* \param out_dir output directory
*/
void generate_mask(const cm::Rply_loader &ply,
  CGAL::Bbox_3& bbox,
  const unsigned int width,
  const unsigned int height,
  const double step,
  const std::string &out_dir){
  const unsigned int ntriangles = ply.num_faces;
  const auto &vertices = ply.vertices;
  const auto &triangles = ply.faces;

  // construct AABB Tree
  std::vector<Triangle_3> cgalfaces;
  std::size_t num = 0;
  LOG_INFO << ntriangles;
  for (std::size_t i = 0; i < ntriangles; i++) {
    const unsigned int v0 = triangles[i * 3];
    const unsigned int v1 = triangles[i * 3 + 1];
    const unsigned int v2 = triangles[i * 3 + 2];
    K::Point_3 p1 = K::Point_3(vertices[v0 * 3], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
    K::Point_3 p2 = K::Point_3(vertices[v1 * 3], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
    K::Point_3 p3 = K::Point_3(vertices[v2 * 3], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);
    Triangle_3 face = Triangle_3(p1, p2, p3);
    if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
      num++;
      continue;
    }
    cgalfaces.push_back(face);
  }
  Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
  LOG_INFO << num << " degenerated faces.";

  int xmin = bbox.xmin();
  int ymin = bbox.ymin();
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        double x = bbox.xmin() + step * (j + 0.5);
        double y = bbox.ymax() - step * (i + 0.5);
        Ray_3 ray(K::Point_3(x, y, bbox.zmax() + 10), K::Point_3(x, y, bbox.zmax()));
        Ray_intersection intersection = AABBTree.first_intersection(ray);
        if (intersection && boost::get<K::Point_3>(&(intersection->first))) {
          mask.at<uchar>(i, j) = 255;
          int id = std::distance(cgalfaces.begin(), intersection->second);
        }
      }
    }
  const std::string mask_name = out_dir + cm::get_config().get<std::string>("prefix") + "_mask.png";
  cv::imwrite(mask_name, mask);
  return;
}


/*!
 * \brief Sample a mesh into orthophoto.
 * \note Assimp can not loads ply with multiple textures properly 
 * \param ply input ply model
 * \param dir mesh directory, to retrieve texture file properly
 * \param bbox sampling bounding region
 * \param width orthophoto width
 * \param height orthophoto height
 */
cv::Mat orthogonal_sample(
  const cm::Rply_loader &ply,
  const std::string &dir,
  const CGAL::Bbox_3 &bbox,
  const unsigned int width,
  const unsigned int height) {
  // split mesh according to different texture
  const unsigned int num_meshes =
    ply.textures.empty() ? 1 : (unsigned int)ply.textures.size();

  // number of faces
  typedef cm::GL_renderer::Buffer Buffer;
  std::vector<Buffer> buffers(num_meshes);
  if (num_meshes == 1)
    buffers[0].num_faces = ply.num_faces;
  else {
    for (auto &b : buffers)
      b.num_faces = 0;
    for (unsigned int i = 0; i < ply.num_faces; ++i)
      buffers[ply.texnumber[i]].num_faces++;
  }

  // geometry
  for (auto &b : buffers)
    b.vertices.reserve(b.num_faces * 3);
  for (unsigned int i = 0; i < ply.num_faces; ++i) {
    const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
    for (unsigned int j = 0; j < 3; ++j) {
      const unsigned int fvidx = i * 3 + j;
      const unsigned int vidx = ply.faces[fvidx];
      buffers[midx].vertices.push_back({
        ply.vertices[vidx * 3],
        ply.vertices[vidx * 3 + 1],
        ply.vertices[vidx * 3 + 2]
      });
    }
  }

  // dummy normal
  for (auto &b : buffers)
    b.normals = std::vector<glm::vec3>(b.num_faces * 3);

  // texture
  if (!ply.textures.empty()) {
    for (unsigned int i = 0; i < num_meshes; ++i) {
      Buffer &b = buffers[i];
      b.texcoords.reserve(b.num_faces * 3);
      b.texture_file = dir + ply.textures[i];
    }
    for (unsigned int i = 0; i < ply.num_faces; ++i) {
      const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
      for (unsigned int j = 0; j < 3; ++j) {
        const unsigned int fvidx = i * 3 + j;
        buffers[midx].texcoords.push_back({
          ply.texcoords[fvidx * 2],
          ply.texcoords[fvidx * 2 + 1]
        });
      }
    }
  }

  // set resolution
  cm::GL_renderer glr;
  glr.set_resolution(width, height);
  // set view and projection matrix
  const float offset = (bbox.zmax() - bbox.zmin()) * 0.01f;
  const glm::vec3 center(
    (bbox.xmin() + bbox.xmax()) / 2.0f,
    (bbox.ymin() + bbox.ymax()) / 2.0f,
    bbox.zmax());
  const glm::mat4 view = glm::lookAt(
    center + glm::vec3(0.0f, 0.0f, offset),
    center,
    glm::vec3(0.0f, 1.0f, 0.0f));
  const float znear = offset;
  const float zfar = bbox.zmax() - bbox.zmin() + offset;
  const glm::mat4 projection = glm::ortho(
    -float(bbox.xmax() - bbox.xmin()) / 2.0f, float(bbox.xmax() - bbox.xmin()) / 2.0f,
    -float(bbox.ymax() - bbox.ymin()) / 2.0f, float(bbox.ymax() - bbox.ymin()) / 2.0f,
    znear, zfar);
  glr.set_view_projection(view, projection);
  glr.initialize(buffers);
  glr.draw_buffers(buffers);
  const auto color_frame = glr.grab_color_frame();

  return color_frame;
}

/*!
* \brief Generate RGB orthophoto, manifold mesh with .off format and 2D building mask.
* \note Surface_mesh need manifold input
* \param wdir base path
*/
double orthogonal(const std::string& input_dir, const std::string& output_dir) {
  // options
  const auto ply_mesh = input_dir + cm::get_config().get<std::string>("prefix") + ".ply";
  const auto step = cm::get_config().get<double>("scene.step");
  // load ply data
  cm::Rply_loader ply(ply_mesh);
  LOG_INFO << "scene #v " << ply.num_vertices << ", #f " << ply.num_faces;

  // compute bounding box
  auto bbox = compute_bounding_box(ply);
  LOG_INFO << "#bbox " << bbox;
  // resize bounding box to avoid margin
  const unsigned int width = (unsigned int)((bbox.xmax() - bbox.xmin()) / step);
  const unsigned int height = (unsigned int)((bbox.ymax() - bbox.ymin()) / step);
  bbox = CGAL::Bbox_3(
    bbox.xmin(), bbox.ymin(), bbox.zmin(),
    bbox.xmin() + width * step, bbox.ymin() + height * step, bbox.zmax());
  LOG_INFO << "#bbox " << bbox;
  
  // orthogonal sample
  const std::string dir = ply_mesh.substr(0, ply_mesh.find_last_of("\\/") + 1);
  const auto orthophoto = orthogonal_sample(ply, dir, bbox, width, height);
  LOG_INFO << "#r " << orthophoto.rows << ", #c " << orthophoto.cols;

  std::stringstream ss;
  ss << output_dir << cm::get_config().get<std::string>("prefix") << ".png";
  cv::imwrite(ss.str(), orthophoto);
  LOG_INFO << "Write image " << ss.str();
  
  // generate manifold mesh with .off format
  generate_manifold_mesh(ply, output_dir);

  // generate mask
  const auto ply_building_mesh = input_dir + cm::get_config().get<std::string>("prefix") + ".ply";
  cm::Rply_loader ply_building(ply_building_mesh);
  LOG_INFO << "building #v " << ply_building.num_vertices << ", #f " << ply_building.num_faces;
  generate_mask(ply_building, bbox, width, height, step, output_dir);
  
  return bbox.xmin(), bbox.ymin(), step;
}
