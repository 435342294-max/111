#include <fstream>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "utils/Logger.h"
#include "utils/Config.h"
#include "utils/Mesh_3.h"
#include "utils/Rply_loader.h"
#include "utils/GL_renderer.h"

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
 * \brief Cut the input image into sub-images of the same size, margins are ignored.
 * \param img input image
 * \param num_rows number of cut rows
 * \param num_cols number of cut columns
 * \param out_dir output directory
 * \return sub-image size, rows by columns
 */
std::pair<int, int> cut_image(
  const cv::Mat &img,
  const int num_rows,
  const int num_cols,
  const std::string &out_dir) {
  // cut image, dump the margins
  const int subimg_rows = img.rows / num_rows;
  const int subimg_cols = img.cols / num_cols;
#pragma omp parallel for
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      // upper left pixel
      const int ulr = r * subimg_rows;
      const int ulc = c * subimg_cols;
      // range is half close half open: [s, e)
      cv::Mat sub_img = img(
        cv::Range(ulr, ulr + subimg_rows), cv::Range(ulc, ulc + subimg_cols));

      std::stringstream ss;
      ss << out_dir << std::setfill('0')
        << std::setw(3) << r
        << std::setw(3) << c << ".png";
      cv::imwrite(ss.str(), sub_img);

      LOG_INFO << "Write image " << ss.str();
    }
  }

  return { subimg_rows, subimg_cols };
}

/*!
 * \brief Cut the mesh into num_rows by num_cols pieces,
 * in the same way as the image and ignore the margins.
 * \param ply input ply model
 * \param step orthophoto sample step
 * \param bbox orthophoto projection bounding box
 * \param num_rows orthophoto cut rows
 * \param num_cols orthophoto cut columns
 * \param subimg_rows sub-image rows size
 * \param subimg_cols sub-image columns size
 * \param out_dir output directory
 */
void cut_mesh(
  const cm::Rply_loader &ply,
  const double step,
  const CGAL::Bbox_3 &bbox,
  const int num_rows,
  const int num_cols,
  const int subimg_rows,
  const int subimg_cols,
  const std::string &out_dir) {
  // stat
  const unsigned int ntriangles = ply.num_faces;
  const auto &vertices = ply.vertices;
  const auto &triangles = ply.faces;

  // write file using memory friendly Surface_mesh
  std::vector<Point_3> centroids(ntriangles);
  for (std::size_t i = 0; i < ntriangles; ++i) {
    const unsigned int v0 = triangles[i * 3];
    const unsigned int v1 = triangles[i * 3 + 1];
    const unsigned int v2 = triangles[i * 3 + 2];
    centroids[i] = CGAL::centroid(
      Point_3(vertices[v0 * 3], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]),
      Point_3(vertices[v1 * 3], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]),
      Point_3(vertices[v2 * 3], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]));
  }

  // TODO: unnecessary
  // each sub mesh have some overlap
  const double enlarge_offset = step * 20;
  const double ystep = step * double(subimg_rows);
  const double xstep = step * double(subimg_cols);
#pragma omp parallel for
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      // compute range
      const double xmin = bbox.xmin() + c * xstep - enlarge_offset;
      const double xmax = xmin + xstep + enlarge_offset;
      const double ymax = bbox.ymax() - r * ystep + enlarge_offset;
      const double ymin = ymax - ystep - enlarge_offset;

      Mesh_3 m;
      std::map<std::size_t, vertex_descriptor> vtx_map;
      for (std::size_t i = 0; i < ntriangles; ++i) {
        const unsigned int v0 = triangles[i * 3];
        const unsigned int v1 = triangles[i * 3 + 1];
        const unsigned int v2 = triangles[i * 3 + 2];
        const Point_3 &centroid = centroids[i];
        if (centroid.x() > xmin && centroid.x() < xmax
          && centroid.y() > ymin && centroid.y() < ymax) {
          const Point_3 p0(vertices[v0 * 3], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
          const Point_3 p1(vertices[v1 * 3], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
          const Point_3 p2(vertices[v2 * 3], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);
          if (vtx_map.find(v0) == vtx_map.end())
            vtx_map.insert({v0, m.add_vertex(p0)});
          if (vtx_map.find(v1) == vtx_map.end())
            vtx_map.insert({v1, m.add_vertex(p1)});
          if (vtx_map.find(v2) == vtx_map.end())
            vtx_map.insert({v2, m.add_vertex(p2)});
          m.add_face(vtx_map[v0], vtx_map[v1], vtx_map[v2]);
          
        }
      }

      std::stringstream ss;
      ss << out_dir << std::setfill('0')
        << std::setw(3) << r
        << std::setw(3) << c << ".off";
      std::ofstream ofs(ss.str());
      ofs << m;

      LOG_INFO << "Write mesh " << ss.str();
    }
  }
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
  LOG_INFO << "OK";
  glr.draw_buffers(buffers);
  LOG_INFO << "OK";
  const auto color_frame = glr.grab_color_frame();

  return color_frame;
}

int main(int argc, char *argv[]) {
  // initialize log
  cm::initialize_logger(cm::severity_level::debug);

  // parse command options
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description desc("Allowed options");
  try {
    desc.add_options()
      ("ply_mesh,P", po::value<std::string>()->required(),
        "Input ply mesh file.")
      ("output_directory,O", po::value<std::string>()->required(),
        "Data and configuration file output directory.")
      ("step", po::value<double>()->required(),
        "Sampling step size.")
      ("rows", po::value<int>()->required(),
        "Number of scene cut rows.")
      ("columns", po::value<int>()->required(),
        "Number of scene cut columns.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
    LOG_INFO << desc;
    return 1;
  }

  // options
  const auto ply_mesh = vm["ply_mesh"].as<std::string>();
  const auto out_dir = vm["output_directory"].as<std::string>();
  const auto step = vm["step"].as<double>();
  const int num_rows = vm["rows"].as<int>();
  const int num_cols = vm["columns"].as<int>();

  // load ply data
  cm::Rply_loader ply(ply_mesh); 
  LOG_INFO << "#v " << ply.num_vertices << ", #f " << ply.num_faces;

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

  // cut orthophoto
  int subimg_rows = 0;
  int subimg_cols = 0;
  std::tie(subimg_rows, subimg_cols) = cut_image(orthophoto, num_rows, num_cols, out_dir);
  LOG_INFO << "#subr " << subimg_rows << ", #subc " << subimg_cols;
  // cut mesh
  cut_mesh(ply, step, bbox, num_rows, num_cols, subimg_rows, subimg_cols, out_dir);
  LOG_INFO << "Cut down.";

  // write configuration
  cm::read_config(out_dir + "/config_modeling.xml");
  LOG_INFO << "Configuration loaded.";
  auto &config = cm::get_config();
  config.put("scene.cut.rows", num_rows);
  config.put("scene.cut.cols", num_cols);
  config.put("scene.sub_image.rows", subimg_rows);
  config.put("scene.sub_image.cols", subimg_cols);
  config.put("scene.bbox.xmin", bbox.xmin());
  config.put("scene.bbox.ymin", bbox.ymin());
  config.put("scene.bbox.zmin", bbox.zmin());
  config.put("scene.bbox.xmax", bbox.xmax());
  config.put("scene.bbox.ymax", bbox.ymax());
  config.put("scene.bbox.zmax", bbox.zmax());
  cm::write_config(out_dir + "/config_modeling.xml");
  LOG_INFO << "Configuration saved.";

  return 0;
}
