#include <fstream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <CGAL/Polygon_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_batched_point_location.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include "utils/Logger.h"
#include "utils/Config.h"
#include "utils/GL_renderer.h"

#include "UBlock.h"
#include "UBlock_building.h"

#include "gco-v3.0/GCoptimization.h"

void UBlock::load_data() {
  // global step size
  m_step = cm::get_config().get<double>("scene.step");

  // load ortho photo
  const std::string file_ortho = m_prefix + ".png";
  m_img = cv::imread(file_ortho, cv::ImreadModes::IMREAD_COLOR);
  if (m_img.empty())
    throw std::runtime_error("Failed to read " + file_ortho);
  m_rows = m_img.rows;
  m_cols = m_img.cols;


  // load segmentation mask
  const std::string file_mask = m_prefix + "_mask.png"; 
  m_bmap = cv::imread(file_mask, cv::ImreadModes::IMREAD_GRAYSCALE);
  //m_bmap = cv::Mat::ones(m_rows, m_cols, cv::ImreadModes::IMREAD_GRAYSCALE);
  if (m_bmap.empty())
    throw std::runtime_error("Failed to read " + file_mask);
  assert(m_rows == m_bmap.rows);
  assert(m_cols == m_bmap.cols);
  // to binary mask
  for (int i = 0; i < m_rows; ++i)
    for (int j = 0; j < m_cols; ++j)
      if (m_bmap.at<unsigned char>(i, j) > 0) {
        m_bmap.at<unsigned char>(i, j) = 1;
      }

  // load mesh
  const std::string file_mesh = m_prefix + ".off";
  std::ifstream ifs2(file_mesh);
  if (!ifs2.is_open())
    throw std::runtime_error("Failed to read " + file_mesh);
  ifs2 >> m_mesh; 
  LOG_INFO << "mesh #vertices: " << m_mesh.vertices().size() << ", #faces: " << m_mesh.faces().size() << std::endl;
  ifs2.close();
  
}

void UBlock::compute_height_and_normal_map(
  const boost::optional<double> ulx,
  const boost::optional<double> uly) {
  // bbox
  const auto vpmap = m_mesh.points();
  CGAL::Bbox_3 bbox;
  for (vertex_descriptor v : m_mesh.vertices())
    bbox += vpmap[v].bbox();
  m_ulx = bbox.xmin();
  m_uly = bbox.ymax();
  if (ulx && uly) {
    m_ulx = *ulx;
    m_uly = *uly;
  }
  std::cout << m_ulx << " " << m_uly << std::endl;

  cm::GL_renderer glr;
  glr.orthogonal_sample(m_mesh,
    m_cols,
    m_rows,
    { m_ulx, m_uly - m_rows * m_step, bbox.zmin() },
    { m_ulx + m_cols * m_step, m_uly, bbox.zmax() });
  // height map, CV_32F to CV_64F, [0, 1] to real height value
  m_height_map = glr.grab_depth_frame().clone();
  m_height_map.convertTo(m_height_map, CV_64F);
  m_height_map = (1.0 - m_height_map) * (bbox.zmax() - bbox.zmin()) + bbox.zmin();
  // normal map, CV_32F to CV_64F
  m_normal_map = glr.grab_normal_frame().clone();
  m_normal_map.convertTo(m_normal_map, CV_64F);

  assert(m_rows == m_height_map.rows);
  assert(m_cols == m_height_map.cols);
  LOG_INFO << "GL_renderer done.";


  const auto &config = cm::get_config();
  // DTM is not available, we assume flat horizontal ground
  // use the block minimum as initial guess of ground if not configured
  if (const auto g = config.get_optional<double>("block.ground_level"))
    m_ground_level = *g;
  else
    cv::minMaxLoc(m_height_map, &m_ground_level);

  if (config.get("block.write.height_map", false))
    write_height_map();
  if (config.get("block.write.normal_map", false))
    write_normal_map();
  if (config.get("block.write.color_map", false))
    write_color_map();

  if (config.get("block.write.normalized_height_map", false)) {
    for (int i = 0; i < m_rows; ++i) {
      for (int j = 0; j < m_cols; ++j) {
        double h = std::max(m_height_map.at<double>(i, j) - m_ground_level, 0.0);
        m_height_map.at<double>(i, j) = 1.0 / (1.0 + std::exp(-2.0 * (h - 3.0)));
      }
    }
    // [0, 1] to [0, 255]
    cv::Mat temp;
    m_height_map.convertTo(temp, CV_8U, 255.0f);
    cv::imwrite(m_prefix + "_phn.png", temp);
  }
}

std::vector<std::unique_ptr<UBlock_building>> UBlock::retrieve_buildings(const std::string &in_dir, const std::string &out_dir, double bbox_xmin, double bbox_ymin, double step) {
  const int morph_size = 10; // 2 meters
  cv::Mat element = cv::getStructuringElement(
    cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1));

  // binary building image to building index image
  cv::Mat bidx_map = cv::Mat::zeros(m_rows, m_cols, CV_32S);
  const int nb_cc = cv::connectedComponents(m_bmap, bidx_map, 4, CV_32S, cv::CCL_DEFAULT) - 1;
  
  // collect building pixels
  std::vector<std::vector<cv::Vec2i>> building_pixels(nb_cc);
  for (int i = 0; i < m_rows; ++i)
    for (int j = 0; j < m_cols; ++j)
      if (bidx_map.at<int>(i, j) > 0)
        building_pixels[bidx_map.at<int>(i, j) - 1].push_back({i, j});
 
  // construct position map
  double building_max_height = -1e9;
  double building_min_height = 1e9;
  cv::Mat pos_map(m_rows, m_cols, CV_64FC3);
  const double ulx = m_ulx - m_step / 2.0;
  const double uly = m_uly + m_step / 2.0;
  for (int i = 0; i < m_rows; ++i)
    for (int j = 0; j < m_cols; ++j) {
      pos_map.at<cv::Vec3d>(i, j) =
        cv::Vec3d(ulx + j * m_step, uly - i * m_step, m_height_map.at<double>(i, j));
      building_max_height = std::max(building_max_height, m_height_map.at<double>(i, j));
      building_min_height = std::min(building_min_height, m_height_map.at<double>(i, j));
    }
 
  assert(nb_cc < int(std::numeric_limits<short>::max()));

  std::vector<std::unique_ptr<UBlock_building>> buildings;
  cv::Mat dilation_map(m_rows, m_cols, CV_16S, cv::Scalar(-1));
  const std::size_t min_pixel = std::size_t(cm::get_config().get<double>(
    "building.min_area") / (m_step * m_step));
    
  for (const auto &pixels : building_pixels) {
    if (pixels.size() < min_pixel)
      continue;

    const short current_building_index = short(buildings.size());
    for (const cv::Vec2i &p : pixels)
      dilation_map.at<short>(p) = current_building_index;

    // Close operation to fill gaps
    cv::morphologyEx(dilation_map, dilation_map, cv::MORPH_CLOSE, element);

    // Find bounding box of the building
    int min_i = std::numeric_limits<int>::max(), max_i = 0;
    int min_j = std::numeric_limits<int>::max(), max_j = 0;
    for (int i = 0; i < m_rows; ++i) {
      for (int j = 0; j < m_cols; ++j) {
        if (dilation_map.at<short>(i, j) == current_building_index) {
          min_j = std::min(min_j, j);
          max_j = std::max(max_j, j);
          min_i = std::min(min_i, i);
          max_i = std::max(max_i, i);
        }
      }
    }
    
    // If no valid pixels found, skip this building
    if (min_j > max_j || min_i > max_i) {
        LOG_WARNING << "No valid pixels for building " << current_building_index << ", skipping.";
        continue;
    }
    
    const int brows = max_i - min_i + 1;
    const int bcols = max_j - min_j + 1;
    const cv::Rect rect(min_j, min_i, bcols, brows);
    
    // Extract building region
    cv::Mat bmap = cv::Mat::zeros(brows, bcols, CV_8U);
    for (int i = 0; i < brows; ++i)
      for (int j = 0; j < bcols; ++j)
        if (dilation_map.at<short>(i + rect.y, j + rect.x) == current_building_index)
          bmap.at<unsigned char>(i, j) = 1;

    cv::Mat cmap = cv::Mat(m_img, rect).clone();
    cv::Mat pmap = cv::Mat(pos_map, rect).clone();
    cv::Mat nmap = cv::Mat(m_normal_map, rect).clone();
    
    // 关键修改：使用正确的坐标计算建筑物边界
    // 使用位置图原点(ulx, uly)而不是整个网格的包围盒
    const double minx = ulx + min_j * m_step;
    const double maxx = ulx + (max_j + 1) * m_step;
    // 注意Y轴方向：图像坐标向下，地理坐标向上
    const double miny = uly - (max_i + 1) * m_step;
    const double maxy = uly - min_i * m_step;
    
    // 关键修改：传递正确的位置信息
    Mesh_3 bmesh = single_building_mesh(minx, maxx, miny, maxy, 
                                        dilation_map, current_building_index);
    
    // 检查生成的网格是否为空
    if (bmesh.number_of_vertices() == 0 || bmesh.number_of_faces() == 0) {
        LOG_WARNING << "Empty mesh generated for building " << current_building_index << ", skipping.";
        continue;
    }
    
    // 保存网格
    std::stringstream ss;
    ss << out_dir << cm::get_config().get<std::string>("prefix") 
       << "_b" << std::to_string(current_building_index) << ".off";
    std::ofstream ofs(ss.str());
    ofs << bmesh;
    
    // 计算建筑物地面高度
    double bmesh_ground = std::numeric_limits<double>::max();
    for (Mesh_3::Vertex_index vi : bmesh.vertices()) {
      if (bmesh.point(vi).z() < bmesh_ground) {
        bmesh_ground = bmesh.point(vi).z();
      }
    }
    
    // 填充非建筑物区域
    for (int i = 0; i < brows; ++i) {
      for (int j = 0; j < bcols; ++j) {
        if (bmap.at<unsigned char>(i, j) == 0) {
          cmap.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
          pmap.at<cv::Vec3d>(i, j)[2] = bmesh_ground;
          nmap.at<cv::Vec3d>(i, j) = cv::Vec3d(0.0, 0.0, 1.0);
        }
      }
    }
    
    // 创建建筑物对象
    buildings.push_back(
      std::make_unique<UBlock_building>(
        m_prefix + "_b" + std::to_string(current_building_index),
        bmap,
        cmap,
        pmap,
        nmap,
        bmesh_ground,
        (building_max_height - building_min_height),
        bmesh));
  }
  
  LOG_INFO << "#buildings " << buildings.size();
  return buildings;
}


Mesh_3 UBlock::single_building_mesh(
    double minx, double maxx, double miny, double maxy, 
    cv::Mat& dilation_map, short current_building_index) 
{
    typedef CGAL::Simple_cartesian<double> K;
    Mesh_3 m;
    std::map<std::size_t, vertex_descriptor> vtx_map;

    // 遍历整个网格的所有面片
    for (auto face : m_mesh.faces()) {
        bool in_building = false;
        std::vector<vertex_descriptor> face_vertices;

        // 收集面片的所有顶点
        for (auto vd : vertices_around_face(m_mesh.halfedge(face), m_mesh)) {
            Point_3 p = m_mesh.point(vd);
            
            // 计算顶点对应的像素坐标
            // 使用位置图原点(ulx, uly)和步长(m_step)
            int j = static_cast<int>((p.x() - m_ulx) / m_step + 0.5);
            int i = static_cast<int>((m_uly - p.y()) / m_step + 0.5);
            
            // 检查坐标是否在图像范围内
            if (i >= 0 && i < m_rows && j >= 0 && j < m_cols) {
                // 检查像素是否属于当前建筑
                if (dilation_map.at<short>(i, j) == current_building_index) {
                    in_building = true;
                }
            }
            face_vertices.push_back(vd);
        }

        // 如果面片属于当前建筑，则添加到新网格
        if (in_building) {
            std::vector<vertex_descriptor> new_vertices;
            for (auto vd : face_vertices) {
                if (vtx_map.find(vd.idx()) == vtx_map.end()) {
                    Point_3 p = m_mesh.point(vd);
                    vertex_descriptor new_vd = m.add_vertex(p);
                    vtx_map[vd.idx()] = new_vd;
                    new_vertices.push_back(new_vd);
                } else {
                    new_vertices.push_back(vtx_map[vd.idx()]);
                }
            }
            
            // 添加面片 (假设是三角形网格)
            if (new_vertices.size() == 3) {
                m.add_face(new_vertices[0], new_vertices[1], new_vertices[2]);
            }
            // 处理多边形面片 (如果需要)
            else if (new_vertices.size() > 3) {
                // 添加多边形面片处理逻辑
                // 这里简化为跳过非三角形面片
            }
        }
    }

    return m;
}

Mesh_3 UBlock::retrieve_building_mesh(const cv::Mat bmap, const cv::Mat pmap) {
  // offset / dilation of 10 pixel
  const int offset = 10;
  // make border
  cv::Mat tmp(bmap.rows + offset * 2, bmap.cols + offset * 2, bmap.depth());
  cv::copyMakeBorder(bmap, tmp,
    offset, offset, offset, offset,
    cv::BORDER_CONSTANT, cv::Scalar(0));

  // dilate
  const cv::Mat kernel = cv::getStructuringElement(
    cv::MORPH_RECT, cv::Size(2 * offset + 1, 2 * offset + 1));
  cv::dilate(tmp, tmp, kernel);

  // retrieves only the extreme outer contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(tmp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  assert(!contours.empty());

  // image contour to polygon
  const double ulx = pmap.at<cv::Vec3d>(0, 0)[0] - offset * m_step;
  const double uly = pmap.at<cv::Vec3d>(0, 0)[1] + offset * m_step;
  std::vector<Kernel_epec::Point_2> points;
  for (const auto &p : contours.back())
    points.push_back({ulx + p.x * m_step, uly - p.y * m_step});
  CGAL::Polygon_2<Kernel_epec> polygon(points.begin(), points.end());

  // simplify contours
  namespace PS = CGAL::Polyline_simplification_2;
  typedef PS::Stop_above_cost_threshold Stop;
  typedef PS::Squared_distance_cost Cost;
  polygon = PS::simplify(polygon, Cost(), Stop(0.5));

  // contour arrangement
  std::list<CGAL::Segment_2<Kernel_epec>> segments;
  for (auto eitr = polygon.edges_begin(); eitr != polygon.edges_end(); ++eitr)
    segments.push_back(*eitr);
  typedef CGAL::Arr_segment_traits_2<Kernel_epec> Arr_traits_epic;
  typedef CGAL::Arrangement_2<Arr_traits_epic> Arr_epic;
  Arr_epic arr;
  CGAL::insert(arr, segments.begin(), segments.end());

  // point location query to build building mesh
  typedef Arr_traits_epic::Point_2 Point_2;
  typedef CGAL::Arr_point_location_result<Arr_epic>  Point_location_result;
  typedef std::pair<Point_2, Point_location_result::Type> Query_result;

  // point to vertex map
  typedef std::map<Point_2, vertex_descriptor> Point_vertex_map;
  struct Transformer {
    const Point_2 &operator()(const Point_vertex_map::value_type &p) const {
      return p.first;
    }
  };
  // location query
  std::list<Query_result> results;
  CGAL::locate(arr,
    boost::make_transform_iterator(m_pv_map.cbegin(), Transformer()),
    boost::make_transform_iterator(m_pv_map.cend(), Transformer()),
    std::back_inserter(results));

  // vertex to arrangement side map
  // attach building side map to retrieve building mesh
  Mesh_3::Property_map<vertex_descriptor, bool> vside_map =
    m_mesh.add_property_map<vertex_descriptor, bool>("v:side_map", false).first;
  for (const vertex_descriptor v : m_mesh.vertices())
    vside_map[v] = false;

  typedef Arr_epic::Face_const_handle Face_const_handle;
  for (const auto &r : results)
    if (const Face_const_handle *f = boost::get<Face_const_handle>(&(r.second)))
      if (!(*f)->is_unbounded())
        vside_map[m_pv_map[r.first]] = true;

  // collect building mesh
  Mesh_3 m;
  std::map<vertex_descriptor, vertex_descriptor> vtx_map;
  for (const face_descriptor f : m_mesh.faces()) {
    const halfedge_descriptor he = m_mesh.halfedge(f);
    const vertex_descriptor v0 = m_mesh.source(he);
    const vertex_descriptor v1 = m_mesh.target(he);
    const vertex_descriptor v2 = m_mesh.target(m_mesh.next(he));
    if (vside_map[v0] || vside_map[v1] || vside_map[v2]) {
      if (vtx_map.find(v0) == vtx_map.end())
        vtx_map[v0] = m.add_vertex(m_mesh.point(v0));
      if (vtx_map.find(v1) == vtx_map.end())
        vtx_map[v1] = m.add_vertex(m_mesh.point(v1));
      if (vtx_map.find(v2) == vtx_map.end())
        vtx_map[v2] = m.add_vertex(m_mesh.point(v2));
      m.add_face(vtx_map[v0], vtx_map[v1], vtx_map[v2]);
    }
  }

  // remove isolated components
  namespace PMP = CGAL::Polygon_mesh_processing;
  PMP::keep_largest_connected_components(m, 1, PMP::parameters::all_default());
  // TODO: investigate VSA crash if without this intentional
  // garbate collection after largest connected components
  if (m.has_garbage())
    m.collect_garbage();

  return m;
}



void UBlock::boost_tree_MRF_labeling() {

  const auto &config = cm::get_config();

  double hground = std::numeric_limits<double>::max();
  for (int i = 0; i < m_rows; ++i) {
    for (int j = 0; j < m_cols; ++j) {
      const double h = m_height_map.at<double>(i, j);
      hground = h < hground ? h : hground;
    }
  } // intial ground height

  hground = 1076;

  // load tree probability map
  const std::string file_pro = m_prefix + "_tbc.txt";
  std::ifstream fi(file_pro);
  std::vector<double> imgp(m_rows * m_cols, 0.0);


  // set up graph
  const int nb_sites = int(m_rows * m_cols);
  const int nb_labels = 4;

  struct pixels {
    double p;
    double h;
    int label = -1;
  };
  std::vector<pixels> pixel(nb_sites);

  // read
  const std::string filep = m_prefix + "_tbc.png";
  cv::Mat pp = cv::imread(filep, cv::ImreadModes::IMREAD_GRAYSCALE);
  
  int iter = 0;
  std::vector<double> imgh(m_rows * m_cols, 0.0);
  while (iter < 1) {
    std::cout << "ground height: " << hground << std::endl;
    iter++;
    for (int x = 0; x < m_cols; ++x)
      for (int y = 0; y < m_rows; ++y) {
        double hs = (m_height_map.at<double>(y, x) - hground);
        double p = std::max(hs, 0.0);
        imgh[x + y * m_cols] = 1 / (1 + exp(-2 * (p - 3)));
        if (iter == 1) {
          imgp[x + y * m_cols] = pp.at<unsigned char>(y, x) / 255.0f;
        }
          //fi >> imgp[x + y * m_cols];
      }


    // setup data term
    double hmax = -1e9;
    double hmin = 1e9;
    std::vector<double> data(nb_sites * nb_labels, 0.0);
    for (int i = 0; i < nb_sites; ++i) {
      double hi = imgh[i];
      double pii = imgp[i];
      data[i * nb_labels + 0] = (hi+0.03) * (pii+0.01); // ground
      data[i * nb_labels + 1] = (hi+0.01) * (1.0 - pii); // grass
      data[i * nb_labels + 2] = (1.01 - hi) * (1.01 - pii); // tree
      data[i * nb_labels + 3] = (1.01 - hi) *  (pii+0.001); // building
      pixel[i].h = hi;
      pixel[i].p = pii;
      if (hi > hmax) hmax = hi;
      if (hi < hmin) hmin = hi;
    }
    std::cout << "normalized height difference: " << hmax-hmin << std::endl;

    class SmoothFn : public GCoptimization::SmoothCostFunctor {
    public:
      SmoothFn(const std::vector<pixels> &pixel_, const double &hdir) : pixel(pixel_), height_dir(hdir){}

      double compute(int p1, int p2, int l1, int l2) {
        if (l1 == l2)
          return 0.0;
        return std::abs(0.05 - std::abs(pixel[p1].h - pixel[p2].h));
      }
    private:
      const std::vector<pixels> &pixel;
      const double height_dir;
    };

    // two faces may be separated by collinear segments, we need mapping
    typedef std::pair<int, int> Adj_facets;
    typedef std::map<Adj_facets, double> Adj_map;

    // set up neighboring
    Adj_map adj_map;
    for (int i = 0; i < m_cols; i++) {
      for (int j = 0; j < m_rows; j++) {
        if (i < m_cols - 1) {
          int si = i + j*m_cols;
          int sj = i + j*m_cols + 1;
          adj_map.insert({ { si, sj }, 1 });
        }
        if (j < m_rows - 1) {
          int si = i + j*m_cols;
          int sj = i + (j + 1)*m_cols;
          adj_map.insert({ { si, sj }, 1 });
        }
      }
    }

    // MRF labeling
    try {
      GCoptimizationGeneralGraph gc(nb_sites, nb_labels);
      gc.setDataCost(data.data());
      SmoothFn smoothfn(pixel, hmax - hmin);
      gc.setSmoothCostFunctor(&smoothfn);

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

      double hg = 0;
      int hg_nb = 0;
      for (int i = 0; i < nb_sites; i++) {
        pixel[i].label = gc.whatLabel(i);
        if (pixel[i].label == 0) { // ground
          hg += m_height_map.at<double>(i % m_rows, i / m_rows);
          hg_nb++;
        }
        
      }
      hground = hg / hg_nb;
    }
    catch (GCException e) {
      e.Report();
    }


    // show result
    cv::Mat img_out = cv::Mat::zeros(m_rows, m_cols, CV_8UC3);
    for (int i = 0; i < m_cols; i++)
      for (int j = 0; j < m_rows; j++) {
        if (pixel[i + m_cols*j].label == 0) // ground
          img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 64, 128);
        if (pixel[i + m_cols*j].label == 1) // grass
          img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(170, 205, 102);//cv::Vec3b(170, 205, 102);
        if (pixel[i + m_cols*j].label == 2) // tree
          img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(170, 205, 102);
        if (pixel[i + m_cols*j].label == 3) // building
          img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(225, 105, 65);
      }

    const std::string fname = m_prefix + std::to_string(iter) + "_mrf_tree.png";
    if (!cv::imwrite(fname, img_out))
      throw std::runtime_error("Failed to write file " + fname);
  }

}

void UBlock::write_height_map() {
  // save to height map, scale to [0, 255]
  double zmin = 0.0, zmax = 0.0;
  cv::minMaxLoc(m_height_map, &zmin, &zmax);
  const double zrange = zmax - zmin;
  cv::Mat img;
  m_height_map.convertTo(img, CV_8U, 255.0 / zrange, -zmin * 255.0 / zrange);

  const std::string fname = m_prefix + "_ph.png";
  if (!cv::imwrite(fname, img))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock::write_normal_map() {
  // save normal map, scaled [-1.0, 1.0] to [0, 255]
  // opencv coloring (x, y, z) => (b, g, r)
  cv::Mat temp;
  m_normal_map.convertTo(temp, CV_32F);
  cv::cvtColor(temp, temp, cv::COLOR_RGB2BGR);
  cv::Mat img;
  temp.convertTo(img, CV_8U, 255.0 / 2.0, 255.0 / 2.0);

  const std::string fname = m_prefix + "_pn.png";
  if (!cv::imwrite(fname, img))
    throw std::runtime_error("Failed to write file " + fname);
}

void UBlock::write_color_map() {
  // save color map, scaled [-1.0, 1.0] to [0, 255]
  // opencv coloring (x, y, z) => (b, g, r)
  cv::Mat img;
  m_color_map.convertTo(img, CV_8UC3);
  const std::string fname = m_prefix + "_pc.png";
  if (!cv::imwrite(fname, img))
    throw std::runtime_error("Failed to write file " + fname);
}

void write_block_lod2(int nb_buildings, std::string m_prefix) {

  // lod2
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int i = 0; i < nb_buildings; i++) {
    std::ifstream ifs(m_prefix + "_b" + std::to_string(i) + "_lod2.off");
    if (!ifs)
      continue;
    std::string f;
    ifs >> f;
    int np, nf;
    ifs >> np >> nf;
    double x, y, z;
    ifs >> x;
    for (int i = 0; i < np; i++) {
      ifs >> x >> y >> z;
      points.push_back(Point_3(x, y, z));
    }
    for (int i = 0; i < nf; i++) {
      int nfp;
      ifs >> nfp;
      std::vector<std::size_t> poly;
      for (int j = 0; j < nfp + 3; j++) {
        int nfp_;
        ifs >> nfp_;
        if (j < nfp)
          poly.push_back(nfp_ + nump);
        else
          poly.push_back(nfp_);
      }
      polygons.push_back(poly);
    }
    nump += np;
  }
  std::ofstream ofs(m_prefix + "_lod2.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    //if (p.z() < 88)
      //ofs << p.x() << ' ' << p.y() << ' ' << 88 << '\n';
    //else
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}


void write_block_lod1(int nb_buildings, std::string m_prefix) {

  // lod2
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int i = 0; i < nb_buildings; i++) {
    std::ifstream ifs(m_prefix + "_b" + std::to_string(i) + "_lod1.off");
    if (!ifs)
      continue;
    std::string f;
    ifs >> f;
    int np, nf;
    ifs >> np >> nf;
    double x, y, z;
    ifs >> x;
    for (int i = 0; i < np; i++) {
      ifs >> x >> y >> z;
      points.push_back(Point_3(x, y, z));
    }
    for (int i = 0; i < nf; i++) {
      int nfp;
      ifs >> nfp;
      std::vector<std::size_t> poly;
      for (int j = 0; j < nfp + 3; j++) {
        int nfp_;
        ifs >> nfp_;
        if (j < nfp)
          poly.push_back(nfp_ + nump);
        else
          poly.push_back(nfp_);
      }
      polygons.push_back(poly);
    }
    nump += np;
  }

  std::ofstream ofs(m_prefix + "_lod1.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}


void write_block_lod0(int nb_buildings, std::string m_prefix) {

  // lod2
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int i = 0; i < nb_buildings; i++) {
    std::ifstream ifs(m_prefix + "_b" + std::to_string(i) + "_lod0.off");
    if (!ifs)
      continue;
    std::string f;
    ifs >> f;
    int np, nf;
    ifs >> np >> nf;
    double x, y, z;
    ifs >> x;
    for (int i = 0; i < np; i++) {
      ifs >> x >> y >> z;
      points.push_back(Point_3(x, y, z));
    }
    for (int i = 0; i < nf; i++) {
      int nfp;
      ifs >> nfp;
      std::vector<std::size_t> poly;
      for (int j = 0; j < nfp + 3; j++) {
        int nfp_;
        ifs >> nfp_;
        if (j < nfp)
          poly.push_back(nfp_ + nump);
        else
          poly.push_back(nfp_);
      }
      polygons.push_back(poly);
    }
    nump += np;
  }

  std::ofstream ofs(m_prefix + "_lod0.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}

void write_scene_lod2(int rows, int cols, std::string wdir) {

  // lod2
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::stringstream ss;
      ss << wdir << std::setfill('0')
        << std::setw(3) << r
        << std::setw(3) << c;

      std::ifstream ifs(ss.str() + "_lod2.off");
      if (!ifs) 
        LOG_INFO << "Open failed.";
      std::string f;
      ifs >> f;
      int np, nf;
      ifs >> np >> nf;
      double x, y, z;
      ifs >> x;
      for (int i = 0; i < np; i++) {
        ifs >> x >> y >> z;
        points.push_back(Point_3(x, y, z));
      }
      for (int i = 0; i < nf; i++) {
        int nfp;
        ifs >> nfp;
        std::vector<std::size_t> poly;
        for (int j = 0; j < nfp + 3; j++) {
          int nfp_;
          ifs >> nfp_;
          if (j < nfp)
            poly.push_back(nfp_ + nump);
          else
            poly.push_back(nfp_);
        }
        polygons.push_back(poly);
      }
      nump += np;
      LOG_INFO << ss.str() << " finished.";
    }
  }
  LOG_INFO << points.size() << " points, " << polygons.size() << " faces";
  std::ofstream ofs(wdir + "lod2.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}

void write_scene_lod1(int rows, int cols, std::string wdir) {

  // lod1
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::stringstream ss;
      ss << wdir << std::setfill('0')
        << std::setw(3) << r
        << std::setw(3) << c;

      std::ifstream ifs(ss.str() + "_lod1.off");
      std::string f;
      ifs >> f;
      int np, nf;
      ifs >> np >> nf;
      double x, y, z;
      ifs >> x;
      for (int i = 0; i < np; i++) {
        ifs >> x >> y >> z;
        points.push_back(Point_3(x, y, z));
      }
      for (int i = 0; i < nf; i++) {
        int nfp;
        ifs >> nfp;
        std::vector<std::size_t> poly;
        for (int j = 0; j < nfp + 3; j++) {
          int nfp_;
          ifs >> nfp_;
          if (j < nfp)
            poly.push_back(nfp_ + nump);
          else
            poly.push_back(nfp_);
        }
        polygons.push_back(poly);
      }
      nump += np;
    }
  }

  std::ofstream ofs(wdir + "_lod1.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}

void write_scene_lod0(int rows, int cols, std::string wdir) {

  // lod0
  int nump = 0;
  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t>> polygons;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::stringstream ss;
      ss << wdir << std::setfill('0')
        << std::setw(3) << r
        << std::setw(3) << c;

      std::ifstream ifs(ss.str() + "_lod0.off");
      std::string f;
      ifs >> f;
      int np, nf;
      ifs >> np >> nf;
      double x, y, z;
      ifs >> x;
      for (int i = 0; i < np; i++) {
        ifs >> x >> y >> z;
        points.push_back(Point_3(x, y, z));
      }
      for (int i = 0; i < nf; i++) {
        int nfp;
        ifs >> nfp;
        std::vector<std::size_t> poly;
        for (int j = 0; j < nfp + 3; j++) {
          int nfp_;
          ifs >> nfp_;
          if (j < nfp)
            poly.push_back(nfp_ + nump);
          else
            poly.push_back(nfp_);
        }
        polygons.push_back(poly);
      }
      nump += np;
    }
  }

  std::ofstream ofs(wdir + "_lod0.off");
  ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
  for (const auto &p : points)
    ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto &plg : polygons) {
    ofs << (plg.size() - 3);
    for (const auto &p : plg)
      ofs << ' ' << p;
    ofs << '\n';
  }

}


#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <boost/function_output_iterator.hpp>
#include <unordered_map>
namespace PMP = CGAL::Polygon_mesh_processing;
struct halfedge2edge
{
  halfedge2edge(const Mesh_3& m, std::vector<edge_descriptor>& edges)
    : m_mesh(m), m_edges(edges)
  {}
  void operator()(const halfedge_descriptor& h) const
  {
    m_edges.push_back(edge(h, m_mesh));
  }
  const Mesh_3& m_mesh;
  std::vector<edge_descriptor>& m_edges;
};

void generate_texture1(const std::string& input_dir, const std::string& output_dir) {

  // load ply data (dense mesh)
  LOG_INFO << "0";
  //const auto ply = input_dir + cm::get_config().get<std::string>("prefix") + ".ply";
  const auto ply1 = input_dir + "Xinghu_Building_ply/Xinghu_Building.ply";
  LOG_INFO << ply1;
  LOG_INFO << "1_0";
  cm::Rply_loader ply_mesh1(ply1);
  
  LOG_INFO << "1";
  // load off data (lod2 model)
  std::ifstream ifs(output_dir + cm::get_config().get<std::string>("prefix") + "_lod2.off");
  Mesh_3 off_mesh;
  if (!ifs || !(ifs >> off_mesh) || off_mesh.is_empty())
  {
    std::cerr << "Not a valid off file." << std::endl;
    return;
  }
  LOG_INFO << "input dense mesh with #v " << ply_mesh1.num_vertices << ", #f " << ply_mesh1.num_faces;
  LOG_INFO << "input lod2 mesh with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();
  LOG_INFO << "2";
  PMP::triangulate_faces(off_mesh);

  // AABB tree
  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Triangle_3   Triangle3;
  typedef std::vector<Triangle3>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

  std::vector<Triangle3> cgalfaces;
  for (Mesh_3::Face_index fi : off_mesh.faces()) {
    Mesh_3::Halfedge_index hf = off_mesh.halfedge(fi);
    std::vector<Mesh_3::Vertex_index> vertexes;
    for (Mesh_3::Halfedge_index hi : halfedges_around_face(hf, off_mesh))
    {
      vertexes.push_back(target(hi, off_mesh));
    }
    K::Point_3 v1 = off_mesh.point(vertexes[0]);
    K::Point_3 v2 = off_mesh.point(vertexes[1]);
    K::Point_3 v3 = off_mesh.point(vertexes[2]);
    Triangle3 face = Triangle3(v1, v2, v3);
    if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
      continue;
    }
    cgalfaces.push_back(face);
  }
  Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
  LOG_INFO << "Construct AABB Tree done.";
  
  // Create a flag array to mark points to delete
  std::vector<bool> to_delete(ply_mesh1.num_vertices, false);

   // adjust vertices and mark points to delete
  // #pragma omp parallel for
  for (int i = 0; i < ply_mesh1.num_vertices; i++) {
    K::Point_3 query(ply_mesh1.vertices[3*i], ply_mesh1.vertices[3 * i+1], ply_mesh1.vertices[3 * i+2]);
    auto pp = AABBTree.closest_point_and_primitive(query);
    auto pi = pp.first;
    double dis = CGAL::squared_distance(query, pi);
    // if (dis > 0.1) continue;
    if (dis > 0.1) {
      // Mark this point for deletion
      to_delete[i] = true;
    }else {
      ply_mesh1.vertices[3 * i] = pi.x(); ply_mesh1.vertices[3 * i+1] = pi.y(); ply_mesh1.vertices[3 * i+2] = pi.z();
    }
  }
  // 统计需要删除的点的数量
  int delete_count = 0;
  for (bool flag : to_delete) {
    if (flag) delete_count++;
  }
  
  // 计算删除点的比例
  double delete_ratio = static_cast<double>(delete_count) / ply_mesh1.num_vertices;
  LOG_INFO << "删除点的数量: " << delete_count << ", 总点数: " << ply_mesh1.num_vertices;
  LOG_INFO << "删除点的比例: " << delete_ratio * 100 << "%";
  ply_mesh1.writeply((input_dir + "/model_lod2_texture.ply").c_str(), 1);
}

void calculate_error(int nb_buildings, const std::string& input_dir, const std::string& output_dir, const int cal_thread_num) {
  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Triangle_3   Triangle3;
  typedef std::vector<Triangle3>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
  // load ply data (dense mesh)
  //const auto ply_building_mesh = input_dir + cm::get_config().get<std::string>("prefix") + "_building.ply";
  //cm::Rply_loader ply_building(ply_building_mesh);
  //LOG_INFO << "Original building #v " << ply_building.num_vertices << ", #f " << ply_building.num_faces;
  std::ofstream ofs(output_dir + cm::get_config().get<std::string>("prefix") + "_error.txt");
  for (int i = 0; i < nb_buildings; i++) {
    // load off data (lod2 model)
    std::ifstream ifs1(output_dir + cm::get_config().get<std::string>("prefix") + "_b" + std::to_string(i) + "_lod2.off");
 
    Mesh_3 off_mesh1;
    if (!ifs1 || !(ifs1 >> off_mesh1) || off_mesh1.is_empty())
    {
      std::cerr << "Not a valid off file." << std::endl;
      continue;;
    }
    LOG_INFO << "input lod2 mesh with #v " << off_mesh1.number_of_vertices() << ", #f " << off_mesh1.number_of_faces();
    PMP::triangulate_faces(off_mesh1);

    // AABB tree

    std::vector<Triangle3> cgalfaces;
    for (Mesh_3::Face_index fi : off_mesh1.faces()) {
      Mesh_3::Halfedge_index hf = off_mesh1.halfedge(fi);
      std::vector<Mesh_3::Vertex_index> vertexes;
      for (Mesh_3::Halfedge_index hi : halfedges_around_face(hf, off_mesh1))
      {
        vertexes.push_back(target(hi, off_mesh1));
      }
      K::Point_3 v1 = off_mesh1.point(vertexes[0]);
      K::Point_3 v2 = off_mesh1.point(vertexes[1]);
      K::Point_3 v3 = off_mesh1.point(vertexes[2]);
      Triangle3 face = Triangle3(v1, v2, v3);
      if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
        continue;
      }
      cgalfaces.push_back(face);
    }
    Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
    LOG_INFO << "Construct AABB Tree done.";

    std::ifstream ifs(output_dir + cm::get_config().get<std::string>("prefix") + "_b" + std::to_string(i) + ".off");
    Mesh_3 off_mesh;
    if (!ifs || !(ifs >> off_mesh) || off_mesh.is_empty())
    {
      std::cerr << "Not a valid off file." << std::endl;
      continue;
    }
    LOG_INFO << "input lod2 mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();
    //PMP::triangulate_faces(off_mesh);
    LOG_INFO << "After triangulation, input origin mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();
    const auto vpmap = off_mesh.points();
    CGAL::Bbox_3 bbox;
    for (vertex_descriptor v : off_mesh.vertices())
      bbox += vpmap[v].bbox();
    double bbox_zmin = bbox.zmin();
    omp_set_num_threads(cal_thread_num);
// #pragma omp parallel for
    double maxerror = 0;
    double meanerror = 0;
    for (Mesh_3::Vertex_index vi : off_mesh.vertices()) {
      K::Point_3 query(off_mesh.point(vi));
      if (query.z() > bbox_zmin + 2) {
        auto pp = AABBTree.closest_point_and_primitive(query);
        auto pi = pp.first;
        double dis = sqrt(CGAL::squared_distance(query, pi));
        meanerror = meanerror + dis;
        if (dis > 5 ) continue;
        if (dis > maxerror) {
          maxerror = dis;
        }
      }
    }
    meanerror = meanerror / off_mesh.number_of_vertices();
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off maxerror" << '\n';
    ofs << maxerror << "\n";
    LOG_INFO << maxerror;
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off meanerror" << '\n';
    ofs << "mean error: " << meanerror << "\n";
    LOG_INFO << meanerror;
  }
}




/*void calculate_error(int nb_buildings, const std::string& input_dir, const std::string& output_dir) {
  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Triangle_3   Triangle3;
  typedef std::vector<Triangle3>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
  // load ply data (dense mesh)
  //const auto ply_building_mesh = input_dir + cm::get_config().get<std::string>("prefix") + "_building.ply";
  //cm::Rply_loader ply_building(ply_building_mesh);
  //LOG_INFO << "Original building #v " << ply_building.num_vertices << ", #f " << ply_building.num_faces;
  std::ofstream ofs(output_dir + cm::get_config().get<std::string>("prefix") + "_error.txt");
  // load off data (lod2 model)

  std::ifstream ifs1(output_dir + cm::get_config().get<std::string>("prefix") + "_building.off");

  Mesh_3 off_mesh1;
  if (!ifs1 || !(ifs1 >> off_mesh1) || off_mesh1.is_empty())
  {
    std::cerr << "Not a valid off file." << std::endl;
    return;
  }
  LOG_INFO << "input origin mesh with #v " << off_mesh1.number_of_vertices() << ", #f " << off_mesh1.number_of_faces();
  PMP::triangulate_faces(off_mesh1);

  // AABB tree

  std::vector<Triangle3> cgalfaces;
  for (Mesh_3::Face_index fi : off_mesh1.faces()) {
    Mesh_3::Halfedge_index hf = off_mesh1.halfedge(fi);
    std::vector<Mesh_3::Vertex_index> vertexes;
    for (Mesh_3::Halfedge_index hi : halfedges_around_face(hf, off_mesh1))
    {
      vertexes.push_back(target(hi, off_mesh1));
    }
    K::Point_3 v1 = off_mesh1.point(vertexes[0]);
    K::Point_3 v2 = off_mesh1.point(vertexes[1]);
    K::Point_3 v3 = off_mesh1.point(vertexes[2]);
    Triangle3 face = Triangle3(v1, v2, v3);
    if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
      continue;
    }
    cgalfaces.push_back(face);
  }
  Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
  LOG_INFO << "Construct AABB Tree done.";
  for (int i = 0; i < nb_buildings; i++) {
    std::ifstream ifs(output_dir + cm::get_config().get<std::string>("prefix") + "_b" + std::to_string(i) + "_lod2.off");
    Mesh_3 off_mesh;
    if (!ifs || !(ifs >> off_mesh) || off_mesh.is_empty())
    {
      std::cerr << "Not a valid off file." << std::endl;
      continue;
    }
    LOG_INFO << "input lod2 mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();
    //PMP::triangulate_faces(off_mesh);
    LOG_INFO << "After triangulation, input lod2 mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();

#pragma omp parallel for
    double maxerror = 0;
    double meanerror = 0;
    for (Mesh_3::Vertex_index vi : off_mesh.vertices()) {
      K::Point_3 query(off_mesh.point(vi));
      auto pp = AABBTree.closest_point_and_primitive(query);
      auto pi = pp.first;
      double dis = sqrt(CGAL::squared_distance(query, pi));
      meanerror = meanerror + dis;
      if (dis > 20) continue;
      if (dis > maxerror) {
        maxerror = dis;
      }
    }
    meanerror = meanerror / off_mesh.number_of_vertices();
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off maxerror" << '\n';
    ofs << maxerror << "\n";
    LOG_INFO << maxerror;
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off meanerror" << '\n';
    ofs << meanerror << "\n";
    LOG_INFO << meanerror;
  }
}*/


/*void calculate_error(int nb_buildings, const std::string& input_dir, const std::string& output_dir) {
  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Triangle_3   Triangle3;
  typedef std::vector<Triangle3>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
  // load ply data (dense mesh)
  //const auto ply_building_mesh = input_dir + cm::get_config().get<std::string>("prefix") + "_building.ply";
  //cm::Rply_loader ply_building(ply_building_mesh);
  //LOG_INFO << "Original building #v " << ply_building.num_vertices << ", #f " << ply_building.num_faces;
  std::ofstream ofs(output_dir + cm::get_config().get<std::string>("prefix") + "_error.txt");
  // load off data (lod2 model)
  std::ifstream ifs1(output_dir + cm::get_config().get<std::string>("prefix") + ".off");
  
  Mesh_3 off_mesh1;
  if (!ifs1 || !(ifs1 >> off_mesh1) || off_mesh1.is_empty())
  {
    std::cerr << "Not a valid off file." << std::endl;
    return;
  }
  LOG_INFO << "input origin mesh with #v " << off_mesh1.number_of_vertices() << ", #f " << off_mesh1.number_of_faces();
  PMP::triangulate_faces(off_mesh1);

  // AABB tree

  std::vector<Triangle3> cgalfaces;
  for (Mesh_3::Face_index fi : off_mesh1.faces()) {
    Mesh_3::Halfedge_index hf = off_mesh1.halfedge(fi);
    std::vector<Mesh_3::Vertex_index> vertexes;
    for (Mesh_3::Halfedge_index hi : halfedges_around_face(hf, off_mesh1))
    {
      vertexes.push_back(target(hi, off_mesh1));
    }
    K::Point_3 v1 = off_mesh1.point(vertexes[0]);
    K::Point_3 v2 = off_mesh1.point(vertexes[1]);
    K::Point_3 v3 = off_mesh1.point(vertexes[2]);
    Triangle3 face = Triangle3(v1, v2, v3);
    if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
      continue;
    }
    cgalfaces.push_back(face);
  }
  Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
  LOG_INFO << "Construct AABB Tree done.";
  for (int i = 0; i < nb_buildings; i++) {
    std::ifstream ifs(output_dir + cm::get_config().get<std::string>("prefix") + "_b" + std::to_string(i) + "_lod2.off");
    Mesh_3 off_mesh;
    if (!ifs || !(ifs >> off_mesh) || off_mesh.is_empty())
    {
      std::cerr << "Not a valid off file." << std::endl;
      continue;
    }
    LOG_INFO << "input lod2 mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();
    //PMP::triangulate_faces(off_mesh);
    LOG_INFO << "After triangulation, input lod2 mesh " << i << " with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();

#pragma omp parallel for
    double maxerror = 0;
    double meanerror = 0;
    for (Mesh_3::Vertex_index vi : off_mesh.vertices()) {
      K::Point_3 query(off_mesh.point(vi));
      auto pp = AABBTree.closest_point_and_primitive(query);
      auto pi = pp.first;
      double dis = sqrt(CGAL::squared_distance(query, pi));
      meanerror = meanerror + dis;
      if (dis > 20) continue;
      if (dis > maxerror) {
        maxerror = dis;
      }
    }
    meanerror = meanerror / off_mesh.number_of_vertices();
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off maxerror" << '\n';
    ofs << maxerror << "\n";
    LOG_INFO << maxerror;
    ofs << cm::get_config().get<std::string>("prefix") << "_b" << std::to_string(i) << "_lod2.off meanerror" << '\n';
    ofs << meanerror << "\n";
    LOG_INFO << meanerror;
  }
}*/

void generate_texture(const std::string& input_dir, const std::string& output_dir, double length, int n_iter) {

  // load ply data (dense mesh)
  const auto ply = input_dir +  "/Xinghu_Building_modified_building.ply";
  cm::Rply_loader ply_mesh(ply);
  // load off data (lod2 model)
  std::ifstream ifs(output_dir + "/Xinghu_Building_modified_building_lod2.off");
  Mesh_3 off_mesh;
  if (!ifs || !(ifs >> off_mesh) || off_mesh.is_empty())
  {
    std::cerr << "Not a valid off file." << std::endl;
    return;
  }
  LOG_INFO << "input dense mesh with #v " << ply_mesh.num_vertices << ", #f " << ply_mesh.num_faces;
  LOG_INFO << "input lod2 mesh with #v " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();

  PMP::triangulate_faces(off_mesh);
  // remeshing
  typedef boost::graph_traits<Mesh_3>::halfedge_descriptor 		halfedge_descriptor;
  typedef boost::graph_traits<Mesh_3>::edge_descriptor 		edge_descriptor;
  typedef Mesh_3::Vertex_index 		vertex_descriptor;
  typedef Mesh_3::Face_index 		face_descriptor;

  if (!CGAL::is_triangle_mesh(off_mesh)) {
    LOG_INFO << "Error: not a valid triangle mesh.";
    return;
  }
  
  double target_edge_length = length;
  int nb_iter = n_iter;
  LOG_INFO << "target_edge_length: " << target_edge_length;
  std::vector<edge_descriptor> border;
  PMP::border_halfedges(faces(off_mesh),
    off_mesh,
    boost::make_function_output_iterator(halfedge2edge(off_mesh, border)));
  // recondition: split long edges
  PMP::split_long_edges(border, target_edge_length, off_mesh);
  // isotropic remeshing
  PMP::isotropic_remeshing(
    faces(off_mesh),
    target_edge_length,
    off_mesh,
    PMP::parameters::number_of_iterations(nb_iter)
    .protect_constraints(true)//i.e. protect border, here
  );
  LOG_INFO << "After remeshing: #v: " << off_mesh.number_of_vertices() << ", #f " << off_mesh.number_of_faces();

  // AABB tree
  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Triangle_3   Triangle3;
  typedef std::vector<Triangle3>::iterator Iterator;
  typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
  typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

  std::vector<K::Point_3> mesh_points(ply_mesh.num_vertices);
  std::vector<Triangle3> cgalfaces;
  for (int i = 0; i < mesh_points.size(); i++) {
    mesh_points[i] = K::Point_3(ply_mesh.vertices[3*i], ply_mesh.vertices[3 *i + 1], ply_mesh.vertices[3 *i + 2]);
  }
  for (int i = 0; i < ply_mesh.num_faces; i++) {
    Triangle3 face = Triangle3(mesh_points[ply_mesh.faces[3 * i]], mesh_points[ply_mesh.faces[3 * i + 1]], mesh_points[ply_mesh.faces[3 * i + 2]]);
    if (face.is_degenerate() || K().is_degenerate_3_object()(face)) {
      continue;
    }
    cgalfaces.push_back(face);
  }
  Tree AABBTree(cgalfaces.begin(), cgalfaces.end());
  LOG_INFO << "Construct AABB Tree done.";

  cm::Rply_loader oply; // output
  oply.num_vertices = off_mesh.number_of_vertices();
  oply.num_faces = off_mesh.number_of_faces();
  oply.textures = ply_mesh.textures;
  std::map<K::Point_3, int> nn; // vertex-id
  for (Mesh_3::Face_index fi : off_mesh.faces()) {
    Mesh_3::Halfedge_index hf = off_mesh.halfedge(fi);
    std::vector<Mesh_3::Vertex_index> vertexes;
    for (Mesh_3::Halfedge_index hi : halfedges_around_face(hf, off_mesh))
    {
      vertexes.push_back(target(hi, off_mesh));
    }
    K::Point_3 v1 = off_mesh.point(vertexes[0]);
    K::Point_3 v2 = off_mesh.point(vertexes[1]);
    K::Point_3 v3 = off_mesh.point(vertexes[2]);
    K::Point_3 centor = CGAL::centroid(v1, v2, v3);
    int idv1, idv2, idv3;
    if (nn.find(v1) != nn.end()) {
      idv1 = nn[v1];
    }
    else {
      idv1 = nn.size();
      nn[v1] = idv1;
      oply.vertices.push_back(v1.x()); oply.vertices.push_back(v1.y()); oply.vertices.push_back(v1.z());
    }
    if (nn.find(v2) != nn.end()) {
      idv2 = nn[v2];
    }
    else {
      idv2 = nn.size();
      nn[v2] = idv2;
      oply.vertices.push_back(v2.x()); oply.vertices.push_back(v2.y()); oply.vertices.push_back(v2.z());
    }
    if (nn.find(v3) != nn.end()) {
      idv3 = nn[v3];
    }
    else {
      idv3 = nn.size();
      nn[v3] = idv3;
      oply.vertices.push_back(v3.x()); oply.vertices.push_back(v3.y()); oply.vertices.push_back(v3.z());
    }
    oply.faces.push_back(idv1); oply.faces.push_back(idv2); oply.faces.push_back(idv3);
    
    int idf = std::distance(cgalfaces.begin(), AABBTree.closest_point_and_primitive(centor).second);
    oply.texnumber.push_back(ply_mesh.texnumber[idf]);
    std::vector<K::Point_3> mv(3);
    mv[0] = K::Point_3(ply_mesh.vertices[3 * ply_mesh.faces[3 * idf]], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf] + 1], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf] + 2]);
    mv[1] = K::Point_3(ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 1]], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 1] + 1], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 1] + 2]);
    mv[2] = K::Point_3(ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 2]], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 2] + 1], ply_mesh.vertices[3 * ply_mesh.faces[3 * idf + 2] + 2]);
    std::map<double, std::pair<K::Point_3, int>> dis;
    int num = -1;
    for (auto v : mv) {
      num++;
      dis[(v - v1).squared_length()] = { v1, num };
      dis[(v - v2).squared_length()] = { v2, num };
      dis[(v - v3).squared_length()] = { v3, num };
    }
    int nnp[3];
    std::set<K::Point_3> repeated1; std::set<int> repeated2;
    std::map<K::Point_3, int> idxf;
    for (auto d : dis) {
      if (repeated1.count(d.second.first) == 0 && repeated2.count(d.second.second) == 0) {
        idxf[d.second.first] = d.second.second;
        repeated1.insert(d.second.first);
        repeated2.insert(d.second.second);
      }
    }
    oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v1]]); oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v1] + 1]);
    oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v2]]); oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v2] + 1]);
    oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v3]]); oply.texcoords.push_back(ply_mesh.texcoords[6 * idf + 2 * idxf[v3] + 1]);
  }
  //output
  oply.writeply((input_dir + "/model_lod2_texture.ply").c_str());
}
