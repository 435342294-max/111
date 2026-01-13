#include <vector>
#include <string>
#include <fstream>

#include <boost/program_options.hpp>

#include <CGAL/Bbox_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Shape_detection_3.h>
#include <CGAL/regularize_planes.h>
#include <CGAL/convex_hull_2.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/BlendFunc>

#include "utils/Logger.h"
#include "utils/Kernel.h"
#include "utils/Palette.h"
#include "utils/View_helper.h"

#include "segment_modeling/Horizontal_plane.h"

// Type declarations
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef CGAL::Shape_detection_3::Shape_detection_traits<
  Kernel, Point_set, Point_set::Point_map, Point_set::Vector_map> Traits;
typedef CGAL::Shape_detection_3::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection_3::Plane<Traits> Ransac_plane;
typedef CGAL::Shape_detection_3::Horizontal_plane<Traits> Horizontal_plane;

osg::Node *ransac_to_geometry(
  Efficient_ransac &ransac,
  const Point_set &point_set,
  const double epsilon,
  const bool color_wrt_type) {
  // report coverage
  const Efficient_ransac::Shape_range shapes = ransac.shapes();
  const double coverage =
    double(point_set.size() - ransac.number_of_unassigned_points()) /
    double(point_set.size());
  LOG_INFO << "#primitives: " << shapes.size() << ", #coverage: " << coverage;

  // convex hulls of shape points
  std::vector<std::vector<Point_3>> convex_hulls;
  for (const auto s : shapes) {
    LOG_DEBUG << s->info();

    std::list<Point_3> pts;
    for (const auto &p : s->indices_of_assigned_points())
      pts.push_back(point_set.point(*(point_set.begin() + p)));

    Plane_3 plane;
    if (Ransac_plane *p = dynamic_cast<Ransac_plane *>(s.get()))
      plane = static_cast<Plane_3>(*p);
    else if (Horizontal_plane *p = dynamic_cast<Horizontal_plane *>(s.get()))
      plane = static_cast<Plane_3>(*p);
    const Point_3 origin = plane.projection(pts.front());

    Vector_3 base1 = plane.base1();
    Vector_3 base2 = plane.base2();
    base1 = base1 / std::sqrt(base1.squared_length());
    base2 = base2 / std::sqrt(base2.squared_length());

    Kernel::Line_3 baseLine1(origin, base1);
    Kernel::Line_3 baseLine2(origin, base2);

    std::vector<Kernel::Point_2> coord_2;
    for (const auto &p : pts) {
      const Point_3 point = plane.projection(p);
      Vector_3 xvector(origin, baseLine1.projection(point));
      Vector_3 yvector(origin, baseLine2.projection(point));
      double x = std::sqrt(xvector.squared_length());
      double y = std::sqrt(yvector.squared_length());
      x = xvector * base1 < 0 ? -x : x;
      y = yvector * base2 < 0 ? -y : y;
      coord_2.push_back(Kernel::Point_2(x, y));
    }

    std::vector<Kernel::Point_2> cvx_hull_2;
    CGAL::convex_hull_2(
      coord_2.begin(),
      coord_2.end(),
      std::back_inserter(cvx_hull_2)
    );

    std::vector<Point_3> cvx_hull_3;
    for (const auto &p : cvx_hull_2)
      cvx_hull_3.push_back(origin + p.x() * base1 + p.y() * base2);

    convex_hulls.push_back(cvx_hull_3);
  }

  // primitives
  auto vertices = new osg::Vec3Array;
  auto colors = new osg::Vec4Array;
  auto geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_PER_PRIMITIVE_SET);
  auto blend_func = new osg::BlendFunc;
  blend_func->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  geometry->getOrCreateStateSet()->setAttributeAndModes(blend_func);
  geometry->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
  // vertices
  for (const auto &cvh : convex_hulls) {
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::POLYGON, vertices->size(), cvh.size()));
    for (const auto &p : cvh)
      vertices->push_back(cm::to_vec3(p));
  }
  // diverging color w.r.t the shape type
  const cm::Palette::Diverging palette;
  for (const auto s : shapes) {
    double sum_distances = 0.0;
    for (const auto &p : s->indices_of_assigned_points())
      sum_distances += CGAL::sqrt(s->squared_distance(
        point_set.point(*(point_set.begin() + p))));
    const double dist = sum_distances / s->indices_of_assigned_points().size();
    LOG_INFO << "#points: " << s->indices_of_assigned_points().size()
      << ", #distance: " << dist;
    if (color_wrt_type) {
      if (dynamic_cast<Ransac_plane *>(s.get()))
        colors->push_back({palette.color(0.0), 0.9f});
      else if (dynamic_cast<Horizontal_plane *>(s.get()))
        colors->push_back({palette.color(1.0), 0.9f});
    }
    else
      colors->push_back({palette.color(dist > epsilon ? 1.0 : dist / epsilon), 0.9f});
  }

  return geometry;
}

// test shape detection with data
int main(int argc, char *argv[]) {
  // initialize log
  cm::initialize_logger(cm::severity_level::debug);

  // parse command options
  namespace po = boost::program_options;
  po::variables_map vm;
  po::options_description desc("Allowed options");
  try {
    desc.add_options()
      ("point_set,P", po::value<std::string>()->required(),
        "input point cloud file")
      ("probability", po::value<double>()->required(),
        "Sets probability to miss the largest primitive at each iteration.")
      ("min_points", po::value<std::size_t>()->required(),
        "Detect shapes with at minimum points.")
      ("epsilon", po::value<double>()->required(),
        "Sets maximum Euclidean distance between a point and a shape.")
      ("cluster_epsilon", po::value<double>()->required(),
        "Sets maximum Euclidean distance between points to be clustered.")
      ("normal_threshold", po::value<double>()->required(),
        "Sets maximum normal deviation.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
    LOG_INFO << desc;
    return 1;
  }

  // Loads point cloud from a file.
  Point_set point_set;
  const std::string fname = vm["point_set"].as<std::string>();
  std::ifstream ifs(fname);
  ifs >> point_set;
  if (!ifs) {
    LOG_ERROR << "Failed to read file " << fname;
    return 1;
  }
  ifs.close();

  // viewer
  cm::Viewer viewer(1, 3, 450, 450);
  viewer.run();
  viewer.add_node(cm::points_to_node(point_set, osg::SILVER), 0);

  Efficient_ransac::Parameters parameters;
  parameters.probability = vm["probability"].as<double>();
  parameters.min_points = vm["min_points"].as<std::size_t>();
  parameters.epsilon = vm["epsilon"].as<double>();
  parameters.cluster_epsilon = vm["cluster_epsilon"].as<double>();
  parameters.normal_threshold = vm["normal_threshold"].as<double>();

  LOG_INFO << "Shape detection...";
  Efficient_ransac ransac;
  ransac.set_input(point_set, point_set.point_map(), point_set.normal_map());
  ransac.add_shape_factory<Ransac_plane>();
  ransac.add_shape_factory<Horizontal_plane>();
  ransac.preprocess();
  ransac.detect(parameters);
  LOG_INFO << "Done.";

  viewer.add_node(ransac_to_geometry(ransac, point_set, vm["epsilon"].as<double>(), true), 1);

  // plane regularization
  // Efficient_ransac::Plane_range planes = ransac.planes();
  // CGAL::regularize_planes(point_set,
  //   point_set.point_map(),
  //   planes,
  //   CGAL::Shape_detection_3::Plane_map<Traits>(),
  //   CGAL::Shape_detection_3::Point_to_shape_index_map<Traits>(point_set, planes),
  //   true, //Regularize parallelism
  //   true, // Regularize orthogonality
  //   false, // Do not regularize co-planarity
  //   true, // Regularize Z-symmetry (default)
  //   10); // 10 degrees of tolerance for parallelism / orthogonality
  viewer.add_node(ransac_to_geometry(ransac, point_set, vm["epsilon"].as<double>(), false), 2);

  return 0;
}
