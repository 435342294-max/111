#include <iostream>
#include <chrono>
#include <random>
#include <thread>

#include <CGAL/Point_set_3.h>

#include <osg/Geode>
#include <osg/Geometry>

#include "utils/Kernel.h"
#include "utils/Logger.h"
#include "utils/View_helper.h"

void add_axes() {
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(2s);

  cm::get_global_viewer().add_node(cm::shape_xyz_axes());
  cm::get_global_viewer().move_to_home_position();

  LOG_INFO << "add_axes done";
}

void add_normls() {
  typedef CGAL::Point_set_3<Point_3> Normal_set;
  // normal with position
  Normal_set normal_set(true);
  normal_set.insert(Point_3(1., 0., 1.), Vector_3(1., 0., 1.));
  normal_set.insert(Point_3(0., 1., 1.), Vector_3(1., 1., 0.));
  normal_set.insert(Point_3(1., 1., 1.), Vector_3(0., 1., 1.));
  cm::get_global_viewer().add_node(
    cm::normals_to_node(normal_set, osg::RED));
  // normal without position
  cm::get_global_viewer().add_node(
    cm::normals_to_node(normal_set, normal_set.normal_map(), osg::GREEN));
  // normals in plain container
  std::vector<Vector_3> normals;
  normals.push_back({1.0, 2.0, 3.0});
  normals.push_back({3.0, 1.0, 2.0});
  normals.push_back({2.0, 3.0, 1.0});
  cm::get_global_viewer().add_node(
    cm::normals_to_node(normals, osg::BLUE));

  LOG_INFO << "add_normls done";
}

void add_points() {
  typedef CGAL::Point_set_3<Point_3> Point_set;
  // pointset with normal
  Point_set point_set(true);
  point_set.insert(Point_3(1., 0., 2.), Vector_3(1., 0., 1.));
  point_set.insert(Point_3(0., 1., 2.), Vector_3(1., 1., 0.));
  point_set.insert(Point_3(1., 1., 2.), Vector_3(0., 1., 1.));
  cm::get_global_viewer().add_node(
    cm::points_to_node(point_set, osg::SILVER));
  // points in container
  std::vector<Point_3> pts;
  pts.push_back({3.0, 4.0, 5.0});
  pts.push_back({5.0, 3.0, 4.0});
  pts.push_back({4.0, 5.0, 3.0});
  cm::get_global_viewer().add_node(
    cm::points_to_node(pts, osg::RED));

  LOG_INFO << "add_points done";
}

void add_random_points() {
  // node insertion stress test
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> disx(0.0f, 10.0f);
  std::uniform_real_distribution<float> disy(0.0f, 10.0f);
  std::uniform_real_distribution<float> disz(0.0f, 10.0f);

  for (int i = 0; i < 1000; ++i) {
    std::vector<Point_3> pts(1000);
    for (auto &p : pts)
      p = Point_3(disx(gen), disy(gen), disz(gen));
    cm::get_global_viewer().add_node(
      cm::points_to_node(pts, { disx(gen), disx(gen), disx(gen), 1.0f }), 1);

    LOG_INFO << "#" << i << ' ' << cm::get_global_viewer().root(1)->getNumChildren();

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2ms);
  }
  LOG_INFO << "add_random_points done";
}

int main() {
  cm::initialize_logger();
  // change sink to have thread id
  namespace log = boost::log;
  log::core::get()->remove_all_sinks();
  log::add_console_log(std::cout, log::keywords::format = "[%ThreadID%]: %Message%");

  // start the global viewer thread
  cm::get_global_viewer(1, 2, 800, 450).run();

  std::thread t0(add_axes);
  std::thread t1(add_normls);
  std::thread t2(add_points);
  std::thread t3(add_random_points);
  std::thread t4(add_random_points);
  std::thread t5(add_random_points);

  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();

  return 0;
}
