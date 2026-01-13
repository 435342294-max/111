#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <CGAL/property_map.h>

#include "utils/RANSAC.h"

struct Point_2 {
  Point_2(double x_, double y_) : x(x_), y(y_) {};

  double x;
  double y;
};

typedef std::vector<std::shared_ptr<Point_2>> Input_range;
typedef Input_range::iterator Input_iterator;
struct Input_map
  : public boost::put_get_helper<Point_2 &, Input_map> {
  typedef std::shared_ptr<Point_2> key_type;
  typedef Point_2 value_type;
  typedef value_type &reference;
  typedef boost::lvalue_property_map_tag category;

  reference operator[](key_type key) const { return *key; }
};

class Line_model_2 {
public:
  typedef ::Input_range Input_range;
  typedef ::Input_map Input_map;

  static const int MINIMUM_SET_SIZE = 2;

  Line_model_2(const std::vector<Input_iterator> &minimum_set, Input_map input_map)
    : m_a(0.0), m_b(0.0), m_c(1.0), m_denominator(1.0) {
    if (minimum_set.size() < MINIMUM_SET_SIZE)
      return;

    const auto &p0 = get(input_map, *minimum_set[0]);
    const auto &p1 = get(input_map, *minimum_set[1]);
    if (p0.x == p1.x && p0.y == p1.y)
      return;

    if (p0.x == p1.x && p0.x == 0.0) {
      // x = 0
      m_a = 1.0; m_b = 0.0; m_c = 0.0;
    }
    else if (p0.y == p1.y && p0.y == 0.0) {
      // y = 0
      m_a = 0.0; m_b = 1.0; m_c = 0.0;
    }
    else {
      // ax + by + 1 = 0
      m_a = (p1.y - p0.y) / (p1.x * p0.y - p0.x * p1.y);
      m_b = (p0.x - p1.x) / (p1.x * p0.y - p0.x * p1.y);
    }
    m_denominator = std::sqrt(m_a * m_a + m_b * m_b);
  };

  double compute_distance(const Point_2 &s) {
    return std::abs(m_a * s.x + m_b * s.y + m_c) / m_denominator;
  };

  // ax + by + c = 0
  double m_a, m_b, m_c;

private:
  double m_denominator;
};

int main(int argc, char * argv[])
{
  if (argc != 1 && argc != 3) {
    std::cout << "[ USAGE ]: " << argv[0] <<
      " [<Image Size> = 1000] [<num_samples> = 500]" << std::endl;
    return -1;
  }

  const int image_size = (argc == 3) ? std::atoi(argv[1]) : 1000;
  const int num_samples = (argc == 3) ? std::atoi(argv[2]) : 500;

  cv::Mat canvas(image_size, image_size, CV_8UC3);
  canvas.setTo(255);

  // generate samples with noise
  std::random_device seed_device;
  std::mt19937 RNG = std::mt19937(seed_device());

  std::uniform_int_distribution<int> uniform_dist(0, image_size - 1);
  std::normal_distribution<> normal_dist(0.0, 25.0);

  std::vector<std::shared_ptr<Point_2>> samples;
  for (int i = 0; i < num_samples; ++i) {
    const int diag = uniform_dist(RNG);
    cv::Point pt(int(diag + normal_dist(RNG)),
      int(diag + normal_dist(RNG)));
    cv::circle(canvas, pt,
      int(image_size / 100.0 + 3.0),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    samples.push_back(std::make_shared<Point_2>(pt.x, pt.y));
  }

  const auto start = std::chrono::high_resolution_clock::now();

  std::shared_ptr<Line_model_2> model;
  std::vector<Input_iterator> inliers;
  cm::RANSAC<Line_model_2> ransac(samples);
  std::tie(model, inliers) = ransac.estimate(20, 1000);
  const auto finish = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "RANSAC took: " << elapsed.count() * 1000.0 << " ms." << std::endl;

  if (model) {
    for (auto &s : inliers) {
      cv::Point pt(int((*s)->x), int((*s)->y));
      cv::circle(canvas, pt, int(image_size / 100.0), cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    // ignore vertical lines
    if (model->m_b != 0.0)
      cv::line(canvas,
        cv::Point(0, int(-1.0 / model->m_b)),
        cv::Point(image_size, int((-1.0 - model->m_a * image_size) / model->m_b)),
        cv::Scalar(0, 0, 255), 2);
  }

  while (true) {
    cv::imshow("RANSAC Example", canvas);

    if (cv::waitKey(1) == 27)
      break;
  }

  return 0;
}
