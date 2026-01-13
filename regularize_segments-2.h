#ifndef REGULARIZE_SEGMENTS_H
#define REGULARIZE_SEGMENTS_H

#include <vector>
#include <queue>

#include <boost/math/constants/constants.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <CGAL/Cartesian_converter.h>

#include "utils/Logger.h"
#include "utils/Kernel.h"

/*!
 * \brief Regularize nearly collinear segments to collinear.
 * Segments that are close enough are merged.
 * \tparam GeomTraits a model of Kernel
 * \param segments input nearly collinear segments
 * \param line target collinear line
 * \param distance_threshold merge distance threshold
 * \return regularized segments
 */
template <typename GeomTraits>
std::vector<typename GeomTraits::Segment_2> regularize_collinear_segments(
  const std::vector<Kernel::Segment_2> &segments,
  const Kernel::Line_2 line,
  const double distance_threshold) {
  // GeomTraits conversion, to accelerate computation
  typedef CGAL::Cartesian_converter<Kernel, GeomTraits> To_geom;
  To_geom to_geom;

  // project segments on the target line
  std::vector<Kernel::Segment_2> projected;
  for (const auto &s : segments)
    projected.push_back({
      line.projection(s.source()), line.projection(s.target())});

  // construct graph
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
  Graph g(projected.size());
  for (std::size_t i = 0; i < projected.size(); ++i) {
    for (std::size_t j = i + 1; j < projected.size(); ++j) {
      const auto dist = CGAL::approximate_sqrt(CGAL::squared_distance(
        to_geom(projected[i]), to_geom(projected[j])));
      if (dist < distance_threshold)
        boost::add_edge(i, j, g);
    }
  }
  std::vector<std::size_t> component_map(boost::num_vertices(g));
  const std::size_t num = boost::connected_components(g, &component_map[0]);
  std::vector<std::vector<std::size_t>> components(num);
  for (std::size_t i = 0; i < component_map.size(); ++i)
    components[component_map[i]].push_back(i);

  // merge
  std::vector<typename GeomTraits::Segment_2> merged;
  const typename GeomTraits::Line_2 co_line = to_geom(line);
  const double dmax = std::numeric_limits<double>::max();
  const double dlowest = std::numeric_limits<double>::lowest();
  for (const auto &c : components) {
    Kernel::Point_2 pmin(dmax, dmax);
    Kernel::Point_2 pmax(dlowest, dlowest);
    for (const auto &i : c) {
      if (projected[i].min() < pmin)
        pmin = projected[i].min();
      if (projected[i].max() > pmax)
        pmax = projected[i].max();
    }
    merged.push_back({
      co_line.projection(to_geom(pmin)),
      co_line.projection(to_geom(pmax))
    });
  }

  return merged;
}

/*!
 * \brief Regularize nearly parallel segments to parallel.
 * In the parallel set, segments that are collinear are also regularized.
 * \tparam GeomTraits a model of Kernel
 * \param segments input nearly parallel segments
 * \param dir target parallel direction
 * \param distance_threshold collinear distance threshold
 * \return regularized segments
 */
template <typename GeomTraits>
std::vector<typename GeomTraits::Segment_2> regularize_parallel_segments(
  const std::vector<Kernel::Segment_2> &segments,
  const Kernel::Vector_2 &dir,
  const double distance_threshold) {
  // segment mid point projected onto orthogonal direction of priori direction
  struct Notch {
    void merge_with(Notch &other) {
      double wei_sum = weight + other.weight;
      pos = (pos * weight + other.pos * other.weight) / wei_sum;
      weight = wei_sum;
      indexes.insert(indexes.end(), other.indexes.begin(), other.indexes.end());
      other.is_valid = false;
    }

    double pos;
    double weight; // > 0
    bool is_valid;
    std::vector<std::size_t> indexes;
  };

  // pair of Notch
  struct Notch_pair {
    Notch_pair(const double d, const std::size_t n1_, const std::size_t n2_)
      : dist(d), n1(n1_), n2(n2_) {
      if (n1 > n2)
        std::swap(n1, n2);
    }

    bool operator<(const Notch_pair &rhs) const {
      return dist > rhs.dist;
    }

    double dist;
    std::size_t n1; // n1 < n2
    std::size_t n2;
  };

  // segment mid points to notch on perpendicular direction
  const auto dir_ortho = dir.perpendicular(CGAL::POSITIVE);
  std::vector<Notch> notches;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const auto &s = segments[i];
    const auto mid = CGAL::midpoint(s.source(), s.target());
    const double len = CGAL::approximate_sqrt(s.squared_length());
    notches.push_back({
      (mid - CGAL::ORIGIN) * dir_ortho, len, true, {i}});
  }
  LOG_DEBUG << "#notches: " << notches.size();

  if (notches.size() < 2)
    return {};

  // sort notches by position field
  std::sort(notches.begin(), notches.end(),
    [](const Notch &nl, const Notch &nr) ->bool {
      return nl.pos < nr.pos;
    });

  std::priority_queue<Notch_pair> merge_queue;
  for (std::size_t n1 = 0, n2 = 1; n2 < notches.size(); ++n2, ++n1)
    merge_queue.emplace(std::abs(notches[n1].pos - notches[n2].pos), n1, n2);
  while (!merge_queue.empty()) {
    const Notch_pair minpair = merge_queue.top();
    merge_queue.pop();
    if (!notches[minpair.n1].is_valid || !notches[minpair.n2].is_valid)
      continue;

    if (minpair.dist < distance_threshold) {
      notches[minpair.n1].merge_with(notches[minpair.n2]);
      std::size_t next_valid = minpair.n2;
      while (next_valid < notches.size() && !notches[next_valid].is_valid)
        ++next_valid;
      if (next_valid < notches.size())
        merge_queue.emplace(
          std::abs(notches[minpair.n1].pos - notches[next_valid].pos),
          minpair.n1,
          next_valid);
      int prev_valid = static_cast<int>(minpair.n1) - 1;
      while (prev_valid >= 0 && !notches[prev_valid].is_valid)
        --prev_valid;
      if (prev_valid >= 0)
        merge_queue.emplace(
          std::abs(notches[minpair.n1].pos - notches[prev_valid].pos),
          static_cast<std::size_t>(prev_valid),
          minpair.n1);
    }
    else
      break;
  }
  LOG_DEBUG << "#notches_valid: " << std::count_if(notches.begin(), notches.end(),
    [](const Notch &n) { return n.is_valid; });

  // result
  std::vector<typename GeomTraits::Segment_2> result;
  for (const auto &n : notches) {
    if (n.is_valid) {
      std::vector<Kernel::Segment_2> collinear_segments;
      for (const auto i : n.indexes)
        collinear_segments.push_back(segments[i]);
      const Kernel::Line_2 line(CGAL::ORIGIN + n.pos * dir_ortho, dir);
      const auto merged = regularize_collinear_segments<GeomTraits>(
        collinear_segments, line, distance_threshold);
      result.insert(result.end(), merged.begin(), merged.end());
    }
  }

  return result;
}

/*!
 * \brief Regularize line segments w.r.t. direction priors.
 * \tparam GeomTraits a model of Kernel
 * \param segments input segments
 * \param dir_priors the (orthogonal) priori directions to be regularized to, unit normals
 * \param distance_threshold collinear regularization threshold, in world coordinate
 */
template <typename GeomTraits>
std::vector<typename GeomTraits::Segment_2> regularize_segments(
  const std::vector<Kernel::Segment_2> &segments,
  const std::vector<Kernel::Vector_2> &dir_priors,
  const double distance_threshold) {
  LOG_INFO << "regularize_segments";

  // direction regularization
  std::vector<std::size_t> parallel_to(segments.size(), -1);
  std::vector<std::vector<std::size_t>> parallel_sets(dir_priors.size());
  for (std::size_t i = 0; i < segments.size(); ++i) {
    // TODO: better use GMM model to calculate the most likely direction
    const auto seg_vec = segments[i].to_vector();
    double max_sim = -1.0;
    std::size_t max_id = 0;
    for (std::size_t j = 0; j < dir_priors.size(); ++j) {
      const double sim = CGAL::abs(seg_vec * dir_priors[j]);
      if (sim > max_sim) {
        max_sim = sim;
        max_id = j;
      }
    }
    parallel_sets[max_id].push_back(i);
    parallel_to[i] = max_id;
  }

  // collinear regularization for each priori direction
  LOG_INFO << "regularize_parallel_segments";
  std::vector<typename GeomTraits::Segment_2> regularized;
  for (std::size_t i = 0; i < dir_priors.size(); ++i) {
    std::vector<Kernel::Segment_2> parallel_segments;
    for (const auto p : parallel_sets[i])
      parallel_segments.push_back(segments[p]);
    const auto result = regularize_parallel_segments<GeomTraits>(
      parallel_segments, dir_priors[i], distance_threshold);
    regularized.insert(regularized.end(), result.begin(), result.end());
  }

  return regularized;
}

/*!
 * \brief Regularize segments without any priori knowledge,
 * we regularizes parallelism, orthogonality and collinearity.
 * \tparam GeomTraits a model of Kernel
 * \param segments input segments
 * \param angle_threshold parallelism and orthogonality angle tolerance, in degree
 * \param distance_threshold collinear regularization threshold, in world coordinate
 * \return regularized segments
 */
template <typename GeomTraits>
std::vector<typename GeomTraits::Segment_2> regularize_segments(
  std::vector<Kernel::Segment_2> &segments,
  const double angle_threshold,
  const double distance_threshold) {
  // typedef
  typedef Kernel::Segment_2 Segment_2;
  typedef Kernel::Vector_2 Vector_2;
  // parallel set
  struct Parallel_set {
    double length;
    Vector_2 direction;
    std::vector<std::size_t> indexes;
  };

  LOG_INFO << "regularize_segments";

  const double parallel_threshold = std::cos(
    angle_threshold * boost::math::double_constants::degree);
  const double orthogonal_threshold = 1.0 - parallel_threshold;

  // descending order of length
  std::sort(segments.begin(), segments.end(),
    [](const Segment_2 &a, const Segment_2 &b) {
      return a.squared_length() > b.squared_length();
    });

  // find parallel sets
  std::vector<std::size_t> parallel_to(segments.size(), -1);
  std::vector<Parallel_set> parallel_sets;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    // segment length and direction
    const double seg_len = CGAL::approximate_sqrt(segments[i].squared_length());
    const Vector_2 seg_dir = segments[i].to_vector() / seg_len;
    // find most parallel set
    double max_sim = -1.0;
    std::size_t max_idx = 0;
    for (std::size_t j = 0; j < parallel_sets.size(); ++j) {
      const auto sim = CGAL::abs(parallel_sets[j].direction * seg_dir);
      if (sim > max_sim) {
        max_sim = sim;
        max_idx = j;
      }
    }
    // create or update an parallel set
    if (max_sim > parallel_threshold) {
      // update
      auto &ps = parallel_sets[max_idx];
      Vector_2 ps_vec = ps.direction * ps.length;
      if (ps.direction * seg_dir < 0.0)
        ps_vec -= segments[i].to_vector();
      else
        ps_vec += segments[i].to_vector();
      ps.length = CGAL::approximate_sqrt(ps_vec.squared_length());
      ps.direction = ps_vec / ps.length;
      ps.indexes.push_back(i);
      parallel_to[i] = max_idx;
    }
    else {
      // create
      parallel_to[i] = parallel_sets.size();
      parallel_sets.push_back({seg_len, seg_dir, {i}});
    }
  }
  LOG_DEBUG << "#parallel_sets: " << parallel_sets.size();

  // regularize orthogonality
  std::sort(parallel_sets.begin(), parallel_sets.end(),
    [](const Parallel_set &a, const Parallel_set &b) {
      return a.length > b.length;
    });
  std::vector<bool> processed(parallel_sets.size(), false);
  for (std::size_t i = 0; i < parallel_sets.size(); ++i) {
    if (processed[i])
      continue;
    for (std::size_t j = i + 1; j < parallel_sets.size(); ++j) {
      if (processed[j])
        continue;
      if (CGAL::abs(parallel_sets[i].direction *
        parallel_sets[j].direction) < orthogonal_threshold) {
        parallel_sets[j].direction =
          parallel_sets[i].direction.perpendicular(CGAL::POSITIVE);
        processed[j] = true;
      }
    }
    processed[i] = true;
  }

  // regularize parallel segments
  LOG_INFO << "regularize_parallel_segments";
  std::vector<typename GeomTraits::Segment_2> regularized;
  for (const auto &ps : parallel_sets) {
    std::vector<Segment_2> parallel_segments;
    for (const auto i : ps.indexes)
      parallel_segments.push_back(segments[i]);
    const auto result = regularize_parallel_segments<GeomTraits>(
      parallel_segments, ps.direction, distance_threshold);
    regularized.insert(regularized.end(), result.begin(), result.end());
  }

  return regularized;
}

#endif // REGULARIZE_SEGMENTS_H
