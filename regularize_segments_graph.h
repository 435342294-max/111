#ifndef REGULARIZE_SEGMENTS_GRAPH_H
#define REGULARIZE_SEGMENTS_GRAPH_H

#include <vector>
#include <queue>
#include <memory>

#include <boost/range.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <CGAL/Cartesian_converter.h>

#include "utils/Kernel.h"

#include "approximate_squared_distance.h"

/*!
 * \brief Epsilons.controlled robust distance of two segments.
 * \param s1 first segment
 * \param s2 second segment
 * \return distance
 */
double robust_distance(const Kernel::Segment_2 &s1, const Kernel::Segment_2 &s2) {
  return std::sqrt(segment_segment_squared_distance(
    s1.source().x(), s1.source().y(), s1.target().x(), s1.target().y(),
    s2.source().x(), s2.source().y(), s2.target().x(), s2.target().y()));
}

/*!
 * \brief Regularize segments with graph decimation,
 * we regularizes parallelism, orthogonality and collinearity (WIP).
 * \tparam GeomTraits a model of Kernel
 * \param segments input segments
 * \param angle_threshold parallelism and orthogonality angle tolerance, in degree
 * \param distance_threshold collinear regularization threshold, in world coordinate
 * \return regularized segments
 */
template <typename GeomTraits>
std::vector<typename GeomTraits::Segment_2> regularize_segments_graph(
  const std::vector<Kernel::Segment_2> &segments,
  const double angle_threshold,
  const double distance_threshold) {
  // typedef
  typedef Kernel::Segment_2 Segment_2;
  typedef Kernel::Vector_2 Vector_2;

  // forward declaration
  struct Queue_element;
  typedef std::shared_ptr<Queue_element> Queue_element_ptr;

  // graph type
  struct Vertex_property {
    Segment_2 segment;
    double length;
    Vector_2 direction;
    bool valid;
  };
  struct Edge_property {
    Queue_element_ptr qelement;
  };
  typedef boost::adjacency_list<boost::listS, boost::vecS,
    boost::undirectedS, Vertex_property, Edge_property> Graph;
  typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;

  // queue element
  struct Queue_element {
    Queue_element(
      const vertex_descriptor &s,
      const vertex_descriptor &t,
      const double &val) : u(s), v(t), value(val) {}
    vertex_descriptor u;
    vertex_descriptor v;
    double value;
  };
  struct Compare {
    bool operator()(const Queue_element_ptr &lhs, const Queue_element_ptr &rhs) const {
      return lhs->value < rhs->value;
    }
  };

  const double parallel_threshold = std::cos(
    angle_threshold * boost::math::double_constants::degree);
  const double orthogonal_threshold = 1.0 - parallel_threshold;

  // construct graph
  Graph g(segments.size());
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const vertex_descriptor v = i;
    g[v].segment = segments[i];
    g[v].length = std::sqrt(segments[i].squared_length());
    g[v].direction = segments[i].to_vector() / g[v].length;
    g[v].valid = true;
  }

  // build the queue
  std::priority_queue<
    Queue_element_ptr,
    std::vector<Queue_element_ptr>,
    Compare> pqueue;
  // local parallel relationship graph
  std::vector<Vector_2> directions;
  for (const auto &s : segments)
    directions.push_back(s.to_vector() / std::sqrt(s.squared_length()));
  for (std::size_t i = 0; i < segments.size(); ++i) {
    for (std::size_t j = i + 1; j < segments.size(); ++j) {
      const double dist = robust_distance(segments[i], segments[j]);
      const double cosine = std::abs(directions[i] * directions[j]);
      if (dist < distance_threshold && cosine > parallel_threshold) {
        const auto qelement = std::make_shared<Queue_element>(i, j, cosine);
        boost::add_edge(i, j, {qelement}, g);
        pqueue.push(qelement);
      }
    }
  }

  // graph decimation
  while (!pqueue.empty()) {
    // process top element
    const Queue_element_ptr top = pqueue.top();
    pqueue.pop();
    const vertex_descriptor u = top->u;
    const vertex_descriptor v = top->v;

    // validity check: edge removed or updated
    edge_descriptor edge;
    bool exist = false;
    std::tie(edge, exist) = boost::edge(u, v, g);
    if (!exist || top != g[edge].qelement)
      continue;

    // merge to the longer segment
    const vertex_descriptor merged = g[u].length > g[v].length ? v : u;
    const vertex_descriptor reserved = g[u].length > g[v].length ? u : v;

    // weighted merged direction
    const auto s1 = g[merged].segment;
    const auto s2 = g[reserved].segment;
    Vector_2 dir = s1.to_vector();
    if (s2.to_vector() * dir < 0)
      dir = -dir;
    dir = s2.to_vector() + dir;
    dir = dir / std::sqrt(dir.squared_length());
    // weighted passing point
    const auto vec1 = CGAL::midpoint(s1.source(), s1.target()) - CGAL::ORIGIN;
    const auto vec2 = CGAL::midpoint(s2.source(), s2.target()) - CGAL::ORIGIN;
    const double len1 = g[merged].length;
    const double len2 = g[reserved].length;
    const double weight1 = len1 / (len1 + len2);
    const double weight2 = len2 / (len1 + len2);
    const auto passing_pt = CGAL::ORIGIN + (vec1 * weight1 + vec2 * weight2);
    // find endpoints
    const Kernel::Line_2 line(passing_pt, dir);
    const auto ps1s = line.projection(s1.source());
    const auto ps1t = line.projection(s1.target());
    const auto ps2s = line.projection(s2.source());
    const auto ps2t = line.projection(s2.target());
    const auto min_pt = std::min(std::min(ps1s, ps1t), std::min(ps2s, ps2t));
    const auto max_pt = std::max(std::max(ps1s, ps1t), std::max(ps2s, ps2t));
    // update segment
    g[reserved].segment = { min_pt, max_pt };
    g[reserved].length = CGAL::squared_distance(min_pt, max_pt);
    g[reserved].direction = dir;

    // collect neighboring vertices and remove edges
    std::set<vertex_descriptor> neighbors;
    for (const auto e : boost::make_iterator_range(boost::out_edges(merged, g))) {
      const vertex_descriptor v = boost::target(e, g);
      if (v != reserved)
        neighbors.insert(v);
    }
    boost::clear_vertex(merged, g);
    for (const auto e : boost::make_iterator_range(boost::out_edges(reserved, g))) {
      const vertex_descriptor v = boost::target(e, g);
      if (v != merged)
        neighbors.insert(v);
    }
    boost::clear_vertex(reserved, g);

    // connect neighbors to the reserved vertex
    for (const auto v : neighbors) {
      const double cosine = std::abs(g[v].direction * g[reserved].direction);
      const double dist = robust_distance(g[v].segment, g[reserved].segment);
      if (dist < distance_threshold && cosine > parallel_threshold) {
        // edge qelement keep track of the up-to-date queue element
        const auto qelement = std::make_shared<Queue_element>(v, reserved, cosine);
        boost::add_edge(v, reserved, {qelement}, g);
        pqueue.push(qelement);
      }
    }

    // mark merged vertex invalid
    g[merged].valid = false;
  }

  // collect valid segments
  typedef CGAL::Cartesian_converter<Kernel, GeomTraits> To_geom;
  To_geom to_geom;
  std::vector<typename GeomTraits::Segment_2> results;
  for (const vertex_descriptor v : boost::make_iterator_range(boost::vertices(g))) {
    if (g[v].valid)
      results.push_back(to_geom((g[v].segment)));
  }
  return results;
}

#endif // REGULARIZE_SEGMENTS_GRAPH_H
