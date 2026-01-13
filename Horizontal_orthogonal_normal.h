#ifndef CGAL_SHAPE_DETECTION_3_HORIZONTAL_ORTHOGONAL_NORMAL_H
#define CGAL_SHAPE_DETECTION_3_HORIZONTAL_ORTHOGONAL_NORMAL_H

#include <CGAL/Shape_detection_3.h>

namespace CGAL {
namespace Shape_detection_3 {

/*!
 * \brief Horizontal_orthogonal_normal derives from Shape_base.
 * This shape:
 *   - always horizontal
 *   - include orthogonal direction
 * \note Input should be unit vector on xy plane and corresponding points.
 * \note Orthogonality is not guaranteed and may be extremely unbalanced.
 */
template <class Traits>
class Horizontal_orthogonal_normal : public CGAL::Shape_detection_3::Shape_base<Traits> {
public:
  typedef typename Traits::FT FT;
  typedef typename Traits::Point_3 Point_3;
  typedef typename Traits::Vector_3 Vector_3;

public:
  Horizontal_orthogonal_normal() : CGAL::Shape_detection_3::Shape_base<Traits>() {}

  // Conversion function
  operator Vector_3() {
    return m_normal;
  }

  // Computes squared Euclidean distance from query point to the shape.
  virtual FT squared_distance(const Point_3 &p) const {
    FT dis[4] = {FT(0.0), FT(0.0), FT(0.0), FT(0.0)};
    dis[0] = CGAL::squared_distance(p, CGAL::ORIGIN + m_normal);
    dis[1] = CGAL::squared_distance(p, CGAL::ORIGIN - m_normal);
    dis[2] = CGAL::squared_distance(p, CGAL::ORIGIN + m_normal_ortho);
    dis[3] = CGAL::squared_distance(p, CGAL::ORIGIN - m_normal_ortho);
    return *std::min_element(dis, dis + 4);
  }

  // Returns a string with shape parameters.
  virtual std::string info() const {
    std::stringstream sstr;
    sstr << "Type: Horizontal_orthogonal_normal ("
      << m_normal.x() << ", " << m_normal.y() << ", " << m_normal.z()
      << ") #Pts: " << this->m_indices.size();
    return sstr.str();
  }

protected:
  // Constructs shape based on minimal set of samples from the input data.
  virtual void create_shape(const std::vector<std::size_t> &indices) {
    m_normal = this->normal(indices[0]) + this->normal(indices[1]) + this->normal(indices[2]);
    m_normal = Vector_3(m_normal.x(), m_normal.y(), FT(0.0));

    if (m_normal.squared_length() <= FT(0.0))
      return;

    m_normal /= CGAL::sqrt(m_normal.squared_length());
    m_normal_ortho = Vector_3(-m_normal.y(), m_normal.x(), FT(0.0));

    // check deviation of the 3 normal
    for (std::size_t i = 0; i < 3; ++i) {
      const Vector_3 n = this->normal(indices[i]);
      if (CGAL::abs(n * m_normal) < this->m_normal_threshold
        && CGAL::abs(n * m_normal_ortho) < this->m_normal_threshold)
        return;
    }

    this->m_is_valid = true;
  }

  // Computes squared Euclidean distance from a set of points.
  virtual void squared_distance(
    const std::vector<std::size_t> &indices,
    std::vector<FT> &dists) const {
    FT d[4] = {FT(0.0), FT(0.0), FT(0.0), FT(0.0)};
    for (std::size_t i = 0; i < indices.size(); i++) {
      d[0] = CGAL::squared_distance(this->point(indices[i]), CGAL::ORIGIN + m_normal);
      d[1] = CGAL::squared_distance(this->point(indices[i]), CGAL::ORIGIN - m_normal);
      d[2] = CGAL::squared_distance(this->point(indices[i]), CGAL::ORIGIN + m_normal_ortho);
      d[3] = CGAL::squared_distance(this->point(indices[i]), CGAL::ORIGIN - m_normal_ortho);
      dists[i] = *std::min_element(d, d + 4);
    }
  }

  // Computes the normal deviation between shape and a set of points with normals.
  virtual void cos_to_normal(
    const std::vector<std::size_t> &indices,
    std::vector<FT> &angles) const {
    for (std::size_t i = 0; i < indices.size(); i++) {
      const FT c0 = CGAL::abs(this->normal(indices[i]) * m_normal);
      const FT c1 = CGAL::abs(this->normal(indices[i]) * m_normal_ortho);
      angles[i] = c0 > c1 ? c0 : c1;
    }
  }

  // Returns the number of required samples for construction.
  virtual std::size_t minimum_sample_size() const {
    // combine few samples gives bigger search space over limited data
    return 3;
  }

  // no connected component concept in this shape
  virtual std::size_t connected_component(std::vector<std::size_t>& indices, FT cluster_epsilon) {
    return indices.size();
  }

private:
  // direction and its orthogonal counterpart, counter clockwise
  Vector_3 m_normal;
  Vector_3 m_normal_ortho;
};

} // Shape_detection_3
} // CGAL

#endif // CGAL_SHAPE_DETECTION_3_HORIZONTAL_ORTHOGONAL_NORMAL_H
