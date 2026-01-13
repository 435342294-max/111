#ifndef CGAL_SHAPE_DETECTION_3_HORIZONTAL_PLANE_H
#define CGAL_SHAPE_DETECTION_3_HORIZONTAL_PLANE_H

#include <CGAL/Shape_detection_3.h>

namespace CGAL {
namespace Shape_detection_3 {

/*!
 * \brief Horizontal_plane derives from Shape_base.
 * The plane is represented by its normal vector and distance to the origin.
 * The plane is guaranteed to be horizontal.
 */
template <class Traits>
class Horizontal_plane : public CGAL::Shape_detection_3::Shape_base<Traits> {
public:
  typedef typename Traits::FT FT;
  typedef typename Traits::Point_3 Point_3;
  typedef typename Traits::Vector_3 Vector_3;

public:
  Horizontal_plane() : CGAL::Shape_detection_3::Shape_base<Traits>() {}

  // Conversion function
  operator typename Traits::Plane_3() {
    return typename Traits::Plane_3(m_point_on_primitive, m_normal);
  }

  // Computes squared Euclidean distance from query point to the shape.
  virtual FT squared_distance(const Point_3 &p) const {
    const FT sd = (p - m_point_on_primitive) * m_normal;
    return sd * sd;
  }

  // Returns a string with shape parameters.
  virtual std::string info() const {
    std::stringstream sstr;
    sstr << "Type: Horizontal_plane (" <<
      m_normal.x() << ", "
      << m_normal.y() << ", "
      << m_normal.z() << ")x - "
      << (m_point_on_primitive - CGAL::ORIGIN) * m_normal << " = 0"
      << " #Pts: " << this->m_indices.size();
    return sstr.str();
  }

protected:
  // Constructs shape based on minimal set of samples from the input data.
  virtual void create_shape(const std::vector<std::size_t> &indices) {
    const Point_3 p0 = this->point(indices[0]);
    const Point_3 p1 = this->point(indices[1]);
    const Point_3 p2 = this->point(indices[2]);

    m_normal = CGAL::cross_product(p1 - p0, p2 - p0);
    const FT length = CGAL::sqrt(m_normal.squared_length());
    // Are the points almost singular?
    if (length <= FT(0.0))
      return;
    m_normal /= length;

    // check deviation of the 3 normal
    for (std::size_t i = 0; i < 3; ++i)
      if (this->normal(indices[i]) * m_normal < this->m_normal_threshold)
        return;
    // check snap to vertical
    if (m_normal.z() < this->m_normal_threshold)
      return;
    m_normal = Vector_3(0.0, 0.0, 1.0);

    m_point_on_primitive = CGAL::ORIGIN + ((p0 - CGAL::ORIGIN)
      + (p1 - CGAL::ORIGIN)
      + (p2 - CGAL::ORIGIN)) / FT(3.0);

    m_base1 = CGAL::cross_product(p1 - p0, m_normal);
    m_base1 /= CGAL::sqrt(m_base1.squared_length());

    m_base2 = CGAL::cross_product(m_base1, m_normal);
    m_base2 /= CGAL::sqrt(m_base2.squared_length());

    this->m_is_valid = true;
  }

  // Computes squared Euclidean distance from a set of points.
  virtual void squared_distance(
    const std::vector<std::size_t> &indices,
    std::vector<FT> &dists) const {
    for (std::size_t i = 0; i < indices.size(); i++) {
      const FT sd = (this->point(indices[i]) - m_point_on_primitive) * m_normal;
      dists[i] = sd * sd;
    }
  }

  // Computes the normal deviation between shape and a set of points with normals.
  // signed cosine, not absolute
  virtual void cos_to_normal(
    const std::vector<std::size_t> &indices,
    std::vector<FT> &angles) const {
    for (std::size_t i = 0; i < indices.size(); i++)
      angles[i] = this->normal(indices[i]) * m_normal;
  }

  // Returns the number of required samples for construction.
  virtual std::size_t minimum_sample_size() const {
    return 3;
  }

  virtual void parameters(
    const std::vector<std::size_t> &indices,
    std::vector<std::pair<FT, FT> > &parameterSpace,
    FT &, FT min[2], FT max[2]) const {
    // Transform first point before to initialize min/max
    Vector_3 p = m_point_on_primitive - this->point(indices[0]);
    FT u = p * m_base1;
    FT v = p * m_base2;
    parameterSpace[0] = std::pair<FT, FT>(u, v);
    min[0] = max[0] = u;
    min[1] = max[1] = v;

    for (std::size_t i = 1; i < indices.size(); ++i) {
      p = m_point_on_primitive - this->point(indices[i]);
      u = p * m_base1;
      v = p * m_base2;
      min[0] = (CGAL::min)(min[0], u);
      max[0] = (CGAL::max)(max[0], u);
      min[1] = (CGAL::min)(min[1], v);
      max[1] = (CGAL::max)(max[1], v);
      parameterSpace[i] = std::pair<FT, FT>(u, v);
    }
  }

  // if the shape is developable and have parameters function
  // set this function to return true to use the internal
  // planar parameterization to detect the connected component
  virtual bool supports_connected_component() const {
    return true;
  };

private:
  Point_3 m_point_on_primitive;
  Vector_3 m_normal;

  // plane base for parameterization
  Vector_3 m_base1, m_base2;
};

} // Shape_detection_3
} // CGAL

#endif // CGAL_SHAPE_DETECTION_3_HORIZONTAL_PLANE_H
