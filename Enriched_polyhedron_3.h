#ifndef ENRICHED_POLYHEDRON_3_H
#define ENRICHED_POLYHEDRON_3_H

#include <memory>
#include <CGAL/Polyhedron_3.h>
#include "Kernel.h"

namespace cm {

/*!
 * \brief Enriched face.
 */
template <typename Refs, typename T>
class Enriched_face : public CGAL::HalfedgeDS_face_base<Refs, T> {
protected:
  // tag
  int m_tag;

public:
  Enriched_face() {}

  // tag
  const int& tag() const { return m_tag; }
  int& tag() { return m_tag; }
  void tag(const int& t)  { m_tag = t; }
};

/*!
 * \brief Enriched halfedge.
 */
template <typename Refs,  typename Tvertex, typename Tface>
class Enriched_halfedge : public CGAL::HalfedgeDS_halfedge_base<Refs, Tprev, Tvertex, Tface> {
protected:
  // tag
  int m_tag;

public:
  Enriched_halfedge() {}

  // tag
  const int& tag() const { return m_tag;  }
  int& tag() { return m_tag;  }
  void tag(const int& t)  { m_tag = t; }
};

/*!
 * \brief Enriched vertex.
 */
template <typename Refs, typename T, typename P>
class Enriched_vertex : public CGAL::HalfedgeDS_vertex_base<Refs, T, P> {
protected:
  // tag
  int m_tag;

public:
  Enriched_vertex() {}

  // repeat mandatory constructors
  Enriched_vertex(const P& pt) : CGAL::HalfedgeDS_vertex_base<Refs, T, P>(pt) {}

  // tag
  int& tag() {  return m_tag; }
  const int& tag() const {  return m_tag; }
  void tag(const int& t)  { m_tag = t; }
};

/*!
 * \brief Enriched items.
 */
struct Enriched_items : public CGAL::Polyhedron_items_3 {
  // wrap vertex
  template <typename Refs, typename Traits>
  struct Vertex_wrapper {
    typedef typename Traits::Point_3  Point;
    typedef Enriched_vertex<Refs,
      CGAL::Tag_true,
      Point> Vertex;
  };

  // wrap face
  template <typename Refs, typename Traits>
  struct Face_wrapper {
    typedef Enriched_face<Refs,
       CGAL::Tag_true> Face;
  };

  // wrap halfedge
  template <typename Refs, typename Traits>
  struct Halfedge_wrapper {
    typedef Enriched_halfedge<Refs,
      CGAL::Tag_true,
      CGAL::Tag_true,
      CGAL::Tag_true> Halfedge;
  };
};

/*!
 * \brief Enriched polyhedron.
 * @tparam GeomTraits model of Kernel
 * @tparam Items model of Enriched_items
 */
template <typename GeomTraits, typename Items>
class Enriched_polyhedron_3 : public CGAL::Polyhedron_3<GeomTraits, Items> {
public:
  typedef typename GeomTraits::FT FT;
  typedef typename GeomTraits::Point_3 Point;
  typedef typename GeomTraits::Vector_3 Vector;
  typedef typename GeomTraits::Iso_cuboid_3 Iso_cuboid;
  typedef typename Enriched_polyhedron_3::Facet_handle Facet_handle;
  typedef typename Enriched_polyhedron_3::Vertex_handle Vertex_handle;
  typedef typename Enriched_polyhedron_3::Halfedge_handle Halfedge_handle;
  typedef typename Enriched_polyhedron_3::Facet_iterator Facet_iterator;
  typedef typename Enriched_polyhedron_3::Vertex_iterator Vertex_iterator;
  typedef typename Enriched_polyhedron_3::Halfedge_iterator Halfedge_iterator;
  typedef typename Enriched_polyhedron_3::Halfedge_around_vertex_circulator Halfedge_around_vertex_circulator;
  typedef typename Enriched_polyhedron_3::Halfedge_around_facet_circulator Halfedge_around_facet_circulator;
  typedef typename Enriched_polyhedron_3::Point_iterator Point_iterator;
  typedef typename Enriched_polyhedron_3::Edge_iterator Edge_iterator;
  typedef typename Enriched_polyhedron_3::HalfedgeDS HalfedgeDS;
  typedef typename HalfedgeDS::Face Facet;
  // typedef typename Facet::Normal_3 Normal;
  // typedef Aff_transformation_3<GeomTraits> Affine_transformation;

public:
  Enriched_polyhedron_3() {}

  // dummy bounding box
  Iso_cuboid bbox() const {
    return {};
  }
};

} // namespace cm

// common typedefs
typedef cm::Enriched_polyhedron_3<Kernel, cm::Enriched_items> Polyhedron;
typedef std::shared_ptr<Polyhedron> PolyhedronPtr;

typedef Polyhedron::Vertex_handle Vertex_handle;
typedef Polyhedron::Halfedge_handle Halfedge_handle;
typedef Polyhedron::Facet_handle Facet_handle;

// CGAL BGL graph_traits specialization
// Reference: CGAL/boost/graph/graph_traits_Polyhedron_3.h
namespace boost
{

template <typename Gt, typename I>
struct graph_traits<cm::Enriched_polyhedron_3<Gt, I>>
  : CGAL::HDS_graph_traits<cm::Enriched_polyhedron_3<Gt, I>> {
  typedef typename Gt::Point_3 vertex_property_type;
};

template<typename Gt, typename I>
struct graph_traits<cm::Enriched_polyhedron_3<Gt, I> const>
  : CGAL::HDS_graph_traits< cm::Enriched_polyhedron_3<Gt, I>> {};

} // namespace boost

#endif // ENRICHED_POLYHEDRON_3_H
