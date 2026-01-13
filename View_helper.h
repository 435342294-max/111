#ifndef CM_VIEW_HELPER_H
#define CM_VIEW_HELPER_H

#include <type_traits>
#include <vector>

#include <CGAL/Kernel_traits.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Kernel/global_functions.h>

#include <osg/Point>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PolygonMode>
#include <osg/PolygonOffset>
#include <osg/MatrixTransform>
#include <osgGA/EventVisitor>
#include <osgUtil/SmoothingVisitor>

#include "Viewer.h"

namespace cm {

/*!
 * \brief Converts a 3d CGAL vector / point to osg::Vec3
 * \tparam T type of `CGAL::Vector_3` or `CGAL::Point_3`
 * \param a input vector / point
 * \return converted osg::Vec3
 */
template <typename T, std::enable_if_t<std::is_same<
  typename CGAL::Ambient_dimension<T>::type, CGAL::Dimension_tag<3>>::value, int> = 0>
osg::Vec3 to_vec3(const T &a) {
  return { static_cast<float>(a.x()), static_cast<float>(a.y()), static_cast<float>(a.z()) };
}

/*!
 * \brief Converts a 2d CGAL vector / point to osg::Vec3
 * \tparam T type of `CGAL::Vector_2` or `CGAL::Point_2`
 * \param a input vector / point
 * \return converted osg::Vec3
 */
template <typename T, std::enable_if_t<std::is_same<
  typename CGAL::Ambient_dimension<T>::type, CGAL::Dimension_tag<2>>::value, int> = 0>
osg::Vec3 to_vec3(const T &a) {
  return { static_cast<float>(a.x()), static_cast<float>(a.y()), 0.0f };
}

/*!
 * \brief Switch node keyboard event callback handler.
 */
class Switch_callback : public osg::NodeCallback {
public:
  /*!
   * \brief Constructor.
   * \param keys child node to key binding
   */
  Switch_callback(const std::vector<int> &keys) : m_keys(keys) {}

  /*!
   * \brief Toggle node visibility.
   * \param node callback node
   * \param nv node visitor
   */
  virtual void operator()(osg::Node *node, osg::NodeVisitor *nv) {
    // switch node
    auto switch_node = node->asSwitch();
    const unsigned int num_children = switch_node->getNumChildren();
    // convert the node visitor to an osg::EventVisitor pointer
    const auto ev = nv->asEventVisitor();
    if (ev) {
      // handle events with the node
      for (const auto e : ev->getEvents()) {
        const auto ea = e->asGUIEventAdapter();
        if (ea->getEventType() == osgGA::GUIEventAdapter::KEYDOWN) {
          const int k = ea->getKey();
          for (unsigned int i = 0; i < m_keys.size() && i < num_children; ++i) {
            if (m_keys[i] == k) {
              switch_node->setValue(i, !switch_node->getValue(i));
              break;
            }
          }
        }
      }
    }
    traverse(node, nv);
  }

private:
  // child node to key binding
  const std::vector<int> m_keys;
};

/*!
 * \brief Construct a vertical arrow shape, the bottom is at origin.
 * \tparam N radial slices
 * \param rh head radius
 * \param rs shaft radius
 * \param lh head length
 * \param ls shaft length
 * \param c color
 * \return arrow shape
 */
template <int N = 16>
osg::Node *shape_arrow(
  const float rh,
  const float rs,
  const float lh,
  const float ls,
  const osg::Vec4 &c) {
  // step
  const float step = 2.0f * osg::PIf / N;
  // head cone, N + 2 points
  auto vertices = new osg::Vec3Array;
  vertices->push_back({0.0f, 0.0f, ls + lh});
  for (int i = 0; i < N + 1; ++i)
    vertices->push_back({rh * std::cos(step * i), rh * std::sin(step * i), ls});
  // shaft cylinder, (N + 1) * 2 points
  for (int i = 0; i < N + 1; ++i) {
    vertices->push_back({rs * std::cos(step * i), rs * std::sin(step * i), ls});
    vertices->push_back({rs * std::cos(step * i), rs * std::sin(step * i), 0.0f});
  }
  // color
  auto colors = new osg::Vec4Array;
  colors->push_back(c);

  auto arrow = new osg::Geometry;
  arrow->setUseDisplayList(true);
  arrow->setDataVariance(osg::Object::STATIC);
  arrow->setVertexArray(vertices);
  arrow->setColorArray(colors, osg::Array::BIND_OVERALL);
  arrow->addPrimitiveSet(
    new osg::DrawArrays(GL_TRIANGLE_FAN, 0, N + 2));
  arrow->addPrimitiveSet(
    new osg::DrawArrays(GL_TRIANGLE_STRIP, N + 2, (N + 1) * 2));
  osgUtil::SmoothingVisitor::smooth(*arrow);

  return arrow;
}

/*!
 * \brief Construct a 2d vertical arrow shape on the XOY plane, the bottom is at origin.
 * \param hw head width
 * \param sw shaft width
 * \param hl head length
 * \param sl shaft length
 * \param c color
 * \return arrow shape
 */
inline osg::Node *shape_arrow_2(
  const float hw,
  const float sw,
  const float hl,
  const float sl,
  const osg::Vec4 &c) {
  // 2d array polygon
  auto vertices = new osg::Vec3Array;
  vertices->push_back({0.0f, sl + hl, 0.0f});
  vertices->push_back({-hw, sl, 0.0f});
  vertices->push_back({-sw, sl, 0.0f});
  vertices->push_back({-sw, 0.0f, 0.0f});
  vertices->push_back({sw, 0.0f, 0.0f});
  vertices->push_back({sw, sl, 0.0f});
  vertices->push_back({hw, sl, 0.0f});
  // color
  auto colors = new osg::Vec4Array;
  colors->push_back(c);

  auto arrow = new osg::Geometry;
  arrow->setUseDisplayList(true);
  arrow->setDataVariance(osg::Object::STATIC);
  arrow->setVertexArray(vertices);
  arrow->setColorArray(colors, osg::Array::BIND_OVERALL);
  arrow->addPrimitiveSet(new osg::DrawArrays(GL_POLYGON, 0, vertices->size()));

  return arrow;
}

/*!
 * \brief Construct a xyz axes.
 * \param l axis length
 * \return axes shape
 */
inline osg::Node *shape_xyz_axes(const float l = 10.0f) {
  // axes
  const float rh = 0.2f;
  const float rs = 0.1f;
  const float lh = 1.0f;
  const float ls = l - lh;
  const auto xaxis = cm::shape_arrow<>(rh, rs, lh, ls, osg::RED);
  const auto yaxis = cm::shape_arrow<>(rh, rs, lh, ls, osg::GREEN);
  const auto zaxis = cm::shape_arrow<>(rh, rs, lh, ls, osg::BLUE);

  // root node
  auto root = new osg::Group;
  auto xtransform = new osg::MatrixTransform;
  xtransform->setMatrix(osg::Matrix::rotate(osg::Z_AXIS, osg::X_AXIS));
  xtransform->addChild(xaxis);
  auto ytransform = new osg::MatrixTransform;
  ytransform->setMatrix(osg::Matrix::rotate(osg::Z_AXIS, osg::Y_AXIS));
  ytransform->addChild(yaxis);
  root->addChild(xtransform);
  root->addChild(ytransform);
  root->addChild(zaxis);

  return root;
}

/*!
 * \brief Construct 2d xy axes.
 * \param l axis length
 * \return axes shape
 */
inline osg::Node *shape_xy_axes(const float l = 10.0f) {
  // axes
  const float hw = 0.2f;
  const float sw = 0.1f;
  const float hl = 1.0f;
  const float sl = l - hl;
  const auto xaxis = cm::shape_arrow_2(hw, sw, hl, sl, osg::RED);
  const auto yaxis = cm::shape_arrow_2(hw, sw, hl, sl, osg::GREEN);

  // root node
  auto root = new osg::Group;
  auto xtransform = new osg::MatrixTransform;
  xtransform->setMatrix(osg::Matrix::rotate(osg::Y_AXIS, osg::X_AXIS));
  xtransform->addChild(xaxis);
  root->addChild(xtransform);
  root->addChild(yaxis);

  return root;
}

/*!
 * \brief Utility function to convert a normal range to an osg node
 * \note the length of the normal is ignored
 * \tparam NormalRange a model of `Range`
 * \tparam NormalMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Vector_2` or `CGAL::Vector_3` as value type
 * \tparam PositionMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Point_3` or `CGAL::Point_2` as value type
 * \tparam ColorMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `osg::Vec4` as value type
 * \param nr input normal range
 * \param nmap input normal map
 * \param pmap input position map
 * \param cmap input color map
 * \return normal range osg node
 */
template <
  typename NormalRange,
  typename NormalMap,
  typename PositionMap,
  typename ColorMap>
osg::Node *normals_to_node(
  const NormalRange &nr,
  const NormalMap &nmap,
  const PositionMap &pmap,
  const ColorMap &cmap) {
  // dimension
  const int dimension = CGAL::Ambient_dimension<typename NormalMap::value_type>::value;
  const auto axis = dimension == 3 ? osg::Z_AXIS : osg::Y_AXIS;
  // each matrix transformation has its own unit normal shape
  auto root = new osg::Group;
  for (const auto &n : nr) {
    auto transform = new osg::MatrixTransform;
    transform->setMatrix(
      osg::Matrix::rotate(axis, cm::to_vec3(get(nmap, n))) *
      osg::Matrix::translate(cm::to_vec3(get(pmap, n))));
    const auto unit_normal = dimension == 3 ?
      cm::shape_arrow<>(0.015f, 0.01f, 0.1f, 0.9f, get(cmap, n)) :
      cm::shape_arrow_2(0.015f, 0.01f, 0.1f, 0.9f, get(cmap, n));
    transform->addChild(unit_normal);
    root->addChild(transform);
  }

  return root;
}

/*!
 * \brief Utility function to convert a normal range to an osg node
 * \note the length of the normal is ignored
 * \tparam NormalRange a model of `Range`
 * \tparam NormalMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Vector_2` or `CGAL::Vector_3` as value type
 * \tparam PositionMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Point_3` or `CGAL::Point_2` as value type
 * \param nr input normal range
 * \param nmap input normal map
 * \param pmap input position map
 * \param c color
 * \return normal range osg node
 */
template <
  typename NormalRange,
  typename NormalMap,
  typename PositionMap>
osg::Node *normals_to_node(
  const NormalRange &nr,
  const NormalMap &nmap,
  const PositionMap &pmap,
  const osg::Vec4 &c) {
  // dimension
  const int dimension = CGAL::Ambient_dimension<typename NormalMap::value_type>::value;
  const auto axis = dimension == 3 ? osg::Z_AXIS : osg::Y_AXIS;
  const auto unit_normal = dimension == 3 ?
    cm::shape_arrow<>(0.015f, 0.01f, 0.1f, 0.9f, c) :
    cm::shape_arrow_2(0.015f, 0.01f, 0.1f, 0.9f, c);
  // all matrix transformations sharing a unit normal shape
  auto root = new osg::Group;
  for (const auto &n : nr) {
    auto transform = new osg::MatrixTransform;
    transform->setMatrix(
      osg::Matrix::rotate(axis, cm::to_vec3(get(nmap, n))) *
      osg::Matrix::translate(cm::to_vec3(get(pmap, n))));
    transform->addChild(unit_normal);
    root->addChild(transform);
  }

  return root;
}

/*!
 * \brief Utility function to convert a normal range to an osg node
 * \tparam NormalRange a model of `Range`
 * \tparam NormalMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Vector_3` or `CGAL::Vector_2` as value type
 * \param nr input normal range
 * \param nmap input normal map
 * \param c color
 * \return normal range osg node
 */
template <typename NormalRange, typename NormalMap>
osg::Node *normals_to_node(
  const NormalRange &nr,
  const NormalMap &nmap,
  const osg::Vec4 &c) {
  // all normals at origin
  CGAL::Constant_property_map<
    typename NormalRange::iterator::value_type,
    typename CGAL::Kernel_traits<
      typename NormalMap::value_type>::Kernel::Point_3>
    origin_pmap(CGAL::ORIGIN);

  return normals_to_node(nr, nmap, origin_pmap, c);
}

/*!
 * \brief Utility function to convert a normal range to an osg node
 * \tparam NormalRange a model of `Range` with value type of `CGAL::Vector_3` or `CGAL::Vector_2`
 * \param nr input normal range
 * \param c color
 * \return normal range osg node
 */
template <typename NormalRange, std::enable_if_t<std::is_class<
  typename CGAL::Kernel_traits<
    typename NormalRange::iterator::value_type>::Kernel::Point_3>::value, int> = 0>
osg::Node *normals_to_node(const NormalRange &nr, const osg::Vec4 &c) {
  // call overload version
  CGAL::Identity_property_map<
    typename NormalRange::iterator::value_type> nmap;
  // all normal at origin
  CGAL::Constant_property_map<
    typename NormalRange::iterator::value_type,
    typename CGAL::Kernel_traits<
      typename NormalRange::iterator::value_type>::Kernel::Point_3>
    origin_pmap(CGAL::ORIGIN);
  return normals_to_node(nr, nmap, origin_pmap, c);
}

/*!
 * \brief Utility function to convert the point set normals to an osg node
 * \tparam PointSet a model of `CGAL::Point_set_3`
 * \param ps input point set, must have the normal property
 * \param c color
 * \return point set normal osg node
 */
template <typename PointSet, std::enable_if_t<std::is_class<
  typename PointSet::Index>::value, int> = 0>
osg::Node *normals_to_node(const PointSet &ps, const osg::Vec4 &c) {
  // call overload version
  return normals_to_node(ps, ps.normal_map(), ps.point_map(), c);
}

/*!
 * \brief Utility function to convert the point range to an osg node
 * \tparam PointRange a model of `Range`
 * \tparam PointMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Point_3` or `CGAL::Point_2` as value type
 * \tparam NormalMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Vector_3` or `CGAL::Vector_2` as value type
 * \param pr input point range
 * \param pmap input point map
 * \param nmap input normal map
 * \param size point size
 * \param c color
 * \return point range osg node
 */
template <
  typename PointRange,
  typename PointMap,
  typename NormalMap>
osg::Node *points_to_node(
  const PointRange &pr,
  const PointMap &pmap,
  const NormalMap &nmap,
  const osg::Vec4 &c,
  const float size) {
  // switch node
  auto switch_node = new osg::Switch;

  // point node
  {
    auto vertices = new osg::Vec3Array;
    auto normals = new osg::Vec3Array;
    for (const auto &p : pr) {
      vertices->push_back(cm::to_vec3(get(pmap, p)));
      normals->push_back(cm::to_vec3(get(nmap, p)));
    }
    auto colors = new osg::Vec4Array;
    colors->push_back(c);
    osg::Geometry *geometry = new osg::Geometry;
    geometry->setUseDisplayList(true);
    geometry->setDataVariance(osg::Object::STATIC);
    geometry->setVertexArray(vertices);
    geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
    geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
    geometry->getOrCreateStateSet()->setAttribute(
      new osg::Point(size), osg::StateAttribute::ON);
    switch_node->addChild(geometry);
  }

  // normal node
  {
    auto vertices = new osg::Vec3Array;
    for (const auto &p : pr) {
      vertices->push_back(cm::to_vec3(get(pmap, p)));
      vertices->push_back(cm::to_vec3(get(pmap, p)) + cm::to_vec3(get(nmap, p)));
    }
    auto colors = new osg::Vec4Array;
    colors->push_back(osg::BLUE);
    osg::Geometry *geometry = new osg::Geometry;
    geometry->setUseDisplayList(true);
    geometry->setDataVariance(osg::Object::STATIC);
    geometry->setVertexArray(vertices);
    geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, vertices->size()));
    geometry->getOrCreateStateSet()->setMode(
      GL_LIGHTING, osg::StateAttribute::OFF);
    switch_node->addChild(geometry);
  }

  // keyboard event call back
  switch_node->setEventCallback(new Switch_callback({'p', 'n'}));

  return switch_node;
}

/*!
 * \brief Utility function to convert the point range to an osg node
 * \tparam PointRange a model of `Range`
 * \tparam PointMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Point_3` or `CGAL::Point_2` as value type
 * \param pr input point range
 * \param pmap input point map
 * \param c color
 * \param size point size
 * \return point range osg node
 */
template <typename PointRange, typename PointMap>
osg::Node *points_to_node(
  const PointRange &pr,
  const PointMap &pmap,
  const osg::Vec4 &c,
  const float size) {
  // point node without normal
  auto vertices = new osg::Vec3Array;
  for (const auto &p : pr)
    vertices->push_back(cm::to_vec3(get(pmap, p)));
  auto colors = new osg::Vec4Array;
  colors->push_back(c);
  osg::Geometry *geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
  geometry->addPrimitiveSet(
    new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
  geometry->getOrCreateStateSet()->setAttribute(
    new osg::Point(size), osg::StateAttribute::ON);
  geometry->getOrCreateStateSet()->setMode(
    GL_LIGHTING, osg::StateAttribute::OFF);

  return geometry;
}

/*!
 * \brief Utility function to convert the point range to an osg node
 * \tparam PointRange a model of `Range` with value type of `CGAL::Point_3` or `CGAL::Point_2`
 * \param pr input point range
 * \param c color
 * \param size point size
 * \return point range osg node
 */
template <typename PointRange, std::enable_if_t<std::is_class<
  typename CGAL::Kernel_traits<
    typename PointRange::iterator::value_type>::Kernel::Point_3>::value, int> = 0>
osg::Node *points_to_node(
  const PointRange &pr,
  const osg::Vec4 &c,
  const float size = 3.0f) {
  // call overload version
  CGAL::Identity_property_map<
    typename PointRange::iterator::value_type> pmap;
  return points_to_node(pr, pmap, c, size);
}

/*!
 * \brief Utility function to convert the point set normals to an osg node
 * \tparam PointSet a model of `CGAL::Point_set_3`
 * \param ps input point set, must have the normal property
 * \param c color
 * \param size point size
 * \return point set osg node
 */
template <typename PointSet, std::enable_if_t<std::is_class<
  typename PointSet::Index>::value, int> = 0>
osg::Node *points_to_node(
  const PointSet &ps,
  const osg::Vec4 &c,
  const float size = 3.0f) {
  // call overload version
  if (ps.has_normal_map())
    return points_to_node(ps, ps.point_map(), ps.normal_map(), c, size);
  else
    return points_to_node(ps, ps.normal_map(), c, size);
}

/*!
 * \brief Utility function to convert a range of segments to an osg node
 * \tparam SegmentRange a model of `Range`
 * \tparam SegmentMap a model of `LvaluePropertyMap` with `Range::iterator::value_type`
 * as key type and `CGAL::Segment_3` or `CGAL::Segment_2` as value type
 * \param sr input segment range
 * \param smap input segment map
 * \param c color
 * \param with_endpoints set true to render endpoints, false otherwise
 * \return segment range osg node
 */
template <typename SegmentRange, typename SegmentMap>
osg::Node *segments_to_node(
  const SegmentRange &sr,
  const SegmentMap &smap,
  const osg::Vec4 &c,
  const bool with_endpoints) {
  // line segments
  auto vertices = new osg::Vec3Array;
  for (const auto &s : sr) {
    vertices->push_back(cm::to_vec3(get(smap, s).source()));
    vertices->push_back(cm::to_vec3(get(smap, s).target()));
  }
  auto colors = new osg::Vec4Array;
  colors->push_back(c);
  osg::Geometry *geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
  geometry->addPrimitiveSet(
    new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, vertices->size()));
  if (with_endpoints)
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
  geometry->getOrCreateStateSet()->setAttribute(
    new osg::Point(3.0f), osg::StateAttribute::ON);
  geometry->getOrCreateStateSet()->setMode(
    GL_LIGHTING, osg::StateAttribute::OFF);

  return geometry;
}

/*!
 * \brief Utility function to convert a segment range to an osg node
 * \tparam SegmentRange a model of `Range`
 * with value type of `CGAL::Segment_3` or `CGAL::Segment_2`
 * \param sr input segment range
 * \param c color
 * \param with_endpoints set true to render endpoints, false otherwise
 * \return segment range osg node
 */
template <typename SegmentRange>
osg::Node *segments_to_node(
  const SegmentRange &sr,
  const osg::Vec4 &c,
  const bool with_endpoints = true) {
  // call overload version
  CGAL::Identity_property_map<
    typename SegmentRange::iterator::value_type> smap;
  return segments_to_node(sr, smap, c, with_endpoints);
}

/*!
 * \brief Utility function to convert a polygon to an osg node
 * \tparam Polygon a model of `CGAL::Polygon_2`
 * \param poly input polygon
 * \param c color
 * \param with_endpoints set true to render endpoints, false otherwise
 * \return polygon osg node
 */
template <typename Polygon>
osg::Node *polygon_to_node(
  const Polygon &poly,
  const osg::Vec4 &c,
  const bool with_endpoints = true) {
  // line segments
  auto vertices = new osg::Vec3Array;
  for (const auto &p : poly)
    vertices->push_back(cm::to_vec3(p));
  auto colors = new osg::Vec4Array;
  colors->push_back(c);
  osg::Geometry *geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
  geometry->addPrimitiveSet(
    new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP, 0, vertices->size()));
  if (with_endpoints)
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
  geometry->getOrCreateStateSet()->setAttribute(
    new osg::Point(3.0f), osg::StateAttribute::ON);
  geometry->getOrCreateStateSet()->setMode(
    GL_LIGHTING, osg::StateAttribute::OFF);

  return geometry;
}

/*!
 * \brief Utility function to convert a ConstrainedTriangulation to an osg node
 * \tparam ConstrainedTriangulation a model of `CGAL::Triangulation_2`
 * \param ct input constrained triangulation
 * \param c color
 * \param with_endpoints set true to render endpoints, false otherwise
 * \return triangulation osg node
 * \todo all CGAL BGL shared implementation
 */
template <typename ConstrainedTriangulation>
osg::Node *constrained_triangulation_to_node(
  const ConstrainedTriangulation &ct,
  const osg::Vec4 &c,
  const bool with_endpoints = true) {
  // ConstrainedTriangulation has specialization of boost::graph_traits
  // Only finite simplices exist when viewed through the scope of these graph traits classes
  const auto vpmap = get(boost::vertex_point, const_cast<ConstrainedTriangulation &>(ct));
  auto vertices = new osg::Vec3Array;
  for (const auto e : edges(ct)) {
    const auto h = halfedge(e, ct);
    if (face(h, ct)->is_in_domain() || face(opposite(h, ct), ct)->is_in_domain()) {
      vertices->push_back(cm::to_vec3(vpmap[source(e, ct)]));
      vertices->push_back(cm::to_vec3(vpmap[target(e, ct)]));
    }
  }
  auto colors = new osg::Vec4Array;
  colors->push_back(c);

  auto geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
  geometry->addPrimitiveSet(
    new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, vertices->size()));
  if (with_endpoints)
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, vertices->size()));
  geometry->getOrCreateStateSet()->setAttribute(
    new osg::Point(3.0f), osg::StateAttribute::ON);
  geometry->getOrCreateStateSet()->setMode(
    GL_LIGHTING, osg::StateAttribute::OFF);

  return geometry;
}

/*!
 * \brief Utility function to convert a mesh to an osg node
 * \tparam TriangleMesh a model of `FaceListGraph`
 * \param tm input triangle mesh
 * \return mesh osg node
 */
template <typename TriangleMesh>
osg::Node *mesh_to_node(const TriangleMesh &tm) {
  // switch node
  auto switch_node = new osg::Switch;
  const auto vpmap = get(boost::vertex_point, const_cast<TriangleMesh &>(tm));
  {
    auto vertices = new osg::Vec3Array;
    auto normals = new osg::Vec3Array;
    for (const auto &f : faces(tm)) {
      const auto h = halfedge(f, tm);
      vertices->push_back(cm::to_vec3(vpmap[source(h, tm)]));
      vertices->push_back(cm::to_vec3(vpmap[target(h, tm)]));
      vertices->push_back(cm::to_vec3(vpmap[target(next(h, tm), tm)]));
      const auto n = cm::to_vec3(CGAL::unit_normal(
        vpmap[source(h, tm)],
        vpmap[target(h, tm)],
        vpmap[target(next(h, tm), tm)]));
      normals->push_back(n);
      normals->push_back(n);
      normals->push_back(n);
    }
    auto colors = new osg::Vec3Array;
    colors->push_back({ 0.8f, 0.8f, 0.8f });

    auto geometry = new osg::Geometry;
    geometry->setUseDisplayList(true);
    geometry->setDataVariance(osg::Object::STATIC);
    geometry->setVertexArray(vertices);
    geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
    geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, vertices->size()));
    auto pm = new osg::PolygonMode;
    pm->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
    geometry->getOrCreateStateSet()->setAttribute(pm);
    geometry->getOrCreateStateSet()->setAttributeAndModes(
      new osg::PolygonOffset(1.0f, 1.0f));

    switch_node->addChild(geometry);
  }

  // wireframe geometry
  {
    auto vertices = new osg::Vec3Array;
    for (const auto &e : edges(tm)) {
      vertices->push_back(cm::to_vec3(vpmap[source(e, tm)]));
      vertices->push_back(cm::to_vec3(vpmap[target(e, tm)]));
    }
    auto colors = new osg::Vec3Array;
    colors->push_back({ 0.0f, 0.0f, 0.0f });

    auto geometry = new osg::Geometry;
    geometry->setUseDisplayList(true);
    geometry->setDataVariance(osg::Object::STATIC);
    geometry->setVertexArray(vertices);
    geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
    geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, vertices->size()));
    geometry->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    switch_node->addChild(geometry);
  }
  switch_node->setEventCallback(new Switch_callback({'f', 'w'}));

  return switch_node;
}

/*!
 * \brief Get the global viewer.
 * Lazy-evaluated, correctly-destroyed and thread-safe.
 * \param row number of rows of sub view
 * \param column number of columns of sub view
 * \param width width of each sub view
 * \param height height of each sub view
 * \return the viewer
 */
inline Viewer &get_global_viewer(
  const std::size_t row = 1,
  const std::size_t column = 1,
  const std::size_t width = 1600,
  const std::size_t height = 900) {
  static Viewer viewer(row, column, width, height);
  return viewer;
}

} // namespace cm

#endif // CM_VIEW_HELPER_H
