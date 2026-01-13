#include <iostream>
#include <fstream>

#include <osg/Geode>
#include <osg/Geometry>

#include "utils/View_helper.h"
#include "utils/Mesh_3.h"
#include "utils/Manhattan_detector.h"

typedef cm::Manhattan_detector<Vector_3> Manhattan_detector;

osg::Node *mesh_to_geometry(const Mesh_3 &m)
{
  auto nmap = m.property_map<face_descriptor, Vector_3>("f:normal").first;

  // mesh face geometry
  osg::Vec3Array *vertices = new osg::Vec3Array;
  osg::Vec3Array *normals = new osg::Vec3Array;
  for (const auto &f : m.faces()) {
    const auto he = m.halfedge(f);
    vertices->push_back(cm::to_vec3(m.point(m.source(he))));
    vertices->push_back(cm::to_vec3(m.point(m.target(he))));
    vertices->push_back(cm::to_vec3(m.point(m.target(m.next(he)))));
    normals->push_back(cm::to_vec3(nmap[f]));
    normals->push_back(cm::to_vec3(nmap[f]));
    normals->push_back(cm::to_vec3(nmap[f]));
  }

  osg::Vec4Array *colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(0.7f, 0.7f, 0.7f, 0.6f));

  osg::Geometry *geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  // geometry->setUseVertexBufferObjects(true);
  geometry->setVertexArray(vertices);
  geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
  geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, vertices->size()));

  return geometry;
}

osg::Node *manhattan_to_geometry(const CGAL::Bbox_3 &bbox, const Manhattan_detector &mc)
{
  const double len = (bbox.xmax() - bbox.xmin()) / 2.0;
  const osg::Vec3 center((bbox.xmin() + bbox.xmax()) / 2.0,
    (bbox.ymin() + bbox.ymax()) / 2.0, (bbox.zmin() + bbox.zmax()) / 2.0);
  const auto &dirs = mc.directions();

  auto vertices = new osg::Vec3Array;
  vertices->push_back(center);
  vertices->push_back(center + cm::to_vec3(dirs[0]) * len);
  vertices->push_back(center);
  vertices->push_back(center + cm::to_vec3(dirs[1]) * len);
  vertices->push_back(center);
  vertices->push_back(center + cm::to_vec3(dirs[2]) * len);
  auto colors = new osg::Vec3Array;
  colors->push_back({ 1.0f, 0.0f, 0.0f });
  colors->push_back({ 1.0f, 0.0f, 0.0f });
  colors->push_back({ 0.0f, 1.0f, 0.0f });
  colors->push_back({ 0.0f, 1.0f, 0.0f });
  colors->push_back({ 0.0f, 0.0f, 1.0f });
  colors->push_back({ 0.0f, 0.0f, 1.0f });

  auto geometry = new osg::Geometry;
  geometry->setUseDisplayList(true);
  geometry->setDataVariance(osg::Object::STATIC);
  // geometry->setUseVertexBufferObjects(true);
  geometry->setVertexArray(vertices);
  geometry->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
  geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, vertices->size()));

  return geometry;
}

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  // load surface
  std::ifstream ifs(argv[1]);
  if (!ifs.is_open())
    return 1;
  Mesh_3 sm;
  ifs >> sm;
  ifs.close();

  // compute bounding box
  CGAL::Bbox_3 bbox;
  for (const auto &v : sm.vertices())
    bbox += sm.point(v).bbox();

  // add normal property
  auto nmap = sm.add_property_map<face_descriptor, Vector_3>(
    "f:normal", CGAL::NULL_VECTOR).first;
  for (const auto &f : sm.faces()) {
    const halfedge_descriptor h = sm.halfedge(f);
    nmap[f] = CGAL::unit_normal(sm.point(sm.source(h)),
      sm.point(sm.target(h)), sm.point(sm.target(sm.next(h))));
  }

  // prepare view data
  cm::Viewer vm;
  vm.run();

  // add mesh to viewer
  vm.add_node(mesh_to_geometry(sm));

  std::vector<Vector_3> normals;
  normals.reserve(sm.number_of_faces());
  for (const auto &f : sm.faces())
    normals.push_back(nmap[f]);

  Manhattan_detector md;
  md.detect(normals,
    Vector_3(0.0, 0.0, 1.0), 0.9,
    Manhattan_detector::Support_mode::Inlier);

  // add Manhattan directions to viewer
  vm.add_node(manhattan_to_geometry(bbox, md));

  return 0;
}
