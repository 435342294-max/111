#ifndef POLYHEDRON_BUILDER_H
#define POLYHEDRON_BUILDER_H

#include <vector>
#include <CGAL/Polyhedron_incremental_builder_3.h>

namespace cm {

/*!
 * \brief Polyhedron incremental builder
 */
template<typename HDS>
class Polyhedron_builder : public CGAL::Modifier_base<HDS> {
  const std::vector<double> &coords;
  const std::vector<int> &tris;

public:
  Polyhedron_builder(
    const std::vector<double> &coords_,
    const std::vector<int> &tris_) :
    coords(coords_), tris(tris_) {}

  void operator() (HDS &hds) {
    typedef typename HDS::Vertex Vertex;
    typedef typename Vertex::Point Point;

    // create a cgal incremental builder
    CGAL::Polyhedron_incremental_builder_3<HDS> b(hds, true);
    b.begin_surface(coords.size() / 3, tris.size() / 3);

    // add the polyhedron vertices
    for (int i = 0; i < (int)coords.size(); i += 3)
      b.add_vertex(Point(coords[i], coords[i + 1], coords[i + 2]));

    // add the polyhedron triangles
    for (int i = 0; i < (int)tris.size(); i += 3) {
      b.begin_facet();
      b.add_vertex_to_facet(tris[i]);
      b.add_vertex_to_facet(tris[i + 1]);
      b.add_vertex_to_facet(tris[i + 2]);
      b.end_facet();
    }

    // finish up the surface
    b.end_surface();
  }
};

} // namespace cm

#endif // POLYHEDRON_BUILDER_H
