#ifndef MESH_3_H
#define MESH_3_H

#include <CGAL/Surface_mesh.h>
#include "Kernel.h"

typedef CGAL::Surface_mesh<Point_3> Mesh_3;

// common typedefs
typedef boost::graph_traits<Mesh_3>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Mesh_3>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<Mesh_3>::edge_descriptor edge_descriptor;
typedef boost::graph_traits<Mesh_3>::face_descriptor face_descriptor;

#endif // MESH_3_H
