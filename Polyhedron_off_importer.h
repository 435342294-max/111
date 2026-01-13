#ifndef Polyhedron_OFF_IMPORTER_H
#define Polyhedron_OFF_IMPORTER_H

#include <vector>
#include <sstream>
#include <fstream>

#include <CGAL/Modifier_base.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>

namespace cm {

/*!
 * \brief OFF Importer for (enriched) Polyhedron
 * Loads vertex color if available
 * Structure:
 * [C]OFF (for simple OFF or vertex color OFF)
 * # comment lines (unlimited)
 * num_vert num_faces num_edges(0)
 * x y z r g b a
 * ...
 * n index0 ... indexn
 * ...
 */
template<typename HDS>
class Polyhedron_off_importer : public CGAL::Modifier_base<HDS> {
  typedef typename HDS::Vertex Vertex;
  typedef typename Vertex::Point Point;
  typedef typename HDS::Vertex_handle Vertex_handle;
public:
  Polyhedron_off_importer(const std::string &fname) : m_fname(fname) {}

  void operator()(HDS& hds) {
    std::ifstream file(m_fname, std::ios::in);
    if (!file.is_open()) {
      return;
    }

    std::string s;

    // comments
    std::getline(file, s);
    while (s[0] == '#')
      std::getline(file, s);

    // name
    bool color = false;
    bool normal = false; // MT
    if (s.compare(0, 4, "COFF") == 0)
      color = true;
    else if (s.compare(0, 4, "NOFF") == 0)
      normal = true;
    else if ((s.compare(0, 5, "NCOFF") == 0) || (s.compare(0, 5, "CNOFF") == 0)) {
      normal = true; 
      color = true; 
    } // MT

    // comments
    file >> s;
    while (s[0] == '#') {
      std::getline(file, s);
      file >> s;
    }

    // header
    stringstream sstream(s);
    sstream >> num_vert;
    file >> num_faces;
    file >> s;

    unsigned int num_vert = 0;
    unsigned int num_faces = 0;
    std::vector<std::vector<float>> Vertices_position;
    std::vector<std::vector<float>> Vertices_color;
    std::vector<std::vector<int>> Facets;

    CGAL::Polyhedron_incremental_builder_3<HDS> builder(hds, true);

    builder.begin_surface((int)num_vert, (int)num_faces);

    float temp_coord[3];
    float temp_color[3];

    float Max_color_value = 0;

    bool Is_color_integer = false;
    // geom + color
    for (unsigned int i = 0; i < num_vert; ++i) {
      std::vector<float> Vert_coord;
      std::vector<float> Vert_color;

      file >> temp_coord[0];
      file >> temp_coord[1];
      file >> temp_coord[2];

      for (unsigned j = 0 ; j < 3; ++j)
        Vert_coord.push_back(temp_coord[j]);

      Vertices_position.push_back(Vert_coord);

      if (normal) { // MT : on ignore les normales
        file >> s;
        file >> s;
        file >> s;
      }

      if (color) {
        file >> temp_color[0];
        file >> temp_color[1];
        file >> temp_color[2];
        for (unsigned j = 0 ; j < 3; ++j)
          Vert_color.push_back(temp_color[j]);

        Vertices_color.push_back(Vert_color);

        for (unsigned j = 0; j < 3; ++j) {
          if (temp_color[j] > Max_color_value)
            Max_color_value = temp_color[j];
        }
      }
    }

    // Color value can be integer between [0; 255]
    // or float between [0.0 ; 1.0]

    // if max of color value > 1.0, we consider that color values are integers;
    if (Max_color_value > 2.0)
      Is_color_integer = true;
    else
      Is_color_integer = false;

    // connectivity
    for (unsigned int i = 0; i < num_faces; ++i) {
      unsigned int face_size;
      unsigned int index;
      std::vector<int> vect_index;

      file >> face_size;

      for (unsigned int j = 0; j < face_size; ++j) {
        file >> index;
        vect_index.push_back(index);
      }

      Facets.push_back(vect_index);
    }

    file.close();

    // construction
    for (unsigned int i = 0; i < num_vert; ++i) {
      Vertex_handle vertex = builder.add_vertex(Point(Vertices_position[i][0], Vertices_position[i][1], Vertices_position[i][2]));

      if (color) {
        if (Is_color_integer) {
          int RGB[3];
          RGB[0] = (int)floor(Vertices_color[i][0] + 0.5);
          RGB[1] = (int)floor(Vertices_color[i][1] + 0.5);
          RGB[2] = (int)floor(Vertices_color[i][2] + 0.5);

          vertex->color((float)RGB[0] / 255.0, (float)RGB[1] / 255.0, (float)RGB[2] / 255.0);
        }
        else
          vertex->color(Vertices_color[i][0], Vertices_color[i][1], Vertices_color[i][2]);
      }
    }

    // connectivity
    for (unsigned int i = 0; i < num_faces; ++i) {
      builder.begin_facet();

      for (unsigned int j = 0; j < Facets[i].size(); ++j) {
        builder.add_vertex_to_facet(Facets[i][j]);
      }

      builder.end_facet();
    }
    builder.end_surface();

    if (builder.check_unconnected_vertices()) {
      builder.remove_unconnected_vertices();
    }
  }

private:
  const std::string m_fname;
};

} // namespace cm

#endif // Polyhedron_OFF_IMPORTER_H
