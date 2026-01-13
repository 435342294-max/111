#ifndef CM_RPLY_LOADER_H
#define CM_RPLY_LOADER_H

#include <string>
#include <vector>
#include <sstream>
#include <iterator>

#include <rply-1.1.4/rply.h>

namespace cm {

 
inline int vertex_cb(p_ply_argument argument) {
  static long i = 0;
  if (argument == NULL) {
    i = 0;
    return 1;
  }
  void *pdata = nullptr;
  ply_get_argument_user_data(argument, &pdata, nullptr);
  ((float *)pdata)[i++] = float(ply_get_argument_value(argument));
  return 1;
}

inline int face_cb(p_ply_argument argument) {
  static long i = 0;
  if (argument == NULL) {
    i = 0;
    return 1;
  }
  void *pdata = nullptr;
  ply_get_argument_user_data(argument, &pdata, nullptr);
  long length = 0, value_index = 0;
  ply_get_argument_property(argument, nullptr, &length, &value_index);
  // triangle surface
  if (length != 3)
    throw std::runtime_error("Not triangle face.");
  else if (value_index >= 0)
    ((unsigned int *)pdata)[i++] = (unsigned int)ply_get_argument_value(argument);
  return 1;
}

inline int texture_cb(p_ply_argument argument) {
  static long i = 0;
  void *pdata = nullptr;
  ply_get_argument_user_data(argument, &pdata, nullptr);
  long length = 0, value_index = 0;
  ply_get_argument_property(argument, nullptr, &length, &value_index);
  if (length != 6)
    throw std::runtime_error("Invalid face texture coordinates.");
  else if (value_index >= 0)
    ((float *)pdata)[i++] = float(ply_get_argument_value(argument));
  return 1;
}

inline int texture_number_cb(p_ply_argument argument) {
  static long i = 0;
  void *pdata = nullptr;
  ply_get_argument_user_data(argument, &pdata, nullptr);
  ((unsigned int *)pdata)[i++] = (unsigned int)ply_get_argument_value(argument);
  return 1;
}

/*!
 * \brief Wrapper for RPLY model loader.
 * Expected properties:
 *   TextureFile: texture file path, optional,
 *   vertex: float x, float y, float z,
 *   face: [unsigned] int vertex_indices, float texcoord, [[unsigned] int texnumber].
 */
class Rply_loader {
public:
  // number of vertices
  unsigned int num_vertices;
  // number of faces
  unsigned int num_faces;
  // vertices, with size of num_vertices * 3
  std::vector<float> vertices;
  // faces, with size of num_faces * 3
  std::vector<unsigned int> faces;
  // texture coordinates, with size of num_faces * 6
  std::vector<float> texcoords;
  // texture number, with size of num_faces
  std::vector<unsigned int> texnumber;
  // texture files
  std::vector<std::string> textures;

  Rply_loader() {}
  Rply_loader(const std::string &path) : num_faces(0), num_vertices(0) {
    // parse header
    p_ply ply = ply_open(path.c_str(), nullptr, 0, nullptr);
    if (!ply)
      throw std::runtime_error("Failed to open file: " + path + ".");
    if (!ply_read_header(ply))
      throw std::runtime_error("Failed to read header: " + path + ".");
    const char *comment = nullptr;
    // texture file
    while (comment = ply_get_next_comment(ply, comment)) {
      std::istringstream iss(comment);
      std::vector<std::string> results(
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>());
      if (results.size() == 2 && results[0] == "TextureFile")
        textures.push_back(results[1]);
    }
    // vertices and faces
    p_ply_element p_ele = nullptr;
    while (p_ele = ply_get_next_element(ply, p_ele)) {
      const char *name = nullptr;
      long count = 0;
      ply_get_element_info(p_ele, &name, &count);
      std::string str(name);
      if (str == "vertex")
        num_vertices = (unsigned int)count;
      else if (str == "face")
        num_faces = (unsigned int)count;
    }
    // read mesh
    vertices = std::vector<float>(num_vertices * 3, 0.0f);
    faces = std::vector<unsigned int>(num_faces * 3, 0);
    if (!ply_set_read_cb(ply, "vertex", "x", vertex_cb, vertices.data(), 0))
      throw std::runtime_error("Failed to set callback vertex::x.");
    if (!ply_set_read_cb(ply, "vertex", "y", vertex_cb, vertices.data(), 0))
      throw std::runtime_error("Failed to set callback vertex::y.");
    if (!ply_set_read_cb(ply, "vertex", "z", vertex_cb, vertices.data(), 0))
      throw std::runtime_error("Failed to set callback vertex::z.");
    if (!ply_set_read_cb(ply, "face", "vertex_indices", face_cb, faces.data(), 0))
      throw std::runtime_error("Failed to set callback face::vertex_indices.");
    if (!textures.empty()) {
      // have texture
      texcoords = std::vector<float>(num_faces * 6, 0.0f);
      if (!ply_set_read_cb(ply, "face", "texcoord", texture_cb, texcoords.data(), 0))
        throw std::runtime_error("Failed to set callback face::texcoord.");
    }
    if (textures.size() > 1) {
      // have texture number
      texnumber = std::vector<unsigned int>(num_faces, 0);
      if (!ply_set_read_cb(ply, "face", "texnumber", texture_number_cb, texnumber.data(), 0))
        throw std::runtime_error("Failed to set callback face::texnumber.");
    }
    if (!ply_read(ply))
      throw std::runtime_error("Failed to read file.");
    ply_close(ply);
    // reset local static variable
    vertex_cb(NULL);
    face_cb(NULL);
  }
  
  // write to .ply file
  bool writeply(const char* fname, int mode = 0) {
    p_ply oply = ply_create(fname, static_cast<e_ply_storage_mode>(mode), NULL, 0 , NULL);
    int tex_number = textures.size();
    while (tex_number--) { // generate comment
      const char* com = ("TextureFile " + textures[textures.size() - tex_number - 1]).c_str();
      if (!ply_add_comment(oply, com)) return 0;
    }
    // generate vertex element
    if (!ply_add_element(oply, "vertex", num_vertices)) return 0;
    if (!ply_add_scalar_property(oply, "x", PLY_FLOAT32)) return 0;
    if (!ply_add_scalar_property(oply, "y", PLY_FLOAT32)) return 0;
    if (!ply_add_scalar_property(oply, "z", PLY_FLOAT32)) return 0;
    // generate face element
    if (!ply_add_element(oply, "face", num_faces)) return 0;
    if (!ply_add_list_property(oply, "vertex_indices", PLY_UCHAR, PLY_INT)) return 0;
    if (texnumber.size()) {
      if (!ply_add_list_property(oply, "texcoord", PLY_UCHAR, PLY_FLOAT32)) return 0;
      if (!ply_add_scalar_property(oply, "texnumber", PLY_INT)) return 0;
    }
    if (!ply_write_header(oply)) return 0;
    // write vertices
    for (int i = 0; i < num_vertices; i++) {
      if (!ply_write(oply, vertices[3*i])) return 0; 
      if (!ply_write(oply, vertices[3 * i + 1])) return 0;
      if (!ply_write(oply, vertices[3 * i + 2])) return 0;
    }
    // write faces and textures
    for (int i = 0; i < num_faces; i++) {
      if (!ply_write(oply, static_cast<int>(3))) return 0;
      if (!ply_write(oply, faces[3*i])) return 0;
      if (!ply_write(oply, faces[3*i+1])) return 0;
      if (!ply_write(oply, faces[3*i+2])) return 0;
      if (texnumber.size()) {
        if (!ply_write(oply, static_cast<int>(6))) return 0;
        if (!ply_write(oply, texcoords[6 * i]) || !ply_write(oply, texcoords[6 * i + 1])) return 0;
        if (!ply_write(oply, texcoords[6 * i + 2]) || !ply_write(oply, texcoords[6 * i + 3])) return 0;
        if (!ply_write(oply, texcoords[6 * i + 4]) || !ply_write(oply, texcoords[6 * i + 5])) return 0;
        if (!ply_write(oply, texnumber[i])) return 0;
      }
    }
    if (!ply_close(oply)) return 0;
    return 1;
  }
}; // Rply_loader

} // namespace cm

#endif // CM_RPLY_LOADER_H
