#ifndef CM_MODEL_LOADER_H
#define CM_MODEL_LOADER_H

#include <vector>
#include <string>

#include "Assimp_loader.h"
#include "Rply_loader.h"
#include "Mesh_3.h"

namespace cm {

/*!
 * \brief Helper class to convert Rply and Assimp loaded data into Mesh_3.
 */
class Model_loader {
public:
  // texture coordinate
  struct Texcoord {
    float u;
    float v;
  };
  typedef Mesh_3::Property_map<vertex_descriptor, Texcoord> Texcoord_map;

  // mesh data
  std::vector<Mesh_3> meshes;
  std::vector<std::string> textures;

  /*!
   * \brief Constructor, load model meshes from file.
   * \param file_name input file path
   */
  Model_loader(const std::string &file_name) {
    const bool is_ply = file_name.substr(file_name.find_last_of('.')) == ".ply";
    try {
      if (is_ply)
        load_rply(file_name);
      else
        load_assimp(file_name);
    }
    catch (std::exception &e) {
      throw std::runtime_error(
        std::string(is_ply ? "RPLY::" : "ASSIMP::") + e.what());
    }
  }

private:
  /*!
   * \brief Loads a ply file.
   * \note Assimp can not loads ply with multiple textures properly 
   * \param file_name input file name
   */
  void load_rply(const std::string &file_name) {
    // split mesh according to different texture
    Rply_loader ply(file_name);
    const unsigned int num_meshes =
      ply.textures.empty() ? 1 : (unsigned int)ply.textures.size();

    // number of faces
    std::vector<unsigned int> num_faces(num_meshes);
    if (num_meshes == 1)
      num_faces[0] = ply.num_faces;
    else {
      for (auto &nf : num_faces)
        nf = 0;
      for (unsigned int i = 0; i < ply.num_faces; ++i)
        num_faces[ply.texnumber[i]]++;
    }
    // reserve (estimated) memory
    meshes = std::vector<Mesh_3>(num_meshes);
    for (unsigned int i = 0; i < num_meshes; ++i)
      meshes[i].reserve(num_faces[i] * 3, num_faces[i] * 3 / 2, num_faces[i]);

    // geometry
    std::vector<vertex_descriptor> vdes(ply.num_vertices, Mesh_3::null_vertex());
    for (unsigned int i = 0; i < ply.num_faces; ++i) {
      const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
      vertex_descriptor face[3];
      for (unsigned int j = 0; j < 3; ++j) {
        const unsigned int fvidx = i * 3 + j;
        const unsigned int vidx = ply.faces[fvidx];
        if (vdes[vidx] == Mesh_3::null_vertex())
          vdes[vidx] = meshes[midx].add_vertex({
            ply.vertices[vidx * 3],
            ply.vertices[vidx * 3 + 1],
            ply.vertices[vidx * 3 + 2]
          });
        face[j] = vdes[vidx];
      }
      meshes[midx].add_face(face[0], face[1], face[2]);
    }

    if (ply.textures.empty())
      return;

    // texture
    const std::string dir = file_name.substr(0, file_name.find_last_of("\\/") + 1);
    textures = std::vector<std::string>(num_meshes);
    for (unsigned int i = 0; i < num_meshes; ++i)
      textures[i] = dir + ply.textures[i];

    std::vector<Texcoord_map> tex_maps;
    for (auto &mesh : meshes)
      tex_maps.push_back(
        mesh.add_property_map<vertex_descriptor, Texcoord>(
          "v:texcord", { 0.0f, 0.0f }).first);

    for (unsigned int i = 0; i < ply.num_faces; ++i) {
      const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
      for (unsigned int j = 0; j < 3; ++j) {
        const unsigned int fvidx = i * 3 + j;
        const auto vd = vdes[ply.faces[fvidx]];
        tex_maps[midx][vd] = {
          ply.texcoords[fvidx * 2],
          ply.texcoords[fvidx * 2 + 1] };
      }
    }
  }

  /*!
   * \brief Loads other format using Assimp.
   * \param file_name input file name
   */
  void load_assimp(const std::string &file_name) {
    Assimp_loader assimp(file_name);
    const unsigned int num_meshes = (unsigned int)assimp.meshes.size();

    // texture
    const std::string dir = file_name.substr(0, file_name.find_last_of("\\/") + 1);
    textures = std::vector<std::string>();
    for (const auto &m : assimp.meshes)
      if (!m.textures.empty())
        textures.push_back(dir + m.textures[0]);
    if (textures.size() != assimp.meshes.size())
      textures.clear();
    const bool has_texture = !textures.empty();

    // geometry
    meshes = std::vector<Mesh_3>(num_meshes);
    std::vector<Texcoord_map> tex_maps;
    if (has_texture) {
      for (auto &mesh : meshes)
        tex_maps.push_back(
          mesh.add_property_map<vertex_descriptor, Texcoord>(
            "v:texcord", { 0.0f, 0.0f }).first);
    }
    for (unsigned int midx = 0; midx < num_meshes; ++midx) {
      const auto &m = assimp.meshes[midx];
      auto &mesh = meshes[midx];
      mesh.reserve(m.num_vertices, m.num_faces * 3 / 2, m.num_faces);
      std::vector<vertex_descriptor> vdes(m.num_vertices, Mesh_3::null_vertex());
      for (unsigned int i = 0; i < m.num_vertices; ++i)
        vdes[i] = mesh.add_vertex({
          m.vertices[i * 3],
          m.vertices[i * 3 + 1],
          m.vertices[i * 3 + 2] });
      for (unsigned int i = 0; i < m.num_faces; ++i) {
        vertex_descriptor face[3];
        for (unsigned int j = 0; j < 3; ++j) {
          const unsigned int vidx = m.faces[i * 3 + j];
          face[j] = vdes[vidx];
          if (has_texture)
            tex_maps[midx][vdes[vidx]] = {
              m.texcoords[vidx * 2],
              m.texcoords[vidx * 2 + 1] };
        }
        mesh.add_face(face[0], face[1], face[2]);
      }
    }
  }
}; // Model_loader

} // namespace cm

#endif // CM_MODEL_LOADER_H
