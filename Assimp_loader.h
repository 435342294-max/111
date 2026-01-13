#ifndef CM_ASSIMP_LOADER_H
#define CM_ASSIMP_LOADER_H

#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace cm {

/*!
 * \brief Wrapper for Assimp model loader.
 * We do not care about the model node structure, so it is flatten.
 */
class Assimp_loader {
public:
  /*!
   * \brief Mesh struct with primitive data types.
   */
  struct Mesh {
    // number of vertices
    unsigned int num_vertices;
    // number of faces
    unsigned int num_faces;
    // vertices, with size of num_vertices * 3
    std::vector<float> vertices;
    // faces, with size of num_faces * 3
    std::vector<unsigned int> faces;
    // vertex texture coordinates, with size of num_vertices * 2
    std::vector<float> texcoords;
    // diffuse textures
    std::vector<std::string> textures;
  };

  // all meshes in the file
  std::vector<Mesh> meshes;

  /*!
   * \brief Constructor
   * \param path filepath to a 3D model
   */
  Assimp_loader(const std::string &path) {
    Assimp::Importer importer;
    importer.SetExtraVerbose(true);
    importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
      aiComponent_TANGENTS_AND_BITANGENTS
      | aiComponent_COLORS
      | aiComponent_BONEWEIGHTS
      | aiComponent_ANIMATIONS
      | aiComponent_TEXTURES
      | aiComponent_LIGHTS
      | aiComponent_CAMERAS
    );
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE,
      aiPrimitiveType_POINT
      | aiPrimitiveType_LINE
    );

    const unsigned int pflags =
      aiProcess_Triangulate
      // | aiProcess_GenNormals
      | aiProcess_SortByPType
      | aiProcess_OptimizeMeshes
      | aiProcess_OptimizeGraph
      // | aiProcess_FlipUVs
      ;
    if (!importer.ValidateFlags(pflags))
      throw std::runtime_error("Unsupported post-processing flags.");
    const aiScene *scene = importer.ReadFile(path, pflags);
    if (!scene)
      throw std::runtime_error(importer.GetErrorString());

    for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
      meshes.push_back(load_mesh(scene, scene->mMeshes[i]));
  }

private:
  /*!
   * \brief Loads mesh.
   * \param mesh assimp mesh
   * \return raw mesh data
   */
  Mesh load_mesh(const aiScene *scene, const aiMesh *mesh) {
    std::vector<float> vertices(mesh->mNumVertices * 3, 0.0f);
    std::vector<float> texcoords;
    if (mesh->HasTextureCoords(0))
      texcoords = std::vector<float>(mesh->mNumVertices * 2, 0.0f);
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      vertices[i * 3] = mesh->mVertices[i].x;
      vertices[i * 3 + 1] = mesh->mVertices[i].y;
      vertices[i * 3 + 2] = mesh->mVertices[i].z;
      // we always take the first texture coordinates set (0)
      if (mesh->HasTextureCoords(0)) {
        texcoords[i * 2] = mesh->mTextureCoords[0][i].x;
        texcoords[i * 2 + 1] = mesh->mTextureCoords[0][i].y;
      }
    }
    std::vector<unsigned int> faces(mesh->mNumFaces * 3, 0);
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      const aiFace &face = mesh->mFaces[i];
      if (face.mNumIndices != 3)
        throw std::runtime_error("Not triangle mesh.");
      faces[i * 3] = face.mIndices[0];
      faces[i * 3 + 1] = face.mIndices[1];
      faces[i * 3 + 2] = face.mIndices[2];
    }
    // diffuse textures from material
    std::vector<std::string> textures;
    const aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    for (unsigned int i = 0; i < material->GetTextureCount(aiTextureType_DIFFUSE); ++i) {
      aiString path;
      material->GetTexture(aiTextureType_DIFFUSE, i, &path);
      textures.push_back(path.C_Str());
    }

    return {
      mesh->mNumVertices,
      mesh->mNumFaces,
      vertices,
      faces,
      texcoords,
      textures
    };
  }
}; // Assimp_loader

} // namespace cm

#endif // CM_ASSIMP_LOADER_H
