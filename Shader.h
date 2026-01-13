#ifndef CM_SHADER_H
#define CM_SHADER_H

#include <string>
#include <fstream>
#include <sstream>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace cm {

/*!
 * \brief Shader class.
 */
class Shader {
  // shader id
  unsigned int m_id;

public:
  /*!
   * \brief Constructor.
   */
  Shader() : m_id(-1) {}

  /*!
   * \brief Loads shader from disk.
   * \param vs_path vertex shader path
   * \param fs_path fragment shader path
   * \param gs_path geometry shader path
   */
  void load_from_path(
    const std::string &vs_path,
    const std::string &fs_path,
    const std::string &gs_path = "") {
    // read shader file
    std::string vs_string;
    std::string fs_string;
    std::string gs_string;
    std::ifstream vs_file;
    std::ifstream fs_file;
    std::ifstream gs_file;
    vs_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fs_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    gs_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    vs_file.open(vs_path);
    std::stringstream vs_stream;
    vs_stream << vs_file.rdbuf();
    vs_file.close();
    vs_string = vs_stream.str();

    fs_file.open(fs_path);
    std::stringstream fs_stream;
    fs_stream << fs_file.rdbuf();
    fs_file.close();
    fs_string = fs_stream.str();

    if (!gs_path.empty()) {
      gs_file.open(gs_path);
      std::stringstream gs_stream;
      gs_stream << gs_file.rdbuf();
      gs_file.close();
      gs_string = gs_stream.str();
    }

    load_from_string(vs_string, fs_string, gs_string);
  }

  /*!
   * \brief Loads shader from string.
   * \param vs_string vertex shader string
   * \param fs_string fragment shader string
   * \param gs_string geometry shader string
   */
  void load_from_string(
    const std::string &vs_string,
    const std::string &fs_string,
    const std::string &gs_string = "") {
    // vertex shader
    const char *vs_code = vs_string.c_str();
    unsigned int vshader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vshader, 1, &vs_code, NULL);
    glCompileShader(vshader);
    check_compile_errors(vshader, "VERTEX");
    // fragment shader
    const char *fs_code = fs_string.c_str();
    unsigned int fshader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fshader, 1, &fs_code, NULL);
    glCompileShader(fshader);
    check_compile_errors(fshader, "FRAGMENT");
    // geometry shader
    unsigned int gshader = 0;
    if (!gs_string.empty()) {
      const char *gs_code = gs_string.c_str();
      gshader = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(gshader, 1, &gs_code, NULL);
      glCompileShader(gshader);
      check_compile_errors(gshader, "GEOMETRY");
    }

    // shader program
    m_id = glCreateProgram();
    glAttachShader(m_id, vshader);
    glAttachShader(m_id, fshader);
    if (!gs_string.empty())
      glAttachShader(m_id, gshader);
    glLinkProgram(m_id);
    check_compile_errors(m_id, "PROGRAM");

    // delete shaders
    glDeleteShader(vshader);
    glDeleteShader(fshader);
    if (!gs_string.empty())
      glDeleteShader(gshader);
  }

  /*!
   * \brief Shader id.
   * \return shader id
   */
  unsigned int id() const { return m_id; }

  /*!
   * \brief Use the shader
   */
  void use() const {
    glUseProgram(m_id);
  }

  /*!
   * \brief Set bool value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_bool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(m_id, name.c_str()), (int)value);
  }

  /*!
   * \brief Set int value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_int(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(m_id, name.c_str()), value);
  }

  /*!
   * \brief Set float value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_float(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(m_id, name.c_str()), value);
  }

  /*!
   * \brief Set vec2 value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_vec2(const std::string &name, const glm::vec2 &value) const {
    glUniform2fv(glGetUniformLocation(m_id, name.c_str()), 1, &value[0]);
  }

  /*!
   * \brief Set vec2 value in the shader.
   * \param name variable name
   * \param x variable first component
   * \param y variable second component
   */
  void set_vec2(const std::string &name, float x, float y) const {
    glUniform2f(glGetUniformLocation(m_id, name.c_str()), x, y);
  }

  /*!
   * \brief Set vec3 value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_vec3(const std::string &name, const glm::vec3 &value) const {
    glUniform3fv(glGetUniformLocation(m_id, name.c_str()), 1, &value[0]);
  }

  /*!
   * \brief Set vec3 value in the shader.
   * \param name variable name
   * \param x variable first component
   * \param y variable second component
   * \param y variable third component
   */
  void set_vec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(m_id, name.c_str()), x, y, z);
  }

  /*!
   * \brief Set vec4 value in the shader.
   * \param name variable name
   * \param value variable value
   */
  void set_vec4(const std::string &name, const glm::vec4 &value) const {
    glUniform4fv(glGetUniformLocation(m_id, name.c_str()), 1, &value[0]);
  }

  /*!
   * \brief Set vec4 value in the shader.
   * \param name variable name
   * \param x variable first component
   * \param y variable second component
   * \param y variable third component
   * \param w variable fourth component
   */
  void set_vec4(const std::string &name, float x, float y, float z, float w) const {
    glUniform4f(glGetUniformLocation(m_id, name.c_str()), x, y, z, w);
  }

  /*!
   * \brief Set mat2 value in the shader.
   * \param name variable name
   * \param mat variable value
   */
  void set_mat2(const std::string &name, const glm::mat2 &mat) const {
    glUniformMatrix2fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
  }

  /*!
   * \brief Set mat3 value in the shader.
   * \param name variable name
   * \param mat variable value
   */
  void set_mat3(const std::string &name, const glm::mat3 &mat) const {
    glUniformMatrix3fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
  }

  /*!
   * \brief Set mat4 value in the shader.
   * \param name variable name
   * \param mat variable value
   */
  void set_mat4(const std::string &name, const glm::mat4 &mat) const {
    glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
  }

private:
  /*!
   * \brief Check shader compile errors.
   * \param shader shader index
   * \param type error type to be checked
   */
  void check_compile_errors(const GLuint shader, const std::string &type) {
    GLint success = 0;
    GLchar info_log[1024];
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, info_log);
        throw std::runtime_error(info_log);
      }
    }
    else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        glGetProgramInfoLog(shader, 1024, NULL, info_log);
        throw std::runtime_error(info_log);
      }
    }
  }
};

} // namespace cm

#endif // CM_SHADER_H
