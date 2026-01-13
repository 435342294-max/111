#ifndef CM_GL_RENDERER_H
#define CM_GL_RENDERER_H

#include <vector>
#include <string>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Logger.h"
#include "Shader.h"
#include "Rply_loader.h"
#include "Assimp_loader.h"
#include "Mesh_3.h"

struct GLFWwindow;

namespace cm {

/*!
 * \brief Process input callback.
 * \param window GLFW window
 */
inline void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

/*!
 * \brief OpenGL render sampler.
 */
class GL_renderer {
  // static mutex to ensure only one instance is alive
  // https://stackoverflow.com/questions/11097170/multithreaded-rendering-on-opengl
  // https://stackoverflow.com/questions/2462961/using-static-mutex-in-a-class
  // https://stackoverflow.com/questions/18860895/how-to-initialize-static-members-in-the-header
  static std::mutex &get_mutex() {
    static std::mutex mx;
    return mx;
  }

  // object life time lock
  std::lock_guard<std::mutex> m_lock;

public:
  // compact packed glm::vec
  static_assert(sizeof(glm::vec2) == sizeof(float) * 2, "glm::vec2 has padding.");
  static_assert(sizeof(glm::vec3) == sizeof(float) * 3, "glm::vec3 has padding.");

  // render enumeration
  enum Option : unsigned char {
    // set to enable backface culling
    BACKFACE_CULL = 0x1,
    // set to render with texture
    WITH_TEXTURE = 0x2,
    // set to render with normal buffer
    WITH_NORMAL = 0x4,
    // set to render with depth buffer
    WITH_DEPTH = 0x8
  };

  // CPU side, OpenGL friendly triangle soup buffer
  struct Buffer {
    // number of faces
    unsigned int num_faces;
    // triangle soup
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    // texture
    std::vector<glm::vec2> texcoords;
    std::string texture_file;

    void release_memory() {
      num_faces = 0;
      std::vector<glm::vec3>().swap(vertices);
      std::vector<glm::vec3>().swap(normals);
      std::vector<glm::vec2>().swap(texcoords);
      texture_file = "";
    }
  };

  /*!
   * \brief Constructor with rendering options.
   * \param options combination of enum Option
   */
  GL_renderer(const unsigned char &options = WITH_TEXTURE | WITH_NORMAL | WITH_DEPTH) :
    m_lock(get_mutex()),
    m_options(options),
    m_offscreen_context(nullptr) {}

  /*!
   * \brief Destructor, terminate OpenGL program and release memory.
   */
  ~GL_renderer() {
    glfwTerminate();
  }

  /*!
   * \brief Top-down orthogonal samples surface mesh.
   * It will set resolution, view and projection matrix accordingly.
   * \param mesh input mesh
   * \param step sampling step
   * \param min_corner minimum corner of the rendering box
   * \param max_corner maximum corner of the rendering box
   */
  void orthogonal_sample(
    const Mesh_3 &mesh,
    const float step,
    const glm::vec3 &min_corner = glm::vec3(1.0f),
    const glm::vec3 &max_corner = glm::vec3(0.0f)) {
    // Mesh_3 to buffer
    std::vector<Buffer> buffers = load_mesh_3(mesh);
    // set rendering box and orthogonal projection
    if (min_corner.x < max_corner.x
      && min_corner.y < max_corner.y
      && min_corner.z < max_corner.z)
      set_orthogonal_projection(step, min_corner, max_corner);
    else {
      const auto bbox = compute_bounding_box(buffers);
      set_orthogonal_projection(step, bbox.first, bbox.second);
    }
    // initialize
    initialize(buffers);
    // draw
    draw_buffers(buffers);
  }

  /*!
   * \brief Top-down orthogonal samples surface mesh.
   * It will set resolution, view and projection matrix accordingly.
   * Used for when x, y have (slightly) different sampling step (TEMPORARILY).
   * \param mesh input mesh
   * \param w sampling resolution width
   * \param h sampling resolution height
   * \param min_corner minimum corner of the rendering box
   * \param max_corner maximum corner of the rendering box
   */
  void orthogonal_sample(
    const Mesh_3 &mesh,
    const unsigned int w,
    const unsigned int h,
    const glm::vec3 &min_corner = glm::vec3(1.0f),
    const glm::vec3 &max_corner = glm::vec3(0.0f)) {
    // Mesh_3 to buffer
    std::vector<Buffer> buffers = load_mesh_3(mesh);
    // set rendering box and orthogonal projection
    if (min_corner.x < max_corner.x
      && min_corner.y < max_corner.y
      && min_corner.z < max_corner.z)
      set_orthogonal_projection(1.0f, min_corner, max_corner);
    else {
      const auto bbox = compute_bounding_box(buffers);
      set_orthogonal_projection(1.0f, bbox.first, bbox.second);
    }
    set_resolution(w, h);
    // initialize
    initialize(buffers);
    // draw
    draw_buffers(buffers);
  }

  /*!
   * \brief Top-down orthogonal samples model from file.
   * It will set resolution, view and projection matrix accordingly.
   * \param file_name input model file
   * \param step sampling step
   * \param min_corner minimum corner of the rendering box
   * \param max_corner maximum corner of the rendering box
   */
  void orthogonal_sample(
    const std::string &file_name,
    const float step,
    const glm::vec3 &min_corner = glm::vec3(1.0f),
    const glm::vec3 &max_corner = glm::vec3(0.0f)) {
    // load from file
    std::vector<Buffer> buffers = load_file(file_name);
    // set rendering box and orthogonal projection
    if (min_corner.x < max_corner.x
      && min_corner.y < max_corner.y
      && min_corner.z < max_corner.z)
      set_orthogonal_projection(step, min_corner, max_corner);
    else {
      const auto bbox = compute_bounding_box(buffers);
      set_orthogonal_projection(step, bbox.first, bbox.second);
    }
    // initialize
    initialize(buffers);
    // draw
    draw_buffers(buffers);
  }

  /*!
   * \brief Grab color frame buffer
   * \return mat with depth of CV_8UC3
   */
  cv::Mat grab_color_frame() const {
    // color buffer
    const unsigned int padding = (4 - (m_width * 3) % 4) % 4;
    const std::size_t n_bytes_per_row = m_width * 3 + padding;
    std::vector<unsigned char> buffer(n_bytes_per_row * m_height, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    cv::Mat color_frame = cv::Mat(
      m_height, m_width, CV_8UC3, (void *)buffer.data(), n_bytes_per_row).clone();
    // RGB to BGR
    cv::cvtColor(color_frame, color_frame, cv::COLOR_RGB2BGR);
    cv::flip(color_frame, color_frame, 0);

    return color_frame;
  }

  /*!
   * \brief Grab normal frame buffer
   * \return mat with depth of CV_8UC3
   */
  cv::Mat grab_normal_frame() const {
    // normal buffer
    // sizeof(float) % 4 == 0, no padding for float
    std::vector<float> buffer(m_width * m_height * 3, 0.0f);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_FLOAT, buffer.data());
    cv::Mat normal_frame = cv::Mat(m_height, m_width, CV_32FC3, (void *)buffer.data()).clone();
    cv::flip(normal_frame, normal_frame, 0);

    return normal_frame;
  }

  /*!
   * \brief Grab depth frame buffer
   * \return mat with depth of CV_32FC1
   */
  cv::Mat grab_depth_frame() const {
    // depth buffer
    // sizeof(float) % 4 == 0, no padding for float
    std::vector<float> buffer(m_width * m_height, 0.0f);
    glReadPixels(0, 0, m_width, m_height, GL_DEPTH_COMPONENT , GL_FLOAT, buffer.data());
    cv::Mat depth_frame = cv::Mat(m_height, m_width, CV_32FC1, (void *)buffer.data()).clone();
    cv::flip(depth_frame, depth_frame, 0);

    return depth_frame;
  }

  /*!
   * \brief Loads a Mesh_3.
   * \param mesh input mesh
   * \return buffers
   */
  std::vector<Buffer> load_mesh_3(const Mesh_3 &mesh) {
    // Mesh_3 to Buffer
    Buffer b;
    // number of faces
    b.num_faces = mesh.number_of_faces();
    // geometry
    b.vertices.reserve(mesh.number_of_faces() * 3);
    const auto vpmap = mesh.points();
    for (face_descriptor f : faces(mesh)) {
      const halfedge_descriptor h = mesh.halfedge(f);
      b.vertices.push_back(to_vec3(vpmap[mesh.source(h)]));
      b.vertices.push_back(to_vec3(vpmap[mesh.target(h)]));
      b.vertices.push_back(to_vec3(vpmap[mesh.target(mesh.next(h))]));
    }
    // normal
    compute_normal(b);

    return { b };
  }

  /*!
   * \brief Loads from a model file.
   * \param file_name input file_name
   * \return buffers
   */
  std::vector<Buffer> load_file(const std::string &file_name) {
    std::vector<Buffer> buffers;
    const bool is_ply = file_name.substr(file_name.find_last_of('.')) == ".ply";
    try {
      if (is_ply)
        buffers = load_rply(file_name);
      else
        buffers = load_assimp(file_name);
    }
    catch (std::exception &e) {
      throw std::runtime_error(
        std::string(is_ply ? "RPLY::" : "ASSIMP::") + e.what());
    }

    return buffers;
  }

  /*!
   * \brief Set sampler resolution manually.
   * \param w width
   * \param h height
   */
  void set_resolution(const unsigned int w, const unsigned int h) {
    m_width = w;
    m_height = h;
  }

  /*!
   * \brief Sets view and projection matrix manually.
   * \param v view matrix
   * \param p projection matrix
   */
  void set_view_projection(const glm::mat4 &v, const glm::mat4 &p) {
    m_view = v;
    m_projection = p;
  }

  /*!
   * \brief Initialize context, buffer and variables.
   * \note Should be called only once.
   * \param buffers input buffers
   */
  void initialize(const std::vector<Buffer> &buffers) {
    // initialize offscreen context
    LOG_INFO << "Resolution: " << m_width << " x " << m_height;
    setup_offscreen_context();
    LOG_INFO << "Context setup done.";

    // setup non-default frame buffer
    setup_frame_buffer();
    LOG_INFO << "Buffer setup done.";

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (m_options & BACKFACE_CULL) {
      glEnable(GL_CULL_FACE);
      glCullFace(GL_BACK);
    }
    glEnable(GL_DEPTH_TEST);

    // reset texture option if buffer has no texture
    for (const auto &b : buffers) {
      if (b.texcoords.empty()) {
        m_options &= (~WITH_TEXTURE);
        break;
      }
    }

    // build and compile shader
    const std::string vertex_shader = m_options & WITH_TEXTURE ? R"(
      #version 330 core
      layout (location = 0) in vec3 vPos;
      layout (location = 1) in vec3 vNormal;
      layout (location = 2) in vec2 vTexCoord;

      out vec3 fNormal;
      out vec2 fTexCoord;

      uniform mat4 model;
      uniform mat4 view;
      uniform mat4 projection;

      void main() {
        fNormal = normalize(vNormal);
        gl_Position = projection * view * model * vec4(vPos, 1.0f);
        fTexCoord = vTexCoord;
      }
      )" : R"(
      #version 330 core
      layout (location = 0) in vec3 vPos;
      layout (location = 1) in vec3 vNormal;

      out vec3 fNormal;

      uniform mat4 model;
      uniform mat4 view;
      uniform mat4 projection;

      void main() {
        fNormal = normalize(vNormal);
        gl_Position = projection * view * model * vec4(vPos, 1.0f);
      }
      )";
    const std::string fragment_shader = m_options & WITH_TEXTURE ? R"(
      #version 330 core
      layout (location = 1) out vec3 out_normal;

      in vec3 fNormal;
      in vec2 fTexCoord;
      out vec3 fFragColor;

      uniform sampler2D textures;

      void main() {
        out_normal = fNormal;
        fFragColor = texture(textures, fTexCoord).rgb;
      }
      )" : R"(
      #version 330 core
      layout (location = 1) out vec3 out_normal;

      in vec3 fNormal;
      out vec3 fFragColor;

      void main() {
        out_normal = fNormal;
        fFragColor = vec3(0.7f, 0.7f, 0.7f);
      }
      )";
    m_shader.load_from_string(vertex_shader, fragment_shader);
    m_shader.use();
    m_shader.set_mat4("model", glm::mat4(1.0f));
    m_shader.set_mat4("view", m_view);
    m_shader.set_mat4("projection", m_projection);

    // generate vertex array object
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    // generate vertex buffer objects
    glGenBuffers(1, &m_vbo_geo);
    if (m_options & WITH_NORMAL)
      glGenBuffers(1, &m_vbo_norm);
    if (m_options & WITH_TEXTURE) {
      glGenBuffers(1, &m_vbo_tex);
      // generate texture buffer
      m_shader.set_int("textures", 0);
      glGenTextures(1, &m_tex);
    }
  }

  /*!
   * \brief Low-level draw Buffer directly.
   * \param buffers input buffers
   */
  void draw_buffers(const std::vector<Buffer> &buffers) {
    // still process events as long as you have at least one window,
    // even if none of them are visible
    process_input(m_offscreen_context);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // draw mesh
    for (const auto &b : buffers) {
      // shader
      m_shader.use();
      m_shader.set_mat4("model", glm::mat4(1.0f));
      m_shader.set_mat4("view", m_view);
      m_shader.set_mat4("projection", m_projection);
      // load texture
      if (m_options & WITH_TEXTURE) {
        glActiveTexture(GL_TEXTURE0);
        load_texture(b.texture_file);
      }
      // load buffer data
      glBindVertexArray(m_vao);
      load_buffer_data(b);
      // draw
      glDrawArrays(GL_TRIANGLES, 0, b.num_faces * 3);
    }
    glfwSwapBuffers(m_offscreen_context);
    glfwPollEvents();
  }

private:
  /*!
   * \brief Set resolution and orthogonal rendering view and projection matrix.
   * \param min_corner minimum corner of the rendering box
   * \param max_corner maximum corner of the rendering box
   * \param step sampling step
   */
  void set_orthogonal_projection(
    const float step,
    const glm::vec3 &min_corner,
    const glm::vec3 &max_corner) {
    // bounding box
    const float xmin = min_corner.x;
    const float ymin = min_corner.y;
    const float zmin = min_corner.z;
    const float xmax = max_corner.x;
    const float ymax = max_corner.y;
    const float zmax = max_corner.z;

    // set resolution
    m_width = (unsigned int)((xmax - xmin) / step);
    m_height = (unsigned int)((ymax - ymin) / step);

    // set view and projection matrix
    const float offset = (zmax - zmin) * 0.01f;
    const glm::vec3 center(
      (xmin + xmax) / 2.0f,
      (ymin + ymax) / 2.0f,
      zmax);
    m_view = glm::lookAt(
      center + glm::vec3(0.0f, 0.0f, offset),
      center,
      glm::vec3(0.0f, 1.0f, 0.0f));
    const float znear = offset;
    const float zfar = zmax - zmin + offset;
    m_projection = glm::ortho(
      -(xmax - xmin) / 2.0f, (xmax - xmin) / 2.0f,
      -(ymax - ymin) / 2.0f, (ymax - ymin) / 2.0f,
      znear, zfar);
  }

  /*!
   * \brief Setup OpenGL offscreen context.
   */
  void setup_offscreen_context() {
    // initialize context
    glfwInit();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    m_offscreen_context = glfwCreateWindow(m_width, m_height, "", nullptr, nullptr);
    if (!m_offscreen_context) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW offscreen context.");
    }

    glfwMakeContextCurrent(m_offscreen_context);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      glfwTerminate();
      throw std::runtime_error("Failed to initialize GLAD.");
    }
    glViewport(0, 0, m_width, m_height);
    glfwSetFramebufferSizeCallback(m_offscreen_context, nullptr);

    // echo run-time versions
    LOG_INFO << "OpenGL version: " <<
      glfwGetWindowAttrib(m_offscreen_context, GLFW_CONTEXT_VERSION_MAJOR) << '.' <<
      glfwGetWindowAttrib(m_offscreen_context, GLFW_CONTEXT_VERSION_MINOR) << '.' <<
      glfwGetWindowAttrib(m_offscreen_context, GLFW_CONTEXT_REVISION);
    int major = 0, minor = 0, rev = 0;
    glfwGetVersion(&major, &minor, &rev);
    LOG_INFO << "GLFW version: " << major << '.' << minor << '.' << rev;
  }

  /*!
   * \brief Sets up OpenGL non-default frame buffer,
   * including three different buffers: color, depth and normal.
   */
  void setup_frame_buffer() {
    // setup non-default frame buffer
    unsigned int frame_buffer = 0;
    glGenFramebuffers(1, &frame_buffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

    unsigned int rbo[3] = {0, 0, 0};
    glGenRenderbuffers(3, rbo);
    // color render buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, m_width, m_height);
    // unbind the render buffer after the memory is allocated
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // attach to current frame buffer object
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo[0]);

    // depth test render buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
    // unbind the render buffer after the memory is allocated
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // attach to current frame buffer object
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);

    // normal render buffer
    if (m_options & WITH_NORMAL) {
      glBindRenderbuffer(GL_RENDERBUFFER, rbo[2]);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB32F, m_width, m_height);
      glBindRenderbuffer(GL_RENDERBUFFER, 0);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, rbo[2]);
    }

    // tell OpenGL which color attachments we'll use (of this FBO) for rendering 
    if (m_options & WITH_NORMAL) {
      unsigned int buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
      glDrawBuffers(2, buffers);
    }
    else {
      unsigned int buffers[] = { GL_COLOR_ATTACHMENT0 };
      glDrawBuffers(1, buffers);
    }

    // check frame buffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      //glfwTerminate();
      //throw std::runtime_error("FRAMEBUFFER::INCOMPLETE");
    }
  }

  /*!
   * \brief Loads texture file to existing buffer.
   * \param texture_file texture file path
   */
  void load_texture(const std::string &texture_file) {
    cv::Mat img = cv::imread(texture_file, cv::ImreadModes::IMREAD_COLOR);
    if (img.empty()) {
      glfwTerminate();
      throw std::runtime_error(std::string("Failed to load texture ") + texture_file + '.');
    }
    // OpenCV image to OpenGL texture image convention
    cv::flip(img, img, 0);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  /*!
   * \brief Loads CPU vertex data to GPU buffer.
   * \param b input buffer
   */
  void load_buffer_data(const Buffer &b) {
    // geometry
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_geo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * b.vertices.size(), b.vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
    glEnableVertexAttribArray(0);

    // normal
    if (m_options & WITH_NORMAL) {
      glBindBuffer(GL_ARRAY_BUFFER, m_vbo_norm);
      glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * b.normals.size(), b.normals.data(), GL_STATIC_DRAW);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
      glEnableVertexAttribArray(1);
    }

    // texture
    if (m_options & WITH_TEXTURE) {
      glBindBuffer(GL_ARRAY_BUFFER, m_vbo_tex);
      glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * b.texcoords.size(), b.texcoords.data(), GL_STATIC_DRAW);
      glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
      glEnableVertexAttribArray(2);
    }
  }

  /*!
   * \brief Loads a ply file.
   * \note Assimp can not loads ply with multiple textures properly 
   * \param file_name input file name
   * \return buffers
   */
  std::vector<Buffer> load_rply(const std::string &file_name) {
    // split mesh according to different texture
    Rply_loader ply(file_name);
    const unsigned int num_meshes =
      ply.textures.empty() ? 1 : (unsigned int)ply.textures.size();

    // number of faces
    std::vector<Buffer> buffers(num_meshes);
    if (num_meshes == 1)
      buffers[0].num_faces = ply.num_faces;
    else {
      for (auto &b : buffers)
        b.num_faces = 0;
      for (unsigned int i = 0; i < ply.num_faces; ++i)
        buffers[ply.texnumber[i]].num_faces++;
    }

    // geometry
    for (auto &b : buffers)
      b.vertices.reserve(b.num_faces * 3);
    for (unsigned int i = 0; i < ply.num_faces; ++i) {
      const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
      for (unsigned int j = 0; j < 3; ++j) {
        const unsigned int fvidx = i * 3 + j;
        const unsigned int vidx = ply.faces[fvidx];
        buffers[midx].vertices.push_back({
          ply.vertices[vidx * 3],
          ply.vertices[vidx * 3 + 1],
          ply.vertices[vidx * 3 + 2]
        });
      }
    }

    // normal
    for (auto &b : buffers)
      compute_normal(b);

    // texture
    if (!ply.textures.empty()) {
      const std::string dir = file_name.substr(0, file_name.find_last_of("\\/") + 1);
      for (unsigned int i = 0; i < num_meshes; ++i) {
        Buffer &b = buffers[i];
        b.texcoords.reserve(b.num_faces * 3);
        b.texture_file = dir + ply.textures[i];
      }
      for (unsigned int i = 0; i < ply.num_faces; ++i) {
        const unsigned int midx = (num_meshes == 1 ? 0 : ply.texnumber[i]);
        for (unsigned int j = 0; j < 3; ++j) {
          const unsigned int fvidx = i * 3 + j;
          buffers[midx].texcoords.push_back({
            ply.texcoords[fvidx * 2],
            ply.texcoords[fvidx * 2 + 1]
          });
        }
      }
    }
    return buffers;
  }

  /*!
   * \brief Loads other format using Assimp.
   * \param file_name input file name
   * \return buffers
   */
  std::vector<Buffer> load_assimp(const std::string &file_name) {
    // assimp load
    Assimp_loader assimp(file_name);
    const std::string dir = file_name.substr(0, file_name.find_last_of("\\/") + 1);
    std::vector<Buffer> buffers;
    for (const auto &m : assimp.meshes) {
      Buffer b;
      // number of faces
      b.num_faces = m.num_faces;
      // geometry
      b.vertices.reserve(m.num_faces * 3);
      for (unsigned int i = 0; i < m.num_faces; ++i) {
        for (unsigned int j = 0; j < 3; ++j) {
          const unsigned int vidx = m.faces[i * 3 + j];
          b.vertices.push_back({
            m.vertices[vidx * 3],
            m.vertices[vidx * 3 + 1],
            m.vertices[vidx * 3 + 2]
          });
        }
      }
      // normal
      compute_normal(b);
      // texture
      if (!m.texcoords.empty()) {
        b.texcoords.reserve(m.num_faces * 3);
        b.texture_file = dir + m.textures[0];
        for (unsigned int i = 0; i < m.num_faces; ++i) {
          for (unsigned int j = 0; j < 3; ++j) {
            const unsigned int vidx = m.faces[i * 3 + j];
            b.texcoords.push_back({
              m.texcoords[vidx * 2],
              m.texcoords[vidx * 2 + 1]});
          }
        }
      }
      buffers.push_back(b);
    }
    return buffers;
  }

  /*!
   * \brief Computes normal for each vertex.
   * \param b input mesh buffer
   */
  void compute_normal(Buffer &b) {
    b.normals.reserve(b.num_faces * 3);
    for (unsigned int i = 0; i < b.vertices.size(); i += 3) {
      const auto &v0 = b.vertices[i];
      const auto &v1 = b.vertices[i + 1];
      const auto &v2 = b.vertices[i + 2];
      const auto n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
      for (unsigned int j = 0; j < 3; ++j)
        b.normals.push_back(n);
    }
  }

  /*!
   * \brief Computes bounding box for the whole model.
   * \param buffers input buffers
   * \return pair of bounding box lowest and highest corners
   */
  std::pair<glm::vec3, glm::vec3> compute_bounding_box(const std::vector<Buffer> &buffers) {
    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymin = std::numeric_limits<float>::max();
    float ymax = std::numeric_limits<float>::lowest();
    float zmin = std::numeric_limits<float>::max();
    float zmax = std::numeric_limits<float>::lowest();
    for (const auto &b : buffers) {
      for (const auto &v : b.vertices) {
        const float x = v[0], y = v[1], z = v[2];
        if (x < xmin)
          xmin = x;
        if (x > xmax)
          xmax = x;
        if (y < ymin)
          ymin = y;
        if (y > ymax)
          ymax = y;
        if (z < zmin)
          zmin = z;
        if (z > zmax)
          zmax = z;
      }
    }

    LOG_INFO << "compute_bounding_box";
    LOG_INFO << "min: " << xmin << ' ' << ymin << ' ' << zmin;
    LOG_INFO << "max: " << xmax << ' ' << ymax << ' ' << zmax;

    return { { xmin, ymin, zmin }, { xmax, ymax, zmax } };
  }

  /*!
   * \brief Converts a CGAL Vector / Point to glm::vec3
   */
  template <typename T>
  glm::vec3 to_vec3(const T &a) {
    return { float(a.x()), float(a.y()), float(a.z()) };
  }

private:
  // options
  unsigned char m_options = 0;

  // OpenGL window context
  GLFWwindow *m_offscreen_context;

  // width
  unsigned int m_width = 0;
  // height
  unsigned int m_height = 0;

  // shader
  cm::Shader m_shader;
  // vertex array object
  unsigned int m_vao = 0;
  // vertex buffer objects
  unsigned int m_vbo_geo = 0;
  unsigned int m_vbo_norm = 0;
  unsigned int m_vbo_tex = 0;
  // texture buffer
  unsigned int m_tex = 0;

  // view matrix
  glm::mat4 m_view;
  // projection matrix
  glm::mat4 m_projection;
};

} // namespace cm

#endif // CM_GL_RENDERER_H
