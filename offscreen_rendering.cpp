// https://learnopengl.com/Advanced-OpenGL/Framebuffers
// https://www.glfw.org/docs/latest/context_guide.html#context_offscreen

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "utils/GL_renderer.h"
#include "utils/Logger.h"

const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;
const unsigned int NUM_FRAMES = 60;
const float PI = 3.14159265358979323846f;

// TODO: need to polish
int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  cm::initialize_logger();

  cm::GL_renderer glr;
  const auto buffers = glr.load_file(argv[1]);
  glr.set_resolution(WIDTH, HEIGHT);
  glr.initialize(buffers);

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

  const float znear = 0.1f;
  const float zfar = 100.0f;
  const glm::mat4 projection = glm::perspective(
    glm::radians(45.0f), (float)WIDTH / (float)HEIGHT, znear, zfar);

  // orbit along the inscribed ellipse
  const float ellipse_h = (xmin + xmax) / 2.0f;
  const float ellipse_k = (ymin + ymax) / 2.0f;
  const float ellipse_a = (xmax - xmin) / 2.0f;
  const float ellipse_b = (ymax - ymin) / 2.0f;
  const float centerx = (xmin + xmax) / 2.0f;
  const float centery = (ymin + ymax) / 2.0f;
  const float centerz = (zmin + zmax) / 2.0f;
  int i = 0;
  while (i++ < NUM_FRAMES) {
    // set view projection matrix
    const float camx = std::sin(PI / NUM_FRAMES * i) * ellipse_a + ellipse_h;
    const float camy = std::cos(PI / NUM_FRAMES * i) * ellipse_b + ellipse_k;
    const glm::mat4 view = glm::lookAt(
      glm::vec3(camx, camy, zmax),
      glm::vec3(centerx, centery, centerz),
      glm::vec3(0.0f, 1.0f, 0.0f));
    glr.set_view_projection(view, projection);

    // render mesh
    glr.draw_buffers(buffers);

    // copy frame buffer to cpu
    const auto color_frame = glr.grab_color_frame();
    cv::imwrite(std::string("dump_color") + std::to_string(i) + ".png", color_frame);

    cv::Mat temp;
    const auto normal_frame = glr.grab_normal_frame();
    // XYZ (RGB) to BGR
    cv::cvtColor(normal_frame, normal_frame, CV_RGB2BGR);
    normal_frame.convertTo(temp, CV_8U, 255.0f / 2.0f, 255.0f / 2.0f);
    cv::imwrite(std::string("dump_normal") + std::to_string(i) + ".png", temp);

    auto depth_frame = glr.grab_depth_frame();
    // perspective depth
    for (int r = 0; r < depth_frame.rows; ++r) {
      for (int c = 0; c < depth_frame.cols; ++c) {
        // to real depth
        depth_frame.at<float>(r, c) =
          znear * zfar / (zfar - depth_frame.at<float>(r, c) * (zfar - znear));
        // to linear normalized depth
        depth_frame.at<float>(r, c) = (depth_frame.at<float>(r, c) - znear) / (zfar -znear);
      }
    }
    // [0, 1] to [65535, 0]
    depth_frame.convertTo(temp, CV_16U, -65535.0f, 65535.0f);
    cv::imwrite(std::string("dump_depth") + std::to_string(i) + ".png", temp);
  }

  return 0;
}
