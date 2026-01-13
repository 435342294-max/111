#ifndef PALETTE_H
#define PALETTE_H

#include <vector>

#include <osg/Vec3>

#include "gencolormap/colormap.hpp"

namespace cm {
namespace Palette {

/*!
 * \brief Wrapper for diverging Moreland 256 colormap
 */
class Diverging {
public:
  /*!
   * \brief Constructor.
   */
  Diverging() {
    unsigned char colormap[256 * 3];
    ColorMap::Moreland(256, colormap);
    for (int i = 0; i < 256 * 3; ++i)
      m_colors[i] = colormap[i] / 255.0f;
  }

  /*!
   * \brief Access color.
   * \param v in [0.0, 1.0]
   * \return color in osg::Vec3
   */
  osg::Vec3 color(const double &v) const {
    const int idx = int(v * 255.0) * 3;
    return { m_colors[idx], m_colors[idx + 1], m_colors[idx + 2] };
  }

  /*!
   * \brief Access color.
   * \param i in [0, 255]
   * \return color in osg::Vec3
   */
  osg::Vec3 color(const int &i) const {
    const int idx = i * 3;
    return { m_colors[idx], m_colors[idx + 1], m_colors[idx + 2] };
  }

private:
  float m_colors[256 * 3];
};

/*!
 * \brief Wrapper for qualitative Brewer-like colormap
 */
class Qualitative {
public:
  /*!
   * \brief Constructor.
   * \param n number of colors
   */
  Qualitative(const int &n) : m_n(n), m_colors(n * 3, 0.0f) {
    std::vector<unsigned char> colormap(m_n * 3, 0);
    ColorMap::BrewerQualitative(m_n, colormap.data(),
      ColorMap::BrewerQualitativeDefaultHue, // hue of the first color
      ColorMap::BrewerQualitativeDefaultDivergence * 3.0f / 2.0f // maximum 2PI divergence
      );
    for (int i = 0; i < m_n * 3; ++i)
      m_colors[i] = colormap[i] / 255.0f;
  }

  /*!
   * \brief Access color.
   * \param v in [0.0, 1.0]
   * \return color in osg::Vec3
   */
  osg::Vec3 color(const double &v) const {
    const int idx = int(v * (m_n - 1.0f)) * 3;
    return { m_colors[idx], m_colors[idx + 1], m_colors[idx + 2] };
  }

  /*!
   * \brief Access color.
   * \param i in [0, n - 1]
   * \return color in osg::Vec3
   */
  osg::Vec3 color(const int &i) const {
    const int idx = i * 3;
    return { m_colors[idx], m_colors[idx + 1], m_colors[idx + 2] };
  }

private:
  const int m_n;
  std::vector<float> m_colors;
};

} // namespace Palette
} // namespace cm

#endif // PALETTE_H
