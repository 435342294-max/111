#ifndef MANHATTAN_DETECTOR_H
#define MANHATTAN_DETECTOR_H

#include <vector>
#include <cstdlib>
#include <array>

namespace cm {

/*!
 * \brief Find Manhattan-world directions of the given normals.
 * Part of this work is inspired by "Structured Indoor Modeling"
 * Webpage: http://www.cse.wustl.edu/~sikehata/sim/
 */
template <typename Vec>
class Manhattan_detector {
public:
  enum Support_mode { Inlier, Energy };

  /*!
   * \brief Constructor
   */
  Manhattan_detector() {}

  /*!
   * \brief Detect
   * \param normals The vector of normals.
   * \param prior_dir prior direction.
   * \param prior_confidence the prior direction conficence.
   * \param smode RANSAC voting mode.
   */
  void detect(const std::vector<Vec> &normals,
    const Vec &prior_dir, const double prior_confidence,
    const Support_mode &smode) {
    m_dirs[0] = Vec(1.0, 0.0, 0.0);
    m_dirs[1] = Vec(0.0, 1.0, 0.0);
    m_dirs[2] = Vec(0.0, 0.0, 1.0);

    // find vertical direction
    std::vector<Vec> candidates;
    candidates.reserve(normals.size() / 3);
    for (const auto &n : normals) {
      // acos(0.9) = 25.85
      // acos(0.995) = 5.732
      if (std::abs(n * prior_dir) > prior_confidence)
        candidates.push_back(n);
    }
    if (smode == Support_mode::Inlier)
      ransac_manhattan_1st(candidates);
    else
      ransac_manhattan_1st_energy(candidates);

    // find two horizontal directions
    candidates.clear();
    for (const auto &n : normals) {
      // acos(0.01) = 89.427
      if (std::abs(n * m_dirs[0]) < 0.01)
        candidates.push_back(n);
    }
    if (smode == Support_mode::Inlier)
      ransac_manhattan_2nd(candidates);
    else
      ransac_manhattan_2nd_energy(candidates);
  }

  /*!
   * \brief Returns the detected directions
   * \return Manhattan directions
   */
  const std::array<Vec, 3> &directions() const { return m_dirs; }

private:
  /*!
   * \brief RANSAC inlier voting to find the 1st Manhattan-world direction
   * \param candidates detection candidate set
   */
  void ransac_manhattan_1st(const std::vector<Vec> &candidates) {
    const auto normal_size = candidates.size();

    unsigned int maxInliers = 0;
    double max_iteration = 1.0e5;
    //double max_iteration = candidates.size() / 2.0;
    for (int i = 0; i < max_iteration; ++i) {
      // random sampling
      // compute the model parameters
      Vec rand_normal = candidates[std::rand() % normal_size];

      // counting inliers
      unsigned int numInliers = 0;
      for (auto itr = candidates.begin();
        itr != candidates.end(); ++itr) {
        if (std::acos(std::abs((*itr) * rand_normal)) < 0.1)
          // cos(0.1) = 0.9999985
          ++numInliers;
      }

      if (numInliers > maxInliers) {
        maxInliers = numInliers;
        m_dirs[0] = rand_normal;

        double w = (numInliers - double(3)) / normal_size;
        double p = (0.1 > w) ? 0.001 : std::pow(w, 3);
        max_iteration = std::log(1 - 0.999) / std::log(1 - p);
      }
    }
  }

  /*!
   * \brief RANSAC inlier voting to find the 2nd Manhattan-world direction
   * \param candidates detection candidate set
   */
  void ransac_manhattan_2nd(const std::vector<Vec>& candidates) {
    const auto normal_size = candidates.size();

    unsigned int maxInliers = 0;
    double max_iteration = 1.0e5;
    for (int i = 0; i < max_iteration; ++i) {
      // random sampling
      // compute the model parameters
      const Vec perpend_normal1 = candidates[std::rand() % normal_size];
      const Vec perpend_normal2 = CGAL::cross_product(perpend_normal1, m_dirs[0]);

      // counting inliers
      unsigned int numInliers = 0;
      for (const auto &c : candidates) {
        if (std::min(std::acos(std::abs(c * perpend_normal1)),
          std::acos(std::abs(c * perpend_normal2))) < 0.01)
          // cos(0.01) = 0.999999985
          ++numInliers;
      }

      if (numInliers > maxInliers) {
        maxInliers = numInliers;
        m_dirs[1] = perpend_normal1;
        m_dirs[2] = perpend_normal2;

        const double w = (numInliers - 3.0) / normal_size;
        const double p = (0.1 > w) ? 0.001 : std::pow(w, 3);
        max_iteration = std::log(1 - 0.999) / std::log(1 - p);
      }
    }
  }

  /*!
   * \brief RANSAC energy accumulation to find the 1st Manhattan-world direction
   * \param candidates detection candidate set
   */
  void ransac_manhattan_1st_energy(const std::vector<Vec> &candidates) {
    const auto normal_size = candidates.size();

    double max_energy_support = 0.0;
    double max_iteration = normal_size / 2.0;
    for (int i = 0; i < max_iteration; ++i) {
      // random sampling
      // compute the model parameters
      const Vec rand_normal = candidates[std::rand() % normal_size];

      // sum total energy support
      double energy_support = 0.0;
      for (const auto &c : candidates)
        energy_support += std::abs(c * rand_normal);

      if (energy_support > max_energy_support) {
        max_energy_support = energy_support;
        m_dirs[0] = rand_normal;
      }
    }
  }

  /*!
   * \brief RANSAC energy accumulation to find the 2nd Manhattan-world direction
   * \param candidates detection candidate set
   */
  void ransac_manhattan_2nd_energy(const std::vector<Vec>& candidates) {
    const auto normal_size = candidates.size();

    double max_energy_support = 0;
    double max_iteration = normal_size / 2.0;
    for (int i = 0; i < max_iteration; ++i) {
      // random sampling
      // compute the model parameters
      const Vec perpend_normal1 = candidates[std::rand() % normal_size];
      const Vec perpend_normal2 = CGAL::cross_product(perpend_normal1, m_dirs[0]);

      // counting inliers
      double energy_support = 0.0;
      for (const auto &c : candidates)
        energy_support += std::max(
          std::abs(c * perpend_normal1), std::abs(c * perpend_normal2));

      if (energy_support > max_energy_support) {
        max_energy_support = energy_support;
        m_dirs[1] = perpend_normal1;
        m_dirs[2] = perpend_normal2;
      }
    }
  }

private:
  // 3 orthogonal directions
  std::array<Vec, 3> m_dirs;
};

} // namespace cm

#endif // MANHATTAN_DETECTOR_H
