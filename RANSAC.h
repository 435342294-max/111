// Simple RANSAC implementation

#ifndef RANSAC_H
#define RANSAC_H

#include <numeric>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>

#include <omp.h>

namespace cm {

/*!
\brief A RANSAC class.

Given a set of samples, this class estimates a model with maximum inliers using the RANSAC algorithm.
The `Sample` and `Model` could be 2D points and line, 3D points and planes, etc.

\tparam `Model` requires
  - One static int:
    - MINIMUM_SET_SIZE: size of minimum set.
  - Two types:
    - `Input_range`: a model `Range` with random access iterators,
      providing input through the following input property map
    - `Input_map`: a model `ReadablePropertyMap` with
      `std::iterator_traits<Input_range::iterator>::value_type` as key type, `Sample` as value type
  - Two functions:
    - `Constructor`: construct from a range of `Input_range::iterator` and `Input_map`
    - `compute_distance`: calculate distance of a `Sample to a `Model`
*/
template <typename Model>
class RANSAC {
public:
  typedef typename Model::Input_range Input_range;
  typedef typename Input_range::iterator Input_iterator;
  typedef typename Model::Input_map Input_map;
  typedef typename Input_map::value_type Sample;

  RANSAC(Input_range &input, Input_map input_pmap = Input_map()) :
    m_input_first(input.begin()),
    m_input_beyond(input.end()),
    m_input_pmap(input_pmap) {};

  std::pair<std::shared_ptr<Model>, std::vector<Input_iterator>>
  estimate(const double threshold, const int iterations = 1000) {
    const int num_samples = int(std::distance(m_input_first, m_input_beyond));

    if (num_samples <= MINIMUM_SET_SIZE)
      return { nullptr, std::vector<Input_iterator>() };

#ifdef _OPENMP
    const int num_threads = std::max(1, omp_get_max_threads());
    // disable dynamic teams
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
#else
    const int num_threads = 1;
#endif

    std::vector<std::mt19937> rand_engines;
    for (int i = 0; i < num_threads; ++i)
      rand_engines.push_back(std::mt19937(std::random_device{}()));

    std::vector<std::shared_ptr<Model>> models(num_threads, nullptr);
    std::vector<std::vector<Input_iterator>> model_inliers(num_threads);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < iterations; ++i) {
      const int thread_id = omp_get_thread_num();

      // Fisher-Yates shuffle
      std::vector<int> sample_ids(num_samples, 0);
      std::iota(sample_ids.begin(), sample_ids.end(), 0);
      int num_selected = 0;
      auto curr = sample_ids.begin();
      while (num_selected++ < MINIMUM_SET_SIZE) {
        auto selected = curr;
        std::uniform_int_distribution<> dis(0, num_samples - num_selected);
        std::advance(selected, dis(rand_engines[thread_id]));
        std::swap(*curr, *selected);
        ++curr;
      }
      std::vector<Input_iterator> minimum_set;
      for (int j = 0; j < MINIMUM_SET_SIZE; ++j)
        minimum_set.push_back(m_input_first + sample_ids[j]);

      std::shared_ptr<Model> model = std::make_shared<Model>(minimum_set, m_input_pmap);
      std::vector<Input_iterator> inliers;
      for (int j = MINIMUM_SET_SIZE; j < num_samples; ++j) {
        const auto sample = m_input_first + sample_ids[j];
        if (model->compute_distance(get(m_input_pmap, *sample)) < threshold)
          inliers.push_back(sample);
      }

      if (!models[thread_id] || model_inliers[thread_id].size() < inliers.size()) {
        models[thread_id] = model;
        model_inliers[thread_id].swap(inliers);
      }
    }

    int idx = 0;
    int max_num_inliers = 0;
    for (int i = 0; i < num_threads; ++i) {
      if (model_inliers[i].size() > max_num_inliers) {
        idx = i;
        max_num_inliers = int(model_inliers[i].size());
      }
    }

    return { models[idx], model_inliers[idx] };
  };

private:
  const Input_iterator m_input_first;
  const Input_iterator m_input_beyond;
  Input_map m_input_pmap;

  static const int MINIMUM_SET_SIZE = Model::MINIMUM_SET_SIZE;
};

} // namespace cm

#endif // RANSAC_H
