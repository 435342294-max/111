#include <iostream>
#include <fstream>

#include <boost/timer/timer.hpp>

#include <CGAL/Cartesian_converter.h>
#include <CGAL/squared_distance_2.h>

#include "utils/Kernel.h"
#include "utils/Kernel_epic.h"
#include "utils/Kernel_epec.h"
#include "utils/Logger.h"

#include "segment_modeling/approximate_squared_distance.h"

typedef CGAL::Cartesian_converter<Kernel, Kernel_epec> To_epeck;
typedef CGAL::Cartesian_converter<Kernel, Kernel_epic> To_epick;
typedef Kernel::Segment_2 Segment_2;

/*!
 * \brief Epsilons.controlled squared distance of two segments.
 * \param s1 first segment
 * \param s2 second segment
 * \return distance
 */
double robust_squared_distance(const Kernel::Segment_2 &s1, const Kernel::Segment_2 &s2) {
  return segment_segment_squared_distance(
    s1.source().x(), s1.source().y(), s1.target().x(), s1.target().y(),
    s2.source().x(), s2.source().y(), s2.target().x(), s2.target().y());
}

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 0;

  // initialize log
  cm::initialize_logger();

  To_epeck to_epeck;
  To_epick to_epick;

  {
    // squared distance of two segments requires exact kernel
    // https://github.com/CGAL/cgal/issues/4189

    const Segment_2 segi = {
      { -4.0380854964382, -1.9947196614192 },
      { 10.43442091460618, -0.5886833953492263 } };
    const Segment_2 segj = {
      { -11.5138934277993, -2.721011070186227 },
      { -8.822747585009402, -2.459560251317805 } };

    LOG_INFO << "simple cartesian: " << CGAL::squared_distance(segi, segj);
    LOG_INFO << "epick: " << CGAL::squared_distance(to_epick(segi), to_epick(segj));
    LOG_INFO << "epeck: " << CGAL::squared_distance(to_epeck(segi), to_epeck(segj));
    LOG_INFO << "epsilon approximation: " << robust_squared_distance(segi, segj);
  }

  // read from file
  std::vector<Segment_2> segments;
  std::ifstream ifs(argv[1]);
  Segment_2 temp;
  while (ifs >> temp)
    segments.push_back(temp);
  LOG_INFO << "#segments: " << segments.size();

  {
    LOG_INFO << "simple cartesian kernel timing";
    boost::timer::auto_cpu_timer t("%w s\n");
    for (int i = 0; i < segments.size(); ++i)
      for (int j = 0; j < segments.size(); ++j)
        CGAL::squared_distance(segments[i], segments[j]);
  }

  {
    LOG_INFO << "epick kernel timing";
    boost::timer::auto_cpu_timer t("%w s\n");
    for (int i = 0; i < segments.size(); ++i)
      for (int j = 0; j < segments.size(); ++j)
        CGAL::squared_distance(to_epick(segments[i]), to_epick(segments[j]));
  }

  {
    LOG_INFO << "epeck kernel timing";
    boost::timer::auto_cpu_timer t("%w s\n");
    for (int i = 0; i < segments.size(); ++i)
      for (int j = 0; j < segments.size(); ++j)
        CGAL::squared_distance(to_epeck(segments[i]), to_epeck(segments[j]));
  }

  {
    LOG_INFO << "epsilon control timing";
    boost::timer::auto_cpu_timer t("%w s\n");
    for (int i = 0; i < segments.size(); ++i)
      for (int j = 0; j < segments.size(); ++j)
        robust_squared_distance(segments[i], segments[j]);
  }

  constexpr double tolerance = 1e-9;

  {
    LOG_INFO << "epsilon approximation vs epeck";
    for (int i = 0; i < segments.size(); ++i) {
      for (int j = 0; j < segments.size(); ++j) {
        const double d1 = robust_squared_distance(segments[i], segments[j]);
        const double d2 = CGAL::to_double(CGAL::squared_distance(
          to_epeck(segments[i]), to_epeck(segments[j])));
        if (std::abs(d1 - d2) > tolerance)
          LOG_INFO << i << ' ' << j << ' ' << d1 << ' ' << d2;
      }
    }
  }

  {
    LOG_INFO << "epsilon approximation vs simpel cartesian";
    for (int i = 0; i < segments.size(); ++i) {
      for (int j = 0; j < segments.size(); ++j) {
        const double d1 = robust_squared_distance(segments[i], segments[j]);
        const double d2 = CGAL::squared_distance(segments[i], segments[j]);
        if (std::abs(d1 - d2) > tolerance)
          LOG_INFO << i << ' ' << j << ' ' << d1 << ' ' << d2;
      }
    }
  }
}
