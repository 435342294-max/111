#include <vector>
#include <fstream>

#include <boost/timer/timer.hpp>

#include "utils/Logger.h"
#include "utils/Kernel.h"
#include "utils/Kernel_epec.h"
#include "utils/View_helper.h"

#include "segment_modeling/regularize_segments_graph.h"

typedef Kernel::Segment_2 Segment_2;

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 0;

  // initialize log
  cm::initialize_logger(cm::severity_level::debug);

  // read from file
  std::vector<Segment_2> segments;
  std::ifstream ifs(argv[1]);
  Segment_2 temp;
  while (ifs >> temp)
    segments.push_back(temp);
  LOG_INFO << "#segments: " << segments.size();

  // viewer
  cm::get_global_viewer(1, 2, 800, 450).run();

  {
    std::vector<Kernel::Segment_3> temp;
    for (const auto &s : segments)
      temp.push_back({
        {s.source().x(), s.source().y(), 0.0},
        {s.target().x(), s.target().y(), 0.0}});
    cm::get_global_viewer().add_node(cm::segments_to_node(temp, osg::BLACK));
  }

  std::vector<Kernel_epec::Segment_2> regularized;
  {
    boost::timer::auto_cpu_timer t("%w s\n");
    //regularized = regularize_segments_graph(segments, 10, 0.2);
    LOG_INFO << "#regularized: " << regularized.size();
  }

  {
    std::vector<Kernel::Segment_3> temp;
    for (const auto &s : regularized)
      temp.push_back({
        {CGAL::to_double(s.source().x()), CGAL::to_double(s.source().y()), 0.0},
        {CGAL::to_double(s.target().x()), CGAL::to_double(s.target().y()), 0.0}});
    cm::get_global_viewer().add_node(cm::segments_to_node(temp, osg::RED), 1);
  }
}
