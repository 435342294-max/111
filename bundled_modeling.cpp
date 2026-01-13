#include "utils/Logger.h"
#include "utils/Config.h"

#include "segment_modeling/UBlock.h"
#include "segment_modeling/UBlock_building.h"

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  // working directory
  const std::string wdir(argv[1]);
  // initialize log
  cm::initialize_logger();
  // load configuration
  cm::read_config(wdir + "/config_modeling.xml");
  LOG_INFO << "Configuration loaded.";

  const auto &config = cm::get_config();
  const int nb_rows = config.get<int>("scene.cut.rows");
  const int nb_cols = config.get<int>("scene.cut.cols");
  const std::string input = config.get<std::string>("input_dir");
  const std::string output = config.get<std::string>("output_dir");
  const int img_rows = config.get<int>("scene.image.rows");
  const int img_cols = config.get<int>("scene.image.cols");
  const int subimg_rows = config.get<int>("scene.sub_image.rows");
  const int subimg_cols = config.get<int>("scene.sub_image.cols");

  const CGAL::Bbox_3 bbox(
    config.get<double>("scene.bbox.xmin"),
    config.get<double>("scene.bbox.ymin"),
    config.get<double>("scene.bbox.zmin"),
    config.get<double>("scene.bbox.xmax"),
    config.get<double>("scene.bbox.ymax"),
    config.get<double>("scene.bbox.zmax")
  );

  LOG_INFO << "#img_rows: " << img_rows << ", #img_cols: " << img_cols;
  LOG_INFO << "#nb_rows: " << nb_rows << ", #nb_cols: " << nb_cols;
  LOG_INFO << "#subimg_rows: " << subimg_rows << ", #subimg_cols: " << subimg_cols;
  LOG_INFO << "#bbox: " << bbox;

  const double step = config.get<double>("scene.step");
  const double bbox_xmin = bbox.xmin(), bbox_xmax = bbox.xmax();
  const double bbox_ymin = bbox.ymin(), bbox_ymax = bbox.ymax();
  const double bbox_zmin = bbox.zmin(), bbox_zmax = bbox.zmax();
  const double ystep = step * double(subimg_rows);
  const double xstep = step * double(subimg_cols);

  int num = 0;
  try {
#pragma omp parallel for
    for (int r = 0; r < nb_rows; ++r) {
      for (int c = 0; c < nb_cols; ++c) {

        // stripped directory prefix
        std::stringstream ss;
        ss << wdir << std::setfill('0')
          << std::setw(3) << r
          << std::setw(3) << c;

        const double ulx = bbox_xmin + c * xstep;
        const double uly = bbox_ymax - r * ystep;
        UBlock bk(ss.str());
        bk.load_data();
        bk.compute_height_and_normal_map(ulx, uly);
        bk.boost_tree_MRF_labeling();

        // retrieve buildings
        
        auto buildings = bk.retrieve_buildings(input, output, bbox_xmin, bbox_ymin, step);
        num += buildings.size();
        for (const auto &b : buildings)
          b->segment_arrangement_modeling();
        // write_scene_lod2();
         // lod2
          int nump = 0;
          std::vector<Point_3> points;
          std::vector<std::vector<std::size_t>> polygons;
          for (int i = 0; i < buildings.size(); i++) {
            std::ifstream ifs(ss.str() + "_b" + std::to_string(i) + "_lod2.off");
            if (!ifs)
              continue;
            std::string f;
            ifs >> f;
            int np, nf;
            ifs >> np >> nf;
            double x, y, z;
            ifs >> x;
            for (int i = 0; i < np; i++) {
              ifs >> x >> y >> z;
              points.push_back(Point_3(x, y, z));
            }
            for (int i = 0; i < nf; i++) {
              int nfp;
              ifs >> nfp;
              std::vector<std::size_t> poly;
              for (int j = 0; j < nfp + 3; j++) {
                int nfp_;
                ifs >> nfp_;
                if (j < nfp)
                  poly.push_back(nfp_ + nump);
                else
                  poly.push_back(nfp_);
              }
              polygons.push_back(poly);
            }
            nump += np;
          }
          std::ofstream ofs(ss.str() + "_lod2.off");
          ofs << "COFF\n" << points.size() << ' ' << polygons.size() << ' ' << "0\n";
          for (const auto &p : points)
            ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
          for (const auto &plg : polygons) {
            ofs << (plg.size() - 3);
            for (const auto &p : plg)
              ofs << ' ' << p;
            ofs << '\n';
          }
        LOG_WARNING << "#<" << r << ", " << c << "> done.";
      }
    }
    write_scene_lod2(nb_rows, nb_cols, static_cast<std::string>(wdir));
    LOG_INFO << "#builging: " << num;
  }
  catch (std::exception &e) {
    LOG_ERROR << e.what();
  }

  return 0;
}
