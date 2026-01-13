#include <iostream>

#include "utils/Config.h"

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  cm::read_config(argv[1]);
  const auto &config = cm::get_config();
  boost::property_tree::write_xml(std::cout, config);

  return 0;
}
