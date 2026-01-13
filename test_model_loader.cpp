#include <iostream>

#include "utils/Model_loader.h"

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  try {
    cm::Model_loader model(argv[1]);
    for (const auto &m : model.meshes) {
      std::cout << "#v " << m.number_of_vertices() << std::endl;
      std::cout << "#f " << m.number_of_faces() << std::endl;
    }
  }
  catch (const std::exception &e) {
    std::cout << "ERROR::" << e.what() << std::endl;
  }

  return 0;
}
