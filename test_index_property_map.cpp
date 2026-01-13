#include <iostream>
#include <vector>
#include <array>
#include <string>

#include "utils/Index_property_map.h"

int main() {
  {
    typedef std::vector<int> Sequence_container;
    typedef cm::Index_property_map<Sequence_container> Ipmap;
    Sequence_container sc = { 1, 3, 5, 23, 57, 31 };
    Ipmap ipmap(sc);

    for (int i = 0; i < sc.size(); ++i)
      std::cout << ipmap[i] << ' ';
    std::cout << std::endl;
    ipmap[0] = 13;
    put(ipmap, 3, 11);
    for (int i = 0; i < sc.size(); ++i)
      std::cout << get(ipmap, i) << ' ';
    std::cout << std::endl;
  }
  {
    typedef std::array<int, 6> Sequence_container;
    typedef cm::Index_property_map_const<Sequence_container> Ipmap;
    const Sequence_container sc = { 1, 3, 5, 23, 57, 31 };
    Ipmap ipmap(sc);

    for (int i = 0; i < sc.size(); ++i)
      std::cout << ipmap[i] << ' ';
    std::cout << std::endl;
    for (int i = 0; i < sc.size(); ++i)
      std::cout << get(ipmap, i) << ' ';
    std::cout << std::endl;
  }
  {
    typedef std::string Sequence_container;
    typedef cm::Index_property_map_const<Sequence_container> Ipmap;
    Sequence_container sc("hello world");
    Ipmap ipmap(sc);

    for (int i = 0; i < sc.size(); ++i)
      std::cout << ipmap[i] << ' ';
    std::cout << std::endl;
    sc[0] = 'H';
    for (int i = 0; i < sc.size(); ++i)
      std::cout << get(ipmap, i) << ' ';
    std::cout << std::endl;
  }
}
