/**
 * @file ndspan_test.cc
 * @author Derek Huang
 * @brief C++ program for ndspan.hh tests
 * @copyright MIT License
 */

#include <cstdlib>
#include <iostream>
#include <vector>

#include "npygl/ctti.hh"
#include "npygl/ndspan.hh"

int main()
{
  // 3D cube
  std::vector vec{
    // matrix 0
    1., 2.,
    3., 4.,
    5., 6.,
    // matrix 1
    7., 8.,
    9., 10.,
    11., 12.
  };
  // row-major view
  npygl::ndspan cv{vec.data(), npygl::ndextents{2u, 3u, 2u}};
  // print slices
  std::cout <<
    "cv: " << cv << "\n" <<
    "cv[1]: " << cv[1] << "\n" <<
    "cv[0][1]: " << cv[0][1] << "\n" <<
    "cv[0][1][1]: " << cv[0][1][1] << std::endl;
  // directly index
  std::cout << "cv(0, 1, 1): " << cv(0, 1, 1) << std::endl;
  return EXIT_SUCCESS;
}
