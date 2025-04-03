/**
 * @file ctti_soak_test.cc
 * @author Derek Huang
 * @brief C++ program testing `type_name<T>` on `type_traits_test_driver`
 * @copyright MIT License
 *
 * @file See npygl::testing::type_traits_test_driver for some historical
 *  background and why we are testing `npygl::type_name<T>` in this manner.
 */

#include <cstdlib>
#include <iostream>

#include "npygl/ctti.hh"
#include "npygl/testing/type_traits_test_driver.hh"

namespace {

// ensure that type_name<T> cam be evaluated at compile-time
using driver_type = npygl::testing::type_traits_test_driver;
constexpr auto driver_type_name = npygl::type_name<driver_type>();

}  // namespace

int main()
{
  if constexpr (npygl::type_name_supported())
    std::cout << driver_type_name << std::endl;
  else
    std::cout << "SKIPPED: npygl::type_name<T> not supported" << std::endl;
  return EXIT_SUCCESS;
}
