/**
 * @file type_traits_test.cc
 * @author Derek Huang
 * @brief type_traits.hh unit tests
 * @copyright MIT License
 */

#include <cstdlib>

#include "npygl/termcolor.hh"
#include "npygl/testing/type_traits_test_driver.hh"

namespace {

// test driver
constexpr npygl::testing::type_traits_test_driver driver;

}  // namespace

int main()
{
  npygl::vts_stdout_context ctx;
  return (driver()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
