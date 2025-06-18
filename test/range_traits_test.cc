/**
 * @file range_traits_test.cc
 * @author Derek Huang
 * @brief range_traits.hh unit tests
 * @copyright MIT License
 */

#include <array>
#include <cstdlib>
#include <deque>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "npygl/range_traits.hh"
#include "npygl/termcolor.hh"
#include "npygl/testing/traits_checker.hh"

namespace {

// test driver type
using driver_type = npygl::testing::traits_checker_driver<
  // is_range
  npygl::testing::traits_checker<
    npygl::is_range,
    std::tuple<
      std::vector<double>,
      std::array<int, 64u>,
      std::string,
      std::set<std::string>,
      std::pair<int, std::false_type>,
      // reference to a fixed-sized array is a range
      double(&)[44],
      // array type itself is not a range (no std::begin overload)
      std::pair<double[44], std::false_type>,
      // unbounded arrays are not ranges
      std::pair<double[], std::false_type>
    >
  >,
  // is_sized_range
  npygl::testing::traits_checker<
    npygl::is_sized_range,
    std::tuple<
      std::pair<int, std::false_type>,
      std::vector<double>,
      std::set<void*>,
      std::array<double, 256u>,
      std::deque<unsigned>,
      // reference to a fixed-sized array is a sized range
      void*(&)[512],
      std::string
    >
  >
>;

// test driver
constexpr driver_type driver;

}  // namespace

int main()
{
  npygl::vts_stdout_context ctx;
  return (driver()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
