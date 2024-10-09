/**
 * @file type_traits_test.cc
 * @author Derek Huang
 * @brief type_traits.hh unit tests
 * @copyright MIT License
 */

#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "npygl/testing/traits_checker.hh"
#include "npygl/type_traits.hh"

namespace {

// test driver type
using driver_type = npygl::testing::traits_checker_driver<
  // is_pair
  npygl::testing::traits_checker<
    npygl::is_pair,
    std::tuple<
      std::pair<int, int>,
      std::pair<double, double>,
      std::pair<std::pair<double, int>, std::true_type>,
      std::pair<double, std::false_type>,
      std::pair<std::pmr::vector<double>, std::false_type>
    >
  >,
  // is_tuple
  npygl::testing::traits_checker<
    npygl::is_tuple,
    std::tuple<
      std::tuple<int, double, int>,
      std::tuple<double, std::string, double>,
      std::pair<std::pair<double, int>, std::false_type>,
      std::pair<std::tuple<std::tuple<unsigned, char>, int>, std::true_type>,
      std::pair<std::map<std::string, std::vector<double>>, std::false_type>
    >
  >,
  // always_true
  npygl::testing::traits_checker<
    npygl::always_true,
    std::tuple<double, std::string, unsigned, char, std::map<unsigned, int>>
  >,
  // has_type_member
  // note: also indirectly tests same_type + type_filter
  npygl::testing::traits_checker<
    npygl::has_type_member,
    std::tuple<
      std::pair<npygl::same_type<int, double, unsigned>, std::false_type>,
      std::enable_if<true>,
      std::pair<std::enable_if<false>, std::false_type>,
      std::pair<std::tuple<std::pair<int, int>, std::string>, std::false_type>,
      std::remove_pointer<const char*>,
      npygl::same_type<char, char, char, char, char, char>,
      std::add_pointer<std::string>,
      npygl::same_type<
        npygl::type_filter_t<
          npygl::has_type_member,
          int,
          std::enable_if<true>,
          std::remove_reference<const std::vector<unsigned>&>,
          double,
          std::string
        >,
        std::tuple<
          std::enable_if<true>,
          std::remove_reference<const std::vector<unsigned>&>
        >
      >,
      npygl::same_type<
        npygl::type_filter_t<npygl::always_true, double, int, unsigned, char>,
        std::tuple<double, int, unsigned, char>
      >
    >
  >
>;

}  // namespace

int main()
{
  driver_type driver;
  return (driver()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
