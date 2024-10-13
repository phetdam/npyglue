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
    std::tuple<
      double,
      std::string,
      unsigned,
      char,
      std::map<unsigned, int>,
      // test skipped<>
      //
      // note:
      //
      // when any more skipped<> are added after skipped<double> on WSL1 Ubuntu
      // 22.04 we are seeing c++filt, abi::__cxa_demangle, nm become unable to
      // demangle the mangled type name. GCC 11.3.0 is used here. it appears
      // that once the type name gets long enough, e.g. more than 4151 chars
      // demangled or 1019 chars mangled (1024 may be the limit), the standard
      // demangling facilities stop working. have not cross-checked against
      // llvm-cxxfilt (some people report this as working when c++filt fails).
      //
      // for now, if intending to demangle driver_type and type_name() throws
      // an exception due to abi::__cxa_demangle being unable to demangle the
      // type name, ensure that the overall driver_type is not too long.
      //
      npygl::testing::skipped<double>,
      npygl::testing::skipped<std::pair<std::pair<int, char>, std::false_type>>,
      npygl::testing::skipped<std::pair<double, std::false_type>>
    >
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
  >,
  // is_monomorphic_tuple
  // note: also tests monomorphic_tuple_type, type_filter, has_type_member
  npygl::testing::traits_checker<
    npygl::is_monomorphic_tuple,
    std::tuple<
      std::tuple<int, int>,
      std::pair<std::tuple<double, int, double>, std::false_type>,
      std::pair<std::tuple<std::string, std::string>, std::true_type>,
      std::tuple<unsigned, unsigned, unsigned, unsigned>,
      std::pair<
        std::tuple<std::string, std::string, int, std::string, std::string>,
        std::false_type
      >,
      std::tuple<std::vector<unsigned>, std::vector<unsigned>>,
      std::tuple<void*, void*, void*, void*>,
      std::pair<std::tuple<std::map<unsigned, int>, char>, std::false_type>,
      std::tuple<
        npygl::type_filter_t<npygl::always_true, int, double, int, unsigned>,
        std::tuple<int, double, int, unsigned>
      >,
      std::tuple<
        npygl::type_filter_t<
          npygl::always_true,
          std::enable_if<true>,
          std::decay<const volatile void*>
        >,
        std::tuple<std::enable_if<true>, std::decay<const volatile void*>>,
        npygl::type_filter_t<
          npygl::has_type_member,
          std::enable_if<true>,
          std::decay<const volatile void*>,
          std::enable_if<false>,
          std::enable_if<false>
        >
      >,
      std::pair<std::tuple<unsigned, int>, std::false_type>
    >
  >
>;
// test driver instance
constexpr driver_type driver;

}  // namespace

int main()
{
  npygl::vts_stdout_context ctx;
  return (driver()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
