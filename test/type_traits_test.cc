/**
 * @file type_traits_test.cc
 * @author Derek Huang
 * @brief type_traits.hh unit tests
 * @copyright MIT License
 */

#include <cstdlib>
#include <functional>
#include <map>
// GCC requires <memory_resource> for std::pmr::allocator to be complete
#include <memory_resource>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "npygl/testing/ostream.hh"
#include "npygl/testing/traits_checker.hh"
#include "npygl/type_traits.hh"
#include "npygl/warnings.h"

// repurpose this preprocessor definition so we can compile this source file as
// a smoke test for using llvm::itaniumDemangle on driver_type. we have noted
// that for GCC 11.3.0, abi::__cxa_demangle can't demangle driver_type, likely
// because driver_type produces a mangled name that is "too long". at the very
// least, the generated mangled name exceeds 1024 characters. this means that
// tools like nm and c++filt will not be able to correctly demangle the name.
//
// fortunately, llvm::itaniumDemangle works properly. type_name() has also been
// updated to act like boost::core::demangle() and return the mangled name in
// case demangling with abi::__cxa_demangle fails.
//
#if defined(NPYGL_USE_LLVM_DEMANGLE)
// not strictly necessary but we do it for correctness
#include <iostream>
#include <string_view>
// for well-formed typeid() usage
#include <typeinfo>
#include "npygl/demangle.hh"
#endif  // defined(NPYGL_USE_LLVM_DEMANGLE)

// allow repurposing this test program as a smoke test for npygl::type_name<T>.
// this is more efficient (possible compile-time computation) and does not
// require any additional libraries to be included
#ifdef NPYGL_USE_CTTI_DEMANGLE
// not meaningful if both macros are defined
#ifdef NPYGL_USE_LLVM_DEMANGLE
#error "type_traits_test.cc: cannot define both NPYGL_USE_CTTI_DEMANGLE and " \
  "NPYGL_USE_LLVM_DEMANGLE"
#endif  // NPYGL_USE_LLVM_DEMANGLE
// for npygl::type_name<T>
#include "npygl/ctti.hh"
#endif  // NPYGL_USE_CTTI_DEMANGLE

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
  // test partially_fixed base specialization
  npygl::testing::traits_checker<
    npygl::partially_fixed<npygl::is_same_type, char, char, char, char>::type,
    std::tuple<char, std::pair<int, std::false_type>>
  >,
  // test partially_fixed with fix_first_types
  npygl::testing::traits_checker<
    npygl::partially_fixed<npygl::is_same_type, npygl::fix_first_types, int>::
      type,
    std::tuple<std::pair<int, std::true_type>, std::pair<double, std::false_type>>
  >,
  // test partially_fixed with fix_last_types
  npygl::testing::traits_checker<
    npygl::partially_fixed<npygl::is_same_type, npygl::fix_last_types, int>::
      type,
    std::tuple<int, std::pair<std::string, std::false_type>>
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
  >,
  // has_static_size
  npygl::testing::traits_checker<
    npygl::has_static_size,
    std::tuple<
      std::tuple<int, double, char>,
      std::pair<std::vector<double>, std::false_type>,
      std::pair<std::tuple<char, unsigned, std::string>, std::true_type>,
      std::pair<double, const volatile void*>,
      std::array<unsigned, 100>,
      char[256],
      std::pair<std::string, std::false_type>,
      std::pair<std::map<std::string, unsigned>, std::false_type>,
      int[2][3][4],
      std::pair<double[], std::false_type>
    >
  >,
  // static_size (indirectly tests static_size_traits)
  npygl::testing::traits_checker<
    npygl::static_size,
    std::tuple<
      std::pair<
        std::vector<double>,
        npygl::testing::traits_value_is_not_equal<unsigned, 2u>
      >,
      std::pair<
        std::pmr::vector<unsigned>,
        npygl::testing::traits_value_is_equal<unsigned, 1u>
      >,
      std::pair<
        const double[256],
        npygl::testing::traits_value_is_equal<unsigned, 256u>
      >,
      std::pair<
        std::array<int, 32>,
        npygl::testing::traits_value_is_equal<unsigned, 32u>
      >,
      // some silly comparisons
      std::pair<
        std::array<int, 64>,
        npygl::testing::traits_value_is_greater<unsigned, 24>
      >,
      std::pair<
        unsigned[1024],
        npygl::testing::traits_value_is_less_equal<unsigned long, 2048>
      >,
      // multidimensional fixed-size array
      std::pair<
        double[24][32][46],
        npygl::testing::traits_value_is_equal<unsigned, 24>
      >,
      // skip testing for traits_value_comparison
      // TODO: move these to dedicated traits_checker.hh tests
      npygl::testing::skipped<
        std::pair<
          const volatile void*[2][4][8][16],
          npygl::testing::traits_value_is_equal<unsigned, 2>
        >
      >,
      npygl::testing::skipped<
        std::pair<
          std::pmr::vector<double>,
          npygl::testing::traits_value_is_greater<unsigned, 100>
        >
      >,
      // multidimensional array of unknown first bound
      std::pair<
        double[][32][16],
        npygl::testing::traits_value_is_equal<unsigned, 1>
      >
    >
  >,
  // is_ostreamable
  npygl::testing::traits_checker<
    npygl::is_ostreamable,
    std::tuple<
      int,
      double,
      char,
      const char*,
      std::string,
      std::pair<std::vector<std::pmr::vector<unsigned>>, std::false_type>,
      std::pair<std::tuple<std::tuple<std::tuple<std::string>>>, std::false_type>,
      std::pair<npygl::testing::not_ostreamable_type, std::false_type>,
      npygl::testing::ostreamable_type
    >
  >,
  // is_iterable
  npygl::testing::traits_checker<
    npygl::is_iterable,
    std::tuple<
      std::array<double, 100u>,
      double(&)[1024],
      std::pmr::vector<unsigned>,
      std::pair<double, std::false_type>,
      std::string,
      std::pair<npygl::testing::not_ostreamable_type, std::false_type>,
      std::pair<std::tuple<double, int, std::string>, std::false_type>
    >
  >
>;
// test driver instance. but if testing demangling we won't be needing it
#if !defined(NPYGL_USE_LLVM_DEMANGLE) && !defined(NPYGL_USE_CTTI_DEMANGLE)
constexpr driver_type driver;
#endif  // !defined(NPYGL_USE_LLVM_DEMANGLE) || !defined(NPYGL_USE_CTTI_DEMANGLE)

}  // namespace

int main()
{
// smoke test for llvm::itaniumDemangle
#if defined(NPYGL_USE_LLVM_DEMANGLE)
  std::cout << npygl::type_name(typeid(driver_type)) << std::endl;
// smoke test for npygl::type_name<T>
#elif defined(NPYGL_USE_CTTI_DEMANGLE)
  std::cout << npygl::type_name<driver_type>() << std::endl;
// standard traits checker driver main
#else
  npygl::vts_stdout_context ctx;
  return (driver()) ? EXIT_SUCCESS : EXIT_FAILURE;
#endif  // !defined(NPYGL_USE_LLVM_DEMANGLE)
}
