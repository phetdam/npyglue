/**
 * @file ctti_test.cc
 * @author Derek Huang
 * @brief C++ program to test ctti.hh compile-time type info helpers
 * @copytighr MIT License
 */

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>  // for well-formed typeid() usage
#include <utility>
#include <vector>

#include "npygl/common.h"
#include "npygl/ctti.hh"
#include "npygl/demangle.hh"
#include "npygl/termcolor.hh"
#include "npygl/warnings.h"

namespace {

/**
 * Test runner for `type_name<T>` tests.
 *
 * This collects a running total of the executed tests and their status.
 */
class type_name_test_suite {
public:
  /**
   * Result struct.
   *
   * @param success Indicate if test succeeded or not
   * @param test_name Name of test, e.g. "T"
   * @param reason Failure reason, empty if success
   */
  struct result {
    bool success;
    std::string test_name;
    std::string reason;
  };

  /**
   * Return the vector of test results.
   */
  const auto& results() const noexcept { return results_; }

  /**
   * Return the number of failed tests.
   */
  auto failed() const noexcept { return failed_; }

  /**
   * Evaluate a test case and add to the existing results.
   *
   * @tparam T Input type
   *
   * @param expected Expected `npygl::type_name<T>` output
   */
  template <typename T>
  auto& add(std::string_view expected) noexcept
  {
    // test: force constexpr computation
    constexpr auto actual = npygl::type_name<T>();
    // check
    bool match = (expected == actual);
    // create cresult. on failure, update fail count + add reason
    result res{match, npygl::type_name(typeid(T))};
    // note: no operator+ for string_view yet until C++20
    if (!match) {
      failed_++;
      res.reason += "[expected \"";
      res.reason += expected;
      res.reason += "\" != actual \"";
      res.reason += actual;
      res.reason += "\"]";
    }
    // append to results
    results_.push_back(std::move(res));
    return *this;
  }

  /**
   * Return the total number of test results.
   */
  auto size() const noexcept
  {
    return results_.size();
  }

private:
  std::vector<result> results_;
  std::size_t failed_{};
};

/**
 * Write the test suite results to the output stream.
 *
 * Colored results are written only if the stream is `std::cout`.
 *
 * @param out Output stream
 * @param suite Test suite
 */
auto& operator<<(std::ostream& out, const type_name_test_suite& suite)
{
  using namespace npygl::vts;
  // color text formatter
  auto format = [&out](npygl::sgr_value color, const std::string& str)
  {
    // if printing to stdout, use color
    if (&out == &std::cout) {
      std::stringstream ss;
      ss << color << str << fg_normal;
      return ss.str();
    }
    // otherwise, no color
    return str;
  };
  // format results
  for (const auto& res : suite.results()) {
    // formatting color + message
    auto color = (res.success) ? fg_green : fg_red;
    auto message = "[ " + std::string{(res.success) ? "PASS" : "FAIL"} + " ]";
    // print test result
    out << format(color, message) << ' ' << res.test_name;
    if (!res.success)
      out << " " << res.reason;
    // end line
    out << '\n';
  }
  // number of failed and total tests
  auto n_failed = suite.failed();
  auto n_total = suite.size();
  // percentage of passed tests to 2 decimal places
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4244 5219)
  unsigned pass_pct = (10000 * (1 - (1. * n_failed) / n_total) / 100);
NPYGL_MSVC_WARNING_POP()
  // percentage format color + failed format color
  auto pct_color = (n_failed) ? fg_normal : fg_green;
  auto fail_color = (n_failed) ? fg_red : fg_normal;
  // print summary
  out << '\n' <<
    format(pct_color, std::to_string(pass_pct) + "% tests passed") << ", " <<
    format(fail_color, std::to_string(n_failed) +  " failed") << " out of " <<
    n_total;
  return out;
}

/**
 * Global test suite for the `type_name<T>` tests.
 */
auto& type_tests()
{
  static type_name_test_suite suite;
  return suite;
}

/**
 * User defined test type.
 */
struct test_type_1 {};

/**
 * User defined test template type.
 */
template <typename... Ts>
struct test_template_type {};

// conditionally-defined macros for expected type name strings

// std::string
#if defined(_WIN32)
#define STD_STRING_NAME \
  "class std::basic_string<char," \
    "struct std::char_traits<char>,class std::allocator<char> >"
#elif defined(__clang__)
#define STD_STRING_NAME "std::basic_string<char>"
#else
#define STD_STRING_NAME "std::__cxx11::basic_string<char>"
#endif  // !defined(_WIN32) && !defined(__clang__)
// std::vector<std::string>
#if defined(_WIN32)
#define STD_STRING_VECTOR_NAME \
  "class std::vector<" STD_STRING_NAME "," \
    "class std::allocator<" STD_STRING_NAME " > >"
#elif defined(__clang__)
#define STD_STRING_VECTOR_NAME "std::vector<" STD_STRING_NAME ">"
#else
#define STD_STRING_VECTOR_NAME "std::vector<" STD_STRING_NAME " >"
#endif  // !defined(_WIN32) && !defined(__clang__)
// const volatile double*
#if defined(_WIN32)
#define DOUBLE_CV_PTR_NAME "volatile const double*"
#else
#define DOUBLE_CV_PTR_NAME "const volatile double*"
#endif  // !defined(_WIN32)
// test_type_1
#if defined(_WIN32)
#define TEST_TYPE_1_NAME "struct `anonymous-namespace'::test_type_1"
#elif defined(__clang__)
#define TEST_TYPE_1_NAME "(anonymous namespace)::test_type_1"
#else
#define TEST_TYPE_1_NAME "{anonymous}::test_type_1"
#endif  // !defined(_WIN32) && !defined(__clang__)
// test_template_type<>
#if defined(_WIN32)
#define TEST_TEMPLATE_TYPE_NAME "struct `anonymous-namespace'::test_template_type<>"
#elif defined(__clang__)
#define TEST_TEMPLATE_TYPE_NAME "(anonymous namespace)::test_template_type<>"
#else
#define TEST_TEMPLATE_TYPE_NAME "{anonymous}::test_template_type<>"
#endif  // !defined(_WIN32) && !defined(__clang__)
// test_template_type<std::string, double, const char*>
#if defined(_WIN32)
#define TEST_TEMPLATE_TYPE_STD_STRING_DOUBLE_CHAR_C_PTR_NAME \
  "struct `anonymous-namespace'::test_template_type<" STD_STRING_NAME "," \
    "double,char const *>"
#elif defined(__clang__)
#define TEST_TEMPLATE_TYPE_STD_STRING_DOUBLE_CHAR_C_PTR_NAME \
  "(anonymous namespace)::test_template_type<" STD_STRING_NAME \
    ", double, const char *>"
#else
// note: STD_STRING_NAME is the "short" string type. it is not used when
// std::string is part of this user-defined template. maybe a bug?
#define TEST_TEMPLATE_TYPE_STD_STRING_DOUBLE_CHAR_C_PTR_NAME \
  "{anonymous}::test_template_type<std::__cxx11::basic_string<char, " \
    "std::char_traits<char>, std::allocator<char> >, double, const char*>"
#endif  // !defined(_WIN32) && !defined(__clang__)
// std::ostream
#if defined(_WIN32)
#define STD_OSTREAM_NAME \
  "class std::basic_ostream<char,struct std::char_traits<char> >"
#else
#define STD_OSTREAM_NAME "std::basic_ostream<char>"
#endif  // !defined(_WIN32)
// const void* const
#if defined(_WIN32)
#define VOID_C_PTR_C "const void*const " /* extra whitespace is not a typo */
#elif defined(__clang__)
#define VOID_C_PTR_C "const void *const"
#else
#define VOID_C_PTR_C "const void* const"
#endif  // !defined(_WIN32) && !defined(__clang__)

/**
 * Program main.
 *
 * @tparam supported `true` if `type_name<T>` supported, `false` otherwise
 * @returns `true` on success, `false` on testing failure
 */
template <bool supported>
bool main()
{
  if constexpr (supported) {
    // enables color printing on Windows
    npygl::vts_stdout_context ctx;
    // evaluate tests
    type_tests()
      .add<double>("double")
      .add<void*>("void*")
      .add<std::string>(STD_STRING_NAME)
      .add<std::vector<std::string>>(STD_STRING_VECTOR_NAME)
      .add<const volatile double*>(DOUBLE_CV_PTR_NAME)
      .add<test_type_1>(TEST_TYPE_1_NAME)
      .add<test_template_type<>>(TEST_TEMPLATE_TYPE_NAME)
#define EXPECTED_NAME TEST_TEMPLATE_TYPE_STD_STRING_DOUBLE_CHAR_C_PTR_NAME
      .add<test_template_type<std::string, double, const char*>>(EXPECTED_NAME)
#undef EXPECTED_NAME
      .add<std::ostream>(STD_OSTREAM_NAME)
      // note: the runtime-demangled name omits the pointer const qualification
      .add<const void* const>(VOID_C_PTR_C);
    // print result
    std::cout << type_tests() << std::endl;
    // success if none failed
    return !type_tests().failed();
  }
  else {
    std::cerr << "npygl::type_name<T> is not supported" << std::endl;
    return true;
  }
}

}  // namespace

int main()
{
  return main<npygl::type_name_supported()>() ? EXIT_SUCCESS : EXIT_FAILURE;
}
