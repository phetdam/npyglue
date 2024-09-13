/**
 * @file py_format_test.cc
 * @author Derek Huang
 * @brief C++ program to test that Python format string helpers work properly
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <utility>

#include "npygl/demangle.hh"
#include "npygl/python.hh"

namespace {

/**
 * Functor to check that a type pack yields the expected Python format string.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct format_checker {

  /**
   * Check that the types produces the expected Python format string.
   *
   * @tparam Ts... types
   * @tparam N Length of format string literal
   *
   * @param failed Number of failures to update (untouched if no error)
   * @param fmt String literal expected format string
   */
  template <std::size_t N>
  void operator()(unsigned int& failed, const char (&fmt)[N]) const
  {
    // need to have enough types to match the given format string
    static_assert(sizeof...(Ts) + 1 == N, "not enough types to match format");
    // call implementation
    check(failed, std::index_sequence_for<Ts...>{}, fmt);
  }

private:
  /**
   * Check that the types produce the expected Python format string.
   *
   * This is the implementation function that uses the index pack for indexing.
   *
   * @tparam Is... Index values from 0 through N - 1
   * @tparam N Length of format string literal
   *
   * @param failed Number of failures to update (untouched if no error)
   * @param seq Unused index sequence to deduce index pack
   * @param fmt String literal expected format string
   */
  template <std::size_t... Is, std::size_t N>
  void check(
    unsigned int& failed,
    std::index_sequence<Is...> /*seq*/,
    const char (&fmt)[N]) const
  {
    // tuple of types for convenient indexing
    using tuple_type = std::tuple<Ts...>;
    // format type being tested (also tests the tuple partial specialization)
    using format_type = npygl::py_format_type<tuple_type>;
    // each test case announces the comparison being done
    std::cout << "Checking format of " <<
      npygl::type_name(typeid(format_type)) << "... " << std::flush;
    // character by character check
    bool mismatch = false;
    (
      [&]
      {
        // no mismatch
        if (npygl::py_format<tuple_type>[Is] == fmt[Is])
          return;
        // mismatch
        std::cout << "\n  Mismatch at index " << Is << ": " <<
          npygl::py_format<tuple_type>[Is] << " != " << fmt[Is] <<
          " (expected)" << std::flush;
        mismatch = true;
      }()
      ,
      ...
    );
    // at least one failure
    if (mismatch)
      failed++;
    // print success
    else
      std::cout << "ok";
    // need final newline
    std::cout << std::endl;
  }
};

/**
 * Partial specialization to allow use of a tuple of types.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct format_checker<std::tuple<Ts...>> : format_checker<Ts...> {};

/**
 * Global object to allow function-like usage of the `format_checker`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
constexpr format_checker<Ts...> check_format{};

}  // namespace

int main()
{
  // TODO: have some main function that takes std::tuple<std::tuple<Ts...>, ...>
  // so we can know how many inputs are being sent. also may need an input type
  unsigned int failed = 0;
  // 1.
  using types_1 = std::tuple<int, short, npygl::py_optional_args, PyObject*>;
  check_format<types_1>(failed, "ih|O");
  // 2.
  using types_2 = std::tuple<
    double, float, PyObject*,
    npygl::py_optional_args, Py_ssize_t, const char*, Py_complex
  >;
  check_format<types_2>(failed, "dfO|nsD");
  // 3.
  using types_3 = std::tuple<
    const char*, const char*, PyObject*,
    npygl::py_optional_args, Py_complex, Py_complex, int, Py_ssize_t
  >;
  check_format<types_3>(failed, "ssO|DDin");
  // all the input types + number of tests in total
  using all_types = std::tuple<types_1, types_2, types_3>;
  constexpr auto n_tests = std::tuple_size_v<all_types>;
  // report failure/success
  if (failed)
    std::cout << "Failed: " << failed << " of " << n_tests << std::endl;
  else
    std::cout << "Passed " << n_tests << " of " << n_tests << std::endl;
  return !!failed;
}
