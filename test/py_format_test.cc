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

/**
 * Input type holding a format type and the input types.
 *
 * @tparam N Length of format string literal
 * @tparam Ts... Types to use in the format string
 *
 * @param fmt Reference to the expected string literal format string
 */
template <std::size_t N, typename... Ts>
struct format_input {
  using format_types = std::tuple<Ts...>;
  const char (&fmt)[N];
};

/**
 * Partial specialization taking a tuple of input types.
 *
 * @tparam N Length of format string literal
 * @tparam Ts... Types to use in the format string
 *
 * @param fmt Reference to the expected string literal format string
 */
template <std::size_t N, typename... Ts>
struct format_input<N, std::tuple<Ts...>> {
  using format_types = std::tuple<Ts...>;
  const char (&fmt)[N];
};

/**
 * Traits helper to check if several types are `std::tuple`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct is_tuple : std::false_type {};

/**
 * True specialization when the type is a single `std::tuple`.
 *
 * @tparam Ts... Tuple types
 */
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

/**
 * Recursive specialization when there is more than one `std::tuple`.
 *
 * @tparam Ts... Tuple types
 * @tparam Us... Subsequent types to check
 */
template <typename... Ts, typename... Us>
struct is_tuple<std::tuple<Ts...>, Us...> : is_tuple<Us...> {};

/**
 * Check if all the types are `std::tuple`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr bool is_tuple_v = is_tuple<Ts...>::value;

/**
 * Make a `format_input` from a type and a string literal.
 *
 * This allows deducing the string literal length from a function parameter.
 *
 * @note For convenience of template deduction the
 *  `format_input<N, std::tuple<Ts...>>` is used when creating the inputs.
 *
 * @tparam T Tuple of input types
 * @tparam N Length of format string literal
 */
template <typename T, std::size_t N, typename = std::enable_if_t<is_tuple_v<T>>>
constexpr auto make_input(const char (&fmt)[N])
{
  return format_input<N, T>{fmt};
}

/**
 * Test main function.
 *
 * @tparam Ns... String literal expected format lengths
 * @tparam Inputs... Tuples of input types
 *
 * @param inputs... Testing inputs
 * @returns 0 on success, 1 on failure
 */
template <std::size_t... Ns, typename... Inputs>
std::enable_if_t<is_tuple_v<Inputs...>, int>
test_main(const format_input<Ns, Inputs>&... inputs)
{
  // number of inputs/tests + number of failed
  constexpr auto n_tests = sizeof...(Ns);
  unsigned int n_failed = 0;
  // loop through pack to update number of failed
  (check_format<Inputs>(n_failed, inputs.fmt), ...);
  // report failure/success
  if (n_failed)
    std::cout << "Failed: " << n_failed << " of " << n_tests << std::endl;
  else
    std::cout << "Passed " << n_tests << " of " << n_tests << std::endl;
  return !!n_failed;
}

}  // namespace

int main()
{
  // input types
  using types_1 = std::tuple<int, short, npygl::py_optional_args, PyObject*>;
  using types_2 = std::tuple<
    double, float, PyObject*,
    npygl::py_optional_args, Py_ssize_t, const char*, Py_complex
  >;
  using types_3 = std::tuple<
    const char*, const char*, PyObject*,
    npygl::py_optional_args, Py_complex, Py_complex, int, Py_ssize_t
  >;
  using types_4 = std::tuple<
    Py_ssize_t, PyBytesObject*,
    npygl::py_optional_args, double, Py_complex, PyObject*
  >;
  // run tests
  return test_main(
    make_input<types_1>("ih|O"),
    make_input<types_2>("dfO|nsD"),
    make_input<types_3>("ssO|DDin"),
    make_input<types_4>("nS|dDO")
  );
}
