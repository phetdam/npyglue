/**
 * @file py_convert_test.cc
 * @author Derek Huang
 * @brief C++ program testing Python to C++ value conversions
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <tuple>
#include <type_traits>

#include "npygl/ctti.hh"
#include "npygl/ostream.hh"
#include "npygl/python.hh"
#include "npygl/termcolor.hh"

namespace {

/**
 * Traits type for getting the `i`th input test case.
 *
 * If if the index `I` exceeds the number of input types this is a hard error.
 *
 * @tparam I Index
 * @tparam T Type
 * @tparam Ts Type pack
 */
template <std::size_t I, typename T, typename... Ts>
struct type_at {
  static_assert(I < 1 + sizeof...(Ts), "I >= the number of types");
  using type = typename type_at<I - 1, Ts...>::type;
};

/**
 * Get the `i`th type in the given parameter pack.
 *
 * @tparam I Index
 * @tparam Ts Type pack
 */
template <std::size_t I, typename... Ts>
using type_at_t = typename type_at<I, Ts...>::type;

/**
 * Specialization when the index is zero.
 *
 * @tparam T Type
 * @tparam Ts Type pack (may be empty)
 */
template <typename T, typename... Ts>
struct type_at<0u, T, Ts...> {
  using type = T;
};

/**
 * Type to represent an entire suite of test inputs.
 *
 * @tparam Ts Input case types
 */
template <typename... Ts>
struct converter_test_list {};

/**
 * Partial specialization for a `converter_test_list<Ts...>`.
 *
 * @tparam Ts Input types
 */
template <std::size_t I, typename... Ts>
struct type_at<I, converter_test_list<Ts...>> : type_at<I, Ts...> {};

/**
 * Test runner for the conversion tests.
 *
 * Each input case must implement the following:
 *
 * @code{.cc}
 * static auto expected() noexcept;
 * @endcode
 *
 * Function can be `constexpr`. If the value returned by `expected()` cannot be
 * directly constructed into a `py_object` you can provide your own method of
 * constructing the `py_object` input with the following:
 *
 * @code{.cc}
 * py_object operator()() noexcept;
 * @endcode
 *
 * This can be `const` as appropriate and on error sets a Python exception.
 *
 * @tparam Ts Input case types
 */
template <typename... Ts>
class converter_test_runner {
public:
  static_assert(sizeof...(Ts), "no test inputs");
  static constexpr auto n_tests = sizeof...(Ts);

  /**
   * Execute the test cases and update state accordingly.
   *
   * @returns `true` if all cases passed
   */
  bool operator()()
  {
    (test<Ts>(), ...);
    return passed_ == n_tests;
  }

  /**
   * Return the number of passed tests.
   */
  auto passed() const noexcept { return passed_; }

  /**
   * Return the number of failed tests.
   */
  auto failed() const noexcept { return failed_; }

private:
  unsigned passed_{};
  unsigned failed_{};

  /**
   * Evaluate a single test case.
   *
   * Updates `passed_` and `failed_` appropriately on completion.
   *
   * @tparam T Input case
   * @returns `true` on success, `false` on failure
   */
  template <typename T>
  bool test()
  {
    // get expected value
    auto expected = T::expected();
    // initialize Python object
    auto obj = [&expected]() -> npygl::py_object
    {
      // if custom operator() implemented
      if constexpr (std::is_invocable_r_v<npygl::py_object, T>)
        return T{}();
      // otherwise, use ctor
      else
        return expected;
    }();
    // convert using as<> + check if Python exception occurred
    auto actual = npygl::as<decltype(expected)>(obj);
    auto py_err = !!PyErr_Occurred();
    // check if we passed or not
    //
    // TODO:
    //
    // for floats use Knuth's "essentially equal" comparison. this helps us
    // avoid some truncation error involved in converting to double and back
    //
    auto pass = (!py_err && expected == actual);
    // type message
    // TODO: this can be constructed at compile time
    auto type_msg = std::string{npygl::type_name<T>()} + " [T = " +
      std::string{npygl::type_name<decltype(expected)>()} + "]";
    // print message + update state
    // TODO: clean this up more
    using namespace npygl::vts;
    if (pass) {
      npygl::cout << fg_green << "[ PASS ] " << fg_normal << type_msg <<
        std::endl;
      passed_++;
    }
    else {
      npygl::cout << fg_red << "[ FAIL ] " << fg_normal << type_msg << ": ";
      // Python exception
      if (py_err) {
        npygl::cout << "could not convert PyObject* constructed from " <<
          expected << ": " << std::flush;
        // print Python exception + clear
        // note: this is on another stream so writing to file is a problem
        // unless we redirect stderr to stdout
        PyErr_Print();
      }
      // otherwise, conversion error
      else
        npygl::cout << "expected " << expected << " != actual " << actual <<
          std::endl;
      failed_++;
    }
    // done
    return pass;
  }
};

/**
 * Partial specialization for the `converter_test_list<Ts...>`.
 *
 * @tparam Ts Input types
 */
template <typename... Ts>
class converter_test_runner<converter_test_list<Ts...> >
  : public converter_test_runner<Ts...> {};

// test cases

struct bool_input_true {
  static constexpr auto expected() noexcept
  {
    return true;
  }
};

struct bool_input_false {
  static constexpr auto expected() noexcept
  {
    return false;
  }
};

struct double_input {
  static constexpr auto expected() noexcept
  {
    return 1.999;
  }
};

struct float_input {
  static constexpr auto expected() noexcept
  {
    return 4.2f;
  }
};

struct long_double_input {
  static constexpr auto expected() noexcept
  {
    // TODO: originally 1.555l but was changed to avoid truncation errors
    return 1.0l;
  }

  npygl::py_object operator()() const noexcept
  {
    // just silently truncate
    return static_cast<double>(expected());
  }
};

// list of test cases
using test_cases = converter_test_list<
  bool_input_true,
  bool_input_false,
  double_input,
  float_input,
  long_double_input
>;

}  // namespace

int main()
{
  // initialize + execute tests
  npygl::vts_stdout_context ctx;
  npygl::py_init();
  converter_test_runner<test_cases> runner;
  return (runner()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
