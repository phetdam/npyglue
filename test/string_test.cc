/**
 * @file string_test.cc
 * @author Derek Huang
 * @brief C++ program for string.hh tests
 * @copyright MIT License
 */

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>

#include "npygl/string.hh"
#include "npygl/termcolor.hh"

// TODO: add fixed_string tests

namespace {

/**
 * Test status message with appropriate VTS foreground color.
 */
class test_status {
public:
  /**
   * Ctor.
   *
   * @param passed Indicate if test status should indicate passing
   */
  constexpr test_status(bool passed) noexcept
    : color_{passed ? npygl::vts::fg_green : npygl::vts::fg_red},
      banner_{passed ? "[ PASS ]" : "[ FAIL ]"}
  {}

  /**
   * Return the SGR color for the status text.
   */
  constexpr auto color() const noexcept { return color_; }

  /**
   * Return the status text banner.
   */
  constexpr auto banner() const noexcept { return banner_; }

private:
  npygl::sgr_value color_;
  const char* banner_;
};

/**
 * Write the test status with color to the output stream.
 *
 * @param out Output stream
 * @param status Test status object
 */
auto& operator<<(std::ostream& out, const test_status& status)
{
  return out << status.color() << status.banner() << npygl::vts::fg_normal;
}

/**
 * Test case for compile-time `strlen` testing.
 */
class constexpr_strlen_test {
public:
  /**
   * Ctor.
   *
   * This evaluates the test case by calling `npygl::strlen` on the input.
   *
   * @param input Null-terminated string
   * @param length Expected length of `input` excluding null terminator
   */
  constexpr constexpr_strlen_test(const char* input, std::size_t length) noexcept
    : input_{input}, expected_{length}, actual_{npygl::strlen(input)}
  {}

  /**
   * Return the input string.
   */
  constexpr auto input() const noexcept { return input_; }

  /**
   * Return the input string [expected] length.
   */
  constexpr auto expected() const noexcept { return expected_; }

  /**
   * Return the actual length computed by `npygl::strlen`.
   */
  constexpr auto actual() const noexcept { return actual_; }

  /**
   * Indicate if the test passed or not.
   */
  constexpr bool passed() const noexcept
  {
    return expected_ == actual_;
  }

  /**
   * Indicate if the test passed or not.
   */
  constexpr operator bool() const noexcept
  {
    return passed();
  }

  /**
   * Print the test case input and result to the given stream.
   *
   * @param out Output stream to write to
   *
   * @returns `true` if the test passed, `false` otherwise
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    // print
    out << test_status{*this} << " strlen(\"" << input_ << "\")";
    // on failure, print reason
    if (!*this)
      out << " [expected " << expected_ << " != actual " << actual_ << "]";
    // write newline and flush + return status
    out << std::endl;
    return *this;
  }

private:
  const char* input_;
  std::size_t expected_;
  std::size_t actual_;
};

// set up test cases
constexpr constexpr_strlen_test strlen_tests[] = {
  {"hello world", 11u},
  {"the quick brown fox jumped over the lazy dog", 44u},
  {"despair not till your last breath / make your death count", 57u}
};

/**
 * Test case for compile-time `fixed_string<N> checking.
 *
 * @tparam Op Binary operator to compare expected and actual
 * @tparam N1 Length of first fixed string
 * @tparam N2 Length of second fixed string
 */
template <typename Op, std::size_t N1, std::size_t N2>
class constexpr_fixed_string_test {
  /**
   * Traits to map the operator to a string representation.
   *
   * @tparam O Binary operator type
   */
  template <typename O>
  struct op_traits {};

  // note: full specializations are not allowed. but T still must be void

  template <typename T>
  struct op_traits<std::equal_to<T>> {
    static constexpr auto str = "==";
  };

  template <typename T>
  struct op_traits<std::not_equal_to<T>> {
    static constexpr auto str = "!=";
  };

  template <typename T>
  struct op_traits<std::less<T>> {
    static constexpr auto str = "<";
  };

  template <typename T>
  struct op_traits<std::greater<T>> {
    static constexpr auto str = ">";
  };

  template <typename T>
  struct op_traits<std::less_equal<T>> {
    static constexpr auto str = "<=";
  };

  template <typename T>
  struct op_traits<std::greater_equal<T>> {
    static constexpr auto str = ">=";
  };

public:
  /**
   * Ctor.
   *
   * @param op Binary function object for comparison
   * @param expected Expected fixed string value
   * @param actual Actual fixed string value
   */
  constexpr constexpr_fixed_string_test(
    Op&& op,
    const npygl::fixed_string<N1>& expected,
    const npygl::fixed_string<N2>& actual) noexcept
    : op_{std::move(op)}, expected_{expected}, actual_{actual}
  {}

  /**
   * Return the binary operator object.
   */
  constexpr auto& op() const noexcept { return op_; }

  /**
   * Return the expected fixed string.
   */
  constexpr auto& expected() const noexcept { return expected_; }

  /**
   * Return the actual fixed string.
   */
  constexpr auto& actual() const noexcept { return actual_; }

  /**
   * Indicate that the test has passed.
   */
  constexpr bool passed() const noexcept
  {
    return op_(expected_, actual_);
  }

  /**
   * Indicate that the test has passed.
   */
  constexpr operator bool() const noexcept
  {
    return passed();
  }

  /**
   * Print the test case result to the given stream.
   *
   * @param out Output stream to write to
   *
   * @returns `true` if the test passed, `false` otherwise
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    // print
    constexpr auto op_str = op_traits<Op>::str;
    out << test_status{*this} << " \"" << expected_ << "\" " << op_str <<
      " \"" << actual_ << "\"" << std::endl;
    // done
    return *this;
  }

private:
  Op op_;
  npygl::fixed_string<N1> expected_;
  npygl::fixed_string<N2> actual_;
};

/**
 * Helper for creating a `constexpr_fixed_string_test<Op, N1, N2>`.
 *
 * @tparam Op Binary operators
 * @tparam T1 First type convertible to `fixed_string<N>`
 * @tparam T2 Second type convertible to `fixed_string<N>`
 */
template <typename Op, typename T1, typename T2>
constexpr auto make_test(Op&& op, T1&& expected, T2&& actual) noexcept
{
  // T1, T2 must be convertible to fixed_string
  return constexpr_fixed_string_test{
    std::move(op),
    npygl::fixed_string{std::forward<T1>(expected)},
    npygl::fixed_string{std::forward<T2>(actual)}
  };
}

/**
 * Print the fixed string test results and return the number of failed tests.
 *
 * We have to store the tests in a tuple, not an array, because technically
 * each fixed string test object is of a differen type.
 *
 * @tparam Is Indices 0 up to number of tuple elements
 * @tparam Ops Binary operators to compare expected and actual
 * @tparam N1s Lengths of first fixed strings
 * @tparam N2s Lengths of second fixed strings
 *
 * @param tests Tuple of `constexpr_fixed_string_test<Op, N1, N2>` tests
 */
template <
  std::size_t... Is,
  typename... Ops,
  std::size_t... N1s,
  std::size_t... N2s >
auto report_failed(
  std::index_sequence<Is...>,
  const std::tuple<constexpr_fixed_string_test<Ops, N1s, N2s>...>& tests)
{
  static_assert(sizeof...(Is) == sizeof...(Ops));
  return ([&tests]() -> std::size_t { return !std::get<Is>(tests)(); }() + ...);
}

/**
 * Print the fixed string test results and return the number of failed tests.
 *
 * @tparam Ops Binary operators to compare expected and actual
 * @tparam N1s Lengths of first fixed strings
 * @tparam N2s Lengths of second fixed strings
 *
 * @param tests Tuple of `constexpr_fixed_string_test<Op, N1, N2>` tests
 */
template <typename... Ops, std::size_t... N1s, std::size_t... N2s>
auto report_failed(
  const std::tuple<constexpr_fixed_string_test<Ops, N1s, N2s>...>& tests)
{
  return report_failed(std::index_sequence_for<Ops...>{}, tests);
}

// set up test cases
constexpr auto fixed_string_tests = std::make_tuple(
  make_test(std::equal_to<>{}, "abc", "abc"),
  make_test(std::equal_to<>{}, "abc", npygl::fixed_string{"a", "b", "c"}),
  make_test(std::not_equal_to<>{}, "abc", "def"),
  make_test(std::equal_to<>{}, "hello", npygl::fixed_string{"he", "l", "lo"}),
  make_test(std::less{}, "ABC", "abc"),
  make_test(std::less_equal{}, "abc", npygl::fixed_string{"ab", "c"}),
  make_test(std::greater{}, npygl::fixed_string{"a", "b", "c"}, "ABC"),
  make_test(std::greater_equal{}, npygl::fixed_string{"D", "e", "f"}, "DEf")
);

}  // namespace

int main()
{
  // note: on Windows Terminal virtual terminal sequences are already enabled.
  // this is still necessary for the old Windows CMD shell however
  npygl::vts_stdout_context ctx;
  // number of failed tests
  std::size_t n_failed = 0u;
  // report strlen tests + update failed
  for (const auto& test : strlen_tests)
    n_failed += !test();
  // report fixed_string tests + update failed
  n_failed += report_failed(fixed_string_tests);
  return (n_failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
