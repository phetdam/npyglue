/**
 * @file string_test.cc
 * @author Derek Huang
 * @brief C++ program for string.hh tests
 * @copyright MIT License
 */

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "npygl/string.hh"
#include "npygl/termcolor.hh"

// TODO: add fixed_string tests

namespace {

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
   * Print the test case input and result to the given stream
   *
   * @param out Output stream to write to
   *
   * @returns `true` if the test passed, `false` otherwise
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    using namespace npygl::vts;
    // color + status
    auto color = *this ? fg_green : fg_red;
    auto status = *this ? "[ PASS ]" : "[ FAIL ]";
    // print
    out << color << status << fg_normal << " strlen(\"" << input_ << "\")";
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

}  // namespace

int main()
{
  // number of failed tests
  std::size_t n_failed = 0u;
  // report strlen tests + update failed
  for (const auto& test : strlen_tests)
    n_failed += !test();
  return (n_failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
