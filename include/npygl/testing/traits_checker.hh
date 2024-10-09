/**
 * @file testing/traits_checker.hh
 * @author Derek Huang
 * @brief C++ header for performing type traits checks
 * @copyright MIT License
 */

#ifndef NPYGL_TESTING_TRAITS_CHECKER_HH_
#define NPYGL_TESTING_TRAITS_CHECKER_HH_

#include <cstdint>
#include <iostream>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "npygl/demangle.hh"
#include "npygl/type_traits.hh"

namespace npygl {
namespace testing {

/**
 * Check if a type satisfies the given traits.
 *
 * This class knows at compile-time the number of failed and total tests.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker {
  /**
   * Return total number of tests (always 1).
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return 1u;
  }

  /**
   * Return number of failed tests, either 1 or 0.
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return !Traits<T>::value;
  }

  /**
   * Check the type against the traits and indicate if test succeeded.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << "Test: " << npygl::type_name(typeid(Traits<T>)) <<
      "::value == true\n  " << (!n_failed() ? "Passed" : "Failed") << std::endl;
    return !n_failed();
  }
};

/**
 * Partial specialization for a `std::pair<T, std::bool_constant<B>>`.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T type
 * @tparam B Expected truth value
 */
template <template <typename> typename Traits, typename T, bool B>
struct traits_checker<Traits, std::pair<T, std::bool_constant<B>>> {
  /**
   * Return total number of tests (always 1)
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return 1u;
  }

  /**
   * Return number of failed tests, either 1 or 0.
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return !(B == Traits<T>::value);
  }

  /**
   * Check the type against the traits and indicate if test succeeded.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << "Test: " << npygl::type_name(typeid(Traits<T>)) << "::value == " <<
      (B ? "true" : "false") << "\n  " <<
      (!n_failed() ? "Passed" : "Failed") << std::endl;
    return !n_failed();
  }
};

/**
 * Placeholder type.
 */
struct placeholder {};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam Ts... Type pack where types can be `std::pair<U, V>`, `V` either
 *  `std::true_type` or `std::false_type`, to indicate that the type `U` should
 *  induce `Traits<U>::value` to be a specific value. A non-pair type `U` is
 *  expected to induce `Traits<U>::value` to be `true`.
 */
template <template <typename> typename Traits, typename... Ts>
struct traits_checker<Traits, std::tuple<Ts...>> {
private:
  /**
   * Conditionally wrap a tuple to prevent recursion into this specialization.
   *
   * Without this, if a `Ts` is a `std::tuple` specialization, the compiler
   * will recurse into this partial specialization, which is not what we want.
   *
   * @tparam T type
   */
  template <typename T>
  using tuple_wrap = std::conditional_t<
    npygl::is_tuple_v<T>, std::pair<T, std::true_type>, T
  >;

public:
  /**
   * Return total number of tests.
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return sizeof...(Ts);
  }

  /**
   * Return number of failed tests.
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return (traits_checker<Traits, tuple_wrap<Ts>>::n_failed() + ...);
  }

  /**
   * Check each type with the traits and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << "Running " << n_tests() << " tests on " <<
      npygl::type_name(typeid(Traits<placeholder>)) << "..." << std::endl;
    return (traits_checker<Traits, tuple_wrap<Ts>>{}(out) && ...);
  }
};

/**
 * Print a CTest-like summary for the traits checker test driver.
 *
 * @tparam Driver Traits checker driver type
 *
 * @param out Stream to write output to
 * @param driver Traits checker driver
 * @returns `true` if all tests passed, `false` otherwise
 */
template <typename Driver>
bool print_summary(std::ostream& out = std::cout)
{
  // failed and total tests
  constexpr auto n_fail = Driver::n_failed();
  constexpr auto n_total = Driver::n_tests();
  // print CTest-like output
  out << '\n' <<
    100 * (1 - n_fail / static_cast<double>(n_total)) << "% tests passed, " <<
    n_fail << " failed out of " << n_total << std::endl;
  return !n_fail;
}

/**
 * Main traits checker test driver.
 *
 * @tparam Ts... `traits_checker` specializations
 */
template <typename... Ts>
struct traits_checker_driver {
  /**
   * Get total number of tests registered.
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return (traits_checker_driver<Ts>::n_tests() + ...);
  }

  /**
   * Get number of failed tests.
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return (traits_checker_driver<Ts>::n_failed() + ...);
  }

  /**
   * Run the traits checker test suites and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << "Running " << n_tests() << " tests from " << sizeof...(Ts) <<
      " test suites..." << std::endl;
    // print messages for all test suites
    // note: can discard result since pass/fail known at compile time
    (Ts{}(out), ...);
    // print summary
    return print_summary<traits_checker_driver<Ts...>>(out);
  }
};

/**
 * Partial specialization for a single `traits_checker` specialization.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Type, either a `std::tuple<Ts...>` or single type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker_driver<traits_checker<Traits, T>> {
  /**
   * Get total number of tests registered.
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return traits_checker<Traits, T>::n_tests();
  };

  /**
   * Get number of failed tests.
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return traits_checker<Traits, T>::n_failed();
  }

  /**
   * Run the traits checker test suite and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << "Running " << n_tests() << " tests from 1 test suite..." << std::endl;
    // print messages for test suite
    // note: can discard result since pass/fail known at compile time
    traits_checker<Traits, T>{}(out);
    // print summary
    return print_summary<traits_checker_driver<traits_checker<Traits, T>>>(out);
  }
};

/**
 * Partial specialization for a tuple of `traits_checker` specializations.
 *
 * @tparam Ts... `traits_checker` specializations
 */
template <typename... Ts>
struct traits_checker_driver<std::tuple<Ts...>> : traits_checker_driver<Ts...> {};

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_TRAITS_CHECKER_HH_
