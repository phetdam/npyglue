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
#include "npygl/termcolor.hh"
#include "npygl/type_traits.hh"

namespace npygl {
namespace testing {

/**
 * Formatting traits for a compile-time traits test pass/fail.
 *
 * @tparam passed Pass/fail result for the traits test
 * @tparam truth Expected truth result for the traits test
 */
template <bool passed, bool truth = true>
struct traits_checker_formatter {
  static constexpr auto status_text = passed ? "PASS" : "FAIL";
  static constexpr auto status_color = passed ? vts::fg_green : vts::fg_red;
  static constexpr auto truth_text = truth ? "true" : "false";
};

/**
 * Traits class to represent a single traits checker test case.
 *
 * This defines the type member that represents how the test is represented.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Input type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker_case {
  using type = std::pair<Traits<T>, std::true_type>;
};

/**
 * Type alias for the traits checker test case type member.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Input type
 */
template <template <typename> typename Traits, typename T>
using traits_checker_case_t = typename traits_checker_case<Traits, T>::type;

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
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair<Traits<T>, std::true_type>`.
   */
  using failed_cases = std::conditional_t<
    Traits<T>::value,
    std::tuple<>,
    std::tuple<traits_checker_case_t<Traits, T>>
  >;

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = std::tuple<>;

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Return number of skipped tests (always 0).
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return 0u;
  }

  /**
   * Check the type against the traits and indicate if test succeeded.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    using formatter = traits_checker_formatter<!n_failed()>;
    // print formatted output
    out << formatter::status_color << "[ " << formatter::status_text <<
      " ] " << vts::fg_normal << npygl::type_name(typeid(Traits<T>)) <<
      "::value == " << formatter::truth_text << std::endl;
    return !n_failed();
  }
};

/**
 * Partial specialization for `std::pair<T, std::bool_constant<B>>` inputs.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Input type
 * @tparam B Truth value
 */
template <template <typename> typename Traits, typename T, bool B>
struct traits_checker_case<Traits, std::pair<T, std::bool_constant<B>>> {
  using type = std::pair<Traits<T>, std::bool_constant<B>>;
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
private:
  using input_type = std::pair<T, std::bool_constant<B>>;

public:
  /**
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair<Traits<T>, std::bool_constant<B>>`.
   */
  using failed_cases = std::conditional_t<
    B == Traits<T>::value,
    std::tuple<>,
    std::tuple<traits_checker_case_t<Traits, input_type>>
  >;

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = std::tuple<>;

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Return number of skipped tests (always 0).
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return 0u;
  }

  /**
   * Check the type against the traits and indicate if test succeeded.
   *
   * @todo Duplicates `operator()` of the `traits_checker<Traits, T>`. We
   *  should have a common CRTP base for these two specializations.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    using formatter = traits_checker_formatter<!n_failed(), B>;
    // print formatted output
    out << formatter::status_color << "[ " << formatter::status_text << " ] " <<
      vts::fg_normal << npygl::type_name(typeid(Traits<T>)) << "::value == " <<
      formatter::truth_text << std::endl;
    return !n_failed();
  }
};

namespace detail {

/**
 * Traits type that giving the `value` member or a compile-time invocation.
 *
 * Valid partial specializations will have a `value` member that is either a
 * copy of `T::value` or the result of the `T()` constant expression.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct invoke_result_or_value {};

/**
 * Partial specialization for invocable types.
 *
 * @tparam T Invocable type
 */
template <typename T>
struct invoke_result_or_value<T, std::enable_if_t<std::is_invocable_v<T>>> {
  static constexpr auto value = T();
};

/**
 * Partial specialization for types with a `value` static member.
 *
 * @tparam T Type with a constexpr `value` static member
 */
template <typename T>
struct invoke_result_or_value<T, std::void_t<decltype(T::value)>> {
  static constexpr auto value = T::value;
};

}  // namespace detail

/**
 * Input type representing a comparison operation and value to operate on.
 *
 * @tparam C Comparator type, e.g. `std::equal_to<>`
 * @tparam V `std::integral_constant<T, v>` specialization
 */
template <typename C, typename V>
struct traits_value_comparison {};

/**
 * Type alias for an equality comparison operation and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_equal = traits_value_comparison<
  std::equal_to<>, std::integral_constant<T, v_>
>;

/**
 * Type alias for an inequality comparison operation and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_not_equal = traits_value_comparison<
  std::not_equal_to<>, std::integral_constant<T, v_>
>;

/**
 * Type alias for a strict less than comparison operation and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_less = traits_value_comparison<
  std::less<>, std::integral_constant<T, v_>
>;

/**
 * Type alias for a less than or equal comparison operation and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_less_equal = traits_value_comparison<
  std::less_equal<>, std::integral_constant<T, v_>
>;

/**
 * Type alias for a strict greater than comparison and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_greater = traits_value_comparison<
  std::greater<>, std::integral_constant<T, v_>
>;

/**
 * Type alias for a greater than or equal comparison operation and input.
 *
 * @tparam T Input type
 * @tparam v_ Input value
 */
template <typename T, T v_>
using traits_value_is_greater_equal = traits_value_comparison<
  std::greater_equal<>, std::integral_constant<T, v_>
>;

/**
 * Formatting traits for the `traits_value_comparison` comparator.
 *
 * This base template is for non-standard comparators, e.g. something that is
 * not one of the standard `<functional>` comparison types.
 *
 * @tparam C Comparator type, e.g. `std::equal_to<>`
 */
template <typename C>
struct traits_comparator_formatter {
  static constexpr bool standard = false;
};

/**
 * Partial specialization for `std::equal_to<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::equal_to<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = "==";
};

/**
 * Partial specializaton for `std::not_equal_to<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::not_equal_to<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = "!=";
};

/**
 * Partial specialization for `std::less<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::less<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = "<";
};

/**
 * Partial specialization for `std::greater<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::greater<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = ">";
};

/**
 * Partial specialization for `std::greater_equal<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::greater_equal<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = ">=";
};

/**
 * Partial specialization for `std::less_equal<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct traits_comparator_formatter<std::less_equal<T>> {
  static constexpr bool standard = true;
  static constexpr const char op_string[] = "<=";
};

/**
 * Partial specialization for a `std::pair<T, traits_value_comparison<C, V>>`.
 *
 * @tparam Traits Traits template type invocable or with a `value` member
 * @tparam T Traits input type
 * @tparam C Comparator type, e.g. `std::equal_to<>`
 * @tparam V Expression input type
 * @tparam v_ Expression input value
 */
template <
  template <typename> typename Traits,
  typename T,
  typename C,
  typename V,
  V v_>
struct traits_checker_case<
  Traits,
  std::pair<T, traits_value_comparison<C, std::integral_constant<V, v_>>> > {
  using type = std::pair<
    std::pair<
      Traits<T>,
      traits_value_comparison<C, std::integral_constant<V, v_>>
    >,
    std::true_type
  >;
};

/**
 * Partial specialization for a `traits_value_comparison<C, v_>`.
 *
 * @tparam Traits Traits template type invocable or with a `value` member
 * @tparam C Comparator type, e.g. `std::equal_to<>`
 * @tparam T Traits input type
 * @tparam V Expression input type
 * @tparam v_ Expression value type
 */
template <
  template <typename> typename Traits,
  typename C,
  typename T,
  typename V,
  V v_ >
struct traits_checker<
  Traits,
  std::pair<T, traits_value_comparison<C, std::integral_constant<V, v_>>> > {
private:
  using input_type = std::pair<
    T,
    traits_value_comparison<C, std::integral_constant<V, v_>>
  >;
  using invoke_type = detail::invoke_result_or_value<Traits<T>>;

public:
  /**
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair` where the first type is a pair with `Traits<T>`
   * as the first element and `traits_value_comparison<...>` as the second
   * element. This lets the failed cases formatter show a comparison was done.
   *
   * The second type of the `std::pair` case is always `std::true_type`.
   */
  using failed_cases = std::conditional_t<
    C{}(invoke_type::value, v_),
    std::tuple<>,
    std::tuple<traits_checker_case_t<Traits, input_type>>
  >;

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = std::tuple<>;

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Return number of skipped tests (always 0).
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return 0u;
  }

  /**
   * Check the type against the traits and indicate if test succeeded.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    using formatter = traits_checker_formatter<!n_failed()>;
    using comp_formatter = traits_comparator_formatter<C>;
    // print formatted output
    out << formatter::status_color << "[ " << formatter::status_text << " ] " <<
      vts::fg_normal;
    // if standard comparator, we can print a nicer format
    if constexpr (comp_formatter::standard) {
      out << npygl::type_name(typeid(invoke_type)) << "::value " <<
        comp_formatter::op_string << ' ' << v_;
    }
    // else fall back to a more verbose and explicit representation
    else {
      out << npygl::type_name(typeid(C)) << "{}(" <<
        npygl::type_name(typeid(invoke_type)) << "::value, " << v_ << ") == " <<
        formatter::truth_text;
    }
    // flush and finish
    out << std::endl;
    return !n_failed();
  }
};

/**
 * Skip indication type.
 *
 * @tparam T Input case type
 */
template <typename T>
struct skipped { using type = T; };

/**
 * Traits class to unwrap an input type.
 *
 * This yields the `T` from a `std::pair<T, std::bool_constant<B>>` or a
 * `std::pair<T, traits_value_comparison<C, V>>` and the correct unwrapped
 * inputy type from a `skipped<T>` specialization.
 *
 * @tparam T Input type
 */
template <typename T>
struct traits_checker_input_unwrapper {
  using type = T;
  static constexpr bool truth = true;
};

/**
 * Partial specialization for a `std::pair<T, std::bool_constant<B>>`.
 *
 * @tparam T Input type
 * @tparam B Expected truth value
 */
template <typename T, bool B>
struct traits_checker_input_unwrapper<std::pair<T, std::bool_constant<B>>> {
  using type = T;
  static constexpr bool truth = B;
};

/**
 * Partial specialization for a `std::pair<T, traits_value_comparison<...>>`.
 *
 * @tparam T Traits input type
 * @tparam C Comparator type
 * @tparam V Expression input type
 * @tparam v_ Expression input value
 */
template <typename T, typename C, typename V, V v_>
struct traits_checker_input_unwrapper<
  std::pair<T, traits_value_comparison<C, std::integral_constant<V, v_>>> > {
  using type = T;
  static constexpr bool truth = true;
};

/**
 * Partial specialization for a `skipped<T>`.
 *
 * @tparam T Input type
 */
template <typename T>
struct traits_checker_input_unwrapper<skipped<T>>
  : traits_checker_input_unwrapper<T> {};

/**
 * Partial specialization for a skipped input case.
 *
 * @tparam Traits Traits template type invocable or with a `value` member
 * @tparam T Input type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker<Traits, skipped<T>> {
private:
  // unwrapped traits input type and expected truth value
  using input_type = typename traits_checker_input_unwrapper<T>::type;
  static constexpr bool truth = traits_checker_input_unwrapper<T>::truth;

public:
  /**
   * Provides the tuple of failing input cases (empty).
   */
  using failed_cases = std::tuple<>;

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = std::tuple<traits_checker_case_t<Traits, T>>;

  /**
   * Return total number of tests (always 1).
   */
  static constexpr std::size_t n_tests() noexcept
  {
    return 1u;
  }

  /**
   * Return number of failed tests (always 0).
   */
  static constexpr std::size_t n_failed() noexcept
  {
    return 0u;
  }

  /**
   * Return number of skipped tests (always 1).
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return 1u;
  }

  /**
   * Indicate that this test was skipped.
   *
   * @param out Stream to write messages to
   * @return `true` to indicate no failure (due to skip)
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    // borrowing traits_checker_formatter to use truth_text
    using formatter = traits_checker_formatter<true /*ignored*/, truth>;
    // print formatted output
    // FIXME: format not correct for traits_value_comparison<...> inputs
    out << vts::fg_yellow << "[ SKIP ] " << vts::fg_normal <<
      npygl::type_name(typeid(Traits<T>)) << "::value == " <<
      formatter::truth_text << std::endl;
    return true;
  }
};

/**
 * Placeholder type.
 */
struct placeholder {};

/**
 * Partial specialization for a tuple of types.
 *
 * This represents a single test suite that is run by the testing driver.
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
   * Wrapped traits checker that will not recurse into this specialization.
   *
   * Without this, if a `Ts` is a `std::tuple` specialization, the compiler
   * will recurse into this partial specialization, which is not what we want.
   *
   * @tparam T type
   */
  template <typename T>
  using wrapped_checker = traits_checker<
    Traits,
    std::conditional_t<is_tuple_v<T>, std::pair<T, std::true_type>, T>
  >;

public:
  /**
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair<Traits<T>, std::bool_constant<B>>`.
   */
  using failed_cases = decltype(
    std::tuple_cat(std::declval<typename wrapped_checker<Ts>::failed_cases>()...)
  );

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = decltype(
    std::tuple_cat(std::declval<typename wrapped_checker<Ts>::skipped_cases>()...)
  );

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Return number of skipped tests.
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return std::tuple_size_v<skipped_cases>;
  }

  /**
   * Check each type with the traits and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << vts::fg_green << "[======] " << vts::fg_normal <<
      "Ran " << n_tests() << " tests on " <<
      npygl::type_name(typeid(Traits<placeholder>)) << "." << std::endl;
    // to prevent short-circuiting we check that there are no failed
    return !(static_cast<unsigned>(!wrapped_checker<Ts>{}(out)) + ...);
  }
};

namespace detail {

/**
 * List test cases into a printable format.
 *
 * This is used with `display_failed` and `display_skipped`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct input_case_lister {};

/**
 * Partial specialization for a tuple of input cases.
 *
 * @note The tuple must contain at least one input case.
 *
 * @tparam T First `std::pair<T, std:bool_constant<B>>`
 * @tparam Ts... Pack of `std::pair<T, std:bool_constant<B>>`
 */
template <typename T, typename... Ts>
struct input_case_lister<std::tuple<T, Ts...>> {
  // text color
  sgr_value text_color = vts::fg_normal;
  // left padding + delimiter
  unsigned int left_pad = 0u;
  char delim = '\n';

  /**
   * Stream the input cases to the given output stream.
   *
   * No delimiter will trail the last input case streamed.
   *
   * @param out Stream to write output to
   */
  void operator()(std::ostream& out) const
  {
    // borrowing traits_checker_formatter to use truth_text
    using truth_type = typename T::second_type;
    using formatter = traits_checker_formatter<true /*ignored*/, truth_type::value>;
    // input case
    out << std::string(left_pad, ' ') << text_color <<
      type_name(typeid(typename T::first_type)) << " == " <<
      formatter::truth_text << vts::fg_normal;
    // recurse if more types
    if constexpr (sizeof...(Ts)) {
      out << delim;
      input_case_lister<std::tuple<Ts...>>{text_color, left_pad, delim}(out);
    }
  }
};

/**
 * Global failed test case formatter for `display_failed`.
 *
 * @note Odd padding number is to ensure printed inputs are aligned in the same
 *  same column as the inputs printed under each test suite.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr input_case_lister<Ts...> format_failed{vts::fg_red, 9u};

/**
 * Global skipped test case formatter for `display_skipped`.
 *
 * @note Odd padding number is to ensure printed inputs are aligned in the same
 *  same column as the inputs printed under each test suite.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr input_case_lister<Ts...> format_skipped{vts::fg_blue, 9u};

}  // namespace detail

/**
 * Display the list of the traits checker test driver's failed test cases.
 *
 * @tparam Driver Traits checker driver type
 *
 * @param out Stream to write output to
 */
template <typename Driver>
void display_failed(std::ostream& out = std::cout)
{
  // no-op if no failed
  if constexpr (Driver::n_failed()) {
    out << "\nThe following tests FAILED:\n";
    detail::format_failed<typename Driver::failed_cases>(out);
    out << std::endl;
  }
}

/**
 * Display the list of the traits checker test driver's skipped test cases.
 *
 * @tparam Driver Traits checker driver type
 *
 * @param out Stream to write output to
 */
template <typename Driver>
void display_skipped(std::ostream& out = std::cout)
{
  // no-op if no skipped
  if constexpr (Driver::n_skipped()) {
    out << "\nThe following tests were skipped:\n";
    detail::format_skipped<typename Driver::skipped_cases>(out);
    out << std::endl;
  }
}

/**
 * Print a CTest-like summary for the traits checker test driver.
 *
 * @tparam Driver Traits checker driver type
 *
 * @param out Stream to write output to
 * @returns `true` if all tests passed, `false` otherwise
 */
template <typename Driver>
bool print_summary(std::ostream& out = std::cout)
{
  // failed and total tests
  constexpr auto n_fail = Driver::n_failed();
  constexpr auto n_total = Driver::n_tests();
  // percent passed + tests failed colors
  constexpr auto pass_color = (!n_fail) ? vts::fg_green : vts::fg_normal;
  constexpr auto fail_color = (!n_fail) ? vts::fg_normal : vts::fg_red;
  // print CTest-like output
  out << '\n' << pass_color <<
    100 * (1 - n_fail / static_cast<double>(n_total)) << "% tests passed" <<
    vts::fg_normal << ", " << fail_color << n_fail << " failed " <<
    vts::fg_normal << "out of " << n_total << std::endl;
  // display failed + skipped test cases if any
  display_failed<Driver>(out);
  display_skipped<Driver>(out);
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
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair<Traits<T>, std::bool_constant<B>>`.
   */
  using failed_cases = decltype(
    std::tuple_cat(
      std::declval<typename traits_checker_driver<Ts>::failed_cases>()...)
  );

  /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = decltype(
    std::tuple_cat(
      std::declval<typename traits_checker_driver<Ts>::skipped_cases>()...)
  );

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Get number of skipped tests.
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return std::tuple_size_v<skipped_cases>;
  }

  /**
   * Run the traits checker test suites and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << vts::fg_green << "[======] " << vts::fg_normal <<
      "Ran " << n_tests() << " tests from " << sizeof...(Ts) <<
      " test suites." << std::endl;
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
   * Provides the tuple of failing input cases.
   *
   * Each case is a `std::pair<Traits<T>, std::bool_constant<B>>`.
   */
  using failed_cases = typename traits_checker<Traits, T>::failed_cases;

   /**
   * Provides the tuple of skipped input cases.
   */
  using skipped_cases = typename traits_checker<Traits, T>::skipped_cases;

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
    return std::tuple_size_v<failed_cases>;
  }

  /**
   * Get number of skipped tests.
   */
  static constexpr std::size_t n_skipped() noexcept
  {
    return std::tuple_size_v<skipped_cases>;
  }

  /**
   * Run the traits checker test suite and indicate success or failure.
   *
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    out << vts::fg_green << "[======] " << vts::fg_normal <<
      "Ran " << n_tests() << " tests from 1 test suite." << std::endl;
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
