/**
 * @file testing/traits_checker.hh
 * @author Derek Huang
 * @brief C++ header for performing type traits checks
 * @copyright MIT License
 */

#ifndef NPYGL_TESTING_TRAITS_CHECKER_HH_
#define NPYGL_TESTING_TRAITS_CHECKER_HH_

#include <cstdint>
#include <functional>
#include <iostream>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "npygl/demangle.hh"
#include "npygl/termcolor.hh"
#include "npygl/type_traits.hh"

// TODO:
//
// replace use of std::pair and std::tuple with type_tuple (probably want some
// pair-like version of type_tuple too). this allows us to test with incomplete
// types without the constraint of not being able to instantiate std::tuple
// when one of the types is incomplete (storage size is not known)
//

namespace npygl {
namespace testing {

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
 * Enum for the traits checker test case status.
 *
 * @note Members are ordered such that casting a boolean to this strongly-typed
 *  enum will result in either `pass` or `fail` status.
 */
enum class traits_checker_case_status {
  fail,
  pass,
  skip
};

/**
 * Formatter and traits class for printing the test case status banner.
 *
 * @tparam status Pass/fail/skip status
 */
template <traits_checker_case_status status>
struct traits_checker_banner {
  // status text message
  static constexpr auto text = []
  {
    switch (status) {
      case traits_checker_case_status::pass:
        return "PASS";
      case traits_checker_case_status::fail:
        return "FAIL";
      case traits_checker_case_status::skip:
        return "SKIP";
      default:
        return "XXXX";
    }
  }();
  // VTS foreground color
  static constexpr auto color = []
  {
    switch (status) {
      case traits_checker_case_status::pass:
        return vts::fg_green;
      case traits_checker_case_status::fail:
        return vts::fg_red;
      case traits_checker_case_status::skip:
        return vts::fg_yellow;
      default:
        return vts::fg_normal;
    }
  }();
};

/**
 * Print the test case status banner to the given stream.
 *
 * @tparam status Pass/fail/skip status
 *
 * @param out Output stream
 */
template <traits_checker_case_status status>
auto& operator<<(std::ostream& out, traits_checker_banner<status> banner)
{
  out << banner.color << "[ " << banner.text << " ]" << vts::fg_normal;
  return out;
}

/**
 * Traits test case formatter type.
 *
 * This assists with printing the actual test case name as a string.
 *
 * @tparam T Traits test case
 */
template <typename T>
struct traits_checker_case_formatter {};

/**
 * Stream the traits test case name using the given formatter.
 *
 * @tparam T Traits test case
 *
 * @param formatter Formatter instance
 */
template <typename T>
auto& operator<<(std::ostream& out, traits_checker_case_formatter<T> formatter)
{
  formatter(out);
  return out;
}

/**
 * Partial specialization for "standard" traits test case.
 *
 * This is typically of the form `std::pair<Traits<T>, std::bool_constant<B>>`,
 * but `T` is used as a general type to work with `partially_fixed<...>`.
 *
 * @tparam T Traits type
 * @tparam B Truth value
 */
template <typename T, bool B>
struct traits_checker_case_formatter<std::pair<T, std::bool_constant<B>> > {
  /**
   * Write the formatted test case to the output stream.
   *
   * @param out Output stream
   */
  void operator()(std::ostream& out = std::cout) const
  {
    constexpr auto truth = B ? "true" : "false";
    out << npygl::type_name(typeid(T)) << "::value == " << truth;
  }
};

/**
 * Check if a type satisfies the given traits.
 *
 * This class knows at compile-time the number of failed and total tests.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Input type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker;

namespace detail {

/**
 * Traits type to obtain the traits checker case type from the traits checker.
 *
 * This deduces the unary `Traits<T>` and input type `T`.
 *
 * @tparam T Traits checker type
 */
template <typename T>
struct traits_checker_to_case {};

/**
 * Partial specialization for a standard `traits_checker<Traits, T>`.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Input type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker_to_case<traits_checker<Traits, T>> {
  using type = traits_checker_case_t<Traits, T>;
};

/**
 * Get the `traits_checker_case<Traits, T>` from a `traits_checker<Traits, T>`.
 *
 * @tparam T `traits_checker<Traits, T>` specialization
 */
template <typename T>
using traits_checker_to_case_t = typename traits_checker_to_case<T>::type;

/**
 * Run a `traits_checker<Traits, T>` to check the traits and report results.
 *
 * This is used to implement the `operator()` shared amongst the different
 * `traits_checker<Traits, T>` single-input partial specializations. Each
 * specialization must implement the following:
 *
 * @code{.cc}
 * // number of tests (always 1)
 * static constexpr std::size_t n_tests() noexcept;
 * // number of failed tests, either 0 or 1
 * static constexpr std::size_t n_failed() noexcept;
 * // number of skipped tests, either 0 or 1
 * static constexpr std::size_t n_skipped() noexcept;
 * @endcode
 *
 * The following relationship between the functions must also be true:
 *
 * @code{.cc}
 * n_tests() >= n_failed() + n_skipped()
 * @endcode
 *
 * @tparam T `traits_checker<Traits, T>` specialization
 *
 * @param out Stream to write messages to
 * @return `true` on success, `false` on failure
 */
template <typename T>
bool run(const T*, std::ostream& out = std::cout)
{
  // validate implementation
  static_assert(
    T::n_tests() == 1u,
    "traits_checker_base is for single-input traits_checker specializations"
  );
  static_assert(
    T::n_tests() >= T::n_failed() + T::n_skipped(),
    "expected n_test() >= n_failed() + n_skipped() does not hold"
  );
  // determine test status
  constexpr auto res = []
  {
    if constexpr (T::n_failed())
      return traits_checker_case_status::fail;
    else if constexpr (T::n_skipped())
      return traits_checker_case_status::skip;
    else
      return traits_checker_case_status::pass;
  }();
  // print banner and case
  out << traits_checker_banner<res>{} << ' ' <<
    traits_checker_case_formatter<traits_checker_to_case_t<T>>{} << std::endl;
  return !T::n_failed();
}

}  // namespace detail

/**
 * Traits type checker main definition for a single input type.
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
    return detail::run(this, out);
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
   * @param out Stream to write messages to
   * @return `true` on success, `false` on failure
   */
  bool operator()(std::ostream& out = std::cout) const
  {
    return detail::run(this, out);
  }
};

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
 * Partial specialization for a `traits_value_comparison` traits test case.
 *
 * @tparam T Traits type that is either invocable or with a `value` member
 * @tparam C Comparator type, e.g. `std::equal_to<>`
 * @tparam V Expression input type
 * @tparam v_ Expression input value
 */
template <typename T, typename C, typename V, V v_>
struct traits_checker_case_formatter<
  std::pair<
    std::pair<T, traits_value_comparison<C, std::integral_constant<V, v_>>>,
    std::true_type
  > > {
  // traits type with the traits type value member
  using invoke_type = invoke_result_or_value<T>;

  /**
   * Write the formatted test case to the output stream.
   *
   * @param out Output stream
   */
  void operator()(std::ostream& out = std::cout) const
  {
    using comp_formatter = traits_comparator_formatter<C>;
    // TODO:
    //
    // if we use T, the actual input type, then we don't have an extra
    // invoke_result_or_value<...> around the traits type. however, if we have
    // an invocable traits (currently, this should not be true), then we need
    // to use std::is_invocable_v<T> to change "::value" to "()" instead
    //
    // if standard comparator, we can print a nicer format
    if constexpr (comp_formatter::standard) {
      out << npygl::type_name(typeid(invoke_type)) << "::value " <<
        comp_formatter::op_string << ' ' << v_;
    }
    // else fall back to a more verbose and explicit representation
    else {
      out << npygl::type_name(typeid(C)) << "{}(" <<
        npygl::type_name(typeid(invoke_type)) << "::value, " << v_ <<
          ") == true";
    }
  }
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
  using invoke_type = invoke_result_or_value<Traits<T>>;

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
    // note: may want to suppress MSVC C4388 (signed/unsigned mismatch)
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
    return detail::run(this, out);
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
 * Traits type to get the `T` from a `skipped<T>`.
 *
 * The base template simply provides the incoming type verbatim.
 *
 * @tparam T type
 */
template <typename T>
struct remove_skipped {
  using type = T;
};

/**
 * Partial specialization for a `skipped<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct remove_skipped<skipped<T>> {
  using type = T;
};

/**
 * Type alias for the `T` in a `skipped<T>`.
 *
 * If not a `skipped<T>` the type is provided verbatim.
 *
 * @tparam T type
 */
template <typename T>
using remove_skipped_t = typename remove_skipped<T>::type;

/**
 * Traits class to unwrap an input type.
 *
 * This yields the `T` from a `std::pair<T, std::bool_constant<B>>` or a
 * `std::pair<T, traits_value_comparison<C, V>>` and the correct unwrapped
 * input type from a `skipped<T>` specialization.
 *
 * @todo Consider removing this as we do not use it anywhere.
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
    // print skip banner + formatted test case
    out << traits_checker_banner<traits_checker_case_status::skip>{} << ' ' <<
      traits_checker_case_formatter<traits_checker_case_t<Traits, T>>{} <<
      std::endl;
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
    // print test case with padding and given color
    out << std::string(left_pad, ' ') << text_color <<
      traits_checker_case_formatter<T>{} << vts::fg_normal;
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
 * @note
 *
 * It is unclear why the `traits_checker_driver<traits_checker<Traits, T>>`
 * exists as this driver can be implemented by using the members of the
 * `traits_checker<Traits, T>` specializations directly.
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
