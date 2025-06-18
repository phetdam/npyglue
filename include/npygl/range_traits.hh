/**
 * @file range_traits.hh
 * @author Derek Huang
 * @brief C++ header for range metaprogramming helpers
 * @copyright MIT License
 *
 * This provides some C++17 support to simulate C++20 ranges.
 */

#ifndef NPYGL_RANGE_TRAITS_HH_
#define NPYGL_RANGE_TRAITS_HH_

#include <iterator>     // for std::size definition
#include <type_traits>
#include <utility>

namespace npygl {

/**
 * Traits to check that a type is compatible with `std::begin`.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_begin_compatible : std::false_type {};

/**
 * True specialization for types compatible with `std::begin`.
 */
template <typename T>
struct is_begin_compatible<
  T,
  std::void_t<decltype(std::begin(std::declval<T>()))> > : std::true_type {};

/**
 * Helper to indicate if a type is compatible with `std::begin`.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_begin_compatible_v = is_begin_compatible<T>::value;

/**
 * Traits to check that a type is compatible with `std::end`.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_end_compatible : std::false_type {};

/**
 * True specialization for types compatible with `std::end`.
 *
 * @tparam T type
 */
template <typename T>
struct is_end_compatible<
  T,
  std::void_t<decltype(std::end(std::declval<T>()))> > : std::true_type {};

/**
 * Helper to indicate if a type is compatible with `std::end`.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_end_compatible_v = is_end_compatible<T>::value;

/**
 * Traits to check if a type is a range-like type.
 *
 * Such a type works with `std::begin` and `std::end`.
 *
 * @note This supersedes `is_iterable<T>` in `type_traits.hh`
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_range : std::false_type {};

/**
 * True specialization for a range-like type.
 *
 * @note Unlike the C++ standard's legacy iterator requirements the begin and
 *  end iterators are not required to yield the same value type.
 *
 * @tparam T type
 */
template <typename T>
struct is_range<
  T,
  std::enable_if_t<is_begin_compatible_v<T> && is_end_compatible_v<T>> >
  : std::true_type {};

/**
 * Helper to indicate if a type is range-like.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_range_v = is_range<T>::value;

/**
 * Traits to get the value type of a range's iterator.
 *
 * @tparam R Range-like type
 */
template <typename R, typename = void>
struct range_value {};

/**
 * True specialization for range-like types.
 *
 * @tparam R Range-like type
 */
template <typename R>
struct range_value<R, std::enable_if_t<is_range_v<R>>> {
  using type = std::decay_t<decltype(*std::begin(std::declval<R>()))>;
};

/**
 * SFINAE-capable type alias for the value type of the range-like type.
 *
 * @tparam R Range-like type
 */
template <typename R>
using range_value_t = typename range_value<R>::type;

/**
 * Traits type for a range with a floating point value type.
 *
 * This uses `std::is_floating_point<T>` on the range's value type.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_floating_point_range : std::false_type {};

/**
 * True specialization for a range with a floating point value type.
 *
 * @tparam T type
 */
template <typename T>
struct is_floating_point_range<
  T,
  std::enable_if_t<is_range_v<T> && std::is_floating_point_v<range_value_t<T>>> >
  : std::true_type {};


/**
 * Helper to indicate if a type is range-like with a floating point value type.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_floating_point_range_v = is_floating_point_range<T>::value;

/**
 * SFINAE helper for a range with a floating point value type.
 *
 * @tparam T type
 */
template <typename T>
using floating_point_range_t = std::enable_if_t<is_floating_point_range_v<T>>;

/**
 * Traits type for a sized range.
 *
 * A sized range is one where `std::size` can be applied to get the number of
 * hops between the begin and end iterators. User-defined overloads need not
 * return an unsigned type but are of course encouraged to.
 *
 * @todo Support checking for a `size()` member function as well.
 *
 * @tparam T type
 */
template <typename T, typename = void, typename = void>
struct is_sized_range : std::false_type {};

/**
 * True specialization for a range-like type supporting `std::size`.
 *
 * @tparam T type
 */
template <typename T>
struct is_sized_range<
  T,
  std::enable_if_t<is_range_v<T>>,
  std::void_t<decltype(std::size(std::declval<T>()))> > : std::true_type {};

/**
 * Helper to indicate if a type is a sized range.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_sized_range_v = is_sized_range<T>::value;

}  // namespace npygl

#endif  // NPYGL_RANGE_TRAITS_HH_
