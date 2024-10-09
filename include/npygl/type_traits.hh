/**
 * @file type_traits.hh
 * @author Derek Huang
 * @brief C++ type traits helpers
 * @copyright MIT License
 */

#ifndef NPYGL_TYPE_TRAITS_HH_
#define NPYGL_TYPE_TRAITS_HH_

#include <type_traits>

namespace npygl {

/**
 * Traits to check if a type can be considered a pair of types.
 *
 * @tparam T type
 */
template <typename T>
struct is_pair : std::false_type {};

/**
 * True specialization when a type is a `std::pair<T1, T2>`.
 *
 * @tparam T1 First type
 * @tparam T2 Second type
 */
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};

/**
 * Helper to check if a type is a pair of types.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool is_pair_v = is_pair<T>::value;

/**
 * Traits to check if a type can be considered a tuple of types.
 *
 * @tparam T type
 */
template <typename T>
struct is_tuple : std::false_type {};

/**
 * True specialization when a type is a `std::tuple<Ts...>`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

/**
 * Helper to check if a type is a tuple of types.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

}  // namespace npygl

#endif  // NPYGL_TYPE_TRAITS_HH_
