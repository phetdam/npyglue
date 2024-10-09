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

/**
 * Check if a traits type has the `type` type member.
 *
 * @tparam T Traits class
 */
template <typename T, typename = void>
struct has_type_member : std::false_type {};

/**
 * Partial specialization when a traits type has the `type` type member.
 *
 * @tparam T Traits class
 */
template <typename T>
struct has_type_member<T, std::void_t<typename T::type>> : std::true_type {};

/**
 * Helper to check if a traits type has the `type` type member
 *
 * @tparam T Traits class
 */
template <typename T>
inline constexpr bool has_type_member_v = has_type_member<T>::value;

/**
 * Traits type to ensure all types in a parameter pack are the same.
 *
 * This extends `std::is_same<T, U>` to a parameter pack.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct same_type {};

/**
 * Partial specialization for a single type.
 *
 * @tparam T type
 */
template <typename T>
struct same_type<T> {
  using type = T;
};

/**
 * Partial specialization when two types are the same.
 *
 * @tparam T type
 */
template <typename T>
struct same_type<T, T> {
  using type = T;
};

/**
 * Partial specialization for more than two types.
 *
 * @tparam T First type
 * @tparam Ts... Subsequent types
 */
template <typename T, typename... Ts>
struct same_type<T, T, Ts...> : same_type<T, Ts...> {};

/**
 * Helper to get the type member of `same_type<Ts...>`.
 *
 * This can be used as a convenient type alias or as a specialized version of
 * `std::enable_if_t<...>` for ensuring a pack of types all have the same type.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
using same_type_t = typename same_type<Ts...>::type;

/**
 * Traits class to indicate if a parameter pack contains all the same types.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct is_same_type : has_type_member<same_type<Ts...>> {};

/**
 * Helper to indicate if a parameter pack contains all the same types.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr bool is_same_type_v = is_same_type<Ts...>::value;

}  // namespace npygl

#endif  // NPYGL_TYPE_TRAITS_HH_
