/**
 * @file type_traits.hh
 * @author Derek Huang
 * @brief C++ type traits helpers
 * @copyright MIT License
 */

#ifndef NPYGL_TYPE_TRAITS_HH_
#define NPYGL_TYPE_TRAITS_HH_

#include <type_traits>
#include <utility>

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
struct same_type<T, T> : same_type<T> {};

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

/**
 * Traits type to ensure that all the types in a tuple are the same.
 *
 * We cannot create a `std::tuple<Ts...>` partial specialization for
 * `same_type<Ts...>` because any tuple that does not have all same types will
 * fall back to the partial specialization for a single type.
 *
 * @tparam T type
 */
template <typename T>
struct monomorphic_tuple_type {};

/**
 * Partial specialization for a tuple with a single type.
 *
 * @tparam T type
 */
template <typename T>
struct monomorphic_tuple_type<std::tuple<T>> {
  using type = T;
};

/**
 * Partial specialization for a tuple with two types that are the same.
 *
 * @tparam T type
 */
template <typename T>
struct monomorphic_tuple_type<std::tuple<T, T>> {
  using type = T;
};

/**
 * Partial specialization for a tuple with more than two types.
 *
 * @tparam T First type
 * @tparam Ts... Subsequent types
 */
template <typename T, typename... Ts>
struct monomorphic_tuple_type<std::tuple<T, T, Ts...>>
  : monomorphic_tuple_type<std::tuple<T, Ts...>> {};

/**
 * Helper to get the type member of `monomorphic_tuple_type<Ts...>`.
 *
 * This can be used as a convenient type alias or as a specialized version of
 * `std::enable_if_t<...>` for ensuring a `std::tuple<Ts...>` is monomorphic.
 *
 * @tparam T type
 */
template <typename T>
using monomorphic_tuple_type_t = typename monomorphic_tuple_type<T>::type;

/**
 * Traits class to indicate if a type is a monomorphic tuple.
 *
 * @tparam T type
 */
template <typename T>
struct is_monomorphic_tuple : has_type_member<monomorphic_tuple_type<T>> {};

/**
 * Helper to indicate if a type is a monomorphic tuple.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool is_monomorphic_tuple_v = is_monomorphic_tuple<T>::value;

/**
 * Traits type that always has a `true` value.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct always_true : std::true_type {};

/**
 * Helper to get the booleam truth of `always_true`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr bool always_true_v = always_true<Ts...>::value;

/**
 * Helper type to conditionally filter a parameter pack into a tuple.
 *
 * @tparam Filter Traits type with boolean `value` member indicating inclusion
 * @tparam Ts... types
 */
template <template <typename> typename Filter, typename... Ts>
struct type_filter {};

/**
 * Partial specialization for a single type.
 *
 * @tparam Filter Traits type with boolean `value` member indicating inclusion
 * @tparam T type
 */
template <template <typename> typename Filter, typename T>
struct type_filter<Filter, T> {
  using type = std::conditional_t<Filter<T>::value, std::tuple<T>, std::tuple<>>;
};

/**
 * Partial specialization for more than one type.
 *
 * @tparam Filter Traits type with boolean `value` member indicating inclusion
 * @tparam T type
 * @tparam Ts... Subsequent types
 */
template <template <typename> typename Filter, typename T, typename... Ts>
struct type_filter<Filter, T, Ts...> {
  using type = std::conditional_t<
    Filter<T>::value,
    // type of the T and Ts... tuples concatenated
    decltype(
      std::tuple_cat(
        std::declval<std::tuple<T>>(),
        std::declval<typename type_filter<Filter, Ts...>::type>()
      )
    ),
    // plain recursion to Ts...
    typename type_filter<Filter, Ts...>::type
  >;
};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Filter Traits type with boolean `value` member indicating inclusion
 * @tparam Ts.. types
 */
template <template <typename> typename Filter, typename... Ts>
struct type_filter<Filter, std::tuple<Ts...>> : type_filter<Filter, Ts...> {};

/**
 * Type alias for the tuple type created from the filtered types.
 *
 * @tparam Filter Traits type with boolean `value` member indicating inclusion
 * @tparam Ts... types
 */
template <template <typename> typename Filter, typename... Ts>
using type_filter_t = typename type_filter<Filter, Ts...>::type;

/**
 * Indicate that the first few types of a variadic template should be fixed.
 */
struct fix_first_types {};

/**
 * Indicate that the last few types of a variadic template should be fixed.
 */
struct fix_last_types {};

/**
 * Allow fixing the first/last few types of a variadic template.
 *
 * This is different from a `template <typename... Ts>` using-declaration
 * because we can package the template template in an actual type and then
 * choose to "evaluate" (instantiate) the actual type at some later point.
 *
 * @tparam T Template template type
 * @tparam F Fix type, either `fix_first_type` or `fix_last_type`
 * @tparam Ts... types
 */
template <template <typename...> typename T, typename F, typename... Ts>
struct partially_fixed {
  static_assert(
    std::is_same_v<F, fix_first_types> ||
    std::is_same_v<F, fix_last_types>,
    "F must be either fix_first_types or fix_last_types"
  );

  /**
   * Indicate whether the first few or last few types are fixed.
   */
  using fix_type = F;

  /**
   * Tuple of the types fixed in the `T<...>` specialization.
   */
  using fixed_types = std::tuple<Ts...>;

  /**
   * Template type alias to instantiate the partially fixed template template.
   *
   * @tparam Us... types
   */
  template <typename... Us>
  using type = std::conditional_t<
    std::is_same_v<F, fix_first_types>, T<Ts..., Us...>, T<Us..., Ts...>
  >;
};

}  // namespace npygl

#endif  // NPYGL_TYPE_TRAITS_HH_
