/**
 * @file string.hh
 * @author Derek Huang
 * @brief C++ header for compile- and run-time string manipulation
 * @copyright MIT License
 */

#ifndef NPYGL_STRING_HH_
#define NPYGL_STRING_HH_

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace npygl {

/**
 * Return the length of a null-terminated string.
 *
 * This is only different from `strlen` in the fact that it is `constexpr`.
 *
 * Behavior is undefined if `str` is `nullptr` or not null-terminated.
 *
 * @param str Null-terminated string
 */
constexpr auto strlen(const char* str) noexcept
{
  std::size_t n = 0u;
  while (*str++)
    n++;
  return n;
}

// forward decl for traits
template <std::size_t N>
class fixed_string;

namespace detail {

/**
 * Traits type for `fixed_string<N>` ctor inputs providing the input size.
 *
 * @tparam T type
 */
template <typename T>
struct fixed_string_input_size {};

/**
 * Partial specialization for a character array.
 *
 * The reported size is `N - 1` to exclude the null terminator.
 *
 * @tparam N Number of elements in array
 */
template <std::size_t N>
struct fixed_string_input_size<char[N]> {
  static constexpr auto value = N - 1u;
};

/**
 * Partial specialization for a single `fixed_string<N>`.
 *
 * @tparam N String length excluding null terminator
 */
template <std::size_t N>
struct fixed_string_input_size<fixed_string<N>> {
  static constexpr auto value = N;
};

}  // namespace detail

/**
 * Helper to obtain the size of the `fixed_string<N>` input.
 *
 * @note Originally this was defined using a fold expression and took a type
 *  pack but in C++17 MSVC ended up emitting C7516 (unary fold expression over
 *  + must have a non-empty expansion). Therefore we stick with using a single
 *  type parameter and require explicit folding over type packs.
 *
 * @tparam T Input type with cv-ref qualifiers
 */
template <typename T>
constexpr auto fixed_string_input_size_v = detail::
  fixed_string_input_size<std::remove_cv_t<std::remove_reference_t<T>>>::value;

/**
 * Class template for a fixed-length string.
 *
 * This is used to represent a copyable string literal, modeling a character
 * array of fixed size that is the only member of the class. For compatibility
 * with C functions that expect null-terminated strings, the array member
 * includes a null terminator and an implicit conversion to
 * `const char (&)[N + 1]` that can further decay to `const char*`.
 *
 * @tparam N String length excluding null terminator
 */
template <std::size_t N>
class fixed_string {
public:
  /**
   * Array type used for storage with room for a null terminator.
   */
  using storage_type = char[N + 1];

  /**
   * Ctor.
   *
   * Constructs from multiple string literals or `fixed_string<N>`.
   *
   * @tparam Ts Input types
   *
   * @param inputs String literals of `fixed_string<N>` inputs
   */
  template <
    typename... Ts,
    typename = std::enable_if_t<N == (fixed_string_input_size_v<Ts> + ...)> >
  constexpr fixed_string(Ts&&... inputs) noexcept
  {
    // current position in data_
    decltype(N) count = 0u;
    // copy characters from inputs using fold
    (
      [this, &count, &inputs]
      {
        for (decltype(N) i = 0u; i < fixed_string_input_size_v<Ts>; i++)
          data_[count++] = inputs[i];
      }(), ...
    );
  }

  /**
   * Return a pointer to the first character in the buffer.
   */
  constexpr auto data() const noexcept { return data_; }

  /**
   * Return the number of elements in the string.
   */
  static constexpr auto size() noexcept
  {
    return N;
  }

  /**
   * Return a const reference the `i`th character in the string.
   *
   * @param i Index of desired character
   */
  constexpr auto& operator[](std::size_t i) const noexcept
  {
    return data_[i];
  }

  /**
   * Implicitly convert to a const reference to a character array.
   *
   * This can be further implicitly converted to `const char*`.
   */
  constexpr operator const storage_type&() const noexcept
  {
    return data_;
  }

  /**
   * Return an iterator to the first character in the buffer.
   */
  constexpr auto begin() const noexcept
  {
    return data_;
  }

  /**
   * Return an iterator one past the last character in the buffer.
   */
  constexpr auto end() const noexcept
  {
    return data_ + N;
  }

private:
  storage_type data_{};  // value-initialized
};

/**
 * Deduction guide for CTAD when constructing from `fixed_string<N>` inputs.
 *
 * @tparam Ts Input types
 */
template <typename... Ts>
fixed_string(Ts&&...) -> fixed_string<(fixed_string_input_size_v<Ts> + ...)>;

/**
 * Concatenate two `fixed_string<N>` instances into a new `fixed_string<N>`.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr auto
operator+(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  return fixed_string{s1, s2};
}

/**
 * Concatenate a `fixed_string<N>` and a string literal.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 Fixed string
 * @param s2 String literal
 */
template <std::size_t N1, std::size_t N2>
constexpr auto
operator+(const fixed_string<N1>& s1, const char (&s2)[N2]) noexcept
{
  return fixed_string{s1, s2};
}

/**
 * Concatenate a string literal and a `fixed_string<N>`.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 String literal
 * @param s2 Fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr auto
operator+(const char (&s1)[N1], const fixed_string<N2>& s2) noexcept
{
  return fixed_string{s1, s2};
}

/**
 * Compare two `fixed_string<N>` for equality.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator==(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  // if sizes differ, short-circuit
  if constexpr (N1 != N2)
    return false;
  // otherwise check
  else {
    for (decltype(N1) i = 0u; i < N1; i++)
      if (s1[i] != s2[i])
        return false;
    return true;
  }
}

/**
 * Compare two `fixed_string<N>` for inequality.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator!=(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  return !(s1 == s2);
}

/**
 * Check if one `fixed_string<N>` is ordered before the other.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator<(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  constexpr auto len = (N1 < N2) ? N1 : N2;
  // compare shared length
  for (decltype(N1) i = 0u; i < len; i++)
    if (s1[i] < s2[i])
      return true;
  return false;
}

/**
 * Check if one `fixed_string<N>` is ordered after the other.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator>(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  constexpr auto len = (N1 < N2) ? N1 : N2;
  // compare shared length
  for (decltype(N1) i = 0u; i < len; i++)
    if (s1[i] > s2[i])
      return true;
  return false;
}

/**
 * Check if one `fixed_string<N>` is ordered before or is equal to the other.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator<=(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  return (s1 == s2) || (s1 < s2);
}

/**
 * Check if one `fixed_string<N>` is ordered after or is equal to the other.
 *
 * @tparam N1 First length
 * @tparam N2 Second length
 *
 * @param s1 First fixed string
 * @param s2 Second fixed string
 */
template <std::size_t N1, std::size_t N2>
constexpr
bool operator>=(const fixed_string<N1>& s1, const fixed_string<N2>& s2) noexcept
{
  return (s1 == s2) || (s1 > s2);
}

}  // namespace npygl

#endif  // NPYGL_STRING_HH_
