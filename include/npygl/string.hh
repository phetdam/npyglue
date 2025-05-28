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

/**
 * Class template for a fixed-length string.
 *
 * This is used to represent a copyable string literal, modeling a character
 * array of fixed size that is the only member of the class. For compatibility
 * with C functions that expect null-terminated strings, the array member
 * includes a null terminator and an implicit conversion to
 * `const char (&)[N + 1]` with decay to `const char*` exists.
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
   * This enables construction from multiple string literals.
   *
   * @tparam Ns Null-terminated character array sizes
   *
   * @param strs String literals to construct from
   */
  template <std::size_t... Ns, typename = std::enable_if_t<N == ((Ns - 1) + ...)> >
  constexpr fixed_string(const char (&...strs)[Ns]) noexcept
  {
    decltype(N) count = 0u;
    (
      [this, &count, &strs]
      {
        // copy characters except trailing null terminator
        for (decltype(Ns) i = 0u; i < Ns - 1; i++)
          data_[count++] = strs[i];
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
 * Deduction guide for CTAD when constructing from string literals.
 *
 * @tparam Ns Null-terminated character array sizes
 *
 * @param strs String literals to construct from
 */
template <std::size_t... Ns>
fixed_string(const char (&...strs)[Ns]) -> fixed_string<((Ns - 1) + ...)>;

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
