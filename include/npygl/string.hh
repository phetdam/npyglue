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
 * array of fixed size that is the only member of the class. The array member
 * includes a null terminator to enable compatibility with C functions that
 * expect string literals or other null-terminated character buffers.
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
   * @todo Allow construction from `const char*` and `std::string_view`.
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
   * Implicit convert to a const reference to a character array.
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

}  // namespace npygl

#endif  // NPYGL_STRING_HH_
