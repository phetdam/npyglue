/**
 * @file ctti.hh
 * @author Derek Huang
 * @brief C++ header for compile-time type information
 * @copyright MIT License
 */

#ifndef NPYGL_CTTI_HH_
#define NPYGL_CTTI_HH_

#include <cstddef>

#include "npygl/features.h"

namespace npygl {

namespace detail {

/**
 * `type_name<T>()` value when not compiling with MSVC, GCC, or Clang.
 */
constexpr auto unknown_type_name = "(unknown)";

#ifdef _MSC_VER
/**
 * Return string view of the specified type's name for MSVC.
 *
 * This uses the `__FUNCSIG__` function signature provided by MSVC.
 *
 * @tparam T type
 */
template <typename T>
constexpr std::string_view msvc_type_name() noexcept
{
  // begin + end iterators
  auto begin = std::begin(__FUNCSIG__);
  auto end = std::end(__FUNCSIG__);
  // for MSVC, the function signature is as follows:
  //
  //    class std::basic_string_view<char,struct std::char_traits<char> >
  //    __cdecl npygl::msvc_type_name<type-name>(void)
  //
  // unlike the GCC or Clang sigantures, there are far less identifying chars
  // we can use, in particular '=' to mark the beginning of the type. so we
  // actually go in reverse here, pre-decrementing the end iterator to the
  // rightmost '>' that we see to mark the one past the end of the type.
  while (begin != end && *--end != '>');
  // it is harder to locate the beginning because the Microsoft format does
  // not have any special chars that cannot be used in type names. therefore,
  // our approach is to count the number of unmatched angle brackets. each
  // '>' increases the count and each '<' decreases the count. at 0,
  //
  // number of unmatched angle brackets + iterator (starting at '>')
  std::size_t unmatched = 0u;
  auto it = end;
  // loop
  while (begin != it) {
    // if '>', increase unmatched
    if (*it == '>')
      unmatched++;
    // if '<', decrease unmatched
    else if (*it == '<')
      unmatched--;
    // if no unmatched, stop. for well-formed function signature, the next
    // char should be the first char of the type name, so advance + break
    if (!unmatched) {
      it++;
      break;
    }
    // otherwise, continue
    it--;
  }
  // done
#if NPYGL_HAS_CC_20
  return {it, end};
#else
  // cast to avoid narrowing conversion error
  return {it, static_cast<std::size_t>(end - it)};
#endif  // !NPYGL_HAS_CC_20
}
#endif  // _MSC_VER

#ifdef __GNUC__
/**
 * Return string view of the specified type's name for GCC/Clang.
 *
 * This uses the `__PRETTY_FUNCTION__` function signature defined by GCC/Clang.
 *
 * @tparam T type
 */
template <typename T>
constexpr std::string_view gnu_type_name() noexcept
{
  // begin iterator
  auto begin = std::begin(__PRETTY_FUNCTION__);
  // for GCC, the function signature is as follows:
  //
  //    constexpr std::string_view npygl::gnu_type_name()
  //    [with T = type-name; std::string_view = std::basic_string_view<char>]
  //
  // for Clang, the function signature is as follows:
  //
  //    std::string_view npygl::gnu_type_name() [T = type-name]
  //
  // therefore, we move the begin iterator to '='
  while (*begin && *begin != '=')
    begin++;
  // move to first char of type name (not ' ' or '=')
  while (*begin && (*begin == ' ' || *begin == '='))
    begin++;
  // for Clang, since ']' can show up as part of an array type, we move
  // backwards and pre-decrement to get to the rightmost ']'. for GCC, we can
  // just keep moving rightwards from begin to the ';'
#if defined(__clang__)
  auto end = std::end(__PRETTY_FUNCTION__);
  while (begin != end && *--end != ']');
#else
  auto end = begin;
  while (*end && *end != ';')
    end++;
#endif  // !defined(__clang__)
  // done
#if NPYGL_HAS_CC_20
  return {begin, end};
#else
  // cast to avoid narrowing conversion error
  return {begin, static_cast<std::size_t>(end - begin)};
#endif  // !NPYGL_HAS_CC_20
}
#endif  // __GNUC__

}  // namespace detail

/**
 * Return a string view of the specified type's name.
 *
 * This relies on compiler predefined string literals providing the function
 * signature. If the compiler used is not MSVC, GCC, or Clang, then
 * `detail::unknown_type_name` is returned instead.
 *
 * @tparam T type
 */
template <typename T>
constexpr std::string_view type_name() noexcept
{
// MSVC
#if defined(_MSC_VER)
  return detail::msvc_type_name<T>();
// GCC/Clang
#elif defined(__GNUC__)
  return detail::gnu_type_name<T>();
// unsupported
#else
  return detail::unknown_type_name;
#endif  // !defined(_MSC_VER) && !defined(__GNUC__)
}

/**
 * Check if `type_name<T>()` will provide the actual type name of `T`.
 *
 * For MSVC, GCC, and Clang, this will be true, and is false otherwise.
 */
constexpr bool type_name_supported() noexcept
{
  return detail::unknown_type_name != type_name<void*>();
}

}  // namespace npygl

#endif  // NPYGL_CTTI_HH_
