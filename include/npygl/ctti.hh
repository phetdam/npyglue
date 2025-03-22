/**
 * @file ctti.hh
 * @author Derek Huang
 * @brief C++ header for compile-time type information
 * @copyright MIT License
 */

#ifndef NPYGL_CTTI_HH_
#define NPYGL_CTTI_HH_

#include <cstddef>

#include "npygl/common.h"
#include "npygl/features.h"

namespace npygl {

namespace detail {

/**
 * `type_name<T>()` value when not compiling with MSVC, GCC, or Clang.
 */
constexpr auto unsupported_type_name = "(unknown)";

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
  // begin iterator
  auto begin = std::begin(__FUNCSIG__);
  // for MSVC, the function signature is as follows:
  //
  //    class std::basic_string_view<char,struct std::char_traits<char> >
  //    __cdecl npygl::msvc_type_name<type-name>(void)
  //
  // therefore, we advance forward to the first '('. we cannot go in reverse
  // because *std::end(__FUNCSIG__) is undefined behavior
  auto end = begin;
  while (*end && *end != '(')
    end++;
  // this is too far, so bring it back to the rightmost '>'
  while (begin != end && *end != '>')
    end--;
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
  // move end to ';' for GCC but to ']' for Clang
  auto end = begin;
  constexpr auto term =
#if defined(__clang__)
    ']'
#else
    ';'
#endif  // !defined(__clang__)
    ;
  while (*end && *end != term)
    end++;
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
 * `detail::unsupported_type_name` is returned instead.
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
  return detail::unsupported_type_name;
#endif  // !defined(_MSC_VER) && !defined(__GNUC__)
}

/**
 * Check if `type_name<T>()` will provide the actual type name of `T`.
 *
 * For MSVC, GCC, and Clang, this will be true, and is false otherwise.
 */
constexpr bool type_name_supported() noexcept
{
  return detail::unsupported_type_name != type_name<void*>();
}

}  // namespace npygl

#endif  // NPYGL_CTTI_HH_
