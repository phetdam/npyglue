/**
 * @file demangle.hh
 * @author Derek Huang
 * @brief C++ header for demangling Itanium ABI type names
 * @copyright MIT License
 */

#ifndef NPYGL_DEMANGLE_HH_
#define NPYGL_DEMANGLE_HH_

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "npygl/features.h"

#if NPYGL_HAS_ITANIUM_ABI
#include <cxxabi.h>
#endif  // NPYGL_HAS_ITANIUM_ABI

namespace npygl {

// TODO: consider putting memory helpers in their own header
#if NPYGL_HAS_ITANIUM_ABI
/**
 * Custom deleter for unique pointers allocating memory with `malloc`.
 *
 * @tparam T
 */
struct malloc_deleter {
  void operator()(void* ptr) const noexcept
  {
    std::free(ptr);
  }
};

/**
 * Unique pointer type alias using `malloc_deleter` as a deleter.
 *
 * @tparam T Element type
 */
template <typename T>
using unique_malloc_ptr = std::unique_ptr<T, malloc_deleter>;

/**
 * Demangle the mangled type name.
 *
 * @param name Unique pointer with the null-terminated demangled name buffer
 * @param mangled_name The mangled type name
 * @returns 0 on success, <0 on error (see `abi::__cxa_demangle` docs)
 */
inline auto demangle(
  unique_malloc_ptr<char[]>& name, const char* mangled_name) noexcept
{
  int status;
  name.reset(abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status));
  return status;
}

/**
 * Get the type name of the `std::type_info` into a heap-allocated buffer.
 *
 * @note This is only useful if you really want a `noexcept` guarantee.
 *
 * @param name Unique pointer with the null-terminated demangled name buffer
 * @param info Type info object
 * @returns 0 on success, <0 on error (see `abi::__cxa_demangle` docs)
 */
inline auto demangle(
  unique_malloc_ptr<char[]>& name, const std::type_info& info) noexcept
{
  return demangle(name, info.name());
}
#endif  // NPYGL_HAS_ITANIUM_ABI

/**
 * Demangle the mangled type name.
 *
 * The return value points to a null-terminated string that resides in
 * thread-local storage and therefore should not be deallocated.
 *
 * @param mangled_name The mangled type name
 */
inline const char* demangle(const char* mangled_name)
{
  // note: thread local storage is very convenient here
#if NPYGL_HAS_ITANIUM_ABI
  thread_local unique_malloc_ptr<char[]> buf;
  // switch on status
  switch (demangle(buf, mangled_name)) {
    case 0:
      break;
    case -1:
      throw std::runtime_error{"memory allocation failure"};
    case -2:
      throw std::runtime_error{"mangled_name is not a valid mangled type name"};
    case -3:
      throw std::runtime_error{"__cxa_demangle provided an invalid argument"};
    default:
      throw std::logic_error{"unreachable"};
  }
  return buf.get();
#else
  // using MSVC as an example all we need is to copy the name
  thread_local std::string name;
  name = mangled_name;
  return name.c_str();
#endif  // !NPYGL_HAS_ITANIUM_ABI
}

/**
 * Return the demangled type name as a null-terminated string.
 *
 * The return value points to a null-terminated string that resides in
 * thread-local storage and therefore should not be deallocated.
 *
 * @param info Type info object
 */
inline auto type_name(const std::type_info* info)
{
  return demangle(info->name());
}

/**
 * Return the demangled type name as a null-terminated string.
 *
 * The return value points to a null-terminated string that resides in
 * thread-local storage and therefore should not be deallocated.
 *
 * @param info Type info object
 */
inline auto type_name(const std::type_info& info)
{
  return demangle(info.name());
}

/**
 * Functor to return a comma-separated list of type names.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct type_name_lister {};

/**
 * Partial specialization for a single type.
 *
 * @tparam T type
 */
template <typename T>
struct type_name_lister<T> {
  std::string operator()() const
  {
    return type_name(typeid(T));
  }
};

/**
 * Partial specialization for more than one type.
 *
 * @tparam T type
 * @tparam Ts... Other types
 */
template <typename T, typename... Ts>
struct type_name_lister<T, Ts...> {
  auto operator()() const
  {
    return type_name_lister<T>{}() + ", " + type_name_lister<Ts...>{}();
  }
};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct type_name_lister<std::tuple<Ts...>> : type_name_lister<Ts...> {};

/**
 * Global functor to provide a comma-separated list of type names.
 *
 * This provides a functional interface to the `type_name_lister<Ts...>`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr type_name_lister<Ts...> type_name_list;

}  // namespace npygl

#endif  // NPYGL_DEMANGLE_HH_
