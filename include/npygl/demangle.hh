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

#if NPYGL_HAS_CXX_ABI_H
#include <cxxabi.h>
#endif  // NPYGL_HAS_CXX_ABI_H

namespace npygl {

// TODO: consider putting memory helpers in their own header
#if NPYGL_HAS_CXX_ABI_H
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
 * Return the demangled type name as a string.
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
#endif  // NPYGL_HAS_CXX_ABI_H

/**
 * Demangle the mangled type name.
 *
 * @param mangled_name The mangled type name
 */
inline std::string demangle(const char* mangled_name)
{
#if NPYGL_HAS_CXX_ABI_H
  unique_malloc_ptr<char[]> buf;
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
  return mangled_name;
#endif  // !NPYGL_HAS_CXX_ABI_H
}

/**
 * Return the demangled type name as a string.
 *
 * @param info Type info object
 */
inline auto demangle(const std::type_info* info)
{
  return demangle(info->name());
}

/**
 * Return the demangled type name as a string.
 *
 * @param info Type info object
 */
inline auto demangle(const std::type_info& info)
{
  return demangle(info.name());
}

}  // namespace npygl

#endif  // NPYGL_DEMANGLE_HH_
