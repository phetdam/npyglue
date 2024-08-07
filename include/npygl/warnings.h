/**
 * @file warnings.h
 * @author Derek Huang
 * @brief C/C++ header for warnings helper macros
 * @copyright MIT License
 */

#ifndef NPYGL_WARNINGS_H_
#define NPYGL_WARNINGS_H_

#include "npygl/common.h"

/**
 * `_Pragma` helper to allow use without quoting and with macro expansion.
 */
#define NPYGL_PRAGMA(x) _Pragma(NPYGL_STRINGIFY(x))

// MSVC warning control macros
#if defined(_MSC_VER)
#define NPYGL_MSVC_WARNING_PUSH() __pragma(warning(push))
#define NPYGL_MSVC_WARNING_DISABLE(wnos) __pragma(warning(disable : wnos))
#define NPYGL_MSVC_WARNING_POP() __pragma(warning(pop))
#else
#define NPYGL_MSVC_WARNING_PUSH()
#define NPYGL_MSVC_WARNING_DISABLE(wnos)
#define NPYGL_MSVC_WARNING_POP()
#endif  // !defined(_MSC_VER)

// GCC/Clang warning control macros (requires C99/C++11)
#if defined(__GNUC__)
#define NPYGL_GNU_WARNING_PUSH() _Pragma("GCC diagnostic push")
#define NPYGL_GNU_WARNING_DISABLE(w) \
  NPYGL_PRAGMA(GCC diagnostic ignored NPYGL_STRINGIFY(NPYGL_CONCAT(-W, w)))
#define NPYGL_GNU_WARNING_POP() _Pragma("GCC diagnostic pop")
#else
#define NPYGL_GNU_WARNING_PUSH()
#define NPYGL_GNU_WARNING_DISABLE(w)
#define NPYGL_GNU_WARNING_POP()
#endif  // !defined(__GNUC__)

#endif  // NPYGL_WARNINGS_H_
