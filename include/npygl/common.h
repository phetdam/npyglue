/**
 * @file common.g
 * @author Derek Huang
 * @brief C/C++ header for common shared macro
 * @copyright MIT License
 */

#ifndef NPYGL_COMMON_H_
#define NPYGL_COMMON_H_

/**
 * C++ version macro that works with MSVC.
 *
 * This allows us to avoid using `/Zc:__cplusplus` during compilation.
 */
#if defined(__cplusplus)
// MSVC
#if defined(_MSVC_LANG)
#define NPYGL_CPLUSPLUS _MSVC_LANG
#else
#define NPYGL_CPLUSPLUS __cplusplus
#endif  // !defined(_MSVC_LANG)
#endif  // !defined(__cplusplus)

/**
 * Stringify argument without macro expansion.
 */
#define NPYGL_STRINGIFY_I(x) #x

/**
 * Stringify argument with macro expansion.
 */
#define NPYGL_STRINGIFY(x) NPYGL_STRINGIFY_I(x)

#endif  // NPYGL_COMMON_H_
