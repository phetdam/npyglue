/**
 * @file common.g
 * @author Derek Huang
 * @brief C/C++ header for common shared macros
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
 * Concatenate arguments without macro expansion.
 */
#define NPYGL_CONCAT_I(a, b) a ## b

/**
 * Concatenate arguments with macro expansion.
 */
#define NPYGL_CONCAT(a, b) NPYGL_CONCAT_I(a, b)

/**
 * Stringify argument without macro expansion.
 */
#define NPYGL_STRINGIFY_I(x) #x

/**
 * Stringify argument with macro expansion.
 */
#define NPYGL_STRINGIFY(x) NPYGL_STRINGIFY_I(x)

/**
 * Macro indicating an unused argument.
 */
#define NPYGL_UNUSED(x)

/**
 * Macro for expanding arguments unchanged.
 *
 * This is useful when dealing with multi-argument templates and macros because
 * without it, a template like `A<T, U>` is expanded into two macro arguments.
 */
#define NPYGL_IDENTITY(...) __VA_ARGS__

#endif  // NPYGL_COMMON_H_
