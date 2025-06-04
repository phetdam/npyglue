/**
 * @file common.h
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
 *
 * @note Doxygen does not correctly process this macro which causes instances
 *  of `NPYGL_UNUSED(arg)` to show up in documentation as `NPYGL_UNUSEDarg`.
 *  Doxygen will also erroneously report that `NPYGL_UNUSED` is an undocumented
 *  parameter and that `arg` is not found in the function's argument list.
 */
#define NPYGL_UNUSED(x)

/**
 * Macro for expanding arguments unchanged.
 *
 * This is useful when dealing with multi-argument templates and macros because
 * without it, a template like `A<T, U>` is expanded into two macro arguments.
 */
#define NPYGL_IDENTITY(...) __VA_ARGS__

/**
 * Macro for a string variable giving the function signature.
 *
 * Falls back to `__func__` if not compiling with GCC/Clang/MSVC.
 *
 * @note As per documentation this is *not* a string literal but a variable. In
 *  particular, string literal concatenation cannot be done.
 */
#if defined(_MSC_VER)
#define NPYGL_PRETTY_FUNCTION_NAME __FUNCSIG__
#elif defined(__GNUC__)
#define NPYGL_PRETTY_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#define NPYGL_PRETTY_FUNCTION_NAME __func__
#endif  // !defined(_MSC_VER) && !defined(__GNUC__)

#endif  // NPYGL_COMMON_H_
