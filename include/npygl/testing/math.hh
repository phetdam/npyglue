/**
 * @file testing/math.hh
 * @author Derek Huang
 * @brief C++ header for math functions used in tests
 * @copyright MIT License
 */

#ifndef NPYGL_TESTING_MATH_HH_
#define NPYGL_TESTING_MATH_HH_

#include <algorithm>
#include <cmath>
#include <numeric>

#include "npygl/features.h"
#include "npygl/npy_helpers.hh"

#if NPYGL_HAS_CC_20
#include <span>
#endif  // NPYGL_HAS_CC_20

namespace npygl {
namespace testing {

// SWIG doesn't allow defining macros to values so we allow this block to be
// enabled via compile-time -DNPYGL_SWIG_CC_20 flag
#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Function that simply doubles its argument.
 *
 * @tparam T type
 *
 * @param view Span to operate on
 */
template <typename T>
void array_double(std::span<T> view) noexcept
{
  for (auto& v : view)
    v = 2 * v;
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

// make only std::span overloads visible to SWIG when compiling with C++20
#ifndef NPYGL_SWIG_CC_20
/**
 * Function that simply doubles its argument.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void array_double(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  // explicitly mention std::span in case flat_view gets similar ctor
  array_double(std::span{view.begin(), view.end()});
#else
  for (auto& v : view)
    v = 2 * v;
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Compute the sine of the view elements.
 *
 * @tparam T type
 *
 * @param view Span to operate on
 */
template <typename T>
void sine(std::span<T> view) noexcept
{
  for (auto& v : view)
    v = std::sin(v);
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

#ifndef NPYGL_SWIG_CC_20
/**
 * Compute the sine of the view elements.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void sine(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  sine(std::span{view.begin(), view.end()});
#else
  for (auto& v : view)
    v = std::sin(v);
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Compute the inverse sine of the view elements.
 *
 * @tparam T type
 *
 * @param view Span to operate on
 */
template <typename T>
void asine(std::span<T> view) noexcept
{
  for (auto& v : view)
    v = std::asin(v);
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

#ifndef NPYGL_SWIG_CC_20
/**
 * Compute the inverse sine of the view elements.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void asine(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  asine(std::span{view.begin(), view.end()});
#else
  for (auto& v : view)
    v = std::asin(v);
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Function that compresses the values of the argument to the unit circle.
 *
 * In other words, all the values will fall in `[-1, 1]`.
 *
 * @tparam T type
 *
 * @param view Span to operate on
 */
template <typename T>
void unit_compress(std::span<T> view) noexcept
{
  auto radius = *std::ranges::max_element(view);
  std::ranges::for_each(view, [&radius](auto& x) { x /= radius; });
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

#ifndef NPYGL_SWIG_CC_20
/**
 * Function that compresses the values of the argument to the unit circle.
 *
 * In other words, all the values will fall in `[-1, 1]`.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void unit_compress(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  unit_compress(std::span{view.begin(), view.end()});
#else
  auto radius = *std::max_element(view.begin(), view.end());
  std::for_each(view.begin(), view.end(), [&radius](auto& x) { x /= radius; });
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Return the 1-norm of the given view.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
T norm1(std::span<T> view) noexcept
{
  return std::accumulate(
    view.begin(),
    view.end(),
    T{},
    [](const T& sum, const T& a) { return sum + std::abs(a); }
  );
}

/**
 * Return the 2-norm of the given view.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
T norm2(std::span<T> view) noexcept
{
  // sum of squared values
  auto total = std::accumulate(
    view.begin(),
    view.end(),
    T{},
    [](const T& sum, const T& a) { return sum + a * a; }
  );
  return std::sqrt(total);
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

#ifndef NPYGL_SWIG_CC_20
/**
 * Return the 1-norm of the given view.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
T norm1(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  return norm1(std::span{view.begin(), view.end()});
#else
  return std::accumulate(
    view.begin(),
    view.end(),
    T{},
    [](const T& sum, const T& a) { return sum + std::abs(a); }
  );
#endif  // !NPYGL_HAS_CC_20
}

/**
 * Return the 2-norm of the given view.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
T norm2(ndarray_flat_view<T> view) noexcept
{
#if NPYGL_HAS_CC_20
  return norm2(std::span{view.begin(), view.end()});
#else
  return std::sqrt(
    std::accumulate(
      view.begin(),
      view.end(),
      T{},
      [](const T& sum, const T& a) { return sum + a * a; }
    )
  );
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_MATH_HH_
