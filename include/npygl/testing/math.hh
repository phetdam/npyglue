/**
 * @file testing/math.hh
 * @author Derek Huang
 * @brief C++ header for math functions used in tests
 * @copyright MIT License
 */

#ifndef NPYGL_TESTING_MATH_HH_
#define NPYGL_TESTING_MATH_HH_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <stdexcept>
#include <type_traits>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/ndarray.hh"

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
inline void unit_compress(std::span<T> view) noexcept
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
inline void unit_compress(ndarray_flat_view<T> view) noexcept
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
inline T norm1(std::span<T> view) noexcept
{
  return std::accumulate(
    view.begin(),
    view.end(),
    T{},
    [](const T& sum, const T& a) { return sum + std::abs(a); }
  );
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
inline T norm1(ndarray_flat_view<T> view) noexcept
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
#endif  // NPYGL_SWIG_CC_20

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Return the 2-norm of the given view.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
inline T norm2(std::span<T> view) noexcept
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
 * Return the 2-norm of the given view.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
inline T norm2(ndarray_flat_view<T> view) noexcept
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

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Compute the vector inner product.
 *
 * The return type is the common type amongst the two inputs' element types. If
 * there is no common type then there will be a template substitution failure.
 *
 * @note We use an `assert()` instead of throwing an exception to keep the
 *  function `noexcept` and to show how to work around this using `%inline`.
 *
 * @tparam T type
 * @tparam U type
 *
 * @param in1 First input span
 * @param in2 Second input span
 */
template <typename T, typename U, typename V = std::common_type_t<T, U>>
inline V inner(std::span<T> in1, std::span<U> in2) noexcept
{
  assert(in1.size() == in2.size());
  return std::inner_product(in1.begin(), in1.end(), in2.begin(), V{});
}
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20

#ifndef NPYGL_SWIG_CC_20
/**
 * Compute the vector inner product.
 *
 * The return type is the common type amongst the two inputs' element types. If
 * there is no common type then there will be a template substitution failure.
 *
 * @note We use an `assert()` instead of throwing an exception to keep the
 *  function `noexcept` and to show how to work around this using `%inline`.
 *
 * @tparam T type
 * @tparam U type
 *
 * @param in1 First NumPy array view
 * @param in2 Second NumPy array view
 */
template <typename T, typename U, typename V = std::common_type_t<T, U>>
inline V inner(ndarray_flat_view<T> in1, ndarray_flat_view<U> in2) noexcept
{
#if NPYGL_HAS_CC_20
  using std::span;
  return inner(span{in1.begin(), in2.end()}, span{in2.begin(), in2.end()});
#else
  assert(in1.size() == in2.size());
  return std::inner_product(in1.begin(), in1.end(), in2.begin(), V{});
#endif  // !NPYGL_HAS_CC_20
}
#endif  // NPYGL_SWIG_CC_20

/**
 * Enumeration for selecting a PRNG to use.
 */
enum class rng_type {
  mersenne,     // Mersenne Twister
  mersenne64,   // 64-bit Mersenne Twister
  ranlux48      // 48-bit RANLUX
};

// implementation details SWIG should not process
#ifndef SWIG
/**
 * Type used as an argument delineator.
 */
struct delineator {};

/**
 * A type-erasure wrapper for a distribution type and its PRNG type.
 *
 * This models an object that draws pseudo-random values from a distribution.
 *
 * @tparam T Type of the generated value
 */
template <typename T>
class rng_wrapper {
public:
  using result_type = T;

  /**
   * Destroy the distribution object and the PRNG object.
   */
  ~rng_wrapper()
  {
    deleter_(dist_, rng_);
  }

  /**
   * Get a pseudo-random value from the distribution using the internal PRNG.
   */
  T operator()() const
  {
    return invoker_(dist_, rng_);
  }

private:
  /**
   * Ctor.
   *
   * We disallow client creaton of the `rng_wrapper`.
   */
  rng_wrapper() = default;

  void* dist_;
  void* rng_;
  std::function<void(void*, void*)> deleter_;
  std::function<T(void*, void*)> invoker_;

  // factory function is friend
  template <typename DistType, typename RngType, typename... DTs, typename... RTs>
  friend auto make_rng_wrapper(DTs&&..., delineator, RTs&&...);
};

/**
 * Create a new `rng_wrapper` from the given distribution and RNG types.
 *
 * @tparam DistType *RandomNumberDistribution*
 * @tparam RngType *UniformRandomBitGenerator*
 * @tparam DTs... Argument types for the `DistType` ctor
 * @tparam RTs... Argument types for the `RngType` ctor
 */
template <typename DistType, typename RngType, typename... DTs, typename... RTs>
auto make_rng_wrapper(
  DTs&&... dist_args, delineator /*delim*/, RTs&&... rng_args)
{
  rng_wrapper<typename DistType::result_type> w;
  // set distribution and RNG pointers
  w.dist_ = new DistType{std::forward<DTs>(dist_args)...};
  w.rng_ = new RngType{std::forward<RTs>(rng_args)...};
  // set deleter
  w.deleter_ = [](void* dist, void* rng)
  {
    static_cast<DistType*>(dist)->~DistType();
    static_cast<RngType*>(rng)->~RngType();
  };
  // set invoker
  w.invoker_ = [](void* dist, void* rng)
  {
    return (*static_cast<DistType*>(dist))(*static_cast<RngType*>(rng));
  };
  return w;
}

/**
 * Return a vector of random values drawn from `[0, 1]`.
 *
 * @todo Try directly exposing this to SWIG after creating typemaps for
 * `std::optional<T>` for signed/unsigned integral types.
 *
 * @tparam T Floating type
 * @tparam A Allocator type
 *
 * @param n Vector size
 * @param type PRNG type
 * @param seed Seed value to use
 */
template <typename T, typename A = std::allocator<double>>
auto urand_vector(
  std::size_t n, rng_type type, std::optional<std::uint_fast64_t> seed = {})
{
  // produce generator based on PRNG type
  auto gen = [type, seed]
  {
    using dist_type = std::uniform_real_distribution<T>;
    // use random_device is no seed value
    auto sv = (seed) ? *seed : std::random_device{}();
    // switch over RNG type
    switch (type) {
      case rng_type::mersenne:
        return make_rng_wrapper<dist_type, std::mt19937>(delineator{}, sv);
      case rng_type::mersenne64:
        return make_rng_wrapper<dist_type, std::mt19937_64>(delineator{}, sv);
      case rng_type::ranlux48:
        return make_rng_wrapper<dist_type, std::ranlux48>(delineator{}, sv);
      default:
        throw std::logic_error{
          NPYGL_PRETTY_FUNCTION_NAME +
          std::string{": invalid PRNG type specifier"}
        };
    }
  }();
  // allocate vector to return + populate
  std::vector<T, A> vec(n);
  for (auto& v : vec)
    v = gen();
  return vec;
}

/**
 * Return a vector of random values drawn from `[0, 1]`.
 *
 * The underlying PRNG used is the 32-bit Mersenne Twister.
 *
 * @todo Overloads confuse SWIG; probably keep in conditional block.
 *
 * @tparam T Floating type
 * @tparam A Allocator type
 *
 * @param n Vector size
 * @param type PRNG type
 */
template <typename T, typename A = std::allocator<double>>
inline auto urand_vector(
  std::size_t n, std::optional<std::uint_fast64_t> seed = {})
{
  return urand_vector<T, A>(n, rng_type::mersenne, seed);
}
#endif  // SWIG

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_MATH_HH_
