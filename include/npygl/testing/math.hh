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
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>

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
enum class rngs {
  mersenne,     // Mersenne Twister
  mersenne64,   // 64-bit Mersenne Twister
  ranlux48,     // 48-bit RANLUX
};

// implementation details SWIG should not process
#ifndef SWIG

/**
 * Traits type to map a `rngs` value to a PRNG type.
 *
 * This provides the default PRNG type for any member without a specialization.
 *
 * @tparam R `rngs` value
 */
template <rngs R>
struct rng_type_traits {
  using type = std::mt19937;
};

/**
 * Specialization for the Mersenne Twister.
 */
template <>
struct rng_type_traits<rngs::mersenne> {
  using type = std::mt19937;
};

/**
 * Specialization for the 64-bit Mersenne Twister.
 */
template <>
struct rng_type_traits<rngs::mersenne64> {
  using type = std::mt19937_64;
};

/**
 * Specialization for the 48-bit RANLUX.
 */
template <>
struct rng_type_traits<rngs::ranlux48> {
  using type = std::ranlux48;
};

/**
 * Provide the PRNG type given a `rngs` value.
 *
 * @tparam R `rngs` value
 */
template <rngs R>
using rng_type_t = typename rng_type_traits<R>::type;

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

  // builder needs to be friend
  template <typename DistType, typename RngType>
  friend class rng_wrapper_builder;
};

/**
 * Provides the input type pair for the `rng_wrapper`.
 *
 * @tparam DistType *RandomNumberDistribution*
 * @tparam Rng `rngs` value
 */
template <typename DistType, rngs Rng>
using rng_wrapper_type_pair = std::pair<DistType, rng_type_t<Rng>>;

/**
 * Builder for the `rng_wrapper` that provides a fluent builder API.
 *
 * This replaces an older factory function template and enhances exception
 * safety guarantees if a method throws during building.
 *
 * @tparam DistType *RandomNumberDistribution*
 * @tparam RngType *UniformRandomBitGenerator*
 */
template <typename DistType, typename RngType>
class rng_wrapper_builder {
public:
  /**
   * Construct the distribution object the `rng_wrapper` will use.
   *
   * @tparam Args... `DistType` ctor arguments
   */
  template <typename... Args>
  auto& dist(Args&&... args)
  {
    dist_.reset(new DistType{std::forward<Args>(args)...});
    return *this;
  }

  /**
   * Construct the PRNG object the `rng_wrapper` will use.
   *
   * @tparam Args... `RngType` ctor arguments
   */
  template <typename... Args>
  auto& rng(Args&&... args)
  {
    rng_.reset(new RngType{std::forward<Args>(args)...});
    return *this;
  }

  /**
   * Finish building the `rng_wrapper` by setting its members.
   */
  auto operator()()
  {
    // set deleter
    w_.deleter_ = [](void* dist, void* rng)
    {
      static_cast<DistType*>(dist)->~DistType();
      static_cast<RngType*>(rng)->~RngType();
    };
    // set invoker
    w_.invoker_ = [](void* dist, void* rng)
    {
      return (*static_cast<DistType*>(dist))(*static_cast<RngType*>(rng));
    };
    // release distribution and PRNG objects and return
    w_.dist_ = dist_.release();
    w_.rng_ = rng_.release();
    return std::move(w_);
  }

private:
  // private distribution and PRNG objects. unique_ptr prevents leaks
  std::unique_ptr<DistType> dist_;
  std::unique_ptr<RngType> rng_;
  // target PRNG wrapper
  rng_wrapper<typename DistType::result_type> w_;
};

/**
 * Traits class for `rng_wrapper_builder`.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct rng_wrapper_builder_traits {};

/**
 * Partial specialization for a distribution and PRNG type.
 *
 * @tparam DistType *RandomNumberDistribution*
 * @tparam RngType *UniformRandomBitGenerator*
 */
template <typename DistType, typename RngType>
struct rng_wrapper_builder_traits<DistType, RngType> {
  using type = rng_wrapper_builder<DistType, RngType>;
};

/**
 * Partial specialization for a distribution and PRNG type pair.
 *
 * @tparam DistType *RandomNumberDistribution*
 * @tparam RngType *UniformRandomBitGenerator*
 */
template <typename DistType, typename RngType>
struct rng_wrapper_builder_traits<std::pair<DistType, RngType>>
  : rng_wrapper_builder_traits<DistType, RngType> {};

/**
 * Retrieve the `rng_wrapper_builder` specialization.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
using rng_wrapper_builder_t = typename rng_wrapper_builder_traits<Ts...>::type;

/**
 * Type alias for `std::optional<std::uint_fast_32_t>`.
 *
 * This is helpful as a way to specify a seed value or to use `random_device`.
 */
using optional_seed_type = std::optional<std::uint_fast32_t>;

/**
 * Return a vector of random values drawn from `[0, 1]`.
 *
 * @todo Try directly exposing this to SWIG after creating typemaps for
 * `std::optional<T>` for signed/unsigned integral types.
 *
 * @tparam T Floating type
 *
 * @param n Vector size
 * @param type PRNG type
 * @param seed Seed value to use
 */
template <typename T>
auto uniform(std::size_t n, rngs type, optional_seed_type seed = {})
{
  // produce generator based on PRNG type
  auto gen = [type, seed]
  {
    using dist_type = std::uniform_real_distribution<T>;
    // use random_device is no seed value
    auto sv = (seed) ? *seed : std::random_device{}();
    // switch over RNG type
    switch (type) {
      case rngs::mersenne:
        return rng_wrapper_builder<dist_type, rng_type_t<rngs::mersenne>>{}
          .dist()
          .rng(sv)();
      case rngs::mersenne64:
        return rng_wrapper_builder<dist_type, rng_type_t<rngs::mersenne64>>{}
          .dist()
          .rng(sv)();
      case rngs::ranlux48:
        return rng_wrapper_builder<dist_type, rng_type_t<rngs::ranlux48>>{}
          .dist()
          .rng(sv)();
      default:
        throw std::logic_error{
          NPYGL_PRETTY_FUNCTION_NAME +
          std::string{": invalid PRNG type specifier"}
        };
    }
  }();
  // allocate vector to return + populate
  std::vector<T> vec(n);
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
 *
 * @param n Vector size
 * @param seed Seed value to use
 */
template <typename T>
inline auto uniform(std::size_t n, optional_seed_type seed = {})
{
  return uniform<T>(n, rngs::mersenne, seed);
}
#endif  // SWIG

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_MATH_HH_
