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
#include <cstddef>
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
#include "npygl/range_traits.hh"
#include "npygl/warnings.h"

#if NPYGL_HAS_CC_20
#include <span>
#endif  // NPYGL_HAS_CC_20

// note: SWIG doesn't under auto as a placeholder type, e.g. without a trailing
// return type. therefore any wrapped functions need a "real" return type

// TODO: once we develop the apply<Range>(R1&&, R2&&, UnaryFunc) template, we
// can replace a lot of the repeated looping with the new template

// TODO: need better range traits support

namespace npygl {
namespace testing {

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Traits type for `make_vector(R&&, F&&)` constaints.
 *
 * @tparam R Range-like type
 * @tparam F Unary callable taking and return `range_value_t<R>`
 */
template <typename R, typename F, typename = void>
struct make_vector_constraints {};

/**
 * True specialization that provides a `void` type member.
 *
 * @tparam R Range-like type
 * @tparam F Unary callable taking and return `range_value_t<R>`
 */
template <typename R, typename F>
struct make_vector_constraints<
  R,
  F,
  std::enable_if_t<
    is_iterable_v<R> &&
    std::is_invocable_r_v<range_value_t<R>, F, range_value_t<R>>> > {
  using type = void;
};

/**
 * SFINAE helper for `make_vector_constraints<R, F>`.
 *
 * @tparam R Range-like type
 * @tparam F Unary callable taking and return `range_value_t<R>`
 */
template <typename R, typename F>
using make_vector_constraints_t = typename make_vector_constraints<R, F>::type;

/**
 * Return a new vector obtained from applying a unary callable to a range.
 *
 * @note Definitely can be better constrained.
 *
 * @tparam R Range-like type
 * @tparam F Unary callable
 *
 * @param range Input range
 * @param func Unary callable
 */
template <typename R, typename F, typename = make_vector_constraints_t<R, F>>
auto make_vector(R&& range, F&& func)
{
  // begin and end iterators
  auto begin = std::begin(range);
  auto end = std::end(range);
  // result vector + index
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4365)
  std::vector<range_value_t<R>> res(std::distance(begin, end));
NPYGL_MSVC_WARNING_POP()
  std::size_t i = 0u;
  // populate + return
  for (auto it = begin; it != end; it++)
    res[i++] = func(*it);
  return res;
}

}  // namespace detail

/**
 * Return a new vector of the input range's elements multiplied by 2.
 *
 * @note We can definitely constrain the input more as the iterator must be a
 *  forward iterator that yields arithmetic types but that is for later.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = iterable_range_t<R>>
auto range_double(R&& range)
{
  using detail::make_vector;
  return make_vector(std::forward<R>(range), [](auto v) { return 2 * v; });
}
#endif  // SWIG

// SWIG doesn't allow defining macros to values so we allow this block to be
// enabled via compile-time -DNPYGL_SWIG_CC_20 flag
#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Function that simply doubles its argument.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
std::vector<T> array_double(std::span<const T> view)
{
  return range_double(view);
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
std::vector<T> array_double(ndarray_flat_view<const T> view)
{
  return range_double(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
// TODO: to simplify C++17/C++20 conditional compilation of the SWIG module we
// lock range-based implementation templates up in a separate namespace for
// now. eventually we want to remove these from detail::
namespace detail {

/**
 * Return a new vector of the sine of the input range's elements.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = floating_point_range_t<R>>
auto sine(R&& range)
{
  // sine functor since std::sin is overloaded (won't properly deduce)
  auto sine = [](auto v) { return std::sin(v); };
  return make_vector(std::forward<R>(range), sine);
}

}  // namespace detail
#endif  // SWIG

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Compute the sine of the view elements.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
std::vector<T> sine(std::span<const T> view)
{
  return detail::sine(view);
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
std::vector<T> sine(ndarray_flat_view<const T> view)
{
  return detail::sine(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Return a new vector of the inverse sine of the input range's elements.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = floating_point_range_t<R>>
auto asine(R&& range)
{
  // asine functor since std::asin is overloaded (won't properly deduce)
  auto asine = [](auto v) { return std::asin(v); };
  return make_vector(std::forward<R>(range), asine);
}

}  // namespace detail
#endif  // SWIG

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Compute the inverse sine of the view elements.
 *
 * @tparam T type
 *
 * @param view Span to operate on
 */
template <typename T>
std::vector<T> asine(std::span<const T> view)
{
  return detail::asine(view);
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
std::vector<T> asine(ndarray_flat_view<const T> view)
{
  return detail::asine(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Function that compresses the values of the argument to the unit circle.
 *
 * In other words, all the values will fall in `[-1, 1]`.
 *
 * @note We should constrain this to floating-point types.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = floating_point_range_t<R>>
auto unit_compress(R&& range)
{
  auto radius = *std::max_element(std::begin(range), std::end(range));
  auto func = [radius](auto v) { return v / radius; };
  return make_vector(std::forward<R>(range), std::move(func));
}

}  // namespace detail
#endif  // SWIG

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
std::vector<T> unit_compress(std::span<const T> view)
{
  return detail::unit_compress(view);
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
std::vector<T> unit_compress(ndarray_flat_view<const T> view)
{
  return detail::unit_compress(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Return the 1-norm of the given range.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = floating_point_range_t<R>>
auto norm1(R&& range) noexcept
{
  return std::accumulate(
    std::begin(range),
    std::end(range),
    range_value_t<R>{},
    [](const auto& sum, const auto& a) { return sum + std::abs(a); }
  );
}

}  // namespace detail
#endif  // SWIG

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Return the 1-norm of the given view.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
T norm1(std::span<const T> view) noexcept
{
  return detail::norm1(view);
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
T norm1(ndarray_flat_view<const T> view) noexcept
{
  return detail::norm1(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Return the 2-norm of the given range.
 *
 * @tparam R Range-like type
 *
 * @param range Input range
 */
template <typename R, typename = floating_point_range_t<R>>
auto norm2(R&& range) noexcept
{
  // sum of squared values
  auto total = std::accumulate(
    std::begin(range),
    std::end(range),
    range_value_t<R>{},
    [](const auto& sum, const auto& a) { return sum + a * a; }
  );
  // square root for norm
  return std::sqrt(total);
}

}  // namespace detail
#endif  // SWIG

#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
/**
 * Return the 2-norm of the given view.
 *
 * @tparam T type
 *
 * @param view Input span
 */
template <typename T>
T norm2(std::span<const T> view) noexcept
{
  return detail::norm2(view);
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
T norm2(ndarray_flat_view<const T> view) noexcept
{
  return detail::norm2(view);
}
#endif  // NPYGL_SWIG_CC_20

// implementation details SWIG should not process
#ifndef SWIG
namespace detail {

/**
 * Constraints for `inner` that requires both ranges have compatible types.
 *
 * Both ranges must have floating point value types that share commonality.
 * The provided type will be the common floating type amongst the ranges.
 *
 * @tparam T1 Range-like type
 * @tparam T2 Range-like type
 */
template <typename T1, typename T2>
using inner_constraints_t = std::enable_if_t<
  // common value type needs to be floating
  std::is_floating_point_v<
    std::common_type_t<range_value_t<T1>, range_value_t<T2>>
  >,
  // if instantiable, provide the common value type as a convenience
  std::common_type_t<range_value_t<T1>, range_value_t<T2>>
>;

/**
 * Compute the vector inner product of the two ranges.
 *
 * The return type is the common type amongst the two inputs' element types. If
 * there is no common type then there will be a template substitution failure.
 *
 * @note We use an `assert()` instead of throwing an exception to keep the
 *  function `noexcept` and to show how to work around this using `%inline`.
 *
 * @tparam R1 Range-like type
 * @tparam R2 Range-like type
 *
 * @param r1 First input range
 * @param r2 Second input range
 */
template <typename R1, typename R2, typename = inner_constraints_t<R1, R2>>
auto inner(R1&& r1, R2&& r2) noexcept
{
  // also the common value type
  using T = inner_constraints_t<R1, R2>;
  // TODO: need to allow sized range here + maybe not rely on assert()
  assert(std::size(r1) == std::size(r2));
  return std::inner_product(std::begin(r1), std::end(r1), std::begin(r2), T{});
}

}  // namespace detail
#endif  // SWIG

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
V inner(std::span<const T> in1, std::span<const U> in2) noexcept
{
  return detail::inner(in1, in2);
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
V inner(ndarray_flat_view<const T> in1, ndarray_flat_view<const U> in2) noexcept
{
  return detail::inner(in1, in2);
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
 * Create a `rng_wrapper<T>` instance given the specified PRNG type and seed.
 *
 * @param type PRNG type
 * @param seed Seed value to use
 */
template <typename T>
auto make_wrapped_rng(rngs type, optional_seed_type seed = {})
{
  using dist_type = std::uniform_real_distribution<T>;
  // use random_device if no seed value
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
}

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
  auto gen = make_wrapped_rng<T>(type, seed);
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
 * @tparam T Floating type
 *
 * @param n Vector size
 * @param seed Seed value to use
 */
template <typename T>
auto uniform(std::size_t n, optional_seed_type seed = {})
{
  return uniform<T>(n, rngs::mersenne, seed);
}
#endif  // SWIG

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_MATH_HH_
