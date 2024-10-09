/**
 * @file npy_traits_test.cc
 * @author Derek Huang
 * @brief C++ tests for ndarray.hh type traits
 * @copyright MIT License
 */

#include <complex>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/type_traits.hh"

#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3

namespace {

/**
 * Check if a type satisfies the given traits.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T type
 */
template <template <typename> typename Traits, typename T>
struct traits_checker {
  /**
   * Check the type against the traits and return number of failures.
   *
   * @param out Stream to write messages to
   * @return 0 on success, 1 on failure
   */
  unsigned operator()(std::ostream& out = std::cout) const
  {
    constexpr bool match = Traits<T>::value;
    out << "Test: " << npygl::type_name(typeid(Traits<T>)) <<
      "::value == true\n  " << ((match) ? "PASS" : "FAIL") << std::endl;
    return !match;
  }
};

/**
 * Partial specialization for a `std::pair<T, std::bool_constant<B>>`.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T type
 * @tparam B Expected truth value
 */
template <template <typename> typename Traits, typename T, bool B>
struct traits_checker<Traits, std::pair<T, std::bool_constant<B>>> {
  /**
   * Check the type against the traits and return number of failures.
   *
   * @param out Stream to write messages to
   * @return 0 on success, 1 on failure
   */
  unsigned operator()(std::ostream& out = std::cout) const
  {
    constexpr bool match = (B == Traits<T>::value);
    out << "Test: " << npygl::type_name(typeid(Traits<T>)) << "::value == " <<
      (B ? "true" : "false") << "\n  " <<
      (match ? "PASS" : "FAIL") << std::endl;
    return !match;
  }
};

/**
 * Placeholder type.
 */
struct placeholder {};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam Ts... Type pack where types can be `std::pair<U, V>`, `V` either
 *  `std::true_type` or `std::false_type`, to indicate that the type `U` should
 *  induce `Traits<U>::value` to be a specific value. A non-pair type `U` is
 *  expected to induce `Traits<U>::value` to be `true`.
 */
template <template <typename> typename Traits, typename... Ts>
struct traits_checker<Traits, std::tuple<Ts...>> {
private:
  /**
   * Conditionally wrap a tuple to prevent recursion into this specialization.
   *
   * Without this, if a `Ts` is a `std::tuple` specialization, the compiler
   * will recurse into this partial specialization, which is not what we want.
   *
   * @tparam T type
   */
  template <typename T>
  using tuple_wrap = std::conditional_t<
    npygl::is_tuple_v<T>, std::pair<T, std::true_type>, T
  >;

public:
  /**
   * Check each type with the traits and return failures.
   *
   * @param out Stream to write messages to
   * @returns Number of failed types
   */
  auto operator()(std::ostream& out = std::cout) const
  {
    out << "Running " << sizeof...(Ts) << " tests on " <<
      npygl::type_name(typeid(Traits<placeholder>)) << "..." << std::endl;
    return (traits_checker<Traits, tuple_wrap<Ts>>{}(out) + ...);
  }
};

/**
 * Global functor to check if a type or tuple of types satisfies traits.
 *
 * @tparam Traits Traits template type with a boolean `value` member
 * @tparam T Type, either a `std::tuple<Ts...>` or single type
 */
template <template <typename> typename Traits, typename T>
inline constexpr traits_checker<Traits, T> traits_check;

// has_npy_type_traits test
using input_types_1 = std::tuple<
  double,
  float,
  int,
  unsigned,
  long,
  unsigned long,
  std::complex<double>,
  std::complex<float>,
  std::pair<std::string, std::false_type>,
  std::pair<std::map<std::string, double>, std::false_type>,
  std::pair<std::pair<std::pair<int, int>, std::string>, std::false_type>
>;
// can_make_ndarray test
using input_types_2 = std::tuple<
  std::vector<double>,
  std::pair<std::vector<float>, std::true_type>,
  std::pmr::vector<unsigned>,
  std::pmr::vector<double>,
  std::pair<std::map<unsigned, std::vector<double>>, std::false_type>,
#if NPYGL_HAS_EIGEN3
  Eigen::MatrixXd,
  Eigen::MatrixXf,
  Eigen::Matrix4cd,
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  arma::mat,
  arma::fmat,
  arma::cube,
  arma::rowvec,
  arma::fvec,
#endif  // NPYGL_HAS_ARMADILLO
  std::pair<std::vector<std::vector<double>>, std::false_type>,
  std::pair<
    std::tuple<std::pair<unsigned, std::true_type>, std::pair<double, int>>,
    std::false_type
  >
>;

}  // namespace

int main()
{
  // total failed
  unsigned n_fail = 0u;
  n_fail += traits_check<npygl::has_npy_type_traits, input_types_1>();
  n_fail += traits_check<npygl::can_make_ndarray, input_types_2>();
  // summary
  constexpr auto n_total = std::tuple_size_v<input_types_1> +
    std::tuple_size_v<input_types_2>;
  // CTest-like output
  std::cout << '\n' <<
    100 * (1 - n_fail / static_cast<double>(n_total)) << "% tests passed, " <<
    n_fail << " failed out of " << n_total << '\n' << std::endl;
  return (!n_fail) ? EXIT_SUCCESS : EXIT_FAILURE;
}
