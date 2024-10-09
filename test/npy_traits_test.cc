/**
 * @file npy_traits_test.cc
 * @author Derek Huang
 * @brief C++ tests for ndarray.hh type traits
 * @copyright MIT License
 */

#include <complex>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/testing/traits_checker.hh"

#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3

namespace {

// has_npy_type_traits test inputs
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
// can_make_ndarray test inputs
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
// test driver type
using test_driver_type = npygl::testing::traits_checker_driver<
  npygl::testing::traits_checker<npygl::has_npy_type_traits, input_types_1>,
  npygl::testing::traits_checker<npygl::can_make_ndarray, input_types_2>
>;

}  // namespace

int main()
{
  test_driver_type driver;
  return driver() ? EXIT_SUCCESS : EXIT_FAILURE;
}
