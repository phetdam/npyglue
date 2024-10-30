/**
 * @file torch_test.cc
 * @author Derek Huang
 * @brief torch.hh unit tests
 * @copyright MIT License
 */

#include <cstdlib>
#include <iostream>
#include <memory_resource>
#include <typeinfo>
#include <vector>

#include <c10/util/complex.h>
#include <torch/torch.h>

#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/torch.hh"

#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYL_HAS_ARMADILLO
#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3

namespace {

/**
 * Helper function to print out a PyTorch tensor and its source type.
 *
 * @tparam T Source type
 *
 * @param out Stream to write output to
 * @param ten Tensor created from the source type
 */
template <typename T>
void tensor_summary(std::ostream& out, const torch::Tensor& ten)
{
  out << "-- " << npygl::type_name(typeid(ten)) << " (" <<
    npygl::type_name(typeid(T)) << ")\n" << ten << std::endl;
}

}  // namespace

int main()
{
  // create complex double PyTorch tensor from a vector with custom resource
  {
    // resource with stack-based initial buffer (fine for most systems)
    unsigned char buf[64 * sizeof(c10::complex<double>)];
    std::pmr::monotonic_buffer_resource res{buf, sizeof buf};
    // vector using custom resource (should not exceed stack-based buffer)
    std::pmr::vector<c10::complex<double>> vec{
      {
        {1.3, 4.2},
        {4.55, 2.32},
        {4.3, 2.322},
        {1.1, 8.76},
        {4.32, 0.},
        {0., 12.11}
      },
      &res
    };
    auto ten = npygl::make_tensor(std::move(vec));
    // note: operator<< of a complex tensor only prints the real part
    tensor_summary<decltype(vec)>(std::cout, ten);
  }
  // create double PyTorch tensor from a vector
  {
    // note: roughly powers of 1.05 but with some truncation
    std::vector<double> vec{1., 1.05, 1.1025, 1.157625, 1.21550625, 1.2762815625};
    auto ten = npygl::make_tensor(std::move(vec));
    tensor_summary<decltype(vec)>(std::cout, ten);
  }
#if NPYGL_HAS_EIGEN3
  // create float PyTorch tensor from an Eigen matrix
  {
    Eigen::MatrixXf mat{
      {1.f, 2.3f, 4.34f, 1.222f, 2.12f},
      {5.44f, 12.f, 2.11f, 3.43f, 3.14159f},
      {7.66f, 5.99f, 2.122f, 6.51f, 1.23f}
    };
    auto ten = npygl::make_tensor(std::move(mat));
    tensor_summary<decltype(mat)>(std::cout, ten);
  }
  // create double PyTorch tensor from a row-major Eigen matrix
  {
    Eigen::Matrix<
      double,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::StorageOptions::RowMajor
    > mat{
      {1.3, 4.22, 4.13, 1.766},
      {6.5, 3.33, 6.46, 1.8},
      {6.65, 1.23, 14.11, 6.21}
    };
    auto ten = npygl::make_tensor(std::move(mat));
    tensor_summary<decltype(mat)>(std::cout, ten);
  }
  // create float PyTorch tensor from fixed-size Eigen vector outer product
  {
    // print vector
    Eigen::Vector4f vec{1.f, 1.5f, 2.4f, 5.4f};
    auto ten = npygl::make_tensor(std::move(vec));
    tensor_summary<decltype(vec)>(std::cout, ten);
    // print outer product
    std::cout << ten * ten.t() << std::endl;
  }
#endif  /// NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
  // create float PyTorch tensor from an Armadillo matrix
  {
    arma::fmat mat{
      {1.f, 3.f},
      {1.44f, 5.4f},
      {3.44f, 2.1f},
      {3.23f, 4.2f}
    };
    auto ten = npygl::make_tensor(std::move(mat));
    tensor_summary<decltype(mat)>(std::cout, ten);
  }
  // create double PyTorch tensor from an Armadillo column vector
  {
    arma::vec vec{3., 2.3, 1.222, 3.51, 7.34, 4.12, 31.22, 5.33};
    auto ten = npygl::make_tensor(std::move(vec));
    tensor_summary<decltype(vec)>(std::cout, ten);
  }
  // create float PyTorch tensor from an Armadillo row vector
  {
    // sequence created by computing log2(5i) for i = 1, ... 6 and truncating
    // to 5 decimal places from the original double values
    // note: with Visual Studio 2022 the halfway values with 5 as least
    // significant decimal are rounded down instead of up for some reason
    // note: WSL1 Ubuntu 22.04 GCC 11.3 also exhibiting same behavior when it
    // did not exhibit that behavior previously for some odd reason
    arma::frowvec vec{2.80735f, 3.80735f, 4.39231f, 4.80735f, 5.12928f, 5.39231f};
    auto ten = npygl::make_tensor(std::move(vec));
    tensor_summary<decltype(vec)>(std::cout, ten);
  }
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
  return EXIT_SUCCESS;
}
