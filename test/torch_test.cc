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
    auto ten = npygl::make_tensor(std::move(vec), at::kComplexDouble);
    // note: operator<< of a complex tensor only prints the real part
    tensor_summary<decltype(vec)>(std::cout, ten);
  }
  // create float PyTorch tensor from an Eigen matrix
#if NPYGL_HAS_EIGEN3
  {
    Eigen::MatrixXf mat{
      {1.f, 2.3f, 4.34f, 1.222f, 2.12f},
      {5.44f, 12.f, 2.11f, 3.43f, 3.14159f},
      {7.66f, 5.99f, 2.122f, 6.51f, 1.23f}
    };
    auto ten = npygl::make_tensor(std::move(mat));
    tensor_summary<decltype(mat)>(std::cout, ten);
  }
#endif  /// NPYGL_HAS_EIGEN3
  return EXIT_SUCCESS;
}
