/**
 * @file npy_intp_alias_test.cc
 * @author Derek Huang
 * @brief C++ program to verify when npy_intp can alias an `at::IntArrayRef`
 * @copyright MIT License
 */

#include <cstdlib>
#include <iostream>

#include <ATen/core/TensorBase.h>  // for at::TensorBase::size
#include <numpy/npy_common.h>  // for npy_intp

#include "npygl/type_traits.hh"

namespace {

// should be int64_t
using torch_dim_type = decltype(std::declval<at::TensorBase>().size(0));
// test result
constexpr bool aliasable = npygl::is_accessible_as_v<torch_dim_type, npy_intp>;
// message components
constexpr auto status = (aliasable) ? "  OK" : "FAIL";
constexpr auto maybe_neg = (aliasable) ? "" : "NOT ";

}  // namespace

int main()
{
  // Torch tensor stride type is same size as npy_intp
  if constexpr (sizeof(torch_dim_type) == sizeof(npy_intp)) {
    std::cout << status << ": torch_dim_type " << maybe_neg <<
      "accessible as npy_intp" << std::endl;
    return (aliasable) ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  // skip
  else
    std::cout << "SKIPPED: sizeof(torch_dim_type) != sizeof(npy_intp)" <<
      std::endl;
  return EXIT_SUCCESS;
}
