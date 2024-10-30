/**
 * @file torch_traits_test.cc
 * @author Derek Huang
 * @brief C++ program testing PyTorch-related type traits
 * @copyright MIT License
 */

#include <complex>
#include <cstdlib>
#include <deque>
// note: libstdc++ requires explicit include for std::pmr::vector. when using
// std::pmr::vector<float> and std::pmr::vector<std::complex<float>> GCC 11.3
// was giving errors about incomplete types without this included
#include <memory_resource>
#include <tuple>
#include <utility>
#include <vector>

#include "npygl/termcolor.hh"
#include "npygl/testing/traits_checker.hh"
#include "npygl/torch.hh"

// namespace contiuation for tensor_info_context specializations
namespace npygl {

/**
 * Invalid `tensor_info_context<T>` specialization.
 *
 * @tparam T Element type
 */
template <typename T>
struct tensor_info_context<std::pmr::deque<T>> {};

}  // namespace npygl

using driver_type = npygl::testing::traits_checker_driver<
  // is_tensor_info_context
  npygl::testing::traits_checker<
    npygl::is_tensor_info_context,
    std::tuple<
      npygl::tensor_info_context<std::vector<double>>,
      npygl::tensor_info_context<std::pmr::deque<unsigned>>,
      std::pair<double, std::false_type>,
      std::pair<std::vector<double>, std::false_type>,
      npygl::tensor_info_context<int>
    >
  >,
  // is_tensor_info_context_constructible
  npygl::testing::traits_checker<
    npygl::is_tensor_info_context_constructible,
    std::tuple<
      npygl::tensor_info_context<std::pmr::vector<unsigned>>,
      npygl::tensor_info_context<std::vector<int>>,
      std::pair<npygl::tensor_info_context<int>, std::false_type>,
      std::pair<npygl::tensor_info_context<std::deque<int>>, std::false_type>
    >
  >,
  // is_tensor_info_context_with_data
  npygl::testing::traits_checker<
    npygl::is_tensor_info_context_with_data,
    std::tuple<
      std::pair<npygl::tensor_info_context<int>, std::false_type>,
      npygl::tensor_info_context<std::pmr::vector<float>>,
      std::pair<std::vector<std::complex<float>>, std::false_type>,
      // note: must use c10::complex instead of std::complex
      std::pair<
        npygl::tensor_info_context<std::pmr::vector<std::complex<double>>>,
        std::false_type
      >,
      // PyTorch C++ types work just fine
      npygl::tensor_info_context<std::pmr::vector<c10::complex<c10::Half>>>
    >
  >
>;
constexpr driver_type driver;

int main()
{
  npygl::vts_stdout_context ctx;
  return driver() ? EXIT_SUCCESS : EXIT_FAILURE;
}
