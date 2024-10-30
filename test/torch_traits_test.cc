/**
 * @file torch_traits_test.cc
 * @author Derek Huang
 * @brief C++ program testing PyTorch-related type traits
 * @copyright MIT License
 */

#include <cstdlib>
#include <deque>
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
  >
  // TODO: test other tensor_info_context traits
>;
constexpr driver_type driver;

int main()
{
  npygl::vts_stdout_context ctx;
  return driver() ? EXIT_SUCCESS : EXIT_FAILURE;
}
