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
struct tensor_info_context<std::deque<T>> {
  /**
   * Ctor.
   *
   * The pointer type is incorrect.
   */
  tensor_info_context(T** /*unused*/) noexcept {}

  /**
   * Data pointer is not convertible into a Torch type value.
   */
  std::deque<T>* data() const noexcept
  {
    return nullptr;
  }

  /**
   * Shape is returned as prvalue instead of lvalue reference.
   */
  std::vector<std::int64_t> shape() const
  {
    return {8, 4};
  }

  /**
   * Strides are returned as prvalue instead of lvalue reference.
   */
  std::vector<std::int64_t> strides() const
  {
    return {1, 8};
  }
};

}  // namespace npygl

namespace {

/**
 * Shorthand for the invalid `tensor_info_context<std::deque<T>>`.
 *
 * @tparam T Element type
 */
template <typename T>
using bad_tensor_info_context = npygl::tensor_info_context<std::deque<T>>;

}  // namespace

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
      std::pair<bad_tensor_info_context<int>, std::false_type>
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
      npygl::tensor_info_context<std::pmr::vector<c10::complex<c10::Half>>>,
      std::pair<bad_tensor_info_context<double>, std::false_type>
    >
  >,
  // is_tensor_context_with_shape
  npygl::testing::traits_checker<
    npygl::is_tensor_info_context_with_shape,
    std::tuple<
      std::pair<npygl::tensor_info_context<void*>, std::false_type>,
      npygl::tensor_info_context<std::vector<double>>,
      std::pair<bad_tensor_info_context<unsigned>, std::false_type>,
      // note: although the std::vector<std::complex<T>> partial specialization
      // is overall invalid the resultant shape() member is still valid
      npygl::tensor_info_context<std::vector<std::complex<double>>>,
      // as usual PyTorch C++ types are ok
      npygl::tensor_info_context<std::vector<c10::complex<double>>>
    >
  >
>;
constexpr driver_type driver;

int main()
{
  npygl::vts_stdout_context ctx;
  return driver() ? EXIT_SUCCESS : EXIT_FAILURE;
}
