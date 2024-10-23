/**
 * @file torch.hh
 * @author Derek Huang
 * @brief C++ header for LibTorch helpers
 * @copyright MIT License
 */

#ifndef NPYGL_TORCH_HH_
#define NPYGL_TORCH_HH_

#include <memory>
#include <type_traits>
#include <vector>

#include <torch/torch.h>

#include "npygl/features.h"

// enable Eigen3 LibTorch helpers if desired
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

namespace npygl {

/**
 * Create a 1D PyTorch tensor from a `std::vector<T>`.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 *
 * @param vec Vector to consume
 * @param opts Tensor creation options
 */
template <typename T, typename A>
auto make_tensor(std::vector<T, A>&& vec, const torch::TensorOptions& opts = {})
{
  using V = std::remove_reference_t<decltype(vec)>;
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(V));
  auto buf_vec = new(buf.get()) V{std::move(vec)};
  // create new 1D tensor
  auto ten = torch::from_blob(
    // data and shape
    buf_vec->data(),
    {buf_vec->size()},
    // deleter calls ~V() explicitly
    [buf_vec](void*)
    {
      // buffer is deleted on scope exit
      std::unique_ptr<unsigned char[]> buf{(unsigned char*) buf_vec};
      buf_vec->~V();
    },
    // use c10 traits specializations to map C++ type to tensor type value
    opts.dtype(torch::CppTypeToScalarType<T>::value)
  );
  // release after from_blob in case exception is thrown
  buf.release();
  return ten;
}

#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
/**
 * Create a 2D PyTorch tensor from a `Eigen::Matrix`.
 *
 * @tparam T Element type
 * @tparam R Number of compile-time rows
 * @tparam C Number of compile-time columns
 * @tparam O Matrix options
 * @tparam RMax Max number of rows
 * @tparam CMarx Max number of columns
 *
 * @param mat Dense matrix to consume
 * @param opts Tensor creation options
 */
template <typename T, int R, int C, int O, int RMax, int CMax>
auto make_tensor(
  Eigen::Matrix<T, R, C, O, RMax, CMax>&& mat,
  const torch::TensorOptions& opts = {})
{
  using M = std::remove_reference_t<decltype(mat)>;
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(M));
  auto buf_mat = new(buf.get()) M{std::move(mat)};
  // create new 2D tensor
  auto ten = torch::from_blob(
    // data and shape
    buf_mat->data(),
    {buf_mat->rows(), buf_mat->cols()},
    // column-major by default but may be row-major
    {
      [buf_mat]() -> Eigen::Index
      {
        if constexpr (O & Eigen::StorageOptions::RowMajor)
          return buf_mat->cols();
        else
          return 1;
      }(),
      [buf_mat]() -> Eigen::Index
      {
        if constexpr (O & Eigen::StorageOptions::RowMajor)
          return 1;
        else
          return buf_mat->rows();
      }()
    },
    // deleter
    [buf_mat](void*)
    {
      // buffer deleted on scope exit
      std::unique_ptr<unsigned char[]> buf{(unsigned char*) buf_mat};
      buf_mat->~M();
    }
  );
  // again, release buf in case of exception throw
  buf.release();
  return ten;
}
#endif  // NPYGL_HAS_EIGEN3

}  // namespace npygl

#endif  // NPYGL_TORCH_HH_
