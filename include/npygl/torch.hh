/**
 * @file torch.hh
 * @author Derek Huang
 * @brief C++ header for LibTorch helpers
 * @copyright MIT License
 */

#ifndef NPYGL_TORCH_HH_
#define NPYGL_TORCH_HH_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <torch/torch.h>

#include "npygl/features.h"

// enable Armadillo LibTorch helpers if desired
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
// enable Eigen3 LibTorch helpers if desired
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

namespace npygl {

/**
 * Create a 1D PyTorch tensor from a `std::vector<T, A>`.
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
  using Vec = std::remove_reference_t<decltype(vec)>;
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(Vec));
  auto buf_vec = new(buf.get()) Vec{std::move(vec)};
  // create new 1D tensor
  auto ten = torch::from_blob(
    // data + shape (cast required to silence warning)
    buf_vec->data(),
    {static_cast<std::int64_t>(buf_vec->size())},
    // deleter calls ~V() explicitly
    [buf_vec](void*)
    {
      // buffer is deleted on scope exit
      std::unique_ptr<unsigned char[]> buf{(unsigned char*) buf_vec};
      buf_vec->~Vec();
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
 * @tparam CMax Max number of columns
 *
 * @param mat Dense matrix to consume
 * @param opts Tensor creation options
 */
template <typename T, int R, int C, int O, int RMax, int CMax>
auto make_tensor(
  Eigen::Matrix<T, R, C, O, RMax, CMax>&& mat,
  const torch::TensorOptions& opts = {})
{
  using Mat = std::remove_reference_t<decltype(mat)>;
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(Mat));
  auto buf_mat = new(buf.get()) Mat{std::move(mat)};
  // create new 2D tensor
  auto ten = torch::from_blob(
    // data + shape
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
      buf_mat->~Mat();
    },
    // use c10 traits specializations to map C++ type to tensor type value
    opts.dtype(torch::CppTypeToScalarType<T>::value)
  );
  // again, release buf in case of exception throw
  buf.release();
  return ten;
}
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
/**
 * Create a 2D PyTorch tensor from an Armadillo matrix.
 *
 * @note `arma::Row<T>` or `arma::Col<T>` objects will be sliced on the
 *  placement new. Although they do not provide any extra members or virtual
 *  functions compared to the standard `arma::Mat<T>`, in general you probably
 *  don't want to slice an object on placement new.
 *
 * @todo Provide `arma::Row<T>` overloads for PyTorch row tensor.
 *
 * @tparam T Element type
 *
 * @param mat Dense matrix to consume
 * @param opts Tensor creation options
 */
template <typename T>
auto make_tensor(arma::Mat<T>&& mat, const torch::TensorOptions& opts = {})
{
  using Mat = std::remove_reference_t<decltype(mat)>;
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(Mat));
  auto buf_mat = new(buf.get()) Mat{std::move(mat)};
  // create new 2D tensor
  auto ten = torch::from_blob(
    // data + shape (cast required to silence narrowing warnings)
    buf_mat->memptr(),
    {
      static_cast<std::int64_t>(buf_mat->n_rows),
      static_cast<std::int64_t>(buf_mat->n_cols)
    },
    // deleter
    [buf_mat](void*)
    {
      // buffer deleted on scope exit
      std::unique_ptr<unsigned char[]> buf{(unsigned char*) buf_mat};
      buf_mat->~Mat();
    },
    // use c10 traits specializations to map C++ type to tensor type value
    opts.dtype(torch::CppTypeToScalarType<T>::value)
  );
  // again, release buf in case of exception throw
  buf.release();
  return ten;
}
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)

}  // namespace npygl

#endif  // NPYGL_TORCH_HH_
