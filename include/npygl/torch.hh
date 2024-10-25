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
#include <utility>
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
 * Context object providing information for `torch::from_blob`.
 *
 * Any explicit or partial specialization needs to provide the following:
 *
 * 1. Ability to construct from a `T*` (usually from placement new buffer)
 * 2. `data()` member to yield a typed pointer to the data buffer
 * 3. `shape()` member to yield tensor shape (convertible to `IntArrayRef`)
 * 4. `strides()` member to yield tensor strides (optional, default `shape()`)
 *
 * @tparam T type
 */
template <typename T>
class tensor_info_context {};

/**
 * Indicate that a type is a `tensor_info_context<T>` specialization.
 *
 * @note The `tensor_info_context<T>` may not have a valid specialization, e.g.
 *  it is just the empty struct, but this will still evaluate true.
 *
 * @tparam T type
 */
template <typename T>
struct is_tensor_info_context : std::false_type {};

/**
 * True specialization for a `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context<tensor_info_context<T>> : std::true_type {};

/**
 * Indicate that the `tensor_info_context<T>` is `T*` constructible.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_tensor_info_context_constructible : std::false_type {};

/**
 * True specialization for a `T*` constructible `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context_constructible<
  tensor_info_context<T>,
  std::void_t<decltype(tensor_info_context<T>{static_cast<T*>(nullptr)})>
> : std::true_type {};

/**
 * Type alias for the `data()` pointer type.
 *
 * Can be used for SFINAE as it is defined for valid `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
using tensor_info_context_data_t =
  decltype(std::declval<tensor_info_context<T>>().data());

/**
 * Type alias for the `*data()` cv-unqualified value type.
 *
 * Can be used for SFINAE as it is defined for valid `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
using tensor_info_context_data_value_t = std::remove_cv_t<
  std::remove_pointer_t<tensor_info_context_data_t<T>> >;

/**
 * Indicate that the `tensor_info_context<T>` has a valid `data()` member.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_tensor_info_context_with_data : std::false_type {};

/**
 * True specialization for a `tensor_info_context<T>` with a `data()` member.
 *
 * @note `data()` must be implicitly convertible to `void*` and must also have
 *  a valid `CppTypeToScalarType<decltype(data())>` specialization
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context_with_data<
  tensor_info_context<T>,
  std::enable_if_t<
    std::is_convertible_v<tensor_info_context_data_t<T>, void*> &&
    std::is_same_v<
      c10::ScalarType,
      // note: need to remove cv-qualifiers
      std::remove_cv_t<
        decltype(
          torch::CppTypeToScalarType<tensor_info_context_data_value_t<T>>::value)
      >
    >
  > > : std::true_type {};

/**
 * Type alias for the `shape()` cv-unqualified element type.
 *
 * Can be used for SFINAE as it is defined for valid `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
using tensor_info_context_shape_t = std::remove_cv_t<
  decltype(std::declval<tensor_info_context<T>>().shape())>;

/**
 * Indicate that the `tensor_info_context<T>` has a valid `shape()` member.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_tensor_info_context_with_shape : std::false_type {};

/**
 * True specialization of a `tensor_info_context<T>` with a `shape()` member.
 *
 * @note `shape()` must be implicitly convertible to `IntArrayRef`.
 */
template <typename T>
struct is_tensor_info_context_with_shape<
  tensor_info_context<T>,
  std::enable_if_t<
    std::is_convertible_v<tensor_info_context_shape_t<T>, at::IntArrayRef>>
> : std::true_type {};

/**
 * Type alias for the `strides()` cv-unqualified element type.
 *
 * Can be used for SFINAE as it is defined for valid `tensor_info_context<T>`.
 *
 * @tparam T Parent type
 */
template <typename T>
using tensor_info_context_strides_t = std::remove_cv_t<
  decltype(std::declval<tensor_info_context<T>>().strides())>;

/**
 * Indicate that the `tensor_info_context<T>` has a valid `strides()` member.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_tensor_info_context_with_strides : std::false_type {};

/**
 * True specialization of a `tensor_info_context<T>` with a `strides()` member.
 *
 * @note `strides()` must be implicitly convertible to `IntArrayRef`.
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context_with_strides<
  tensor_info_context<T>,
  std::enable_if_t<
    std::is_convertible_v<tensor_info_context_strides_t<T>, at::IntArrayRef>
  > > : std::true_type {};

namespace detail {

/**
 * Indicate that a type has a `strides()` member.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct has_strides : std::false_type {};

/**
 * True specialization for types with a `strides()` member.
 *
 * @tparam T type
 */
template <typename T>
struct has_strides<T, std::void_t<decltype(std::declval<T>().strides())>>
  : std::true_type {};

}  // namespace detail

/**
 * Indicate that the `tensor_info_context<T>` uses default strides.
 *
 * This indicates that `shape()` is used to provide strides as well.
 *
 * @tparam T type
 */
template <typename T>
struct is_tensor_info_context_with_default_strides : std::false_type {};

/**
 * True specialization of a `tensor_info_context<T>` using default strides.
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context_with_default_strides<tensor_info_context<T>>
  : std::bool_constant<
      is_tensor_info_context_with_shape<tensor_info_context<T>>::value &&
      !detail::has_strides<tensor_info_context<T>>::value
    > {};

/**
 * Indicate that a type is a valid `tensor_info_context<T>` specialization.
 *
 * This can be used to check if a type if a`tensor_info_context<T>` and also
 * satisfies the interface requirements imposed.
 *
 * @tparam T type
 */
template <typename T>
struct is_valid_tensor_info_context : std::bool_constant<
  is_tensor_info_context_constructible<T>::value &&
  is_tensor_info_context_with_data<T>::value &&
  is_tensor_info_context_with_shape<T>::value &&
  (
    is_tensor_info_context_with_default_strides<T>::value ||
    is_tensor_info_context_with_strides<T>::value
  )
> {};

/**
 * Indicate that a type is a valid `tensor_info_context<T>` specialization.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool
is_valid_tensor_info_context_v = is_valid_tensor_info_context<T>::value;

/**
 * `tensor_info_context<T>` specialization for `std::vector<T, A>`.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 */
template <typename T, typename A>
class tensor_info_context<std::vector<T, A>> {
public:
  using parent_type = std::vector<T, A>;

  /**
   * Ctor.
   *
   * @param parent Parent object
   */
  tensor_info_context(parent_type* parent) noexcept
    : parent_{parent},
      // note: cast is needed to appease compiler
      // note: braces no longer necessary C++14 and later but MSVC still
      // complains with C5246. see https://stackoverflow.com/a/70127399/14227825
      shape_{{static_cast<std::int64_t>(parent->size())}},
      strides_{{1}}
  {}

  /**
   * Return pointer to the data buffer.
   */
  auto data() const noexcept
  {
    return parent_->data();
  }

  /**
   * Return the desired tensor shape.
   */
  const auto& shape() const noexcept
  {
    return shape_;
  }

  /**
   * Return the tensor strides.
   */
  const auto& strides() const noexcept
  {
    return strides_;
  }

private:
  parent_type* parent_;
  std::array<std::int64_t, 1> shape_;
  std::array<std::int64_t, 1> strides_;
};

namespace experimental {

/**
 * Create a PyTorch tensor from a C++ object.
 *
 * This function can create a PyTorch tensor from any appropriate C++ object
 * that exposes a dense coefficient buffer and has a `tensor_info_context<T>`
 * such that `is_valid_tensor_info_context_v<tensor_info_context<T>>::value` is
 * `true`, i.e. the `tensor_info_context<T>` is valid.
 *
 * @tparam T C++ type with a valid `tensor_info_context<T>` specialization
 *
 * @param obj C++ object to consume
 * @param opts Tensor creation options
 */
template <
  typename T,
  typename = std::enable_if_t<
    !std::is_reference_v<T> &&
    is_valid_tensor_info_context_v<tensor_info_context<T>>> >
auto make_tensor(T&& obj, const torch::TensorOptions& opts = {})
{
  // placement new into buffer
  auto buf = std::make_unique<unsigned char[]>(sizeof(T));
  auto buf_o = new(buf.get()) T{std::move(obj)};
  // get tensor info context
  tensor_info_context<T> info{buf_o};
  // create new tensor
  auto ten = torch::from_blob(
    // data + shape
    info.data(),
    info.shape(),
    // TODO: enable support for default strides, e.g. std::array<int64_t, 1>
    // results in strides of {1}, std::array<int64_t, 2> results in strides of
    // {shape[1], 1} (assume row-major), and so on
    info.strides(),
    // deleter calls ~V() explicitly
    [buf_o](void*)
    {
      // buffer is deleted on scope exit
      std::unique_ptr<unsigned char[]> buf{(unsigned char*) buf_o};
      buf_o->~T();
    },
    // use c10 traits specializations to map C++ type to tensor type value
    opts.dtype(
      torch::CppTypeToScalarType<tensor_info_context_data_value_t<T>>::value)
  );
  // release after from_blob in case exception is thrown
  buf.release();
  return ten;
}

}  // namespace experimental

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
