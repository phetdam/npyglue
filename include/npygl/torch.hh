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
#include "npygl/warnings.h"

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
 * The shape and strides member functions *must* return lvalues as they are
 * converted to `at::IntArrayRef` *views*. If a prvalue is returned one will
 * get a dangling reference after conversion! Also, the data pointer is assumed
 * to live until the dtor of the `T*` parent object is called.
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
 *  a valid `CppTypeToScalarType<U>` specialization where `U` is the type
 *  yielded by `std::remove_cv_t(decltype(*data()))`.
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
 * @note `shape()` must be an lvalue implicitly convertible to `IntArrayRef`.
 */
template <typename T>
struct is_tensor_info_context_with_shape<
  tensor_info_context<T>,
  std::enable_if_t<
    std::is_lvalue_reference_v<tensor_info_context_shape_t<T>> &&
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
 * @note `strides()` must be an lvalue implicitly convertible to `IntArrayRef`.
 *
 * @tparam T Parent type
 */
template <typename T>
struct is_tensor_info_context_with_strides<
  tensor_info_context<T>,
  std::enable_if_t<
    std::is_lvalue_reference_v<tensor_info_context_strides_t<T>> &&
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
// braces no longer necessary C++14 and later but MSVC still complains with
// C5246 so disable. see https://stackoverflow.com/a/70127399/14227825
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(5246)
      // cast required to silence narrowing warnings
      shape_{static_cast<std::int64_t>(parent->size())},
      strides_{1}
NPYGL_MSVC_WARNING_POP()
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

#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
/**
 * `tensor_info_context<T>` specialization for an Eigen matrix.
 *
 * @tparam T Element type
 * @tparam R Number of compile-time rows
 * @tparam C Number of compile-time columns
 * @tparam O Matrix options
 * @tparam RMax Max number of rows
 * @tparam CMax Max number of columns
 */
template <typename T, int R, int C, int O, int RMax, int CMax>
class tensor_info_context<Eigen::Matrix<T, R, C, O, RMax, CMax>> {
public:
  using parent_type = Eigen::Matrix<T, R, C, O, RMax, CMax>;

  /**
   * Ctor.
   *
   * @param parent Parent object
   */
  tensor_info_context(parent_type* parent) noexcept
    : parent_{parent},
// disable C5246 warning about aggregate init requiring more braces
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(5246)
      shape_{parent_->rows(), parent_->cols()},
// disable C4355 complaint about using this
NPYGL_MSVC_WARNING_DISABLE(4355)
      // column-major by default but may be row-major
      strides_{
        [this]() -> Eigen::Index
        {
          if constexpr (O & Eigen::StorageOptions::RowMajor)
            return parent_->cols();
          else
            return 1;
        }(),
        [this]() -> Eigen::Index
        {
          if constexpr (O & Eigen::StorageOptions::RowMajor)
            return 1;
          else
            return parent_->rows();
        }()
      }
NPYGL_MSVC_WARNING_POP()
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
  std::array<std::int64_t, 2> shape_;
  std::array<std::int64_t, 2> strides_;
};
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
/**
 * `tensor_info_context<T>` specialization for an Armadillo matrix.
 *
 * This also handles `arma::Col<T>` which derives from `arma::Mat<T>`.
 *
 * @note Object slicing will be done on placement new but the binary layout of
 *  the `arma::Col<T>` and `arma::Mat<T>` are the same.
 *
 * @tparam T Element type
 */
template <typename T>
class tensor_info_context<arma::Mat<T>> {
public:
  using parent_type = arma::Mat<T>;

  /**
   * Ctor.
   *
   * @param parent Parent object
   */
  tensor_info_context(parent_type* parent) noexcept
    : parent_{parent},
// disable C5246 warning about aggregate init requiring more braces
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(5246)
      // cast required to silence narrowing warnings
      shape_{
        static_cast<std::int64_t>(parent_->n_rows),
        static_cast<std::int64_t>(parent_->n_cols)
      },
      // Armadillo matrices are in column-major order (leading varies fastest)
      strides_{1, shape_[0]}
NPYGL_MSVC_WARNING_POP()
  {}

  /**
   * Return pointer to the data buffer.
   */
  auto data() const noexcept
  {
    return parent_->memptr();
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
  std::array<std::int64_t, 2> shape_;
  std::array<std::int64_t, 2> strides_;
};

/**
 * `tensor_info_context<T>` specialization for an Armadillo column vector.
 *
 * @tparam T Element type
 */
template <typename T>
class tensor_info_context<arma::Col<T>>
  : public tensor_info_context<arma::Mat<T>> {
public:
  using parent_type = arma::Col<T>;
  using base_type = tensor_info_context<arma::Mat<T>>;

   /**
   * Ctor.
   *
   * This delegates to the base class ctor.
   *
   * @param parent Parent object
   */
  tensor_info_context(parent_type* parent) noexcept : base_type{parent} {}
};

/**
 * `tensor_info_context<T>` specialization for an Armadillo row vector.
 *
 * @tparam T Element type
 */
template <typename T>
class tensor_info_context<arma::Row<T>> {
public:
  using parent_type = arma::Row<T>;

  /**
   * Ctor.
   *
   * @param parent Parent object
   */
  tensor_info_context(parent_type* parent) noexcept
    : parent_{parent},
// disable C5246 warning about aggregate init requiring more braces
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(5246)
      // cast required to silence narrowing warnings
      shape_{static_cast<std::int64_t>(parent_->n_cols)},
      strides_{1}
NPYGL_MSVC_WARNING_POP()
  {}

  /**
   * Return pointer to the data buffer.
   */
  auto data() const noexcept
  {
    return parent_->memptr();
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
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)

/**
 * Constraint for `make_tensor` for a given type.
 *
 * This evaluates to `true` for a non-reference type that has a valid
 * `tensor_info_context<T>` specialization.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool convertible_to_tensor = (
  !std::is_reference_v<T> &&
  is_valid_tensor_info_context_v<tensor_info_context<T>>
);

#if NPYGL_HAS_CC_20
/**
 * Concept for a C++ type that `make_tensor` can convert to a PyTorch tensor.
 *
 * This evaluates to `true` for a non-reference type that has a valid
 * `tensor_info_context<T>` specialization.
 *
 * @tparam T type
 */
template <typename T>
concept tensor_convertible = convertible_to_tensor<T>;
#endif  // !NPYGL_HAS_CC_20

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
template <typename T, typename = std::enable_if_t<convertible_to_tensor<T>>>
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

}  // namespace npygl

#endif  // NPYGL_TORCH_HH_
