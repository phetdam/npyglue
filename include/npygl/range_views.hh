/**
 * @file range_views.hh
 * @author Derek Huang
 * @brief C++ header for range view helpers
 * @copyright MIT License
 */

#ifndef NPYGL_RANGE_VIEWS_HH_
#define NPYGL_RANGE_VIEWS_HH_

#include <cstdio>

#include "npygl/features.h"

#if !NPYGL_HAS_CC_17
#include <type_traits>
#endif  // !NPYGL_HAS_CC_17

namespace npygl {

/**
 * Interface for a view of a flat range.
 *
 * @tparam ViewType View class type
 */
template <typename ViewType>
class flat_view_interface {
public:
  /////////////////////////////////////////////////////////////////////////////
  // interface members
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Return pointer to the first element in the view.
   */
  auto data() const noexcept
  {
    return static_cast<const ViewType*>(this)->data();
  }

  /**
   * Return number of elements in the view.
   */
  auto size() const noexcept
  {
    return static_cast<const ViewType*>(this)->size();
  }

  /**
   * Return reference to the `i`th data element without bounds checking.
   */
  auto& operator[](std::size_t i) const noexcept
  {
    return static_cast<const ViewType*>(this)->operator[](i);
  }

  /////////////////////////////////////////////////////////////////////////////
  // inherited members
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Return pointer to the first element in the view.
   */
  auto begin() const noexcept
  {
    return static_cast<const ViewType*>(this)->data();
  }

  /**
   * Return pointer to one past the last element in the view.
   */
  auto end() const noexcept
  {
    using V = const ViewType;
    return static_cast<V*>(this)->data() + static_cast<V*>(this)->size();
  }

  /**
   * Indicate if view contains data or not.
   */
  operator bool() const noexcept
  {
    return static_cast<const ViewType*>(this)->data();
  }

  /**
   * Return reference to the `i`th data element without bounds checking.
   *
   * @note This is provided to have consistent feel with non-flat views.
   */
  auto& operator()(std::size_t i) const noexcept
  {
    return static_cast<const ViewType*>(this)->operator[](i);
  }
};

/**
 * Lightweight view of a flat range.
 *
 * @tparam T Data type
 */
template <typename T>
class flat_view : public flat_view_interface<flat_view<T>> {
public:
  using value_type = T;

  /**
   * Default ctor.
   *
   * Constructs a view with no data.
   */
  flat_view() noexcept : flat_view{nullptr, 0U} {}

  /**
   * Ctor.
   *
   * @param data Pointer to first element in view
   * @param size Number of elements in view
   */
  flat_view(T* data, std::size_t size) noexcept : data_{data}, size_{size} {}

  /**
   * Return pointer to first element in view.
   */
  auto data() const noexcept { return data_; }

  /**
   * Return number of elements in the view.
   */
  auto size() const noexcept { return size_; }

  /**
   * Return reference to the `i`th data element without bounds checking.
   */
  auto& operator[](std::size_t i) const noexcept
  {
    return data_[i];
  }

private:
  T* data_;
  std::size_t size_;
};

/**
 * 2D array element ordering.
 *
 * `c` is C-style row major order, `f` is Fortran-style column-major order.
 */
enum class element_order { c, f };

/**
 * Lightweight view of a range treated as a matrix.
 *
 * @tparam T Data type
 * @tparam R Element ordering
 */
template <typename T, element_order R = element_order::c>
class matrix_view : public flat_view_interface<matrix_view<T>> {
public:
  using value_type = T;
  static constexpr auto data_order = R;

  /**
   * Default ctor.
   *
   * Constructs a view with no data.
   */
  matrix_view() noexcept : matrix_view{nullptr, 0U, 0U} {}

  /**
   * Ctor.
   *
   * @note Number of view elements must be at least `rows` * `cols`.
   *
   * @param data Pointer to first element in view
   * @param rows Number of rows in the view
   * @param cols Number of columns in the view
   */
  matrix_view(T* data, std::size_t rows, std::size_t cols) noexcept
    : data_{data}, rows_{rows}, cols_{cols}
  {}

  /**
   * Return pointer to first element in view.
   */
  auto data() const noexcept { return data_; }

  /**
   * Return number of rows in view.
   */
  auto rows() const noexcept { return rows_; }

  /**
   * Return number of columns in view.
   */
  auto cols() const noexcept { return cols_; }

  /**
   * Return number of elements in the view.
   */
  auto size() const noexcept { return rows_ * cols_; }

  /**
   * Return reference to the `i`th data element without bounds checking.
   */
  auto& operator[](std::size_t i) const noexcept
  {
    return data_[i];
  }

  /**
   * Return reference to the `(i, j)` data element without bounds checking.
   */
  auto& operator()(std::size_t i, std::size_t j) const noexcept
  {
#if NPYGL_HAS_CC_17
    if constexpr (R == element_order::c)
      return data_[cols_ * i + j];
    else
      return data_[i + rows_ * j];
#else
    return indexer<R>{this}(i, j);
#endif  // !NPYGL_HAS_CC17
  }

private:
  T* data_;
  std::size_t rows_;
  std::size_t cols_;

#if !NPYGL_HAS_CC_17
  /**
   * Indexer struct to handle different data layouts.
   *
   * @tparam O Element ordering
   */
  template <element_order O, typename = void>
  struct indexer {
    /**
     * Return row-major reference to the view's `(i, j)` data element.
     */
    auto& operator()(std::size_t i, std::size_t j) const noexcept
    {
      return view->data()[view->cols() * i + j];
    }

    // pointer to parent view
    const matrix_view<T, O>* view;
  };

  /**
   * Specialization for Fortran-style ordering.
   *
   * @tparam O Element ordering
   */
  template <element_order O>
  struct indexer<O, std::enable_if_t<O == element_order::f>> {
    /**
     * Return column-major reference to the view's `(i, j)` data element.
     */
    auto& operator()(std::size_t i, std::size_t j) const noexcept
    {
      return view->data()[i + view->rows() * j];
    }

    // pointer to parent view
    const matrix_view<T, O>* view;
  };
#endif  // !NPYGL_HAS_CC_17
};

}  // namespace npygl

#endif  // NPYGL_RANGE_VIEWS_HH_
