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

namespace npygl {

/**
 * Interface for a view of a flat range.
 *
 * @tparam ViewType View class type
 */
template <typename ViewType>
class flat_view_interface {
public:
  /**
   * Return pointer to the first element in the view.
   */
  auto data() const noexcept
  {
    return static_cast<ViewType*>(this)->data();
  }

  /**
   * Return number of elements in the view.
   */
  auto size() const noexcept
  {
    return static_cast<ViewType*>(this)->size();
  }

  /**
   * Return reference to the `i`th data element without bounds checking.
   */
  auto& operator[](std::size_t i) const noexcept
  {
    return static_cast<ViewType*>(this)->operator[](i);
  }

  /**
   * Return reference to the `i`th data element without bounds checking.
   *
   * @note This is provided to have consistent feel with non-flat views.
   */
  auto& operator()(std::size_t i) const noexcept
  {
    return static_cast<ViewType*>(this)->operator[](i);
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

  /**
   * Return reference to the `i`th data element without bounds checking.
   *
   * @note This is provided to have consistent feel with non-flat views.
   */
  auto& operator()(std::size_t i) const noexcept
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
   * Ctor.
   *
   * @note Number of view elements must be at least `rows` * `cols`.
   *
   * @param data Pointer to first element in view
   * @param rows Number of rows in the view
   * @param cols Number of columns in the view
   */

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
   * Return reference to the `i`th data element without bounds checking.
   *
   * @note This is provided to have consistent feel with non-flat views.
   */
  auto& operator()(std::size_t i) const noexcept
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
    return indexer<T, R>{this}(i, j);
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
   * @note We need the extra `T_` parameter since explicit specializations must
   *  be done at the namespace scope (and so cannot be done in nested scope).
   *
   * @tparam T_ Element type
   * @tparam O Element ordering (C-style)
   */
  template <typename T_, element_order O>
  struct indexer {
    /**
     * Return row-major reference to the view's `(i, j)` data element.
     */
    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return view->data()[view->cols() * i + j];
    }

    // pointer to parent view
    const matrix_view<T_>* view;
  };

  /**
   * Specialization for Fortran-style ordering.
   *
   * @tparam T_ Element type
   */
  template <typename T_>
  struct indexer<T_, element_order::f> {
    /**
     * Return column-major reference to the view's `(i, j)` data element.
     */
    auto operator()(std::size_t i, std::size_t j) const noexcept
    {
      return view->data()[i + view->rows() * j];
    }

    // pointer to parent view
    const matrix_view<T_>* view;
  };
#endif  // !NPYGL_HAS_CC_17
};

}  // namespace npygl

#endif  // NPYGL_RANGE_VIEWS_HH_
