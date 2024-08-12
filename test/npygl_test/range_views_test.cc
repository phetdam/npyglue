/**
 * @file range_views_test.cc
 * @author Derek Huang
 * @brief range_views.hh unit tests
 * @copyright MIT License
 */

#include "npygl/range_views.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/warnings.h"

namespace {

/**
 * `FlatViewTest` input for a `std::vector<float>`.
 */
struct fvt_input_1 {
  using value_type = float;
  using container_type = std::vector<value_type>;
  using view_type = npygl::flat_view<value_type>;

  container_type input() const
  {
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4305)
    return {4.3, 4.222, 0.1, 12.22, 3.4};
NPYGL_MSVC_WARNING_POP()
  }

  auto operator()(value_type x) const noexcept
  {
    return std::sin(x);
  }

  view_type view(container_type& values) const noexcept
  {
    return {values.data(), values.size()};
  }
};

/**
 * `FlatViewTest` input for a `std::vector<double>`.
 */
struct fvt_input_2 {
  using value_type = double;
  using container_type = std::vector<value_type>;
  using view_type = npygl::flat_view<value_type>;

  container_type input() const
  {
    return {0.045, 0.334, 0.12, 6.555};
  }

  auto operator()(value_type x) const noexcept
  {
    return std::sqrt(x);
  }

  view_type view(container_type& values) const noexcept
  {
    return {values.data(), values.size()};
  }
};

/**
 * Array wrapper template to allow array iteration.
 *
 * @tparam T Element type
 * @tparam N Array size
 */
template <typename T, std::size_t N>
class array_wrapper {
public:
  /**
   * Ctor.
   *
   * By decaying a reference to an array to a pointer the `begin()` and
   * `end()` iterator members can be `const` member functions.
   */
  array_wrapper(T (&ar)[N]) noexcept : data_{ar} {}

  /**
   * Return iterator to beginning of array.
   */
  auto begin() const noexcept
  {
    return data_;
  }

  /**
   * Return iterator to one past end of array.
   */
  auto end() const noexcept
  {
    return data_ + N;
  }

  /**
   * Check if two `array_wrapper` instances are equal.
   *
   * This is to support `EXPECT_EQ` usage in Google Test.
   *
   * @note This is different than `std::array<T, N>::operator==` which only
   *  works on arrays that are of the same compile-time length.
   *
   * @tparam N_ Size of the other `array_wrapper`
   */
  template <std::size_t N_>
  bool operator==(array_wrapper<T, N_> other) const noexcept
  {
    // if different sizes, definitely not equal
    if constexpr (N != N_)
      return false;
    // other perform elementwise check
    else {
      // same size so no need to track two sizes
      for (decltype(N) i = 0; i < N; i++)
        if (*(begin() + i) != *(other.begin() + i))
          return false;
      return true;
    }
  }

private:
  T* data_;
};

/**
 * `FlatViewTest` input for a static unsigned int array wrapper.
 */
struct fvt_input_3 {
  using value_type = unsigned int;
  using view_type = npygl::flat_view<value_type>;

  auto input() const noexcept
  {
    static value_type ar[] = {4U, 1U, 13U, 15U, 888U, 71U, 190U};
    return array_wrapper{ar};
  }

  auto operator()(value_type x) const noexcept
  {
    return 2U * x * x + 4U * x + 5;
  }

  template <std::size_t N>
  view_type view(array_wrapper<value_type, N> ar) const noexcept
  {
    // cast to silence warning about narrowing conversion (signed to unsigned)
    return {ar.begin(), static_cast<decltype(N)>(ar.end() - ar.begin())};
  }
};

/**
 * Test fixture template for `flat_view` testing.
 *
 * @tparam InType Class type with `input()`, `operator()`, and `view()` members
 *  that generate the test input, transform each input value, and create a
 *  `flat_view` from the input respectively.
 */
template <typename InType>
class FlatViewTest : public ::testing::Test {};

TYPED_TEST_SUITE(
  FlatViewTest,
  NPYGL_IDENTITY(::testing::Types<fvt_input_1, fvt_input_2, fvt_input_3>)
);

/**
 * Test that modification through a `flat_view` works as expected.
 */
TYPED_TEST(FlatViewTest, TransformTest)
{
  // create an instance of the input (in case it is stateful)
  TypeParam input{};
  // expected values
  auto expected = input.input();
  std::transform(expected.begin(), expected.end(), expected.begin(), input);
  // actual values using view
  auto actual = input.input();
  auto view = input.view(actual);
  std::transform(view.begin(), view.end(), view.begin(), input);
  // expect equality (we used the exact same inputs)
  EXPECT_EQ(expected, actual);
}

/**
 * `MatrixViewTest` input for a `std::vector<float>` and C data ordering.
 */
struct mvt_input_1 {
  using value_type = float;
  using container_type = std::vector<value_type>;
  using view_type = npygl::matrix_view<value_type>;

  constexpr std::size_t rows() const noexcept { return 3u; }
  constexpr std::size_t cols() const noexcept { return 2u; }

  container_type input() const
  {
    return {3.45f, 4.111f, 34.34f, 1.22f, 4.61f, 9.992f};
  }

  auto operator()(value_type x) const noexcept
  {
    return std::sin(x) * 1.34f;
  }

  view_type view(container_type& values) const noexcept
  {
    assert(values.size() == rows() * cols());
    return {values.data(), rows(), cols()};
  }
};

/**
 * `MatrixViewTest` input for a `std::vector<double>` and Fortran data ordering.
 */
struct mvt_input_2 {
  using value_type = double;
  using container_type = std::vector<value_type>;
  using view_type = npygl::matrix_view<value_type, npygl::element_order::f>;

  constexpr std::size_t rows() const noexcept { return 2u; }
  constexpr std::size_t cols() const noexcept { return 3u; }

  container_type input() const
  {
    return {2.334, 41.22, 1.1112, 0.012, 4.161, 5.445};
  }

  auto operator()(value_type x) const noexcept
  {
    return 0.5 * x * x + 2.4 * x - 1.45;
  }

  view_type view(container_type& values) const noexcept
  {
    assert(values.size() == rows() * cols());
    return {values.data(), rows(), cols()};
  }
};

/**
 * `MatrixViewTest` input for a double array and Fortran data ordering.
 */
struct mvt_input_3 {
  using value_type = double;
  using view_type = npygl::matrix_view<value_type, npygl::element_order::f>;

  constexpr std::size_t rows() const noexcept { return 4u; }
  constexpr std::size_t cols() const noexcept { return 2u; }

  auto input() const noexcept
  {
    static value_type ar[] = {3.44, 1.453, 49.11, 2.33, 1.1, 3.232, 3.4, 5.11};
    return array_wrapper{ar};
  }

  auto operator()(value_type x) const noexcept
  {
    return std::cos(x) * std::sin(x) * 0.943;
  }

  template <std::size_t N>
  view_type view(array_wrapper<value_type, N> ar) const noexcept
  {
    assert(static_cast<std::size_t>(ar.end() - ar.begin()) == rows() * cols());
    return {ar.begin(), rows(), cols()};
  }
};

/**
 * Test fixture template for `matrix_view` testing.
 *
 * @tparam InType Class type with `input()`, `operator()`, and `view()` members
 *  that generate the test input, transform each input value, and create a
 *  `matrix_view` from the input respectively.
 */
template <typename InType>
class MatrixViewTest : public ::testing::Test {};

TYPED_TEST_SUITE(
  MatrixViewTest,
  NPYGL_IDENTITY(::testing::Types<mvt_input_1, mvt_input_2, mvt_input_3>)
);

/**
 * Test that modification through the `matrix_view` as a flat view works.
 */
TYPED_TEST(MatrixViewTest, FlatTransformTest)
{
  // input instance (in case of statefulness)
  TypeParam input{};
  // expected values
  auto expected = input.input();
  for (auto& v : expected)
    v = input(v);
  // actual values using view (but treating as flat view)
  auto actual = input.input();
  auto view = input.view(actual);
  for (auto& v : view)
    v = input(v);
  // expect equality since inputs are identical
  EXPECT_EQ(expected, actual);
}

/**
 * Test that modification through the `matrix_view` as a matrix works.
 */
TYPED_TEST(MatrixViewTest, MatrixTransformTest)
{
  // input instance (in case of statefulness)
  TypeParam input{};
  // expected values
  auto expected = input.input();
  for (auto& v : expected)
    v = input(v);
  // actual values using view (but treating as matrix view)
  auto actual = input.input();
  auto view = input.view(actual);
  for (decltype(view.rows()) i = 0; i < view.rows(); i++)
    for (decltype(view.cols()) j = 0; j < view.cols(); j++)
      view(i, j) = input(view(i, j));
  // expect equality since inputs are identical
  EXPECT_EQ(expected, actual);
}

/**
 * Test that C and Fortran data layout index as expected.
 */
TYPED_TEST(MatrixViewTest, IndexingTest)
{
  // input instance (in case of statefulness) + view type
  TypeParam input{};
  using view_type = typename TypeParam::view_type;
  // values + view
  auto values = input.input();
  auto view = input.view(values);
  // test only makes sense if both rows and columns are > 1
  ASSERT_GE(view.rows(), 1u) << "view must have more than one row";
  ASSERT_GE(view.cols(), 1u) << "view must have more than one col";
  // C ordering check (check addresses)
  if constexpr (view_type::data_order == npygl::element_order::c)
    EXPECT_EQ(&view[view.cols() - 1], &view(0u, view.cols() - 1)) <<
      "flat indexing to " << view.cols() - 1 << " != C order [0, " <<
      view.cols() - 1 << "]";
  // Fortran ordering check
  else
    EXPECT_EQ(&view[view.rows() - 1], &view(view.rows() - 1, 0u)) <<
      "flat indexing to " << view.rows() - 1 << " != Fortran order [" <<
      view.rows() - 1 << ", 0]";
}

}  // namespace
