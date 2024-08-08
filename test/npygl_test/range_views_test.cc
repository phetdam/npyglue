/**
 * @file range_views_test.cc
 * @author Derek Huang
 * @brief range_views.hh unit tests
 * @copyright MIT License
 */

#include "npygl/range_views.hh"

#include <cmath>
#include <cstdint>
#include <functional>

#include <gtest/gtest.h>

#include "npygl/common.h"
#include "npygl/warnings.h"

namespace {

/**
 * `FlatViewTest` input for a `std::vector<float>`.
 */
struct flt_input_1 {
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
struct flt_input_2 {
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
  bool operator==(const array_wrapper<T, N_>& other) const noexcept
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
struct flt_input_3 {
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
  NPYGL_IDENTITY(::testing::Types<flt_input_1, flt_input_2, flt_input_3>)
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
  for (auto& v : expected)
    v = input(v);
  // actual values using view
  auto actual = input.input();
  auto view = input.view(actual);
  for (auto& v : view)
    v = input(v);
  // expect equality (we used the exact same inputs)
  EXPECT_EQ(expected, actual);
}

}  // namespace
