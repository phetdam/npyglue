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

namespace {

/**
 * `FlatViewTest` input for a `std::vector<float>`.
 */
struct flt_input_1 {
  std::vector<float> input() const
  {
    return {4.3, 4.222, 0.1, 12.22, 3.4};
  }

  auto operator()(float x) const noexcept
  {
    return std::sin(x);
  }

  npygl::flat_view<float> view(std::vector<float>& values) const noexcept
  {
    return {values.data(), values.size()};
  }
};

/**
 * `FlatViewTest` input for a `std::vector<double>`.
 */
struct flt_input_2 {
  std::vector<double> input() const
  {
    return {0.045, 0.334, 0.12, 6.555};
  }

  auto operator()(double x) const noexcept
  {
    return std::sqrt(x);
  }

  npygl::flat_view<double> view(std::vector<double>& values) const noexcept
  {
    return {values.data(), values.size()};
  }
};

/**
 * `FlatViewTest` input for a static double array wrapper.
 */
struct flt_input_3 {
  /**
   * Nested array wrapper template to allow array iteration.
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

  auto input() const
  {
    static double ar[] = {4.33, 2.344, 1.5151, 8.9};
    return array_wrapper{ar};
  }

  auto operator()(double x) const noexcept
  {
    return 0.5 * x * x + 4 * x + 5;
  }

  template <std::size_t N>
  npygl::flat_view<double> view(array_wrapper<double, N> ar) const noexcept
  {
    // silence warning about narrowing conversion
    return {ar.begin(), static_cast<std::size_t>(ar.end() - ar.begin())};
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

using FlatViewTestTypes = ::testing::Types<flt_input_1, flt_input_2, flt_input_3>;
TYPED_TEST_SUITE(FlatViewTest, FlatViewTestTypes);

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
