/**
 * @file ostream_test.cc
 * @author Derek Huang
 * @brief C++ program to test ostream.hh
 * @copyright MIT License
 */

#include <cstdlib>
#include <map>
#include <ostream>
#include <set>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "npygl/ostream.hh"
#include "npygl/testing/ostream.hh"

namespace {

/**
 * Type wrapper.
 *
 * @tparam T type
 */
template <typename T>
class value_wrapper {
public:
  /**
   * Ctor.
   *
   * @param value Value to copy or move from
   */
  value_wrapper(T&& value) : value_{std::forward<T>(value)} {}

  /**
   * Return a const reference to the value.
   */
  const T& value() const noexcept { return value_; }

private:
  T value_;
};

/**
 * Insertion operator for the value wrapper.
 *
 * @tparam T type
 */
template <typename T>
auto& operator<<(std::ostream& out, const value_wrapper<T>& value)
{
  return out << value.value();
}

/**
 * Insertion operator for a vector with value wrapper instances.
 *
 * @tparam T type
 * @tparam A Allocator type
 */
template <typename T, typename A>
auto& operator<<(std::ostream& out, const std::vector<value_wrapper<T>, A>& vec)
{
  out << '[';
  for (decltype(vec.size()) i = 0; i < vec.size(); i++) {
    if (i)
      out << ", ";
    out << vec[i];
  }
  return out << ']';
}

/**
 * Test driver for the output stream wrapper.
 *
 * Each input type should be either a `std::integral_constant<T, v_>`, a type
 * that has a static `value` member, or an invocable type where invocation
 * returns the input value to use for the test case.
 *
 * @tparam Ts... Input types
 */
template <typename... Ts>
struct ostream_wrapper_tester {
  /**
   * Execute the tests for each input type.
   *
   * @param out Output stream to write to
   */
  void operator()(std::ostream& out = std::cout) const
  {
    npygl::ostream_wrapper sink{out};
    // for each input
    (
      [&sink]
      {
        // print test header
        // FIXME: we actually want the type of the input
        sink << "Test " << npygl::type_name(typeid(Ts)) << '\n';
        // if invocable, use invoked value
        if constexpr (std::is_invocable_v<Ts>)
          sink << Ts{}() << '\n';
        // else assume it has the static value member
        else
          sink << Ts::value << '\n';
      }()
      ,
      ...
    );
    // ensure all results are written
    sink << std::flush;
  }
};

/**
 * Partial specialization for a tuple of input types.
 *
 * @tparam Ts... Input types
 */
template <typename... Ts>
struct ostream_wrapper_tester<std::tuple<Ts...>>
  : ostream_wrapper_tester<Ts...> {};

/**
 * Type to hold a double input.
 *
 * Necessary as pre-C++20 non-type template params don't allow floating types.
 */
struct double_input {
  static constexpr auto value = 1.45;
};

/**
 * Type to hold the `not_ostreamable_type` input.
 *
 * Necessary as pre-C++20 non-type template params don't allow user-defined
 * literal types, just the integral, enum, pointer, etc.
 */
struct not_ostreamable_type_input {
  static constexpr npygl::testing::not_ostreamable_type value{};
};

/**
 * Callable that returns a `std::map`.
 */
struct map_input {
  std::map<std::string, double> operator()() const
  {
    return {{"a", 1.4}, {"b", 1.334}, {"c", 5.66}};
  }
};

/**
 * Type to hold a string literal input.
 */
struct cstring_input {
  static constexpr auto value = "the quick brown fox jumped over the lazy dog";
};

/**
 * Callable that returns a `std::string`.
 */
struct string_input {
  std::string operator()() const
  {
    return "this is a short sentence";
  }
};

/**
 * Callable that returns a `std::vector`.
 */
struct vector_input {
  std::vector<double> operator()() const
  {
    return {1., 3.44, 1.232, 1.554, 1.776};
  }
};

/**
 * Callable that returns a `std::vector<value_wrapper<T>>`.
 */
struct value_wrapper_vector_input {
  std::vector<value_wrapper<unsigned>> operator()() const
  {
    return {1u, 2u, 3u, 4u, 5u, 6u};
  }
};

/**
 * Callable that returns a `std::set`.
 */
struct set_input {
  std::set<double> operator()() const
  {
    return {1., 3.22, 4.23, 4.233, 5.151};
  }
};

/**
 * Output stream wrapper input types.
 */
using input_types = std::tuple<
  std::integral_constant<int, 5>,
  double_input,
  not_ostreamable_type_input,
  map_input,
  cstring_input,
  string_input,
  vector_input,
  value_wrapper_vector_input,
  set_input
>;

// sequential stream test driver
constexpr ostream_wrapper_tester<input_types> tester;

}  // namespace

int main()
{
  // run sequential stream tests
  tester();
  return EXIT_SUCCESS;
}
