/**
 * @file ostream_test.cc
 * @author Derek Huang
 * @brief C++ program to test ostream.hh
 * @copyright MIT License
 */

#include <cstdlib>
#include <map>
#include <mutex>
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
 * Traits type to get the stream wrapper tested type from the input type.
 *
 * Input types have either a static `value` member or are invocable.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct tester_input_type {};

/**
 * Partial specialization for a `std::integral_constant<T, v_>`.
 *
 * This is necessary because it has both `value` and `operator()`.
 *
 * @tparam T type
 * @tparam v_ value
 */
template <typename T, T v_>
struct tester_input_type<std::integral_constant<T, v_>, void> {
  using type = T;
};

/**
 * True specialization for a valid input type with a `value` member.
 *
 * @tparam T type
 */
template <typename T>
struct tester_input_type<T, std::void_t<decltype(T::value)>> {
  using type = decltype(T::value);
};

/**
 * True specialization for a valid callable input type.
 *
 * @tparam T type
 */
template <typename T>
struct tester_input_type<T, std::void_t<decltype(std::declval<T>()())>> {
  using type = decltype(std::declval<T>()());
};

/**
 * Type alias for the input type in the stream wrapper input type.
 *
 * @tparam T Input case type
 */
template <typename T>
using tester_input_type_t = typename tester_input_type<T>::type;

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
    sink << "Running " << sizeof...(Ts) << " test inputs..." << std::endl;
    // for each input
    (
      [&sink]
      {
        // print test header
        sink << "-- " << npygl::type_name(typeid(Ts)) << " [type = " <<
          npygl::type_name(typeid(tester_input_type_t<Ts>)) << "]\n";
        // if invocable, use invoked value
        if constexpr (std::is_invocable_v<Ts>)
          sink << Ts{}() << std::endl;
        // else assume it has the static value member
        else
          sink << Ts::value << std::endl;
      }()
      ,
      ...
    );
    sink << "Finished " << sizeof...(Ts) << " test inputs" << std::endl;
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
 * Test driver for the synchronized stream wrapper.
 *
 * Each input type should be either a `std::integral_constant<T, v_>`, a type
 * that has a static `value` member, or an invocable type where invocation
 * returns the input value to use for the test case.
 *
 * @tparam Ts... Input types
 */
template <typename... Ts>
class synced_ostream_wrapper_tester {
public:
  /**
   * Ctor.
   *
   * @param repeats Number of times to re-schedule the given input types
   */
  constexpr synced_ostream_wrapper_tester(unsigned repeats = 1u) noexcept
    : repeats_{repeats}
  {}

  /**
   * Execute the tests for each input type concurrently.
   *
   * @param out Output stream to write to
   */
  void operator()(std::ostream& out = std::cout) const
  {
    npygl::synced_ostream_wrapper sink{out};
    sink << "Running " << sizeof...(Ts) << " test inputs repeated " <<
      repeats_ << " times concurrently..." << std::endl;
    // threads to launch
    std::vector<std::thread> tasks;
    // for the given number of repeat counts * each input
    for (unsigned i = 0; i < repeats_; i++)
      (
        tasks.emplace_back(
          std::thread{
            [&sink, i]
            {
              // ensures writes are sequential in this scope
              std::lock_guard locker{sink.mut()};
              // print test header
              sink << "-- " << npygl::type_name(typeid(Ts)) <<
                " (" << i << ") [type = " <<
                npygl::type_name(typeid(tester_input_type_t<Ts>)) << "]\n";
              // if invocable, use invoked value
              if constexpr (std::is_invocable_v<Ts>)
                sink << Ts{}() << std::endl;
              // else assume it has the static value member
              else
                sink << Ts::value << std::endl;
            }
          }
        )
        ,
        ...
      );
    // join all threads
    for (auto& task : tasks)
      task.join();
    sink << "Finished " << sizeof...(Ts) << " test inputs repeated " <<
      repeats_ << " times concurrently" << std::endl;
  }

private:
  unsigned repeats_;
};

/**
 * Partial specialization for a tuple of input types.
 *
 * @tparam Ts... Input types
 */
template <typename... Ts>
struct synced_ostream_wrapper_tester<std::tuple<Ts...>>
  : synced_ostream_wrapper_tester<Ts...> {
  using synced_ostream_wrapper_tester<Ts...>::synced_ostream_wrapper_tester;
};

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
// synchronized concurrent stream test driver
constexpr synced_ostream_wrapper_tester<input_types> sync_tester{100};

}  // namespace

int main()
{
  // run sequential stream tests
  tester();
  // run concurrent synchronized stream tests
  sync_tester();
  return EXIT_SUCCESS;
}
