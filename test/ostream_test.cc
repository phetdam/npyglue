/**
 * @file ostream_test.cc
 * @author Derek Huang
 * @brief C++ program to test ostream.hh
 * @copyright MIT License
 */

#include <cstdlib>
#include <deque>
#include <forward_list>
#include <iterator>
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
   * @param value Value to copy from
   */
  value_wrapper(const T& value) : value_{value} {}

  /**
   * Ctor.
   *
   * @param value Value to move from
   */
  value_wrapper(T&& value) : value_{std::move(value)} {}

  /**
   * Return a const reference to the value.
   */
  const T& value() const noexcept { return value_; }

  /**
   * Return a const reference to the value.
   */
  const auto& operator*() const noexcept
  {
    return value_;
  }

private:
  T value_;
};

/**
 * Traits class to check if a type is a `value_wrapper<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct is_value_wrapper : std::false_type {};

/**
 * True specialization for a `value_wrapper<T>` instance.
 *
 * @tparam T type
 */
template <typename T>
struct is_value_wrapper<value_wrapper<T>> : std::true_type {};

/**
 * SFINAE helper for an iterable container of `value_wrapper<T>`.
 *
 * @tparam T type
 */
template <typename T>
using value_wrapper_iterable_t = std::enable_if_t<
  is_value_wrapper<std::decay_t<decltype(*std::begin(std::declval<T>()))>>
    ::value
>;

/**
 * Iterator that yields `value_wrapper<T>` from a `T` iterator.
 *
 * This satisfies the *LegacyInputIterator* named concept.
 *
 * @tparam It Iterator type
 */
template <typename It>
class value_wrapper_iterator {
public:
  // iterator traits types
  using difference_type = std::ptrdiff_t;
  using value_type = value_wrapper<std::decay_t<decltype(*std::declval<It>())>>;
  using pointer = const value_type*;
  using reference = const value_type&;
  using iterator_category = std::input_iterator_tag;

  /**
   * Ctor.
   *
   * @note The `value_wrapper<T>` uses a value-initialized `T` so that tag
   *  dispatch is not required when using a one-past-the-end iterator.
   *
   * @param iter Iterator
   */
  value_wrapper_iterator(It iter) noexcept : iter_{iter}, value_{{}} {}

  /**
   * Return a reference to the current wrapped value.
   *
   * @note Each call performs an additional read + copy from original iterator.
   */
  auto& operator*()
  {
    value_ = *iter_;
    return value_;
  }

  /**
   * Increment the iterator position (pre-increment).
   */
  auto& operator++()
  {
    ++iter_;
    return *this;
  }

  /**
   * Increment the iterator position (post-increment).
   */
  void operator++(int)
  {
    iter_++;
  }

  /**
   * Equality comparison for the iterator.
   */
  bool operator==(const value_wrapper_iterator& other) const noexcept
  {
    return iter_ == other.iter_;
  }

  /**
   * Inequality comparison for the iterator.
   */
  bool operator!=(const value_wrapper_iterator& other) const noexcept
  {
    return !(*this == other);
  }

  /**
   * Provide member access to the yielded value wrapper.
   */
  value_type* operator->()
  {
    return &*(*this);
  }

private:
  It iter_;
  value_type value_;
};

}  // namespace

namespace std {

/**
 * `std::less` partial specialization for `value_wrapper<T>`.
 *
 * Necessary for `value_wrapper<T>` to work with `std::set`, `std::map`.
 *
 * @tparam T type
 */
template <typename T>
struct less<value_wrapper<T>> {
  bool operator()(
    const value_wrapper<T>& a,
    const value_wrapper<T>& b) const noexcept(noexcept(*a < *b))
  {
    return *a < *b;
  }
};

}  // namespace std

namespace {

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
 * Write an iterable of `value_wrapper<T>` to a stream.
 *
 * Values are printed with `", "` as a separator.
 *
 * @tparam C Forward-iterable container of `value_wrapper<T>`
 *
 * @param out Stream to write to
 * @param values Container of `value_wrapper<T>` to write
 */
template <typename C, typename = value_wrapper_iterable_t<C>>
void write(std::ostream& out, const C& values)
{
  auto it_begin = std::begin(values);
  auto it_end = std::end(values);
  // print values
  for (auto it = it_begin; it != it_end; it++) {
    if (it != it_begin)
      out << ", ";
    out << *it;
  }
}

/**
 * Write an iterable of `value_wrapper<T>` to a stream with left/right delims.
 *
 * Values are printed with `", "` as a separator.
 *
 * @tparam C Forward-iterable container of `value_wrapper<T>`
 *
 * @param out Stream to write to
 * @param values Container of `value_wrapper<T>` to write
 * @param ldelim Left delimiter
 * @param rdelim Right delimiter
 */
template <typename C, typename = value_wrapper_iterable_t<C>>
void write(std::ostream& out, const C& values, char ldelim, char rdelim)
{
  out << ldelim;
  write(out, values);
  out << rdelim;
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
  write(out, vec, '[', ']');
  return out;
}

/**
 * Insertion operator for a set with value wrapper instances.
 *
 * @tparam K Key type
 * @tparam C Comparator type
 * @tparam A Allocator type
 */
template <typename K, typename C, typename A>
auto& operator<<(std::ostream& out, const std::set<value_wrapper<K>, C, A>& set)
{
  write(out, set, '{', '}');
  return out;
}

/**
 * Insertion operator for a general iterable value wrapper container.
 *
 * @tparam C Container type
 */
template <typename C, typename = value_wrapper_iterable_t<C>>
auto& operator<<(std::ostream& out, const C& values)
{
  write(out, values);
  return out;
}

/**
 * Traits type to get the stream wrapper tested type from the input type.
 *
 * Input types have either a static `value` member or are invocable.
 *
 * @tparam T type
 */
template <typename T, typename = void, typename = void>
struct tester_input_type {};

/**
 * True specialization for a valid input type with a `value` member.
 *
 * @note Since some types, like `std::integral_constant<T, v_>`, may have both
 *  the `value` static member and an `operator()()`, we need to ensure that in
 *  this case only one partial specialization matches.
 *
 * @tparam T type
 */
template <typename T>
struct tester_input_type<
  T,
  std::enable_if_t<!std::is_invocable_v<T>>,
  std::void_t<decltype(T::value)>
> {
  using type = decltype(T::value);
};

/**
 * True specialization for a valid callable input type.
 *
 * @tparam T type
 */
template <typename T>
struct tester_input_type<T, std::enable_if_t<std::is_invocable_v<T>>> {
  using type = std::invoke_result_t<T>;
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
 *
 * The vector values used are from `vector_input{}()`.
 */
struct value_wrapper_vector_input {
  auto operator()() const
  {
    auto values = vector_input{}();
    auto it_begin = value_wrapper_iterator{values.begin()};
    auto it_end = value_wrapper_iterator{values.end()};
    return std::vector(std::move(it_begin), std::move(it_end));
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
 * Callable that returns a `std::set<value_wrapper<T>>`.
 *
 * The set values used are from `set_input{}()`.
 */
struct value_wrapper_set_input {
  auto operator()() const
  {
    auto values = set_input{}();
    auto it_begin = value_wrapper_iterator{values.begin()};
    auto it_end = value_wrapper_iterator{values.end()};
    return std::set(std::move(it_begin), std::move(it_end));
  }
};

/**
 * Callable that returns a `std::deque`.
 */
struct deque_input {
  std::deque<unsigned> operator()() const
  {
    return {1u, 5u, 4u, 23u, 66u, 18u, 41u, 42u};
  }
};

/**
 * Callable that returns a `std::deque<value_wrapper<T>>`.
 *
 * The deque values used are from `deque_input{}()`.
 */
struct value_wrapper_deque_input {
  auto operator()() const
  {
    auto values = deque_input{}();
    auto it_begin = value_wrapper_iterator{values.begin()};
    auto it_end = value_wrapper_iterator{values.end()};
    return std::deque(std::move(it_begin), std::move(it_end));
  }
};

/**
 * Callable that returns a `std::forward_list`.
 */
struct forward_list_input {
  std::forward_list<float> operator()() const
  {
    return {3.1f, 34.22f, 5.11f, 1.23f, 1.5672f, 5.663f};
  }
};

/**
 * Callable that returns a `std::forward_list<value_wrapper<T>>`.
 *
 * The forward list values used are from `forward_list_input{}()`.
 */
struct value_wrapper_forward_list_input {
  auto operator()() const
  {
    auto values = forward_list_input{}();
    auto it_begin = value_wrapper_iterator{values.begin()};
    auto it_end = value_wrapper_iterator{values.end()};
    return std::forward_list(std::move(it_begin), std::move(it_end));
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
  set_input,
  value_wrapper_set_input,
  deque_input,
  value_wrapper_deque_input,
  forward_list_input,
  value_wrapper_forward_list_input
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
