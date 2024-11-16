/**
 * @file ostream_test.cc
 * @author Derek Huang
 * @brief C++ program to test ostream.hh
 * @copyright MIT License
 */

#include <cstdlib>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

#include "npygl/ostream.hh"

namespace {

/**
 * Non-streamable type.
 */
struct not_ostreamable_type {};

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
    out << vec[i].value();
  }
  return out << ']';
}

}  // namespace

int main()
{
  // int
  npygl::cout << 5 << std::endl;
  // double
  npygl::cout << 1.45 << std::endl;
  // not_ostreamable_type
  npygl::cout << not_ostreamable_type{} << std::endl;
  // std::map
  {
    std::map<std::string, double> map{{"a", 1.4}, {"b", 1.334}, {"c", 5.66}};
    npygl::cout << map << std::endl;
  }
  // string literal
  npygl::cout << "the quick brown fox jumped over the lazy dog" << std::endl;
  // std::string
  npygl::cout << std::string{"this is a short sentence"} << std::endl;
  // std::vector
  {
    std::vector<double> vec{1., 3.44, 1.232, 1.554, 1.776};
    npygl::cout << vec << std::endl;
  }
  // std::vector<value_wrapper<T>> (operator<< defined)
  {
    std::vector<value_wrapper<unsigned>> vec{1u, 2u, 3u, 4u, 5u, 6u};
    npygl::cout << vec << std::endl;
  }
  return EXIT_SUCCESS;
}
