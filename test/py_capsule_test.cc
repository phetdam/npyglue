/**
 * @file py_capsule_test.cc
 * @author Derek Huang
 * @brief C++ program to test PyCapsule creation from arbitrary C++ objects
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <iterator>
#include <ostream>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "npygl/py_helpers.hh"

namespace {

/**
 * Simple variant type.
 */
using data_type = std::variant<double, int, std::string, std::vector<double>>;

/**
 * Visitor for printing the values of the `data_type`.
 */
class data_type_printer {
public:
  /**
   * Ctor.
   *
   * @param out Stream to print to
   */
  data_type_printer(std::ostream& out = std::cout) noexcept : out_{out} {}

  /**
   * Write a double to the stream.
   */
  void operator()(double x) const
  {
    out_ << x;
  }

  /**
   * Write an int to the stream.
   */
  void operator()(int x) const
  {
    out_ << x;
  }

  /**
   * Write a string to the stream.
   */
  void operator()(const std::string& s) const
  {
    out_ << s;
  }

  /**
   * Write the double vector to the stream.
   */
  void operator()(const std::vector<double>& vec) const
  {
    out_ << '[';
    for (auto it = vec.begin(); it != vec.end(); it++) {
      if (std::distance(vec.begin(), it))
        out_ << ", ";
      out_ << *it;
    }
    out_ << ']';
  }

private:
  std::ostream& out_;
};

/**
 * Capsule package type.
 *
 * This is what we will have Python own via a capsule.
 */
using package_type = std::map<std::string, data_type>;

/**
 * `operator<<` overload so we can print our package.
 */
auto& operator<<(std::ostream& out, const package_type& package)
{
  // special formatting if empty
  if (package.empty())
    return out << "{}";
  // otherwise, display like a JSON string
  out << "{\n";
  for (auto it = package.begin(); it != package.end(); it++) {
    if (std::distance(package.begin(), it))
      out << ",\n";
    out << "    {" << it->first << ", ";
    std::visit(data_type_printer{out}, it->second);
    out << '}';
  }
  return out << "\n}";
}

}  // namespace

int main()
{
  // init Python + print version
  npygl::py_init();
  std::cout << Py_GetVersion() << std::endl;
  // C++ map that we will pass to Python
  // note: using ordered map for consistent iteration order
  package_type package{
    {"key_1", 4.333},
    {"key_2", 342},
    {"key_3", "the quick brown fox jumped over the lazy dog"},
    {"key_4", std::vector{4., 3.22, 1.22, 5.645, 3.14159265358979}}
  };
  // print package
  std::cout << "C++ package:\n" << package << std::endl;
  // create capsule from the package via move
  // note: braced context used to test dtor callback
  {
    auto capsule = npygl::py_object::create(std::move(package));
    npygl::py_error_exit();
    // package got moved from
    std::cout << "\nC++ package (moved):\n" << package << std::endl;
    // get pointer from capsule
    auto data = (package_type*) PyCapsule_GetPointer(capsule, nullptr);
    npygl::py_error_exit();
    // print package from capsule
    std::cout << "\ncapsule data:\n" << *data << std::endl;
  }
  // pass the data_type_printer itself to a capsule
  {
    auto capsule = npygl::py_object::create(data_type_printer{});
    npygl::py_error_exit();
    // get printer
    auto data = (data_type_printer*) PyCapsule_GetPointer(capsule, nullptr);
    npygl::py_error_exit();
    // call the printer's methods on some data
    const auto& printer = *data;
    std::cout << "\ndata_type_printer printing from capsule:\n";
    printer(4.34353);
    std::cout << '\n';
    printer(233);
    std::cout << '\n';
    printer("the lazy dog was awakened by the boisterous fox");
    std::cout << '\n';
    printer(std::vector{4.33, 1.23, 1.51424, 1.111});
    std::cout << std::endl;
  }
  return EXIT_SUCCESS;
}
