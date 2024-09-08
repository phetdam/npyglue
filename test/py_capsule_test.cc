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
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/python.hh"

#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO

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
    // get capsule view
    npygl::cc_capsule_view view{capsule};
    npygl::py_error_exit();
    // we already know the type so directly print package from capsule
    std::cout << "\ncapsule data:\n" << *view.as<package_type>() << std::endl;
  }
  // pass the data_type_printer itself to a capsule
  {
    auto capsule = npygl::py_object::create(data_type_printer{});
    npygl::py_error_exit();
    // get capsule view
    npygl::cc_capsule_view view{capsule};
    npygl::py_error_exit();
    // call the printer's methods on some data
    const auto& printer = *view.as<data_type_printer>();
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
#if NPYGL_HAS_EIGEN3
  // create a row-major Eigen matrix and pass it into a capsule
  {
    auto capsule = npygl::py_object::create(
      Eigen::MatrixXf{
        {4.f, 3.222f, 3.41f, 2.3f},
        {5.44f, 2.33f, 2.33f, 5.563f},
        {6.55f, 7.234f, 23.1f, 7.66f}
      }
    );
    npygl::py_error_exit();
    npygl::cc_capsule_view view{capsule};
    npygl::py_error_exit();
    // print the type name + the matrix contents itself
    const auto& mat = *view.as<Eigen::MatrixXf>();
    std::cout << "-- " << npygl::type_name(view.info()) << std::endl;
    std::cout << mat << std::endl;
  }
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  // create an Armadillo complex matrix and pass it into a capsule
  {
    auto capsule = npygl::py_object::create(
      arma::cx_mat{
        {{4.3, 3.422}, {1.3, 2.322}, {5.44, 3.431}},
        {{6.33, 3.413}, {12.12, 5.434}, {5.44, 3.222}}
      }
    );
    npygl::py_error_exit();
    npygl::cc_capsule_view view{capsule};
    npygl::py_error_exit();
    // print the type name + the matrix contents
    const auto& mat = *view.as<arma::cx_mat>();
    std::cout << "-- " << npygl::type_name(view.info()) << std::endl;
    std::cout << mat << std::endl;
  }
#endif  // NPYGL_HAS_ARMADILLO
  return EXIT_SUCCESS;
}
