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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/python.hh"
#include "npygl/type_traits.hh"

#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
#include <torch/torch.h>
#endif  // NPYGL_HAS_LIBTORCH

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
    out << "    {" << it->first << ": ";
    std::visit(data_type_printer{out}, it->second);
    out << '}';
  }
  return out << "\n}";
}

/**
 * SFINAE helper for `capsule_test`.
 *
 * @tparam T C++ type compatible with `py_object::create`
 * @tparam Fs... Callables taking a `const T&`
 */
template <typename T, typename... Fs>
using capsule_testable_t = std::enable_if_t<
  !std::is_reference_v<T> &&
  (std::is_invocable_v<Fs, T> && ...)
>;

/**
 * Move a C++ object into a Python capsule and then access it via capsule view.
 *
 * The name of the original C++ object type will also be written along with the
 * text representation of the object as presented via `operator<<` call.
 *
 * @tparam T C++ type compatible with `py_object::create`
 *
 * @param out Output stream to write to
 * @param obj C++ object to move to capsule
 */
template <typename T, typename... Fs, typename = capsule_testable_t<T, Fs...>>
void capsule_test(std::ostream& out, T&& obj, Fs... funcs)
{
  // create capsule from the moved object
  auto cap = npygl::py_object::create(std::move(obj));
  npygl::py_error_exit();
  // take capsule view
  npygl::cc_capsule_view view{cap};
  npygl::py_error_exit();
  // print the type name
  const auto& val = *view.as<T>();
  out << "-- " << npygl::type_name(view.info()) << '\n';
  // stream the type to the output stream if possible else print a default
  if constexpr (npygl::is_ostreamable_v<T>)
    out << val;
  else
    out << "<object " << npygl::type_name(typeid(val)) << " at " <<
      &val << '>';
  // newline + flush
  out << std::endl;
  // run any custom actions on the object
  if constexpr (sizeof...(Fs))
    (funcs(val), ...);
}

/**
 * Move a C++ object into a Python capsule and then access it via capsule view.
 *
 * Standard output will receive the content written by the function.
 *
 * @tparam T C++ type compatible with `py_object::create`
 *
 * @param obj C++ object to move to capsule
 */
template <typename T, typename... Fs, typename = capsule_testable_t<T, Fs...>>
void capsule_test(T&& obj, Fs... funcs)
{
  capsule_test(std::cout, std::move(obj), std::move(funcs)...);
}

}  // namespace

int main()
{
  // init Python + print version
  npygl::py_init();
  std::cout << Py_GetVersion() << std::endl;
  // C++ map that we will pass to Python
  {
    // note: using ordered map for consistent iteration order
    package_type package{
      {"key_1", 4.333},
      {"key_2", 342},
      {"key_3", "the quick brown fox jumped over the lazy dog"},
      {"key_4", std::vector{4., 3.22, 1.22, 5.645, 3.14159265358979}}
    };
    // create capsule from the package via move
    capsule_test(std::move(package));
    // show that original object has been moved from
    std::cout << "-- " << npygl::type_name(typeid(decltype(package))) <<
      " (moved):\n" << package << std::endl;
  }
  // pass the data_type_printer itself to a capsule
  capsule_test(
    data_type_printer{},
    // perform some printing using the moved printer
    [](const data_type_printer& printer)
    {
      std::cout << "-- " << npygl::type_name(typeid(data_type_printer)) <<
        " printing from capsule:\n";
      printer(4.34353);
      std::cout << '\n';
      printer(233);
      std::cout << '\n';
      printer("the lazy dog was awakened by the boisterous fox");
      std::cout << '\n';
      printer(std::vector{4.33, 1.23, 1.51424, 1.111});
      std::cout << std::endl;
    }
  );
#if NPYGL_HAS_EIGEN3
  // create a column-major Eigen matrix and pass it into a capsule
  {
    Eigen::MatrixXf mat{
      {4.f, 3.222f, 3.41f, 2.3f},
      {5.44f, 2.33f, 2.33f, 5.563f},
      {6.55f, 7.234f, 23.1f, 7.66f}
    };
    capsule_test(std::move(mat));
  }
  // create a fixed-size Eigen matrix and pass it into a capsule
  {
    Eigen::Matrix3d mat{
      {1., 0.45, 0.115},
      {0.45, 1., 0.87},
      {0.115, 0.87, 1.}
    };
    capsule_test(std::move(mat));
  }
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  // create an Armadillo complex matrix and pass it into a capsule
  {
    arma::cx_mat mat{
      {{4.3, 3.422}, {1.3, 2.322}, {5.44, 3.431}},
      {{6.33, 3.413}, {12.12, 5.434}, {5.44, 3.222}}
    };
    capsule_test(std::move(mat));
  }
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
  // create a PyTorch float tensor and pass it into a capsule
  {
    // for reproducibility, create a Generator
    // note: ATen/core/Generator.h says that you should acquire the generator
    // mutex before calling any read-only (??) methods. this is because each
    // Generator is a view of a reference-counted generator implementation, so
    // in a multi-threaded context different Generator instances might actually
    // refer to the same generator implementation (and therefore state). in
    // a single-threaded case where we know the instances are separate we can
    // call whatever methods we would like without fear.
    auto gen = at::make_generator<at::CPUGeneratorImpl>();
    capsule_test(torch::randn({2, 3, 4}, gen));
  }
#endif  // NPYGL_HAS_LIBTORCH
  return EXIT_SUCCESS;
}
