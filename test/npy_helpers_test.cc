/**
 * @file npy_helpers_test.cc
 * @author Derek Huang
 * @brief C++ program to test the npygl NumPy helpers
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cmath>
#include <cstdlib>
#include <ios>
#include <iostream>

#include "npygl/py_helpers.hh"
#include "npygl/npy_helpers.hh"

namespace {
/**
 * Compute the sine of the view elements.
 *
 * @param view NumPy array view
 */
void sine(const npygl::ndarray_flat_view<double>& view)
{
  for (decltype(view.size()) i = 0; i < view.size(); i++)
    view[i] = std::sin(view[i]);
}

}  // namespace

int main()
{
  // initialize Python + print version
  npygl::py_instance python;
  std::cout << Py_GetVersion() << std::endl;
  // import NumPy C API
  npygl::npy_api_import(python);
  npygl::py_error_exit();
  // print the NumPy include directory
  std::cout << "NumPy include dir: " << npygl::npy_include_dir() << std::endl;
  // get the np.random module
  auto np_random = npygl::py_import("numpy.random");
  npygl::py_error_exit();
  // get the np.random.random function
  auto np_random_random = npygl::py_getattr(np_random, "random");
  npygl::py_error_exit();
  // create shape tuple
  auto shape = npygl::py_object{Py_BuildValue("(iii)", 3, 4, 2)};
  npygl::py_error_exit();
  // call np.random.random with the shape tuple
  auto res = npygl::py_call_one(np_random_random, shape);
  npygl::py_error_exit();
  // print NumPy array using repr() + add newline and enable boolalpha
  npygl::py_print(res);
  npygl::py_error_exit();
  std::cout << std::boolalpha << std::endl;
  // check if NumPy array (it is)
  std::cout << "Is NumPy array? " << npygl::is_ndarray(res) << std::endl;
  // print additional info
  auto ar = res.as<PyArrayObject>();
  std::cout << "Is double type? " << npygl::is_type<double>(ar) << std::endl;
  std::cout << "Is behaved? " << PyArray_ISBEHAVED(ar) << std::endl;
  // apply sine function to NumPy array and print it again
  sine(ar);
  npygl::py_print(res);
  npygl::py_error_exit();
  std::cout << std::endl;
  return EXIT_SUCCESS;
}
