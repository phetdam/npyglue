/**
 * @file npy_helpers_test.cc
 * @author Derek Huang
 * @brief C++ program to test the npygl NumPy helpers
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <iostream>

#include "npygl/py_helpers.hh"
#include "npygl/npy_helpers.hh"

int main()
{
  // initialize Python + print version
  npygl::py_instance python;
  std::cout << Py_GetVersion() << std::endl;
  // import NumPy
  auto np = npygl::py_import("numpy");
  npygl::py_error_exit();
  // get np.get_include
  auto np_get_include = npygl::py_getattr(np, "get_include");
  npygl::py_error_exit();
  // get include directories
  auto inc_dirs = PyObject_CallNoArgs(np_get_include);
  npygl::py_error_exit();
  // translate object to C++ string
  // get the np.array function
  auto numpy_array = npygl::py_getattr(np, "array");
  npygl::py_error_exit();
  return EXIT_SUCCESS;
}
