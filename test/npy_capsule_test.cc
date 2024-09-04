/**
 * @file npy_capsule_test.cc
 * @author Derek Huang
 * @brief C++ program to test NumPy arrays backed by PyCapsule objects
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "npygl/ndarray.hh"
#include "npygl/python.hh"

int main()
{
  // initialize Python + import NumPy API
  npygl::npy_api_import(npygl::py_init());
  npygl::py_error_exit();
  // print version
  std::cout << Py_GetVersion() << std::endl;
  // create NumPy arrays backed by double and integer vectors
  auto d_ar = npygl::make_ndarray(std::vector{2.3, 1.222, 14.23, 3.243, 5.556});
  npygl::py_error_exit();
  auto i_ar = npygl::make_ndarray(std::vector{3, 14, 1, 555, 34, 3, 8, 42});
  npygl::py_error_exit();
  // create a "normal" NumPy array
  auto ar_init = Py_BuildValue("ddddd", 3.4, 1.222, 6.745, 5.2, 5.66, 7.333);
  npygl::py_error_exit();
  auto ar = npygl::make_ndarray<double>(ar_init);
  npygl::py_error_exit();
  // print the repr() for each array
  std::cout <<
    "NumPy arrays\n" <<
    "============\n\n" <<
    "from std::vector<double>:\n    " << d_ar << '\n' <<
    "from std::vector<int>:\n    " << i_ar << '\n' <<
    "from tuple[double]:\n    " << ar << '\n' << std::endl;
  npygl::py_error_exit();
  // get base objects for each array
  auto d_base = PyArray_BASE(d_ar.as<PyArrayObject>());
  auto i_base = PyArray_BASE(i_ar.as<PyArrayObject>());
  auto base = PyArray_BASE(ar.as<PyArrayObject>());
  // print base objects for each array
  std::cout <<
    "NumPy array bases\n" <<
    "=================\n\n" <<
    "from std::vector<double>:\n    " << d_base << '\n' <<
    "from std::vector<int>:\n    " << i_base << '\n' <<
    "from tuple[double]:\n    " << base << std::endl;
  npygl::py_error_exit();
  return EXIT_SUCCESS;
}
