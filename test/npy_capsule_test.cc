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

#include "npygl/features.h"
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
#if NPYGL_HAS_EIGEN3
  // create a NumPy array backed by an Eigen column-major matrix
  auto e_ar = npygl::make_ndarray(
    Eigen::MatrixXf{
      {4.333f, 1.44f, 1.532f, 1.222f},
      {5.6634f, 2.2f, 1.555f, 5.64f},
      {6.7774f, 4.87f, 9.875f, 3.22f}
    }
  );
  // create a NumPy array backed by an Eigen row-major matrix
  auto ec_ar = npygl::make_ndarray(
    Eigen::Matrix<
      unsigned int,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::StorageOptions::RowMajor
    >{
      {4, 1, 2, 3, 2},
      {6, 7, 8, 2, 3},
      {31, 6, 44, 1, 23}
    }
  );
  npygl::py_error_exit();
#endif  // NPYGL_HAS_EIGEN3
  // create a "normal" NumPy array
  auto ar_init = Py_BuildValue("ddddd", 3.4, 1.222, 6.745, 5.2, 5.66, 7.333);
  npygl::py_error_exit();
  auto ar = npygl::make_ndarray<double>(ar_init);
  npygl::py_error_exit();
  // print the repr() for each array
  std::cout <<
    "NumPy arrays\n"
    "============\n\n"
    "-- std::vector<double>\n" << d_ar << '\n' <<
    "-- std::vector<int>\n" << i_ar << '\n' <<
#if NPYGL_HAS_EIGEN3
    "-- Eigen::MatrixXf\n" << e_ar << '\n' <<
    "-- Eigen::Matrix<unsigned, Eigen::Dynamic, Eigen::Dynamic, "
      "Eiden::StorageOptions::RowMajor>\n" << ec_ar << '\n' <<
#endif  // NPYGL_HAS_EIGEN3
    "-- tuple[double]\n" << ar << '\n' << std::endl;
  npygl::py_error_exit();
  // get base objects for each array
  auto d_base = PyArray_BASE(d_ar.as<PyArrayObject>());
  auto i_base = PyArray_BASE(i_ar.as<PyArrayObject>());
#if NPYGL_HAS_EIGEN3
  auto e_base = PyArray_BASE(e_ar.as<PyArrayObject>());
  auto ec_base = PyArray_BASE(ec_ar.as<PyArrayObject>());
#endif  // NPYGL_HAS_EIGEN3
  auto base = PyArray_BASE(ar.as<PyArrayObject>());
  // print base objects for each array
  std::cout <<
    "NumPy array bases\n" <<
    "=================\n\n" <<
    "-- std::vector<double>\n" << d_base << '\n' <<
    "-- std::vector<int>\n" << i_base << '\n' <<
#if NPYGL_HAS_EIGEN3
    "-- Eigen::MatrixXf\n" << e_base << '\n' <<
    "-- Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, "
      "Eigen::StorageOptions::RowMajor>\n" << ec_base << '\n' <<
#endif  // NPYGL_HAS_EIGEN3
    "-- tuple[double]\n" << base << std::endl;
  npygl::py_error_exit();
  return EXIT_SUCCESS;
}
