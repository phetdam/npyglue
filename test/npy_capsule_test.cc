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

#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_LIBTORCH
#include <torch/torch.h>
#endif  // NPYGL_HAS_LIBTORCH

// TODO: consider having this not be a giant main() function

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
  npygl::py_error_exit();
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
#if NPYGL_HAS_ARMADILLO
  // create a NumPy array backed by an Armadillo complex matrix
  auto a_ar = npygl::make_ndarray(
    arma::cx_mat{
      {{3.44, 5.423}, {9.11, 4.333}, {4.63563, 1.111}},
      {{4.23, 2.123}, {3.4244, 5.22}, {0.999, 12.213}}
    }
  );
  npygl::py_error_exit();
  // create a NumPy array backed by an Armadillo complex cube
  arma::cx_cube cube{2, 2, 3};
  cube.slice(0) = {
    {{3.4, 2.22}, {3.22, 4.23}},
    {{5.34, 5.111}, {6.66, 1.123}}
  };
  cube.slice(1) = {
    {{6.455, 1.111}, {4.232, 0.989}},
    {{6.1212, 1.1139}, {6.45, 0.2345}}
  };
  cube.slice(2) = {
    {{1.12, 4.412}, {5.34, 6.111}},
    {{4.123, 1.998}, {8.99, 1.114}}
  };
  auto ac_ar = npygl::make_ndarray(std::move(cube));
  npygl::py_error_exit();
  // create a NumPy array backed by an Armadillo float column vector
  auto av_ar = npygl::make_ndarray(arma::fvec{1.f, 3.4f, 4.23f, 3.54f, 5.223f});
  npygl::py_error_exit();
  // create a NumPy array backed by an Armadillo double row vector
  auto arv_ar = npygl::make_ndarray(arma::rowvec{5., 4.33, 2.433, 1.22, 4.34});
  npygl::py_error_exit();
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
  // PyTorch generator object for reproducibility. no need to acquire impl
  // mutex in single-threaded runtime like for this program
  auto gen = at::make_generator<at::CPUGeneratorImpl>();
  // create a NumPy array backed by a random PyTorch float tensor
  auto t_ar = npygl::make_ndarray(torch::randn({2, 3, 4}, gen));
  npygl::py_error_exit();
#endif  // NPYGL_HAS_LIBTORCH
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
#if NPYGL_HAS_ARMADILLO
    "-- arma::cx_mat\n" << a_ar << '\n' <<
    "-- arma::cx_cube\n" << ac_ar << '\n' <<
    "-- arma::fvec\n" << av_ar << '\n' <<
    "-- arma::rowvec\n" << arv_ar << '\n' <<
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
    "-- torch::Tensor\n" << t_ar << '\n' <<
#endif  // NPYGL_HAS_LIBTORCH
    "-- tuple[double]\n" << ar << '\n' << std::endl;
  npygl::py_error_exit();
  // get base objects for each array
  auto d_base = PyArray_BASE(d_ar.as<PyArrayObject>());
  auto i_base = PyArray_BASE(i_ar.as<PyArrayObject>());
#if NPYGL_HAS_EIGEN3
  auto e_base = PyArray_BASE(e_ar.as<PyArrayObject>());
  auto ec_base = PyArray_BASE(ec_ar.as<PyArrayObject>());
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  auto a_base = PyArray_BASE(a_ar.as<PyArrayObject>());
  auto ac_base = PyArray_BASE(ac_ar.as<PyArrayObject>());
  auto av_base = PyArray_BASE(av_ar.as<PyArrayObject>());
  auto arv_base = PyArray_BASE(arv_ar.as<PyArrayObject>());
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
  auto t_base = PyArray_BASE(t_ar.as<PyArrayObject>());
#endif  // NPYGL_HAS_LIBTORCH
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
#if NPYGL_HAS_ARMADILLO
    "-- arma::cx_mat\n" << a_base << '\n' <<
    "-- arma::cx_cube\n" << ac_base << '\n' <<
    "-- arma::fvec\n" << av_base << '\n' <<
    "-- arma::rowvec\n" << arv_base << '\n' <<
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
    "-- torch::Tensor\n" << t_base << '\n' <<
#endif  // NPYGL_HAS_LIBTORCH
    "-- tuple[double]\n" << base << std::endl;
  npygl::py_error_exit();
  return EXIT_SUCCESS;
}
