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
#include <typeinfo>
#include <type_traits>
#include <utility>
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

namespace {

// TODO: document; also unnecessary since we can directly evaluate the lambda
template <typename T, typename = void>
struct produces_ndarray_convertible : std::false_type {};

template <typename T>
struct produces_ndarray_convertible<
  T,
  std::enable_if_t<npygl::can_make_ndarray_v<std::invoke_result_t<T>>>
> : std::true_type {};

/**
 * SFINAE helper to indicate if a type can be used to create a NumPy array.
 *
 * @note May want to move to `ndarray.h` to improve the type traits there.
 *
 * @tparam T type
 */
template <typename T>
using make_ndarray_test_t = std::enable_if_t<
  // require move-only construction + type must have a builder
  (!std::is_reference_v<T> && npygl::can_make_ndarray_v<T>) ||
  // or is invokable and return value type has a builder
  produces_ndarray_convertible<T>::value
>;

/**
 * Create a NumPy array from a C++ object and print the capsule and the array.
 *
 * If there is a Python error then `npygl::py_error_exit` is called.
 *
 * @tparam T type
 *
 * @param out Stream to write output to
 * @param obj C++ object rvalue to create a NumPy array from
 */
template <typename T, typename = make_ndarray_test_t<T>>
void make_ndarray_test(std::ostream& out, T&& obj)
{
  // create NumPy array backed by C++ object
  auto ar = [&obj]
  {
    // if callable
    if constexpr (produces_ndarray_convertible<T>::value)
      return npygl::make_ndarray(obj());
    // object is directly convertible
    else
      return npygl::make_ndarray(std::move(obj));
  }();
  npygl::py_error_exit();
  // get backing base object for the object
  // note: need template qualifier since ar implicitly depends on T
  auto ar_base = PyArray_BASE(ar.template as<PyArrayObject>());
  // write to stream with type name + base
  out << "-- " << npygl::type_name(typeid(T)) << " (" << ar_base << ")\n" <<
    ar << std::endl;
}

/**
 * Create a NumPy array from a C++ object and print the capsule and the array.
 *
 * If there is a Python error then `npygl::py_error_exit` is called. This
 * overload simply writes to `std::cout` as the output stream.
 *
 * @tparam T type
 *
 * @param obj C++ object rvalue to create a NumPy array from
 */
template <typename T, typename = make_ndarray_test_t<T>>
void make_ndarray_test(T&& obj)
{
  make_ndarray_test(std::cout, std::move(obj));
}

}  // namespace

int main()
{
  // initialize Python + import NumPy API
  npygl::npy_api_import(npygl::py_init());
  npygl::py_error_exit();
  // print version
  std::cout << Py_GetVersion() << std::endl;
  // create NumPy arrays backed by double and integer vectors
  make_ndarray_test(std::vector{2.3, 1.222, 14.23, 3.243, 5.556});
  make_ndarray_test(std::vector{3, 14, 1, 555, 34, 3, 8, 42});
#if NPYGL_HAS_EIGEN3
  // create a NumPy array backed by an Eigen column-major matrix
  make_ndarray_test(
    Eigen::MatrixXf{
      {4.333f, 1.44f, 1.532f, 1.222f},
      {5.6634f, 2.2f, 1.555f, 5.64f},
      {6.7774f, 4.87f, 9.875f, 3.22f}
    }
  );
  // create a NumPy array backed by an Eigen row-major matrix
  make_ndarray_test(
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
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  // create a NumPy array backed by an Armadillo complex matrix
  make_ndarray_test(
    arma::cx_mat{
      {{3.44, 5.423}, {9.11, 4.333}, {4.63563, 1.111}},
      {{4.23, 2.123}, {3.4244, 5.22}, {0.999, 12.213}}
    }
  );
  // create a NumPy array backed by an Armadillo complex cube
  make_ndarray_test(
    []
    {
      arma::cx_cube cube{2, 2, 3};
      // need to set slice-by-slice
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
      return cube;
    }
  );
  // create a NumPy array backed by an Armadillo float column vector
  make_ndarray_test(arma::fvec{1.f, 3.4f, 4.23f, 3.54f, 5.223f});
  // create a NumPy array backed by an Armadillo double row vector
  make_ndarray_test(arma::rowvec{5., 4.33, 2.433, 1.22, 4.34});
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
  // create a NumPy array backed by a random PyTorch float tensor
  make_ndarray_test(
    []
    {
      // PyTorch generator object for reproducibility. no need to acquire impl
      // mutex in single-threaded runtime like for this program
      auto gen = at::make_generator<at::CPUGeneratorImpl>();
      return torch::randn({2, 3, 4}, gen);
    }()
  );
#endif  // NPYGL_HAS_LIBTORCH
  // create a "normal" NumPy array
  // TODO: extend make_ndarray_test to allow custom type formatting
  auto ar_init = Py_BuildValue("ddddd", 3.4, 1.222, 6.745, 5.2, 5.66, 7.333);
  npygl::py_error_exit();
  // note: take ownership as we need to Py_DECREF the created object
  auto ar = npygl::make_ndarray<double>(npygl::py_object{ar_init});
  npygl::py_error_exit();
  // get base object
  auto base = PyArray_BASE(ar.as<PyArrayObject>());
  // print the repr()
  std::cout << "-- tuple[double] (" << base << ")\n" << ar << std::endl;
  return EXIT_SUCCESS;
}
