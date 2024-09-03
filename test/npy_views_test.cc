/**
 * @file npy_views_test.cc
 * @author Derek Huang
 * @brief C++ program to test npygl views with NumPy arrays
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <ios>
#include <iostream>

#include "npygl/npy_helpers.hh"
#include "npygl/py_helpers.hh"
#include "npygl/range_views.hh"

namespace {

/**
 * Import the `numpy.random.random` function.
 *
 * On error, the returned `py_object` is empty.
 */
npygl::py_object npy_get_random() noexcept
{
  auto mod = npygl::py_import("numpy.random");
  if (!mod)
    return {};
  return npygl::py_getattr(mod, "random");
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
  // get np.random module + np.random.random
  auto np_random = npy_get_random();
  npygl::py_error_exit();
   // create a random 4x4 matrix
  auto shape = npygl::py_object{Py_BuildValue("ii", 4, 4)};
  npygl::py_error_exit();
  auto mat = npygl::py_call_one(np_random, shape);
  npygl::py_error_exit();
  // print + enable boolalpha
  std::cout << mat << std::boolalpha << std::endl;
  // check that this is actually a 2D ndarray
  auto mat_ar = mat.as<PyArrayObject>();
  std::cout << "Is matrix? " << (PyArray_NDIM(mat_ar) == 2) << std::endl;
  // multiply diagonal by 2 and then halve diagonal using matrix view
  // note: this assumes C data ordering and double data type (the default)
  npygl::ndarray_matrix_view<double> mat_view{mat_ar};
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    mat_view(i, i) *= 2;
  std::cout << "doubled diagonal:\n" << mat_ar << std::endl;
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    mat_view(i, i) /= 2;
  std::cout << "halved diagonal:\n" << mat_ar << std::endl;
  // repeat this but using a flat view
  npygl::ndarray_flat_view<double> flat_view{mat_ar};
  for (auto& v : flat_view)
    v *= 2;
  std::cout << "doubled elements:\n" << mat_ar << std::endl;
  for (auto& v : flat_view)
    v /= 2;
  std::cout << "halved elements:\n" << mat_ar << std::endl;
  // double the upper triangle
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    for (unsigned int j = i; j < mat_view.cols(); j++)
      mat_view(i, j) *= 2;
  std::cout << "doubled upper triangle:\n" << mat_ar << std::endl;
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    for (unsigned int j = i; j < mat_view.cols(); j++)
      mat_view(i, j) /= 2;
  std::cout << "halved upper triangle:\n" << mat_ar << std::endl;
  return EXIT_SUCCESS;
}
