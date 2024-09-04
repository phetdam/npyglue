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

#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/python.hh"
#include "npygl/range_views.hh"
#include "npygl/testing/math.hh"

namespace {

/**
 * Import the `numpy.random.random` function.
 *
 * On error, the returned `py_object` is empty.
 */
npygl::py_object npy_random() noexcept
{
  auto mod = npygl::py_import("numpy.random");
  if (!mod)
    return {};
  return npygl::py_getattr(mod, "random");
}

/**
 * Test modification of a NumPy array through flat views.
 *
 * @param np_random Python object representing `numpy.random.random`
 * @returns `true` on success, `false` on failure
 */
bool test_flat_views(PyObject* np_random)
{
  using npygl::testing::sine;
  using npygl::testing::asine;
  // create a random 3x4x2 tensor
  npygl::py_object shape{Py_BuildValue("(iii)", 3, 4, 2)};
  if (!shape)
    return false;
  auto obj = npygl::py_call_one(np_random, shape);
  if (!obj)
    return false;
  // apply sine + inverse sine functions to NumPy array via view and print
  auto ar = obj.as<PyArrayObject>();
  npygl::ndarray_flat_view<double> view{ar};
  sine(view);
  std::cout << "sine transform:\n" << ar << std::endl;
  asine(view);
  std::cout << "inverse sine transform:\n" << ar << std::endl;
#if NPYGL_HAS_CC_20
  // apply sine + inverse function to NumPy array via span and print
  auto stl_view = npygl::make_span<double>(ar);
  sine(stl_view);
  std::cout << "span sine transform:\n" << ar << std::endl;
  asine(stl_view);
  std::cout << "span inverse sine transform:\n" << ar << std::endl;
#endif  // NPYGL_HAS_CC_20
  return true;
}

/**
 * Test modification of a NumPy array through matrix views.
 *
 * @param np_random Pytho object representing `numpy.random.random`
 * @returns `true` on success, `false` on failure
 */
bool test_matrix_views(PyObject* np_random)
{
  // create a random 4x4 matrix
  npygl::py_object shape{Py_BuildValue("ii", 4, 4)};
  if (!shape)
    return false;
  auto mat = npygl::py_call_one(np_random, shape);
  if (!mat)
    return false;
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
  // double the upper triangle
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    for (unsigned int j = i; j < mat_view.cols(); j++)
      mat_view(i, j) *= 2;
  std::cout << "doubled upper triangle:\n" << mat_ar << std::endl;
  for (unsigned int i = 0; i < mat_view.rows(); i++)
    for (unsigned int j = i; j < mat_view.cols(); j++)
      mat_view(i, j) /= 2;
  std::cout << "halved upper triangle:\n" << mat_ar << std::endl;
  return true;
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
  // get np.random.random
  auto np_random = npy_random();
  npygl::py_error_exit();
  // call test functions
  if (!test_flat_views(np_random) || !test_matrix_views(np_random))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
