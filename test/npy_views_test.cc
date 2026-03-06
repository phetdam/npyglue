/**
 * @file npy_views_test.cc
 * @author Derek Huang
 * @brief C++ program to test npygl views with NumPy arrays
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ios>
#include <iostream>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/python.hh"
#include "npygl/range_views.hh"

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
 * RAII manager to print a test header and footer and manage checks.
 */
class test_frame {
public:
  /**
   * Ctor.
   *
   * @param header Frame header, typically `NPYGL_PRETTY_FUNCTION_NAME`
   */
  test_frame(const char* header) noexcept : header_{header}
  {
    std::cout << "[ START ] " << header_ << ": " << std::endl;
  }

  /**
   * Dtor.
   */
  ~test_frame()
  {
    if (failed_)
      std::cout << "[ FAIL  ] " << header_ << ": failed " << failed_ <<
        " of " << total_tests() << std::endl;
    else
      std::cout << "[ OK    ] " << header_ << ": passed " << passed_ <<
        " of " << total_tests() << std::endl;
  }

  /**
   * Indicate an abort occurred.
   *
   * @param text Abort message text
   * @returns `false`
   */
  bool abort(const char* text)
  {
    failed_++;
    std::cout << "[ ABORT ] " << text << std::endl;
    return false;
  }

  /**
   * Indicate a test success.
   *
   * @param text Success message text
   * @returns `true`
   */
  bool success(const char* text)
  {
    passed_++;
    std::cout << "[ PASS  ] " << text << std::endl;
    return true;
  }

  /**
   * Total number of checks.
   */
  unsigned total_tests() const noexcept
  {
    return passed_ + failed_;
  }

private:
  const char* header_;
  unsigned passed_{};
  unsigned failed_{};
};

/**
 * Test modification of a NumPy array through flat views.
 *
 * @param np_random Python object representing `numpy.random.random`
 * @returns `true` on success, `false` on failure
 */
bool test_flat_views(PyObject* np_random)
{
  test_frame frame{NPYGL_PRETTY_FUNCTION_NAME};
  // create a random 3x4x2 tensor
  npygl::py_object shape{Py_BuildValue("(iii)", 3, 4, 2)};
  if (!shape)
    return frame.abort("failed to create tuple (3, 4, 2)");
  auto obj = npygl::py_call_one(np_random, shape);
  if (!obj)
    return frame.abort("failed to call np.random.random((3, 4, 2))");
  // sine + inverse sine functors
  auto sine = [](auto& v) { v = std::sin(v); };
  auto asine = [](auto& v) { v = std::asin(v); };
  // apply sine + inverse sine functors to NumPy array via view and print
  auto ar = obj.as<PyArrayObject>();
  npygl::ndarray_flat_view<double> view{ar};
  std::for_each(view.begin(), view.end(), sine);
  std::cout << "sine transform:\n" << ar << std::endl;
  std::for_each(view.begin(), view.end(), asine);
  std::cout << "inverse sine transform:\n" << ar << std::endl;
// TODO: since we aren't using testing/math.hh functions should just remove
#if NPYGL_HAS_CC_20
  // apply sine + inverse functors to NumPy array via span and print
  auto stl_view = npygl::make_span<double>(ar);
  std::ranges::for_each(stl_view, sine);
  std::cout << "span sine transform:\n" << ar << std::endl;
  std::ranges::for_each(stl_view, asine);
  std::cout << "span inverse sine transform:\n" << ar << std::endl;
#endif  // NPYGL_HAS_CC_20
  return frame.success("passed trig transform tests");
}

/**
 * Test modification of a NumPy array through 2D views.
 *
 * @param np_random Python object representing `numpy.random.random`
 * @returns `true` on success, `false` on failure
 */
bool test_2d_views(PyObject* np_random)
{
  test_frame frame{NPYGL_PRETTY_FUNCTION_NAME};
  // create a random 4x4 matrix
  npygl::py_object shape{Py_BuildValue("ii", 4, 4)};
  if (!shape)
    return frame.abort("failed to create tuple (4, 4)");
  auto mat = npygl::py_call_one(np_random, shape);
  if (!mat)
    return frame.abort("failed to call np.random.random((4, 4))");
  // print + enable boolalpha
  std::cout << mat << std::boolalpha << std::endl;
  // check that this is actually a 2D ndarray
  auto mat_ar = mat.as<PyArrayObject>();
  std::cout << "Is matrix? " << (PyArray_NDIM(mat_ar) == 2) << std::endl;
  // multiply diagonal by 2 and then halve diagonal using matrix view
  // note: this assumes C data ordering and double data type (the default)
  npygl::ndarray_2d_view<double> mat_view{mat_ar};
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
  return frame.success("passed transform tests");
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
  if (!test_flat_views(np_random) || !test_2d_views(np_random))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
