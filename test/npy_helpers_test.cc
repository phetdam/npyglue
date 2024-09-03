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

#include "npygl/features.h"
#include "npygl/npy_helpers.hh"
#include "npygl/py_helpers.hh"
#include "npygl/testing/math.hh"

namespace {

/**
 * Check if the NumPy array type corresponds to one of the C/C++ types.
 *
 * @tparam Ts... C/C++ types to check NumPy array type against
 *
 * @param arr NumPy array
 */
template <typename... Ts>
bool has_type(PyArrayObject* arr) noexcept
{
  static_assert(sizeof...(Ts), "parameter pack must have at least one type");
  // indicator to check if we have a type match
  bool match = (
    [arr]
    {
      // traits exist
      if constexpr (npygl::has_npy_type_traits_v<Ts>)
        return npygl::is_type<Ts>(arr);
      // no traits, false
      else
        return false;
    }()
    // short circuit if match is found
    ||
    ...
  );
  return match;
}

}  // namespace

int main()
{
  using npygl::testing::sine;
  using npygl::testing::asine;
  // initialize Python + print version
  npygl::py_instance python;
  std::cout << Py_GetVersion() << std::endl;
  // import NumPy C API
  npygl::npy_api_import(python);
  npygl::py_error_exit();
  // print the NumPy include directory
  std::cout << "NumPy include dir: " << npygl::npy_get_include() << std::endl;
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
  // print NumPy array using repr() + enable boolalpha
  std::cout << res << std::boolalpha << std::endl;
  // check if NumPy array (it is)
  std::cout << "Is NumPy array? " << npygl::is_ndarray(res) << std::endl;
  // print additional info
  auto ar = res.as<PyArrayObject>();
  std::cout << "Is double type? " << npygl::is_type<double>(ar) << std::endl;
  std::cout << "Is behaved? " << PyArray_ISBEHAVED(ar) << std::endl;
  std::cout << "Has one of " <<
    npygl::npy_typename_list<int, double, float>() << ": " <<
    has_type<int, double, float>(ar) << std::endl;
  // apply sine + inverse sine functions to NumPy array via view and print
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
  return EXIT_SUCCESS;
}
