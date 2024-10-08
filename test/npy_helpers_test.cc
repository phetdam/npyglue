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
#include <tuple>

#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/python.hh"
#include "npygl/testing/math.hh"

namespace {

/**
 * Functor to check if a NumPy array type is one of the given C/C++ types.
 *
 * @tparam Ts... C/C++ types to check NumPy array type against
 */
template <typename... Ts>
struct ndarray_type_checker {};

/**
 * Partial specialization for a single type.
 *
 * @tparam T C/C++ type to check NumPy array type against
 */
template <typename T>
struct ndarray_type_checker<T> {
  bool operator()(PyArrayObject* ar) const noexcept
  {
    // traits exist
    if constexpr (npygl::has_npy_type_traits_v<T>)
      return npygl::is_type<T>(ar);
    // no traits, false
    else
      return false;
  }
};

/**
 * Partial specialization for more than one type.
 *
 * @tparam T C/C++ type to check NumPy array type against
 * @tparam Ts... Other C/C++ types to check NumPy array type against
 */
template <typename T, typename... Ts>
struct ndarray_type_checker<T, Ts...> {
  bool operator()(PyArrayObject* ar) const noexcept
  {
    return ndarray_type_checker<T>{}(ar) || ndarray_type_checker<Ts...>{}(ar);
  }
};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Ts... C/C++ types to check NumPy array type against
 */
template <typename... Ts>
struct ndarray_type_checker<std::tuple<Ts...>> : ndarray_type_checker<Ts...> {};

/**
 * Global to check if a NumPy array type corresponds to one of the C/C++ types.
 *
 * This provides a functional interface to the `ndarray_type_checker<Ts...>`.
 *
 * @tparam Ts... C/C++ types to check NumPy array type against
 */
template <typename... Ts>
inline constexpr ndarray_type_checker<Ts...> has_type;

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
  npygl::py_object shape{Py_BuildValue("(iii)", 3, 4, 2)};
  npygl::py_error_exit();
  // call np.random.random with the shape tuple
  auto res = npygl::py_call_one(np_random_random, shape);
  npygl::py_error_exit();
  // print NumPy array using repr() + enable boolalpha
  std::cout << res << std::boolalpha << std::endl;
  // check if NumPy array (it is)
  std::cout << "Is NumPy array? " << npygl::is_ndarray(res) << std::endl;
  // print additional info
  using target_types = std::tuple<
    int,
    double,
    float,
    std::complex<double>,
    unsigned int,
    std::complex<float>,
    long
  >;
  auto ar = res.as<PyArrayObject>();
  std::cout << "Is double type? " << npygl::is_type<double>(ar) << std::endl;
  std::cout << "Is behaved? " << PyArray_ISBEHAVED(ar) << std::endl;
  std::cout << "Has one of " << npygl::npy_typename_list<target_types>() <<
    ": " << has_type<target_types>(ar) << std::endl;
  return EXIT_SUCCESS;
}
