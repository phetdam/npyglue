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

#include "npygl/py_helpers.hh"
#include "npygl/npy_helpers.hh"

namespace {
/**
 * Compute the sine of the view elements.
 *
 * @param view NumPy array view
 */
void sine(const npygl::ndarray_flat_view<double>& view) noexcept
{
  for (decltype(view.size()) i = 0; i < view.size(); i++)
    view[i] = std::sin(view[i]);
}

/**
 * Check if the NumPy array type corresponds to one of the C/C++ types.
 *
 * @tparam Ts... C/C++ types to check NumPy array type against
 *
 * @param arr NumPy array
 */
template <typename... Ts>
auto has_type(PyArrayObject* arr)
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
  // NB: on Windows, the embedded Python's program name is deduced to argv[0]
  // instead of to the Python interpreter used by the virtual environment. so
  // we have a workaround here such that if a venv virtual environment is being
  // used, the executable there will be set using Py_SetProgramName().
#ifdef _WIN32
  // get path to venv virtual environment root
  auto venv = std::getenv("VIRTUAL_ENV");
  // found, running in venv
  if (venv) {
    // program name must be in static storage
    static const auto progname = [venv]
    {
      // if capturing by copy, non-mutable lambda cannot modify capture
      auto path = venv;
      // write as wide chars
      std::wstringstream ss;
      while (*path)
        ss.put(ss.widen(*path++));
      // add relative path to python.exe
      ss << L"\\Scripts\\python.exe";
      return ss.str();
    }();
    // set path to the Python executable. deduction fails on Windows with venv
    // virtual envs so we must do this explicitly unfortunately
    Py_SetProgramName(progname.c_str());
  }
#endif  // _WIN32
  // initialize Python + print version
  npygl::py_instance python;
  std::cout << Py_GetVersion() << std::endl;
  // import NumPy C API
  npygl::npy_api_import(python);
  npygl::py_error_exit();
  // print the NumPy include directory
  std::cout << "NumPy include dir: " << npygl::npy_include_dir() << std::endl;
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
  // print NumPy array using repr() + add newline and enable boolalpha
  npygl::py_print(res);
  npygl::py_error_exit();
  std::cout << std::boolalpha << std::endl;
  // check if NumPy array (it is)
  std::cout << "Is NumPy array? " << npygl::is_ndarray(res) << std::endl;
  // print additional info
  auto ar = res.as<PyArrayObject>();
  std::cout << "Is double type? " << npygl::is_type<double>(ar) << std::endl;
  std::cout << "Is behaved? " << PyArray_ISBEHAVED(ar) << std::endl;
  std::cout << "Has one of " <<
    npygl::npy_typename_list<int, double, float>() << ": " <<
    has_type<int, double, float>(ar) << std::endl;
  // apply sine function to NumPy array and print it again
  sine(ar);
  npygl::py_print(res);
  npygl::py_error_exit();
  std::cout << std::endl;
  return EXIT_SUCCESS;
}
