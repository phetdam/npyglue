/**
 * @file npygl_ndarray_test.i
 * @author Derek Huang
 * @brief SWIG C++ test module for testing the ndarray.i SWIG helpers
 * @copyright MIT License
 */

%module npygl_ndarray_test

%include "npygl/ndarray.i"

%{
#include "npygl/npy_helpers.hh"  // include <numpy/ndarrayobject.h>
#include "npygl/py_helpers.hh"
%}

%init %{
import_array();
%}

// using %inline directive so we don't have to declare twice
%inline %{
/**
 * Silly function that simply doubles its argument.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void array_double(npygl::ndarray_flat_view<T> view) noexcept
{
  for (auto& v : view)
    v = 2 * v;
}
%}
