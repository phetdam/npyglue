/**
 * @file npygl_math.i
 * @author Derek Huang
 * @brief SWIG C++ module with math functions testing the ndarray.i helpers
 * @copyright MIT License
 */

// module docstring
%define MODULE_DOCSTRING
"npyglue math functions SWIG test module.\n"
"\n"
"This C++ extension module contains math functions that operate on Python\n"
"sequences and return a new NumPy array, possibly with a specific data type."
%enddef

%module(docstring=MODULE_DOCSTRING) npygl_math

%include "npygl/ndarray.i"

%{
#define SWIG_FILE_WITH_INIT

#include <algorithm>

#include "npygl/npy_helpers.hh"  // include <numpy/ndarrayobject.h>
#include "npygl/py_helpers.hh"
%}

%init %{
import_array();
%}

// using %inline directive so we don't have to declare twice
%inline %{
namespace npygl {
namespace testing {

/**
 * Function that simply doubles its argument.
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

/**
 * Function that compresses the values of the argument to the unit circle.
 *
 * In other words, all the values will fall in `[-1, 1]`.
 *
 * @tparam T type
 *
 * @param view NumPy array view
 */
template <typename T>
void unit_compress(npygl::ndarray_flat_view<T> view) noexcept
{
  auto radius = *std::max_element(view.begin(), view.end());
  std::for_each(view.begin(), view.end(), [&radius](auto& x) { x /= radius; });
}

}  // namespace testing
}  // namespace npygl
%}

// instantiate different versions of array_double and unit_compress
// note: SWIG can understand use of namespaces but we are explicit here
NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAP(double)
NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAP(float)

// note: %feature not tagging correctly when namespaces are involved. we just
// globally apply the docstring each time and overwrite it with the next
%feature(
  "autodoc",
  "Double the incoming input array.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray"
);
%template(array_double) npygl::testing::array_double<double>;

// TODO: define SWIG macros to make writing the NumPy docstring easier
%feature(
  "autodoc",
  "Double the incoming input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray"
);
%template(farray_double) npygl::testing::array_double<float>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray"
);
%template(unit_compress) npygl::testing::unit_compress<double>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  "Returns\n"
  "-------\n"
  "numpy.ndarray"
);
%template(funit_compress) npygl::testing::unit_compress<float>;

NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(double)
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(float)
