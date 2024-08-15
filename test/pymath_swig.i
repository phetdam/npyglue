/**
 * @file pymath_swig.i
 * @author Derek Huang
 * @brief SWIG C++ Python extension module for math functions
 * @copyright MIT License
 */

// module docstring
%define MODULE_DOCSTRING
"npyglue math functions SWIG test module.\n"
"\n"
"This C++ extension module contains math functions that operate on Python\n"
"sequences and return a new NumPy array, possibly with a specific data type."
%enddef  // MODULE_DOCSTRING

%module(docstring=MODULE_DOCSTRING) pymath_swig

%include "npygl/ndarray.i"
%include "npygl/testing/math.hh"

%{
#define SWIG_FILE_WITH_INIT

#include "npygl/npy_helpers.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/py_helpers.hh"
#include "npygl/testing/math.hh"
%}

// NumPy array API needs initialization during module load
%init %{
import_array();
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
  NPYGL_NPYDOC_PARAMETERS()
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS()
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
  NPYGL_NPYDOC_PARAMETERS()
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS()
  "numpy.ndarray"
);
%template(farray_double) npygl::testing::array_double<float>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS()
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS()
  "numpy.ndarray"
);
%template(unit_compress) npygl::testing::unit_compress<double>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS()
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS()
  "numpy.ndarray"
);
%template(funit_compress) npygl::testing::unit_compress<float>;

NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(double)
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(float)
