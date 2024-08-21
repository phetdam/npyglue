/**
 * @file pymath_swig.i
 * @author Derek Huang
 * @brief SWIG C++ Python extension module for math functions
 * @copyright MIT License
 */

// C++ dialect string. this is conditionally defined based on the -D macros
// that are passed to SWIG when it is run; we use these to specify the C++
// standard that the generated C++ code will likely be generated for.
//
// in particular, the NPYGL_SWIG_CC_20 macro is used to determine whether or
// not we want SWIG to generate the wrapper code in a way that enables the
// C++20 features/overloads in the processed headers.
//
#if defined(NPYGL_SWIG_CC_20)
%define NPYGL_CC_STRING
"C++20"
%enddef  // NPYGL_CC_STRING
#else
%define NPYGL_CC_STRING
"C++"
%enddef  // NPYGL_CC_STRING
#endif  // !defined(NPYGL_SWIG_CC_20)

// module docstring
%define MODULE_DOCSTRING
"npyglue math functions " NPYGL_CC_STRING " SWIG test module.\n"
"\n"
"This C++ extension module contains math functions that operate on Python\n"
"sequences and return a new NumPy array, possibly with a specific data type."
%enddef  // MODULE_DOCSTRING

// module name depends on C++ standard
// note: originally we tried to define a MODULE_NAME macro but CMake ended up
// incorrectly generating the Python wrapper as MODULE_NAME.py for the Visual
// Studio generators. Makefiles were fine so maybe it's an escaping issue
#if defined(NPYGL_SWIG_CC_20)
%module(docstring=MODULE_DOCSTRING) pymath_swig_cc20
#else
%module(docstring=MODULE_DOCSTRING) pymath_swig
#endif  // !defined(NPYGL_SWIG_CC_20)

%include "npygl/ndarray.i"
%include "npygl/testing/math.hh"

%{
#define SWIG_FILE_WITH_INIT

#include "npygl/features.h"
#include "npygl/npy_helpers.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/py_helpers.hh"
#include "npygl/testing/math.hh"

#if NPYGL_HAS_CC_20
#include <span>
#endif  // !NPYGL_HAS_CC_20
%}

// NumPy array API needs initialization during module load
%init %{
import_array();
%}

// instantiate different versions of array_double and unit_compress
// note: SWIG can understand use of namespaces but we are explicit here
#if defined(NPYGL_SWIG_CC_20)
NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAPS(double)
NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAPS(float)
#else
NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAPS(double)
NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAPS(float)
#endif  // !defined(NPYGL_SWIG_CC_20)

// SWIG 4.2 changes the way that Python function annotation works. instead of
// passing -py3 to enable Python 3 function annotations, we use %feature
#if SWIG_VERSION >= 0x040200
%feature("python:annotations", "c");
#endif  // SWIG_VERSION >= 0X040200

// note: %feature not tagging correctly when namespaces are involved. we just
// globally apply the docstring each time and overwrite it with the next
%feature(
  "autodoc",
  "Double the incoming input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(array_double) npygl::testing::array_double<double>;

%feature(
  "autodoc",
  "Double the incoming input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(farray_double) npygl::testing::array_double<float>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(unit_compress) npygl::testing::unit_compress<double>;

%feature(
  "autodoc",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(funit_compress) npygl::testing::unit_compress<float>;

%feature(
  "autodoc",
  "Apply the sine function to the input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(sine) npygl::testing::sine<double>;

%feature(
  "autodoc",
  "Apply the sine function to the input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
%template(fsine) npygl::testing::sine<float>;

#if defined(NPYGL_SWIG_CC_20)
NPYGL_CLEAR_STD_SPAN_TYPEMAPS(double)
NPYGL_CLEAR_STD_SPAN_TYPEMAPS(float)
#else
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(double)
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(float)
#endif  // !defined(NPYGL_SWIG_CC_20)
