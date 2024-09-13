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

// uppercase enum members to follow Python convention
%rename("%(upper)s", %$isenumitem) "";
// TODO: not sure how to remove enum class name. we just rename for now
%rename(PRNG) rng_type;

%include "npygl/ndarray.i"
%include "npygl/testing/math.hh"

%{
#define SWIG_FILE_WITH_INIT

#include <stdexcept>

#include "npygl/features.h"
#include "npygl/ndarray.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/python.hh"
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

// input-only typemaps (this time, test application to a single name)
#if defined(NPYGL_SWIG_CC_20)
NPYGL_APPLY_STD_SPAN_IN_TYPEMAP(double, view)
NPYGL_APPLY_STD_SPAN_IN_TYPEMAP(float, view)
#else
NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAP(double, view)
NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAP(float, view)
#endif  // !defined(NPYGL_SWIG_CC_20)

%feature(
  "autodoc",
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(norm1) npygl::testing::norm1<double>;

%feature(
  "autodoc",
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(fnorm1) npygl::testing::norm1<float>;

%feature(
  "autodoc",
  "Compute the 2-norm of the flattened input array.\n"
  "\n"
  "For matrices this would correspond to the Frobenius norm.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to take 2-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(norm2) npygl::testing::norm2<double>;

%feature(
  "autodoc",
  "Compute the 2-norm of the flattened input array.\n"
  "\n"
  "For matrices this would correspond to the Frobenius norm.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "view : collections.Sequence\n"
  "    Input sequence of numeric values to take 2-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(fnorm2) npygl::testing::norm2<float>;

#if defined(NPYGL_SWIG_CC_20)
NPYGL_CLEAR_STD_SPAN_TYPEMAP(double, view)
NPYGL_CLEAR_STD_SPAN_TYPEMAP(float, view)
#else
NPYGL_CLEAR_FLAT_VIEW_TYPEMAP(double, view)
NPYGL_CLEAR_FLAT_VIEW_TYPEMAP(float, view)
#endif  // !defined(NPYGL_SWIG_CC_20)

// input-only typemaps
#if defined(NPYGL_SWIG_CC_20)
NPYGL_APPLY_STD_SPAN_IN_TYPEMAPS(double)
NPYGL_APPLY_STD_SPAN_IN_TYPEMAPS(float)
#else
NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAPS(double)
NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAPS(float)
#endif  // !defined(NPYGL_SWIG_CC_20)

// inner() has an internal assert() but does not do any input validation.
// therefore, we must write our own wrapper for that validates and throws.
// we therefore also need to temporarily use the exception handler so that the
// exception is caught instead of causing the Python interpreter to crash.

NPYGL_ENABLE_EXCEPTION_HANDLER

%inline %{
namespace npygl {
namespace testing {

// like in math.hh we need to enable the C++20 overloads explicitly by defining
// NPYGL_SWIG_CC_20 but during real compilation we allow C++20 detection via
// NPYGL_HAS_CC_20 to ensure the correct overloads are available

/**
 * Wrapper for `inner()` that throws on error and requires the same types.
 */
template <typename T>
#if defined(NPYGL_SWIG_CC_20) || NPYGL_HAS_CC_20
inline T py_inner(std::span<T> v1, std::span<T> v2)
#else
inline T py_inner(ndarray_flat_view<T> v1, ndarray_flat_view<T> v2)
#endif  // !defined(NPYGL_SWIG_CC_20) && !NPYGL_HAS_CC_20
{
  if (v1.size() != v2.size())
    throw std::runtime_error{"v1 and v2 must have the same number of elements"};
  return inner(v1, v2);
}

}  // testing
}  // namespace npygl
%}

%feature(
  "autodoc",
  "Compute the vector inner product.\n"
  "\n"
  ".. note::\n"
  "\n"
  "   No error is raised if the ndarrays have the same size but different\n"
  "   shapes as they are both treated as flat vectors.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "v1 : collections.Sequence\n"
  "    Input sequence of numeric values to treat as a vector\n"
  "v2 : collections.Sequence\n"
  "    Input sequence of numeric values to treat as a vector\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(inner) npygl::testing::py_inner<double>;

%feature(
  "autodoc",
  "Compute the vector inner product.\n"
  "\n"
  "If NumPy arrays are used for input they should have ``dtype=float32``.\n"
  "\n"
  ".. note::\n"
  "\n"
  "   No error is raised if the ndarrays have the same size but different\n"
  "   shapes as they are both treated as flat vectors.\n"
  "\n"
  ".. note::\n"
  "\n"
  "   The return value is cast to double precision (float64) from single\n"
  "   precision (float32) internally so the result may differ from inner's.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "v1 : collections.Sequence\n"
  "    Input sequence of numeric values to treat as a vector\n"
  "v2 : collections.Sequence\n"
  "    Input sequence of numeric values to treat as a vector\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "float"
);
%template(finner) npygl::testing::py_inner<float>;

// demonstration of wrapping a std::vector<T, A> into a NumPy array
// TODO: will extend to feature Eigen::MatrixX[df] as well
%{
#include <memory>
#include <vector>
%}

// note: we don't have std::optional<T> typemaps so we use a simplifying
// wrapper but purposefully keep the ugly return type so we can demonstrate
// that SWIG doest not understand template default arguments
%inline %{
namespace npygl {
namespace testing {

/**
 * Return a vector of random values drawn from `[0, 1]`.
 *
 * @note SWIG's support for `auto` in the 4.0.x series is rather limited so we
 *  still have to spell out the return type in its full glory.
 */
template <typename T>
inline std::vector<T> py_uniform_vector(
  std::size_t n, rng_type type = rng_type::mersenne)
{
  return uniform_vector<T>(n, type);
}

}  // namespace testing
}  // namespace npygl
%}

// note: if we were using std::vector<T, A> as a return type we would have to
// specify all the template arguments as SWIG doesn't understand defaults.
// parentheses would also be needed as otherwise something like
// std::vector<double, std::allocator<double>> is treated as two macro args.
NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(std::vector<double>);
NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(std::vector<float>);

// note: as mentioned previously, SWIG does understand namespaces
namespace npygl::testing {

%feature(
  "autodoc",
  "Return a 1D NumPy array of randomly generated values.\n"
  "\n"
  "The memory backing the returned array is held by a std::vector<double>.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "n : int\n"
  "    Number of elements to generate\n"
  "type : rng_type, default=PRNG_MERSENNE\n"
  "    PRNG generator to use\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray\n"
  "    Array shape ``(n,)`` of values"
);
%template(uniform_vector) py_uniform_vector<double>;

%feature(
  "autodoc",
  "Return a 1D NumPy array of randomly generated values.\n"
  "\n"
  "The memory backing the returned array is held by a std::vector<float>.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "n : int\n"
  "    Number of elements to generate\n"
  "type : rng_type, default=PRNG_MERSENNE\n"
  "    PRNG generator to use\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray\n"
  "    Array shape ``(n,)`` of values"
);
%template(funiform_vector) py_uniform_vector<float>;

}  // namespace npygl::testing

NPYGL_CLEAR_NDARRAY_OUT_TYPEMAP(std::vector<double>);
NPYGL_CLEAR_NDARRAY_OUT_TYPEMAP(std::vector<float>);

NPYGL_DISABLE_EXCEPTION_HANDLER

// clear
#if defined(NPYGL_SWIG_CC_20)
NPYGL_CLEAR_STD_SPAN_TYPEMAPS(double)
NPYGL_CLEAR_STD_SPAN_TYPEMAPS(float)
#else
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(double)
NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(float)
#endif  // !defined(NPYGL_SWIG_CC_20)
