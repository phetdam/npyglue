/**
 * @file pymath.cc
 * @author Derek Huang
 * @brief C++ Python extension module for math functions
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/npy_helpers.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/py_helpers.hh"
#include "npygl/testing/math.hh"

// module name macro. for C++20 we build under a different name
#if NPYGL_HAS_CC_20
#define MODULE_NAME pymath_cc20
#else
#define MODULE_NAME pymath
#endif  // !NPYGL_HAS_CC_20

// C++ version string
#if NPYGL_HAS_CC_20
#define CC_STRING "C++20"
#elif NPYGL_HAS_CC_17
#define CC_STRING "C++17"
#else
#define CC_STRING "C++"
#endif  // !NPYGL_HAS_CC_20 && !NPYGL_HAS_CC_17

// internal linkage to follow Python extension module conventions
namespace {

/**
 * Parse a single argument from Python arguments into a NumPy array.
 *
 * On error the returned `py_object` is empty and an exception is set. In all
 * cases the returned NumPy array object has `NPY_ARRAY_DEFAULT` flags.
 *
 * @tparam T NumPy array element type
 * @tparam C `true` to copy, `false` not to copy if possible
 *
 * @param args Python arguments
 */
template <typename T, bool C = true>
npygl::py_object parse_ndarray(PyObject* args) noexcept
{
  // parse Python arguments as objects
  PyObject* objs[1];
  if (!npygl::parse_args(args, objs))
    return {};
  // create output array (empty on error)
  if constexpr (C)
    return npygl::make_ndarray<T>(*objs);
  // create new array only if necessary; increments refcount if already array
  else
    return npygl::make_ndarray<T>(*objs, NPY_ARRAY_DEFAULT);
}

/**
 * Python wrapper template for `array_double`.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* array_double(PyObject* NPYGL_UNUSED(self), PyObject* args) noexcept
{
  using npygl::testing::array_double;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // double the values + release value back to Python
  // note: template keyword required to tell compiler as() is a member template
#if NPYGL_HAS_CC_20
  array_double(npygl::make_span<T>(ar.template as<PyArrayObject>()));
#else
  array_double<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  return ar.release();
}

/**
 * Python wrapper template for `unit_compress`.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* unit_compress(PyObject* NPYGL_UNUSED(self), PyObject* args) noexcept
{
  using npygl::testing::unit_compress;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // compress + release value back to Python
#if NPYGL_HAS_CC_20
  unit_compress(npygl::make_span<T>(ar.template as<PyArrayObject>()));
#else
  unit_compress<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  return ar.release();
}

/**
 * Python wrapper template for `sine`.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* sine(PyObject* NPYGL_UNUSED(self), PyObject* args) noexcept
{
  using npygl::testing::sine;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // compute sine
#if NPYGL_HAS_CC_20
  sine(npygl::make_span<T>(ar.template as<PyArrayObject>()));
#else
  sine<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  return ar.release();
}

/**
 * Python wrapper template for `norm1`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* norm1(PyObject* NPYGL_UNUSED(self), PyObject* arg) noexcept
{
  using npygl::testing::norm1;
  // input array (if possible, no copy is made)
  auto ar = npygl::make_ndarray<T>(arg, NPY_ARRAY_IN_ARRAY);
  if (!ar)
    return nullptr;
  // compute 1-norm
#if NPYGL_HAS_CC_20
  auto res = norm1(npygl::make_span<T>(ar.template as<PyArrayObject>()));
#else
  auto res = norm1<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // could use npygl::py_object{res}.release() too
  return PyFloat_FromDouble(res);
}

/**
 * Python wrapper template for `norm2`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* norm2(PyObject* NPYGL_UNUSED(self), PyObject* arg) noexcept
{
  using npygl::testing::norm2;
  // input array (if possible, no copy is made)
  auto ar = npygl::make_ndarray<T>(arg, NPY_ARRAY_IN_ARRAY);
  if (!ar)
    return nullptr;
  // compute 2-norm
#if NPYGL_HAS_CC_20
  auto res = norm2(npygl::make_span<T>(ar.template as<PyArrayObject>()));
#else
  auto res = norm2<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // could use npygl::py_object{res}.release() too
  return PyFloat_FromDouble(res);
}

/**
 * Python wrapper template for `inner`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 */
template <typename T>
PyObject* inner(PyObject* NPYGL_UNUSED(self), PyObject* args) noexcept
{
  using npygl::testing::inner;
  // parse input objects
  PyObject* objs[2];
  if (!npygl::parse_args(args, objs))
    return nullptr;
  // create NumPy arrays
  auto ar1 = npygl::make_ndarray<T>(objs[0], NPY_ARRAY_IN_ARRAY);
  if (!ar1)
    return nullptr;
  auto ar2 = npygl::make_ndarray<T>(objs[1], NPY_ARRAY_IN_ARRAY);
  if (!ar2)
    return nullptr;
  // get array views + return inner product
#if NPYGL_HAS_CC_20
  auto v1 = npygl::make_span<T>(ar1.template as<PyArrayObject>());
  auto v2 = npygl::make_span<T>(ar2.template as<PyArrayObject>());
#else
  npygl::ndarray_flat_view<T> v1{ar1.template as<PyArrayObject>()};
  npygl::ndarray_flat_view<T> v2{ar2.template as<PyArrayObject>()};
#endif  // !NPYGL_HAS_CC_20
  // sanity check for sizes
  if (v1.size() != v2.size()) {
    PyErr_SetString(PyExc_RuntimeError, "v1 and v2 must have the same size");
    return nullptr;
  }
  return PyFloat_FromDouble(inner(v1, v2));
}

// wrapper method docstrings
PyDoc_STRVAR(
  array_double_doc,
  "array_double(ar)\n"
  NPYGL_CLINIC_MARKER
  "Double the incoming input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  farray_double_doc,
  "farray_double(ar)\n"
  NPYGL_CLINIC_MARKER
  "Double the incoming input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  unit_compress_doc,
  "unit_compress(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  funit_compress_doc,
  "funit_compress(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  sine_doc,
  "sine(ar)\n"
  NPYGL_CLINIC_MARKER
  "Apply the sine function to the input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  fsine_doc,
  "fsine(ar)\n"
  NPYGL_CLINIC_MARKER
  "Apply the sine function to the input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  norm1_doc,
  "norm1(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  fnorm1_doc,
  "fnorm1(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  norm2_doc,
  "norm2(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compute the 2-norm of the flattened input array.\n"
  "\n"
  "For matrices this would correspond to the Frobenius norm.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 2-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  fnorm2_doc,
  "fnorm2(ar)\n"
  NPYGL_CLINIC_MARKER
  "Compute the 2-norm of the flattened input array.\n"
  "\n"
  "For matrices this would correspond to the Frobenius norm.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 2-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray"
);
PyDoc_STRVAR(
  inner_doc,
  "inner(v1, v2)\n"
  NPYGL_CLINIC_MARKER
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
PyDoc_STRVAR(
  finner_doc,
  "finner(v1, v2)\n"
  NPYGL_CLINIC_MARKER
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

// module method table
PyMethodDef mod_methods[] = {
  // TODO: consider using METH_O for single-argument array functions
  {"array_double", array_double<double>, METH_VARARGS, array_double_doc},
  {"farray_double", array_double<float>, METH_VARARGS, farray_double_doc},
  {"unit_compress", unit_compress<double>, METH_VARARGS, unit_compress_doc},
  {"funit_compress", unit_compress<float>, METH_VARARGS, funit_compress_doc},
  {"sine", sine<double>, METH_VARARGS, sine_doc},
  {"fsine", sine<float>, METH_VARARGS, fsine_doc},
  {"norm1", norm1<double>, METH_O, norm1_doc},
  {"fnorm1", norm1<float>, METH_O, fnorm1_doc},
  {"norm2", norm2<double>, METH_O, norm2_doc},
  {"fnorm2", norm2<float>, METH_O, fnorm2_doc},
  {"inner", inner<double>, METH_VARARGS, inner_doc},
  {"finner", inner<float>, METH_VARARGS, finner_doc},
  {}  // zero-initialized sentinel member
};

// module docstring
PyDoc_STRVAR(
  mod_doc,
  "npyglue math functions " CC_STRING " hand-wrapped test module.\n"
  "\n"
  "This C++ extension module contains math functions that operate on Python\n"
  "sequences and return a new NumPy array, possibly with a specific data type."
);

// module definition struct
PyModuleDef mod_def = {
  PyModuleDef_HEAD_INIT,
  NPYGL_STRINGIFY(MODULE_NAME),
  mod_doc,
  -1,
  mod_methods
};

}  // namespace

// visible module initialization function
PyMODINIT_FUNC
NPYGL_CONCAT(PyInit_, MODULE_NAME)()
{
  // import NumPy C API and create module
  import_array();
  return PyModule_Create(&mod_def);
}
