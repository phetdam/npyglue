/**
 * @file pymath.cc
 * @author Derek Huang
 * @brief C++ Python extension module for math functions
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

// static linkage to follow Python extension module conventions
namespace {

/**
 * Parse a single argument from a list of Python arguments into a NumPy array.
 *
 * On error the returned `py_object` is empty and an exception is set.
 *
 * @tparam T NumPy array element type
 *
 * @param args Python arguments
 */
template <typename T>
npygl::py_object parse_ndarray(PyObject* args) noexcept
{
  // Python input argument
  PyObject* in;
  // parse arguments
  if (!PyArg_ParseTuple(args, "O", &in))
    return {};
  // create output array (empty on error)
  return npygl::make_ndarray<T>(in);
}

// TODO: add Python docstring
template <typename T>
PyObject* array_double(PyObject* /*self*/, PyObject* args) noexcept
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

// TODO: add Python docstring
template <typename T>
PyObject* unit_compress(PyObject* /*self*/, PyObject* args) noexcept
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

// module method table
PyMethodDef mod_methods[] = {
  // TODO: add docstrings
  {"array_double", array_double<double>, METH_VARARGS, "nothing for now"},
  {"farray_double", array_double<float>, METH_VARARGS, "nothing for now"},
  {"unit_compress", unit_compress<double>, METH_VARARGS, "nothing for now"},
  {"funit_compress", unit_compress<float>, METH_VARARGS, "nothing for now"},
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
