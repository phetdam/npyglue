/**
 * @file pymath.cc
 * @author Derek Huang
 * @brief C++ Python extension module for math functions
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/ndarray.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/python.hh"
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
template <typename T, bool C = false>
npygl::py_object parse_ndarray(PyObject* args) noexcept
{
  // parse Python arguments as objects
  PyObject* objs[1];
  if (!npygl::parse_args(args, objs))
    return {};
  // create new NPY_ARRAY_DEFAULT array (empty on error)
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
 *
 * @param args Python argument tuple
 */
template <typename T>
PyObject* array_double(PyObject* args)
{
  using npygl::testing::array_double;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // double the values + release value back to Python
  // note: template keyword required to tell compiler as() is a member template
#if NPYGL_HAS_CC_20
  auto view = npygl::make_span<const T>(ar.template as<PyArrayObject>());
  auto res = array_double(view);
#else
  auto res = array_double<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // create NumPy vector and release (nullptr on error)
  return npygl::make_ndarray(std::move(res)).release();
}

NPYGL_PY_FUNC_DECLARE(
  array_double,
  "(ar)",
  "Double the incoming input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return array_double<double>(args);
}

NPYGL_PY_FUNC_DECLARE(
  farray_double,
  "(ar)",
  "Double the incoming input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to double\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return array_double<float>(args);
}

/**
 * Python wrapper template for `unit_compress`.
 *
 * @tparam T Output element type
 *
 * @param args Python argument tuple
 */
template <typename T>
PyObject* unit_compress(PyObject* args)
{
  using npygl::testing::unit_compress;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // compress + release value back to Python
#if NPYGL_HAS_CC_20
  auto view = npygl::make_span<const T>(ar.template as<PyArrayObject>());
  auto res = unit_compress(view);
#else
  auto res = unit_compress<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // create NumPy vector and release (nullptr on error)
  return npygl::make_ndarray(std::move(res)).release();
}

NPYGL_PY_FUNC_DECLARE(
  unit_compress,
  "(ar)",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return unit_compress<double>(args);
}

NPYGL_PY_FUNC_DECLARE(
  funit_compress,
  "(ar)",
  "Compress incoming values into the range [-1, 1].\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to compress\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return unit_compress<float>(args);
}

/**
 * Python wrapper template for `sine`.
 *
 * @tparam T Output element type
 *
 * @param args Python argument tuple
 */
template <typename T>
PyObject* sine(PyObject* args)
{
  using npygl::testing::sine;
  // create output array
  auto ar = parse_ndarray<T>(args);
  if (!ar)
    return nullptr;
  // compute sine
#if NPYGL_HAS_CC_20
  auto res = sine(npygl::make_span<const T>(ar.template as<PyArrayObject>()));
#else
  auto res = sine<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // create NumPy vector and release (nullptr on error)
  return npygl::make_ndarray(std::move(res)).release();
}

NPYGL_PY_FUNC_DECLARE(
  sine,
  "(ar)",
  "Apply the sine function to the input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return sine<double>(args);
}

NPYGL_PY_FUNC_DECLARE(
  fsine,
  "(ar)",
  "Apply the sine function to the input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to apply sine to\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, args) noexcept
{
  return sine<float>(args);
}

/**
 * Python wrapper template for `norm1`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 *
 * @param arg Python object
 */
template <typename T>
PyObject* norm1(PyObject* arg) noexcept
{
  using npygl::testing::norm1;
  // input array (if possible, no copy is made)
  auto ar = npygl::make_ndarray<T>(arg, NPY_ARRAY_IN_ARRAY);
  if (!ar)
    return nullptr;
  // compute 1-norm
#if NPYGL_HAS_CC_20
  auto res = norm1(npygl::make_span<const T>(ar.template as<PyArrayObject>()));
#else
  auto res = norm1<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // could use npygl::py_object{res}.release() too
  return PyFloat_FromDouble(res);
}

NPYGL_PY_FUNC_DECLARE(
  norm1,
  "(ar)",
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, arg) noexcept
{
  return norm1<double>(arg);
}

NPYGL_PY_FUNC_DECLARE(
  fnorm1,
  "(ar)",
  "Compute the 1-norm of the flattened input array.\n"
  "\n"
  "The returned NumPy array will have ``dtype=float32``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 1-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, arg) noexcept
{
  return norm1<float>(arg);
}

/**
 * Python wrapper template for `norm2`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 *
 * @param arg Python argument
 */
template <typename T>
PyObject* norm2(PyObject* arg) noexcept
{
  using npygl::testing::norm2;
  // input array (if possible, no copy is made)
  auto ar = npygl::make_ndarray<T>(arg, NPY_ARRAY_IN_ARRAY);
  if (!ar)
    return nullptr;
  // compute 2-norm
#if NPYGL_HAS_CC_20
  auto res = norm2(npygl::make_span<const T>(ar.template as<PyArrayObject>()));
#else
  auto res = norm2<T>(ar.template as<PyArrayObject>());
#endif  // !NPYGL_HAS_CC_20
  // could use npygl::py_object{res}.release() too
  return PyFloat_FromDouble(res);
}

NPYGL_PY_FUNC_DECLARE(
  norm2,
  "(ar)",
  "Compute the 2-norm of the flattened input array.\n"
  "\n"
  "For matrices this would correspond to the Frobenius norm.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "ar : collections.Sequence\n"
  "    Input sequence of numeric values to take 2-norm of\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, arg) noexcept
{
  return norm2<double>(arg);
}

NPYGL_PY_FUNC_DECLARE(
  fnorm2,
  "(ar)",
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
  "numpy.ndarray",
  self, arg) noexcept
{
  return norm2<float>(arg);
}

/**
 * Python wrapper template for `inner`.
 *
 * @note Only intended to work with floating point types.
 *
 * @tparam T Output element type
 *
 * @param args Python argument tuple
 */
template <typename T>
PyObject* inner(PyObject* args) noexcept
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
  auto v1 = npygl::make_span<const T>(ar1.template as<PyArrayObject>());
  auto v2 = npygl::make_span<const T>(ar2.template as<PyArrayObject>());
#else
  npygl::ndarray_flat_view<const T> v1{ar1.template as<PyArrayObject>()};
  npygl::ndarray_flat_view<const T> v2{ar2.template as<PyArrayObject>()};
#endif  // !NPYGL_HAS_CC_20
  // sanity check for sizes
  if (v1.size() != v2.size()) {
    PyErr_SetString(PyExc_RuntimeError, "v1 and v2 must have the same size");
    return nullptr;
  }
  return PyFloat_FromDouble(inner(v1, v2));
}

NPYGL_PY_FUNC_DECLARE(
  inner,
  "(v1, v2)",
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
  "float",
  self, args) noexcept
{
  return inner<double>(args);
}

NPYGL_PY_FUNC_DECLARE(
  finner,
  "(v1, v2)",
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
  "float",
  self, args) noexcept
{
  return inner<float>(args);
}

/**
 * Helper for casting a scoped or unscoped enum to its underlying type.
 *
 * @tparam E Enum type
 *
 * @param v Enum value to cast
 */
template <typename E, typename = std::enable_if_t<std::is_enum_v<E>>>
constexpr auto as_underlying(E v) noexcept
{
  return static_cast<std::underlying_type_t<E>>(v);
}

/**
 * Python wrapper for `uniform`.
 *
 * @tparam T Output element type, either `float`, `double`, or `long double`
 *
 * @param args Python argument tuple
 * @param kwargs Python argument dict
 */
template <typename T>
PyObject* uniform(PyObject* args, PyObject* kwargs) noexcept
{
  using npygl::testing::optional_seed_type;
  using npygl::testing::rngs;
  using npygl::testing::uniform;
  // number of values, incoming type, seed value
  Py_ssize_t n;
  auto type = as_underlying(rngs::mersenne);
  int seed = INT_MAX;
  // kwarg names
  const char* kws[] = {"type", "seed"};
  // parse arguments
  if (!npygl::parse_args(args, std::tie(n), kws, kwargs, std::tie(type, seed)))
    return nullptr;
  // type has to be valid
  switch (type) {
    case as_underlying(rngs::mersenne):
    case as_underlying(rngs::mersenne64):
    case as_underlying(rngs::ranlux48):
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "unknown PRNG type value");
      return nullptr;
  }
  // seed has to be nonnegative
  if (seed < 0) {
    PyErr_SetString(PyExc_ValueError, "seed value must be nonnegative");
    return nullptr;
  }
  // compute random vector and return
  auto res = uniform<T>(
    static_cast<std::size_t>(n),
    static_cast<rngs>(type),
    (seed == INT_MAX) ? optional_seed_type{} : optional_seed_type{seed}
  );
  return npygl::make_ndarray(std::move(res)).release();
}

NPYGL_PY_KWFUNC_DECLARE(
  uniform,
  "(n, type=PRNG_MERSENNE, seed=None)",
  "Return a 1D NumPy array of randomly generated values.\n"
  "\n"
  "The memory backing the returned array is held by a std::vector<double>.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "n : int\n"
  "    Number of elements to generate\n"
  "type : rngs, default=PRNG_MERSENNE\n"
  "    PRNG generator to use\n"
  "seed : int, default=None\n"
  "    Seed value to use\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray\n"
  "    Array shape ``(n,)`` of values",
  self, args, kwargs) noexcept
{
  return uniform<double>(args, kwargs);
}

NPYGL_PY_KWFUNC_DECLARE(
  funiform,
  "(n, type=PRNG_MERSENNE, seed=None)",
  "Return a 1D NumPy array of randomly generated values.\n"
  "\n"
  "The memory backing the returned array is held by a std::vector<float>.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "n : int\n"
  "    Number of elements to generate\n"
  "type : rngs, default=PRNG_MERSENNE\n"
  "    PRNG generator to use\n"
  "seed : int, default=None\n"
  "    Seed value to use\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray\n"
  "    Array shape ``(n,)`` of values",
  self, args, kwargs) noexcept
{
  return uniform<float>(args, kwargs);
}

// module method table
PyMethodDef mod_methods[] = {
  // TODO: consider using METH_O for single-argument array functions
  NPYGL_PY_FUNC_METHOD_DEF(array_double, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(farray_double, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(unit_compress, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(funit_compress, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(sine, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(fsine, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(norm1, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(fnorm1, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(norm2, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(fnorm2, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(inner, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(finner, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(uniform, METH_VARARGS | METH_KEYWORDS),
  NPYGL_PY_FUNC_METHOD_DEF(funiform, METH_VARARGS | METH_KEYWORDS),
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

/**
 * Helper macro to add an enum value as an integer constant.
 *
 * @param mod Module object
 * @param name Name to add enum value under
 * @param value Enum value
 * @returns `true` on success, `false` on error
 */
#define ADD_ENUM(mod, name, value) \
  !PyModule_AddIntConstant(mod, name, as_underlying(value))

// visible module initialization function
PyMODINIT_FUNC
NPYGL_CONCAT(PyInit_, MODULE_NAME)()
{
  using npygl::testing::rngs;
  // import NumPy C API and create module
  import_array();
  npygl::py_object mod{PyModule_Create(&mod_def)};
  if (!mod)
    return nullptr;
  // add PRNG selection constants
  if (!ADD_ENUM(mod, "PRNG_MERSENNE", rngs::mersenne)) return nullptr;
  if (!ADD_ENUM(mod, "PRNG_MERSENNE64", rngs::mersenne64)) return nullptr;
  if (!ADD_ENUM(mod, "PRNG_RANLUX48", rngs::ranlux48)) return nullptr;
  // release module object
  return mod.release();
}
