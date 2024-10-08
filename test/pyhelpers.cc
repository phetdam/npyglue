/**
 * @file pyhelpers.cc
 * @author Derek Huang
 * @brief C++ Python extension module for testing C++ Python helpers
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

#include "npygl/common.h"
#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/python.hh"

#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO

// allow compilation without NumPy headers
#if NPYGL_HAS_NUMPY
#include "npygl/ndarray.hh"  // includes <numpy/ndarrayobject.h>
#endif  // NPYGL_HAS_NUMPY

// module name
#define MODULE_NAME pyhelpers

// internal linkage to follow Python extension module conventions
namespace {

/**
 * Test function template for `parse_args`.
 *
 * For each of the passed objects `o`, a tuple of `repr(type(o))` is returned.
 * If there is only one object then no exterior tuple is included.
 *
 * @tparam N Number of Python arguments to accept
 *
 * @param args Python argument tuple
 */
template <std::size_t N>
PyObject* parse_args(PyObject* args) noexcept
{
  // parse the given number of Python objects
  PyObject* objs[N];
  if (!npygl::parse_args(args, objs))
    return nullptr;
  // get string representations for each of the type objects. this is the same
  // as calling repr(type(o)) for each of the Python objects
  npygl::py_object rs[N];
  for (decltype(N) i = 0; i < N; i++)
    if (!(rs[i] = npygl::py_repr(Py_TYPE(objs[i]))))
      return nullptr;
  // create tuple from the repr() results and return
  return npygl::py_object{rs}.release();
}

NPYGL_PY_FUNC_DECLARE(
  parse_args_1,
  "(o)",
  "Return the ``repr(type(o))`` of the input ``o``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : object\n"
  "    Input object\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str",
  self, args) noexcept
{
  return parse_args<1>(args);
}

NPYGL_PY_FUNC_DECLARE(
  parse_args_3,
  "(o1, o2, o3)",
  "Return the ``repr(type(o))`` for each input ``o``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o1 : object\n"
  "    First input object\n"
  "o2 : object\n"
  "    Second input object\n"
  "o3 : object\n"
  "    Third input object\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "tuple[str]",
  self, args) noexcept
{
  return parse_args<3>(args);
}

/**
 * C++ type of the map owned by the capsule returned by `capsule_map`.
 */
using capsule_map_type = std::map<std::string, double>;

/**
 * `operator<<` overload for the `capsule_map_type`.
 */
auto& operator<<(std::ostream& out, const capsule_map_type& map)
{
  // compact format if empty
  if (map.empty())
    return out << "{}";
  // iterate
  out << "{\n";
  for (auto it = map.begin(); it != map.end(); it++) {
    if (std::distance(map.begin(), it))
      out << ",\n";
    out << "    {" << it->first << ", " << it->second << "}";
  }
  // need final newline
  return out << "\n}";
}

// TODO: make this function a try-block to catch all C++ exceptions
NPYGL_PY_FUNC_DECLARE(
  capsule_map,
  "()",
  "Return an opaque capsule object owning a std::map<std::string, double>.\n"
  "\n"
  "The contained map cannot be manipulated at all from Python and will be\n"
  "deleted when the last strong Python object reference is gone.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "PyCapsule",
  self, args) noexcept
{
  capsule_map_type map{
    {"a", 3.444},
    {"b", 1.33141},
    {"c", 3.14159265358979},
    {"d", 45.1111}
  };
  return npygl::py_object::create(std::move(map)).release();
}

/**
 * `operator<<` overload for a vector.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 */
template <typename T, typename A>
auto& operator<<(std::ostream& out, const std::vector<T, A>& vec)
{
  out << '[';
  for (auto it = vec.begin(); it != vec.end(); it++) {
    if (std::distance(vec.begin(), it))
      out << ", ";
    out << *it;
  }
  return out << ']';
}

/**
 * Enum to control what kind of C++ object capsule is created by `make_capsule`.
 *
 * The naming convention uses `UPPER_CASE` to mirror Python naming convention.
 */
enum {
  CAPSULE_STD_MAP = 1,
  CAPSULE_STD_VECTOR = 2,
#if NPYGL_HAS_EIGEN3
  CAPSULE_EIGEN3_MATRIX = 3,
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  CAPSULE_ARMADILLO_MATRIX = 4,
  CAPSULE_ARMADILLO_CUBE = 5,
  CAPSULE_ARMADILLO_ROWVEC = 6,
#endif  // NPYGL_HAS_ARMADILLO
  CAPSULE_STD_MAP_VECTOR = 7
};

// macro to support conditional docstring indication of whether or not an
// Eigen3 matrix capsule object can be created
#if NPYGL_HAS_EIGEN3
#define MAKE_CAPSULE_EIGEN3_MATRIX_OPTION \
  "    CAPSULE_EIGEN3_MATRIX\n" \
  "        Return an Eigen::MatrixXf\n"
#else
#define MAKE_CAPSULE_EIGEN3_MATRIX_OPTION ""
#endif  // !NPYGL_HAS_EIGEN3
// macros to support conditional docstring indication of whether or not
// different Armadillo matrix, cube, vector objects can be created
#if NPYGL_HAS_ARMADILLO
#define MAKE_CAPSULE_ARMADILLO_MATRIX_OPTION \
  "    CAPSULE_ARMADILLO_MATRIX\n" \
  "        Return an arma::fmat\n"
#define MAKE_CAPSULE_ARMADILLO_CUBE_OPTION \
  "    CAPSULE_ARMADILLO_CUBE\n" \
  "        Return an arma::cx_cube\n"
#define MAKE_CAPSULE_ARMADILLO_ROWVEC_OPTION \
  "    CAPSULE_ARMADILLO_ROWVEC\n" \
  "        Return an arma::rowvec\n"
#else
#define MAKE_CAPSULE_ARMADILLO_MATRIX_OPTION ""
#define MAKE_CAPSULE_ARMADILLO_CUBE_OPTION ""
#define MAKE_CAPSULE_ARMADILLO_ROWVEC_OPTION ""
#endif  // NPYGL_HAS_ARMADILLO
// TODO: consider making this a function try block
NPYGL_PY_FUNC_DECLARE(
  make_capsule,
  "(type)",
  "Return an opaque capsule object owning a C++ object.\n"
  "\n"
  "The C++ object created and owned is controlled by ``type``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "type : int\n"
  "    Integral constant to indicate what capsule creation operation should\n"
  "    executed. The accepted options and their effects are described below:\n"
  "\n"
  "    CAPSULE_STD_MAP\n"
  "        Return a std::map<std::string, double>\n"
  "    CAPSULE_STD_VECTOR\n"
  "        Return a std::vector<double>\n"
  MAKE_CAPSULE_EIGEN3_MATRIX_OPTION
  MAKE_CAPSULE_ARMADILLO_MATRIX_OPTION
  MAKE_CAPSULE_ARMADILLO_CUBE_OPTION
  MAKE_CAPSULE_ARMADILLO_ROWVEC_OPTION
  "    CAPSULE_STD_MAP_VECTOR\n"
  "        Return a std::vector<std::map<std::string, double>>\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "PyCapsule",
  self, obj)
{
  using npygl::py_object;
  // parse object type to create
  auto type = PyLong_AsLong(obj);
  if (PyErr_Occurred())
    return nullptr;
  // switch
  switch (type) {
    case CAPSULE_STD_MAP:
    {
      capsule_map_type map{
        {"a", 3.444},
        {"e", 2.71828},
        {"pi", 3.14159265358979},
        {"d", 45.1111}
      };
      return py_object::create(std::move(map)).release();
    }
    case CAPSULE_STD_VECTOR:
    {
      std::vector vec{4.3, 3.222, 4.12, 1.233, 1.66, 6.55, 77.333};
      return py_object::create(std::move(vec)).release();
    }
#if NPYGL_HAS_EIGEN3
    case CAPSULE_EIGEN3_MATRIX:
    {
      Eigen::MatrixXf mat{
        {3.4f, 1.222f, 5.122f, 1.22f},
        {6.44f, 3.21f, 5.345f, 9.66f},
        {6.244f, 3.414f, 1.231f, 4.85f}
      };
      return py_object::create(std::move(mat)).release();
    }
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
    case CAPSULE_ARMADILLO_MATRIX:
    {
      arma::fmat mat{
        {1.f, 2.33f, 4.33f, 1.564f},
        {4.55f, 55.6f, 1.212f, 4.333f},
        {4.232f, 9.83f, 1.56f, 65.34f}
      };
      return py_object::create(std::move(mat)).release();
    }
    case CAPSULE_ARMADILLO_CUBE:
    {
      arma::cx_cube cube{2, 3, 2};
      cube.slice(0) = {
        {{4.33, 2.323}, {4.23, 0.222}, {6.345, 1.1}},
        {{6.55, 4.12}, {4.11, 1.2323}, {6.77, 9.88}}
      };
      cube.slice(1) = {
        {{5.66, 5.11}, {5.66, 11.222}, {5.44, 22.333}},
        {{6.5, 1.22}, {1.222, 11.888}, {9.88, 2.33}}
      };
      return py_object::create(std::move(cube)).release();
    }
    case CAPSULE_ARMADILLO_ROWVEC:
    {
      arma::rowvec vec{4., 2.333, 1.23, 6.45, 28.77, 4.23, 1.115, 12.4};
      return py_object::create(std::move(vec)).release();
    }
#endif  // NPYGL_HAS_ARMADILLO
    case CAPSULE_STD_MAP_VECTOR:
    {
      std::vector<capsule_map_type> vec{
        {{"a", 2.3}, {"b", 1.222}, {"c", 4.6334}, {"d", 7.541}},
        {{"one", 1.}, {"two", 2.}, {"three", 3.}},
        {{"A", 1.23}, {"B", 5.443}, {"C", 1.21}, {"D", 1.5523}, {"E", 231.22}}
      };
      return py_object::create(std::move(vec)).release();
    }
    default:
      break;
  }
  // unknown enum value
  PyErr_SetString(
    PyExc_ValueError,
    ("Unknown capsule creation value " + std::to_string(type)).c_str()
  );
  return nullptr;
}

/**
 * String representation formatter for a capsule view's C++ object.
 *
 * @tparam Ts... Types with `operator<<` overloads to consider
 */
template <typename... Ts>
struct capsule_view_formatter {
  static_assert(sizeof...(Ts), "at least one type required");

  /**
   * Return a Python string representation for the capsule view's C++ object.
   *
   * On error the returned `py_object` is empty and a Python exception is set.
   *
   * @param view C++ capsule view
   */
  auto operator()(const npygl::cc_capsule_view& view) const
  {
    npygl::py_object res;
    (
      ...
      &&
      [&res, &view]
      {
        // not correct type so keep going
        if (!view.is<Ts>())
          return true;
        // format as string
        std::stringstream ss;
        ss << *view.as<Ts>();
        res = npygl::py_object{PyUnicode_FromString(ss.str().c_str())};
        return false;
      }()
    );
    // if not populated, something is wrong. set Python exception if no Python
    // exception is set (just the case that none of the types match)
    if (!res && !PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "capsule view type is not supported");
    return res;
  }
};

/**
 * Formatter partial specialization for a tuple of types.
 *
 * This is useful since one cannot using-declare a parameter pack.
 *
 * @tparam Ts... Types with `operator<<` overloads to consider
 */
template <typename... Ts>
struct capsule_view_formatter<std::tuple<Ts...>>
  : capsule_view_formatter<Ts...> {};

/**
 * Inline global to present a function-like usage for the formatter.
 *
 * @tparam Ts... Types with `operator<<` overloads to consider
 */
template <typename... Ts>
inline constexpr capsule_view_formatter<Ts...> capsule_view_format;

/**
 * Test function that takes a C++ object capsule and returns it as a string.
 */
NPYGL_PY_FUNC_DECLARE(
  capsule_str,
  "(o)",
  "Return the string representation of the C++ object owned by the capsule.\n"
  "\n"
  "This function only supports the C++ types corresponding to the capsules\n"
  "created by the ``make_capsule`` module function.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : PyCapsule\n"
  "    Python capsule object following the ``cc_capsule_view`` protocol\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str\n"
  "    String representation of the owned C++ object",
  self, obj) noexcept
{
  // supported types
  using supported_types = std::tuple<
    capsule_map_type,
    std::vector<double>,
#if NPYGL_HAS_EIGEN3
    Eigen::MatrixXf,
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
    arma::fmat,
    arma::cx_cube,
    arma::rowvec,
#endif  // NPYGL_HAS_ARMADILLO
    std::vector<capsule_map_type>
  >;
  // get capsule view
  npygl::cc_capsule_view view{obj};
  if (!view)
    return nullptr;
  // check type and return string representation if possible
  return capsule_view_format<supported_types>(view).release();
}

NPYGL_PY_FUNC_DECLARE(
  capsule_type,
  "(o)",
  "Return the type name of the C++ object owned by the capsule.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : PyCapsule\n"
  "    Python capsule object following the ``cc_capsule_view`` protocol\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str\n"
  "    String giving the type name of the owned C++ object",
  self, obj) noexcept
{
  // get capsule view
  npygl::cc_capsule_view view{obj};
  if (!view)
    return nullptr;
  // return string from the demangled type name
  return PyUnicode_FromString(npygl::type_name(view.info()));
}

// array_from_capsule only available if compiled with NumPy headers
#if NPYGL_HAS_NUMPY
NPYGL_PY_FUNC_DECLARE(
  array_from_capsule,
  "(o)",
  "Return a new NumPy array backed by the C++ object owned by the capsule.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : PyCapsule\n"
  "    Python capsule object following the ``cc_capsule_view`` protocol\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "numpy.ndarray",
  self, obj) noexcept
{
  // get reference-incremented Python object
  npygl::py_object cap{obj, npygl::py_object::incref};
  // supported types
  using supported_types = std::tuple<
#if NPYGL_HAS_EIGEN3
    Eigen::MatrixXf,
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
    arma::fmat,
    arma::cx_cube,
    arma::rowvec,
#endif  // NPYGL_HAS_ARMADILLO
    std::vector<double>  // note: std::vector<float> would be supported too
  >;
  // return new NumPy array if a supported cc_capsule_view capsule
  return npygl::make_ndarray_from_capsule<supported_types>(std::move(cap))
    .release();
}
#endif  // NPYGL_HAS_NUMPY

NPYGL_PY_FUNC_DECLARE(
  numpy_abi_version,
  "()",
  "Return the ABI version number of the NumPy array used during compilation.\n"
  "\n"
  "If module was compiled without NumPy headers ``None`` is returned instead.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "int or None",
  self, args) noexcept
{
#if NPYGL_HAS_NUMPY
  return PyLong_FromUnsignedLong(PyArray_GetNDArrayCVersion());
#else
  Py_RETURN_NONE;
#endif  // !NPYGL_HAS_NUMPY
}

NPYGL_PY_FUNC_DECLARE(
  numpy_api_version,
  "()",
  "Return the API version number of the NumPy C API used during compilation.\n"
  "\n"
  "If module was compiled without NumPy headers ``None`` is returned instead.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "int or None",
  self, args) noexcept
{
#if NPYGL_HAS_NUMPY
  return PyLong_FromUnsignedLong(PyArray_GetNDArrayCFeatureVersion());
#else
  Py_RETURN_NONE;
#endif  // !NPYGL_HAS_NUMPY
}

NPYGL_PY_FUNC_DECLARE(
  eigen3_version,
  "()",
  "Return the Eigen3 version string.\n"
  "\n"
  "If module was compiled without Eigen3 then ``None`` is returned instead.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str or None",
  self, args) noexcept
{
#if NPYGL_HAS_EIGEN3
  return PyUnicode_FromString(
    NPYGL_STRINGIFY(EIGEN_WORLD_VERSION) "."
    NPYGL_STRINGIFY(EIGEN_MAJOR_VERSION) "."
    NPYGL_STRINGIFY(EIGEN_MINOR_VERSION)
  );
#else
  Py_RETURN_NONE;
#endif  // !NPYGL_HAS_EIGEN3
}

NPYGL_PY_FUNC_DECLARE(
  armadillo_version,
  "()",
  "Return the Armadillo version string.\n"
  "\n"
  "If module was compiled without Armadillo then ``None`` is returned instead.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str or None",
  self, args) noexcept
{
#if NPYGL_HAS_ARMADILLO
  return PyUnicode_FromString(
    NPYGL_STRINGIFY(ARMA_VERSION_MAJOR) "."
    NPYGL_STRINGIFY(ARMA_VERSION_MINOR) "."
    NPYGL_STRINGIFY(ARMA_VERSION_PATCH)
  );
#else
  Py_RETURN_NONE;
#endif  // !NPYGL_HAS_ARMADILLO
}

// module method table
PyMethodDef mod_methods[] = {
  NPYGL_PY_FUNC_METHOD_DEF(parse_args_1, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(parse_args_3, METH_VARARGS),
  NPYGL_PY_FUNC_METHOD_DEF(capsule_map, METH_NOARGS),
  NPYGL_PY_FUNC_METHOD_DEF(make_capsule, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(capsule_str, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(capsule_type, METH_O),
  NPYGL_PY_FUNC_METHOD_DEF(eigen3_version, METH_NOARGS),
  NPYGL_PY_FUNC_METHOD_DEF(armadillo_version, METH_NOARGS),
#if NPYGL_HAS_NUMPY
  NPYGL_PY_FUNC_METHOD_DEF(array_from_capsule, METH_O),
#endif  // NPYGL_HAS_NUMPY
  NPYGL_PY_FUNC_METHOD_DEF(numpy_abi_version, METH_NOARGS),
  NPYGL_PY_FUNC_METHOD_DEF(numpy_api_version, METH_NOARGS),
  {}  // zero-initialized sentinel member
};

// module docstring
PyDoc_STRVAR(
  mod_doc,
  "npyglue C++ Python helpers testing extension module.\n"
  "\n"
  "Some of the C++ Python helpers can only be tested in the context of a C++\n"
  "extension module, e.g. the METH_VARARGS argument parsing functions, so we\n"
  "cannot perform all our tests via an embedded interpreter instance."
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
  // import NumPy C API if built with NumPy headers + create module
#if NPYGL_HAS_NUMPY
  import_array();
#endif  // NPYGL_HAS_NUMPY
  npygl::py_object mod{PyModule_Create(&mod_def)};
  if (!mod)
    return nullptr;
  // add capsule creation constants
  if (PyModule_AddIntMacro(mod, CAPSULE_STD_MAP)) return nullptr;
  if (PyModule_AddIntMacro(mod, CAPSULE_STD_VECTOR)) return nullptr;
#if NPYGL_HAS_EIGEN3
  if (PyModule_AddIntMacro(mod, CAPSULE_EIGEN3_MATRIX)) return nullptr;
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  if (PyModule_AddIntMacro(mod, CAPSULE_ARMADILLO_MATRIX)) return nullptr;
  if (PyModule_AddIntMacro(mod, CAPSULE_ARMADILLO_CUBE)) return nullptr;
  if (PyModule_AddIntMacro(mod, CAPSULE_ARMADILLO_ROWVEC)) return nullptr;
#endif  // NPYGL_HAS_ARMADILLO
  if (PyModule_AddIntMacro(mod, CAPSULE_STD_MAP_VECTOR)) return nullptr;
  // done, create module
  return mod.release();
}
