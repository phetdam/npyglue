/**
 * @file pyhelpers.cc
 * @author Derek Huang
 * @brief C++ Python extension module for testing C++ Python helpers
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <ostream>
#include <map>
#include <utility>

#include "npygl/common.h"
// TODO: currently this module only tests the non-NumPy Python C++ helpers
// #include "npygl/ndarray.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/python.hh"

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
 */
template <std::size_t N>
PyObject* parse_args(PyObject* NPYGL_UNUSED(self), PyObject* args) noexcept
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

/**
 * Test function for creating opaque capsule objects.
 *
 * Currently returns a fixed `std::map<std::string, double>`.
 *
 * @todo Make this a function try-block to catch all C++ exceptions.
 */
auto capsule_map(
  PyObject* NPYGL_UNUSED(self), PyObject* NPYGL_UNUSED(args)) noexcept
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
 * Test function that takes a C++ object capsule and returns it as a string.
 *
 * Currently this only works with the capsule returned by `capsule_map`.
 */
PyObject* cc_capsule_str(PyObject* NPYGL_UNUSED(self), PyObject* obj) noexcept
{
  // get capsule view
  npygl::cc_capsule_view view{obj};
  if (!view)
    return nullptr;
  // check type
  // note: in the future, we can iterate through types
  if (!view.is<capsule_map_type>()) {
    // note: could throw but only in extreme conditions
    std::stringstream ss;
    ss << NPYGL_PRETTY_FUNCTION_NAME <<
      ": capsule view type is not the expected " << view.info()->name();
    PyErr_SetString(PyExc_TypeError, ss.str().c_str());
    return nullptr;
  }
  // format as string
  std::stringstream ss;
  ss << *view.as<capsule_map_type>();
  return PyUnicode_FromString(ss.str().c_str());
}

// function docstrings
PyDoc_STRVAR(
  parse_args_1_doc,
  "parse_args_1(o)\n"
  NPYGL_CLINIC_MARKER
  "Return the ``repr(type(o))`` of the input ``o``.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : object\n"
  "    Input object\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str"
);
PyDoc_STRVAR(
  parse_args_3_doc,
  "parse_args_3(o1, o2, o3)\n"
  NPYGL_CLINIC_MARKER
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
  "tuple[str]"
);
PyDoc_STRVAR(
  capsule_map_doc,
  "capsule_map()\n"
  NPYGL_CLINIC_MARKER
  "Return an opaque capsule object owning a std::map<std::string, double>.\n"
  "\n"
  "The contained map cannot be manipulated at all from Python and will be\n"
  "deleted when the last strong Python object reference is gone.\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "PyCapsule"
);
PyDoc_STRVAR(
  cc_capsule_str_doc,
  "cc_capsule_str(o)\n"
  NPYGL_CLINIC_MARKER
  "Return the string representation of the C++ object owned by the capsule.\n"
  "\n"
  NPYGL_NPYDOC_PARAMETERS
  "o : PyCapsule\n"
  "    Python capsule object following the ``cc_capsule_view`` protocol\n"
  "\n"
  NPYGL_NPYDOC_RETURNS
  "str\n"
  "    String representation of the owned C++ object"
);

// module method table
PyMethodDef mod_methods[] = {
  {"parse_args_1", parse_args<1>, METH_VARARGS, parse_args_1_doc},
  {"parse_args_3", parse_args<3>, METH_VARARGS, parse_args_3_doc},
  {"capsule_map", capsule_map, METH_NOARGS, capsule_map_doc},
  {"cc_capsule_str", cc_capsule_str, METH_O, cc_capsule_str_doc},
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
  return PyModule_Create(&mod_def);
}
