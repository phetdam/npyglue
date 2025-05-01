/**
 * @file python.i
 * @author Derek Huang
 * @brief npyglue SWIG Python helpers
 * @copyright MIT License
 */

// ensure SWIG is running in C++ mode
#ifndef __cplusplus
#error "python.i: SWIG C++ processing must be enabled with -c++"
#endif  // __cplusplus

// ensure SWIG is running to generate Python wrappers
#ifndef SWIGPYTHON
#error "python.i: can only be used with Python as target language"
#endif  // SWIGPYTHON

%{
#include <exception>
#include <stdexcept>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>

#include <npygl/demangle.hh>

namespace npygl {

/**
 * Convert a Python object to an optional value.
 *
 * This is conditionally enabled for integral values.
 *
 * @todo Unable to get SWIG to recognize the `std::optional<T>` typemaps.
 *
 * @tparam T Integral type
 * @returns `true` on success, `false` on failure with Python exception set
 */
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
bool get(std::optional<T>& out, PyObject* in) noexcept
{
  // NULL or None are both treated as no argument
  if (!in || in == Py_None)
    return true;
  // get widest possible value from Python
  auto v = [in]
  {
    // signed types
    if constexpr (std::is_signed_v<T>) {
      // long or smaller than long
      if constexpr (sizeof(T) <= sizeof(long))
        return PyLong_AsLong(in);
      else
        return PyLong_AsLongLong(in);
    }
    // unsigned types
    else {
      // unsigned long or smaller
      if constexpr (sizeof(T) <= sizeof(unsigned long))
        return PyLong_AsUnsignedLong(in);
      else
        return PyLong_AsUnsignedLongLong(in);
    }
  }();
  // need to explicitly check for error
  if (!PyErr_Occurred())
    return false;
  // signed types need check against minimum
  if constexpr (std::is_signed_v<T>) {
    constexpr auto v_min = std::numeric_limits<T>::min();
    if (v < v_min) {
      // note: not strictly noexcept
      std::stringstream ss;
      ss << "Received value " << v << " is less than " <<
        npygl::type_name(typeid(T)) << " minimum " << v_min;
      PyErr_SetString(PyExc_OverflowError, ss.str().c_str());
      return false;
    }
  }
  // both signed and unsigned check against maximum
  constexpr auto v_max = std::numeric_limits<T>::max();
  if (v > v_max) {
    std::stringstream ss;
    ss << "Received value " << v << " exceeds " <<
      npygl::type_name(typeid(T)) << " maximum " << v_max;
    PyErr_SetString(PyExc_OverflowError, ss.str().c_str());
    return false;
  }
  // success. cast appropriately
  out = static_cast<T>(v);
  return true;
}

}  // namespace npygl
%}

/**
 * Typemap macro for converting Python input into a `std::optional<T>`.
 *
 * @note Currently this is only supported for inetgral types.
 *
 * @param type C/C++ type
 */
%define NPYGL_OPTIONAL_IN_TYPEMAP(type)
%typemap(in) std::optional<type> OPT_IN {
  if (!npygl::get($1, $input))
    SWIG_fail;
}
%enddef  // NPYGL_OPTIONAL_IN_TYPEMAP(type)

/**
 * Typemap application macro for applying a `std::optional<T>` input typemap.
 *
 * @param type C/C++ type
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_APPLY_OPTIONAL_IN_TYPEMAP(type, name)
%apply std::optional<type> OPT_IN { std::optional<type> name };
%enddef  // NPYGL_APPLY_OPTIONAL_IN_TYPEMAP(type, name)

/**
 * Typemap cleaning macro for a single `std::optional<T>` typemap.
 *
 * @param type C/C++ type
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_CLEAR_OPTIONAL_TYPEMAP(type, name)
%clear std::optional<type> name;
%enddef  // NPYGL_CLEAR_OPTIONAL_TYPEMAP(type, name)

/**
 * Typemap cleaning macro for `std::optional<T>` typemaps.
 *
 * @param type C/C++ type
 */
%define NPYGL_CLEAR_OPTIONAL_TYPEMAPS(type)
%clear std::optional<type>;
%enddef  // NPYGL_CLEAR_OPTIONAL_TYPEMAPS(type)

// supported std::optional<T> typemaps
NPYGL_OPTIONAL_IN_TYPEMAP(short)
NPYGL_OPTIONAL_IN_TYPEMAP(unsigned short)
NPYGL_OPTIONAL_IN_TYPEMAP(std::uint_fast16_t)
NPYGL_OPTIONAL_IN_TYPEMAP(int)
NPYGL_OPTIONAL_IN_TYPEMAP(unsigned int)
NPYGL_OPTIONAL_IN_TYPEMAP(std::uint_fast32_t)
NPYGL_OPTIONAL_IN_TYPEMAP(long)
NPYGL_OPTIONAL_IN_TYPEMAP(unsigned long)
NPYGL_OPTIONAL_IN_TYPEMAP(std::uint_fast64_t)
NPYGL_OPTIONAL_IN_TYPEMAP(long long)
NPYGL_OPTIONAL_IN_TYPEMAP(unsigned long long)

/**
 * Macro to enable a general C++ exception handler.
 *
 * The exception messages will simply use the `what()` members.
 */
%define NPYGL_ENABLE_EXCEPTION_HANDLER
%exception {
  try {
    $action
  }
  // overflow
  catch (const std::overflow_error& ex) {
    PyErr_SetString(PyExc_OverflowError, ex.what());
    SWIG_fail;
  }
  // incorrect argument value
  catch (const std::invalid_argument& ex) {
    PyErr_SetString(
      PyExc_ValueError,
      (std::string{"std::invalid_argument thrown. "} + ex.what()).c_str()
    );
    SWIG_fail;
  }
  // underflow error mapped to ArithmeticError
  catch (const std::underflow_error& ex) {
    PyErr_SetString(
      PyExc_ArithmeticError,
      (std::string{"std::underflow_error thrown. "} + ex.what()).c_str()
    );
    SWIG_fail;
  }
  // domain error mapped to ValueError
  catch (const std::domain_error& ex) {
    PyErr_SetString(
      PyExc_ValueError,
      (std::string{"std::domain_error thrown. "} + ex.what()).c_str()
    );
    SWIG_fail;
  }
  // range error considered a TypeError
  catch (const std::range_error& ex) {
    PyErr_SetString(
      PyExc_TypeError,
      (std::string{"std::range_error thrown. "} + ex.what()).c_str()
    );
    SWIG_fail;
  }
  // base case. shouldn't use PyExc_Exception directly (RuntimeError instead)
  // this we use to catch std::runtime_error as well
  catch (const std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    SWIG_fail;
  }
  // unknown exception (on Windows, could be SEH for example)
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception thrown");
    SWIG_fail;
  }
}
%enddef  // NPYGL_ENABLE_EXCEPTION_HANDLER

/**
 * Macro to disable any previously defined C++ `%exception` block.
 */
%define NPYGL_DISABLE_EXCEPTION_HANDLER
%exception;
%enddef  // NPYGL_DISABLE_EXCEPTION_HANDLER
