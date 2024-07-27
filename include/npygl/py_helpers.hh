/**
 * @file py_helpers.hh
 * @author Derek Huang
 * @brief C++ header for Python C API helpers
 * @copyright MIT License
 */

#ifndef NPYGL_PY_HELPERS_HH_
#define NPYGL_PY_HELPERS_HH_

// as mentioned in Python C API docs, this goes before standard headers
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <sstream>

#include "npygl/features.h"

#if NPYGL_HAS_CC_17
#include <string_view>
#endif  // !NPYGL_HAS_CC_17

namespace npygl {

/**
 * Python object ownership class.
 *
 * This ensures that a single reference to a `PyObject*` is held onto.
 */
class py_object {
public:
  /**
   * Trivial type and member to support construction with reference increment.
   */
  struct incref_type {};
  static inline constexpr incref_type incref{};

  /**
   * Default ctor.
   */
  py_object() noexcept : py_object{nullptr} {}

  /**
   * Ctor.
   *
   * The Python object to own must have one strong reference or more.
   *
   * @param ref Python object to take ownership of (steal)
   */
  explicit py_object(PyObject* ref) noexcept : ref_{ref} {}

  /**
   * Ctor.
   *
   * Increments the Python object's reference count so on destruction, the net
   * change to the object's reference count is zero.
   *
   * @param ref Python object to take ownership of
   */
  py_object(PyObject* ref, incref_type /*inc*/) noexcept : py_object{ref}
  {
    Py_XINCREF(ref);
  }

  /**
   * Deleted copy ctor.
   */
  py_object(const py_object&) = delete;

  /**
   * Move ctor.
   *
   * @param other Python object to move from
   */
  py_object(py_object&& other) noexcept : ref_{other.release()} {}

  /**
   * Dtor.
   */
  ~py_object()
  {
    Py_XDECREF(ref_);
  }

  /**
   * Move assignment operator.
   *
   * @param other Python object to move from
   */
  auto& operator=(py_object&& other) noexcept
  {
    Py_XDECREF(ref_);
    ref_ = other.release();
    return *this;
  }

  /**
   * Return the Python object pointer.
   */
  auto ref() const noexcept { return ref_; }

  /**
   * Return a reference to the Python object pointer.
   *
   * This is intended for interop with C functions so we can take an address.
   */
  auto& ref() noexcept { return ref_; }

  /**
   * Implicit conversion operator for C function interop.
   */
  operator PyObject*() const noexcept
  {
    return ref_;
  }

  /**
   * Implicit conversion operator for checking if ownership exists.
   */
  operator bool() const noexcept
  {
    return !!ref_;  // silences warning about implicit conversion
  }

  /**
   * Release ownership of the Python object.
   *
   * @note If you do not correctly manage the reference count for the returned
   *  object pointer the Python interpreter may end up leaking memory.
   */
  [[nodiscard]]
  PyObject* release() noexcept
  {
    auto old_ref = ref_;
    ref_ = nullptr;
    return old_ref;
  }

  /**
   * Clone this instance and increment the Python object's reference count.
   *
   * This method is useful for feeding a `PyObject*` into a Python C API
   * function that steals (takes ownership of) a reference while still
   * maintaining a strong reference on the Python object. E.g.
   *
   * @code{.cc}
   * PyList_SetItem(list, 1, obj.clone().release());
   * @endcode
   *
   * This allows `obj` to continue to hold onto a strong reference while
   * allowing `PyList_SetItem` to steal a reference.
   */
  auto clone() const noexcept
  {
    Py_XINCREF(ref_);
    return py_object{ref_};
  }

private:
  PyObject* ref_;
};

/**
 * Python main interpreter instance.
 *
 * Initializes and on destruction finalizes a Python interpreter instance.
 *
 * @note Having more than one `py_instance` alive at a time is meaningless
 *  because `Py_Initialize` is a no-op unless `Py_Finalize[Ex]` was called.
 *
 * @note `Py_NewInterpreter` and `Py_EndInterpreter` can be used to create and
 *  destroy sub-interpreters that are managed by the main interpreter.
 */
class py_instance {
public:
  /**
   * Ctor.
   *
   * Perform default Python interpreter initialization for the current thread.
   *
   * @param fail_fast `true` to exit if `Py_FinalizeEx` returns on error
   */
  explicit py_instance(bool fail_fast = false) noexcept : fail_fast_{fail_fast}
  {
    Py_Initialize();
  }

  /**
   * Deleted copy ctor.
   */
  py_instance(const py_instance&) = delete;

  /**
   * Dtor.
   *
   * If `fail_fast()` returns `true` exit is called if `Py_FinalizeEx` errors.
   */
  ~py_instance()
  {
    if (!Py_FinalizeEx() && fail_fast_)
      std::exit(EXIT_FAILURE);
  }

  /**
   * Indicate whether or not program exits with error if `Py_FinalizeEx` errors.
   */
  bool fail_fast() const noexcept { return fail_fast_; }

private:
  bool fail_fast_;
};

/**
 * Import the given Python module.
 *
 * @param name Name of module to import
 */
inline auto py_import(const char* name) noexcept
{
  return py_object{PyImport_ImportModule(name)};
}

#if NPYGL_HAS_CC_17
/**
 * Import the given Python module.
 *
 * @param name Name of module to import
 */
inline auto py_import(std::string_view name) noexcept
{
  return py_import(name.data());
}
#endif  // !NPYGL_HAS_CC_17

/**
 * Set the Python error indicator for the current thread.
 *
 * @param exc Exception type to set
 * @param message Exception message
 * @returns `nullptr` for use in `PyCFunction` return statements
 */
inline auto py_error(PyObject* exc, const char* message) noexcept
{
  PyErr_SetString(exc, message);
  return nullptr;
}

/**
 * Print the exception trace and exit if the Python error indicator is set.
 */
inline void py_error_exit() noexcept
{
  if (!PyErr_Occurred())
    return;
  PyErr_Print();
  std::exit(EXIT_FAILURE);
}

/**
 * Set the Python error indicator, print the exception trace, and exit.
 *
 * @param exc Exception type to set
 * @param message Exception message
 */
[[noreturn]]
inline void py_error_exit(PyObject* exc, const char* message) noexcept
{
  py_error(exc, message);
  PyErr_Print();
  std::exit(EXIT_FAILURE);
}

#if NPYGL_HAS_CC_17
/**
 * Set the Python error indicator for the current thread.
 *
 * @param exc Exception type to set
 * @param message Exception message
 * @returns `nullptr` for use in `PyCFunction` return statements
 */
inline auto py_error(PyObject* exc, std::string_view message) noexcept
{
  return py_error(exc, message.data());
}

/**
 * Set the Python error indicator for the current thread.
 *
 * @tparam Ts... Argument types
 *
 * @param exc Exception type to set
 * @param args... Arguments to format in exception message
 * @returns `nullptr` for use in `PyCFunction` return statements
 */
template <typename... Ts>
inline auto py_error(PyObject* exc, Ts&&... args)
{
  std::stringstream ss;
  (ss << ... << args);
  return py_error(exc, ss.str().c_str());
}

/**
 * Set the Python error indicator, print the exception trace, and exit.
 *
 * @param exc Exception type to set
 * @param message Exception message
 */
[[noreturn]]
inline void py_error_exit(PyObject* exc, std::string_view message) noexcept
{
  py_error_exit(exc, message.data());
}

/**
 * Set the Python error indicator, print the exception trace, and exit.
 *
 * @param expr Expression to set error and exit if `true`
 * @param exc Exception type to set
 * @param message Exception message
 */
inline void py_error_exit(
  bool expr, PyObject* exc, std::string_view message) noexcept
{
  if (expr)
    py_error_exit(exc, message);
}

/**
 * Set the Python error indicator, print the exception trace, and exit.
 *
 * @tparam Ts... Argument types
 *
 * @param expr Expression to set error and exit if `true`
 * @param exc Exception type to set
 * @param args... Arguments to format in exception message
 */
template <typename... Ts>
inline void py_error_exit(bool expr, PyObject* exc, Ts&&... args)
{
  if (!expr)
    return;
  std::stringstream ss;
  (ss << ... << args);
  py_error_exit(exc, ss.str().c_str());
}
#endif  // !NPYGL_HAS_CC_17

/**
 * Retrieve the attribute with the given name from the Python object.
 *
 * The returned `py_object` is empty on failure.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline auto py_getattr(PyObject* obj, const char* name) noexcept
{
  return py_object{PyObject_GetAttrString(obj, name)};
}

#if NPYGL_HAS_CC_17
/**
 * Retrieve the attribute with the given name from the Python object.
 *
 * The returned `py_object` is empty on failure.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline auto py_getattr(PyObject* obj, std::string_view name) noexcept
{
  return py_getattr(obj, name.data());
}
#endif  // !NPYGL_HAS_CC_17

/**
 * Call the Python object with only a single argument.
 *
 * @param callable Callable Python object
 * @param args Single Python argument
 */
inline auto py_call_one(PyObject* callable, PyObject* arg) noexcept
{
  return py_object{PyObject_CallOneArg(callable, arg)};
}

/**
 * Call the Python object with the given positional arguments.
 *
 * @param callable Callable Python object
 * @param args Python positional args
 */
inline auto py_call(PyObject* callable, PyObject* args) noexcept
{
  return py_object{PyObject_CallObject(callable, args)};
}

}  // namespace npygl

#endif  // NPYGL_PY_HELPERS_HH_
