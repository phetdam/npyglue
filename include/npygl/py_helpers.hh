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

#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

#include "npygl/features.h"
#include "npygl/warnings.h"

namespace npygl {

/**
 * Create a Python hex version number from the given components.
 *
 * @param major Python major version, e.g. the 3 in 3.4.1a2
 * @param minor Python minor version, e.g. the 4 in 3.4.1a2
 * @param micro Python micro version, e.g. the 1 in 3.4.1a2
 * @param level Python release level, e.g. `PY_RELEASE_LEVEL_FINAL`, which
 *  expands to `0xF`. This is the a in 3.4.1a2. Can be `0xB`, `0xC`, `0xF`.
 * @param serial Python release serial, e.g. the 2 in 3.4.1a2
 */
#define NPYGL_PY_VERSION_EX(major, minor, micro, level, serial) \
  (((major) << 24) | ((minor) << 16) | ((micro) << 8) | (level << 4) | (serial))

/**
 * Create a Python release hex version number.
 *
 * The release level is final (0xF) with the final release serial of zero.
 *
 * @param major Python major version, e.g. the 3 in 3.4.1
 * @param minor Python minor version, e.g. the 4 in 3.4.1
 * @param micro Python micro version, e.g. the 1 in 3.4.1
 */
#define NPYGL_PY_VERSION(major, minor, micro) \
  NPYGL_PY_VERSION_EX(major, minor, micro, PY_RELEASE_LEVEL_FINAL, 0)

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
  static constexpr incref_type incref{};

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
   * Return the Python object pointer cast to a different pointer type.
   *
   * @tparam T Target type
   *
   * @note No checking is done. Cast to only ABI-compatible types.
   *
   * @note Generally this is only needed for casting to `PyArrayObject*` or
   *  another `PyObject` binary-compatible struct.
   */
  template <typename T>
  auto as() const noexcept
  {
    return reinterpret_cast<T*>(ref_);
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
    // for Python 3.10.0+ use Py_XNewRef
#if PY_VERSION_HEX >= 0x030a00f0
    return py_object{Py_XNewRef(ref_)};
#else
    Py_XINCREF(ref_);
    return py_object{ref_};
#endif  // PY_VERSION_HEX < 0x030a00f0
  }

private:
  PyObject* ref_;
};

/**
 * Python main interpreter instance.
 *
 * Initializes and on destruction finalizes a Python interpreter instance.
 *
 * @note This is typically only intended to be used when embedding the Python
 *  interpreter in a C/C++ application as Python extension modules are loaded
 *  by an existing Python interpreter process.
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
   * @note May consider making `noexcept` for Windows as well.
   *
   * @param fail_fast `true` to exit if `Py_FinalizeEx` returns on error
   */
  explicit py_instance(bool fail_fast = false) noexcept(!NPYGL_WIN32)
    : fail_fast_{fail_fast}
  {
    // on Windows, the embedded Python's program name is deduced to argv[0]
    // instead of to the Python interpreter used by the virtual environment. so
    // we have a workaround here such that if a venv virtual environment is
    // being used, we set the program name using the venv's Python interpreter
#ifdef _WIN32
    // do this only once
    static const auto name_set = []
    {
      // VIRTUAL_ENV has path to venv virtual env root directory
      auto venv = std::getenv("VIRTUAL_ENV");
      // not in virtual environment
      if (!venv)
        return true;
      // program name must be in static storage
      static const auto progname = [venv]
      {
        // if capturing by copy, non-mutable lambda cannot modify capture
        auto path = venv;
        // widen to wchar_t
        std::wstringstream ss;
        while (*path)
          ss.put(ss.widen(*path++));
        // add relative path to python.exe + return
        ss << L"\\Scripts\\python.exe";
        return ss.str();
      }();
      // set program name
      // FIXME: deprecated in 3.11, use custom initialization
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4996)
      Py_SetProgramName(progname.c_str());
NPYGL_MSVC_WARNING_POP()
      return true;
    }();
#endif  // _WIN32
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
 * Initialize the Python interpreter once for the entire process.
 *
 * @note This is typically only intended to be called when embedding the Python
 *  interpreter in a C/C++ application as Python compiled extension modules are
 *  loaded at runtime by an existing Python interpreter.
 */
inline const auto& py_init() noexcept
{
  static py_instance python;
  return python;
}

/**
 * Import the given Python module.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param name Name of module to import
 */
inline auto py_import(const char* name) noexcept
{
  return py_object{PyImport_ImportModule(name)};
}

/**
 * Import the given Python module.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param name Name of module to import
 */
inline auto py_import(std::string_view name) noexcept
{
  return py_import(name.data());
}

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
 * Print the exception trace if the Python error indicator is set.
 *
 * @returns `true` if an exception trace was printed, `false` otherwise
 */
inline bool py_error_print() noexcept
{
  if (!PyErr_Occurred())
    return false;
  PyErr_Print();
  return true;
}

/**
 * Print the exception trace and exit if the Python error indicator is set.
 */
inline void py_error_exit() noexcept
{
  if (!py_error_print())
    return;
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
inline void py_error_exit(bool expr, PyObject* exc, const char* message) noexcept
{
  if (expr)
    py_error_exit(exc, message);
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
  py_error_exit(expr, exc, message.data());
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

/**
 * Retrieve the attribute with the given name from the Python object.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline auto py_getattr(PyObject* obj, const char* name) noexcept
{
  return py_object{PyObject_GetAttrString(obj, name)};
}

/**
 * Retrieve the attribute with the given name from the Python object.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline auto py_getattr(PyObject* obj, std::string_view name) noexcept
{
  return py_getattr(obj, name.data());
}

/**
 * Check if the Python object has an attribute of the given name.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline bool py_hasattr(PyObject* obj, const char* name) noexcept
{
  // silences MSVC C4800 warning about implicit conversion
  return !!PyObject_HasAttrString(obj, name);
}

/**
 * Check if the Python object has an attribute of the given name.
 *
 * @param obj Python object
 * @param name Attribute name
 */
inline bool py_hasattr(PyObject* obj, std::string_view name) noexcept
{
  return py_hasattr(obj, name.data());
}

/**
 * Call the Python object with the given positional arguments.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param callable Callable Python object
 * @param args Python positional args
 */
inline auto py_call(PyObject* callable, PyObject* args) noexcept
{
  return py_object{PyObject_CallObject(callable, args)};
}

#if PY_VERSION_HEX >= NPYGL_PY_VERSION(3, 9, 0)
/**
 * Call the Python object with no arguments.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param callable Callable Python object
 */
inline auto py_call(PyObject* callable) noexcept
{
  return py_object{PyObject_CallNoArgs(callable)};
}

/**
 * Call the Python object with only a single argument.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param callable Callable Python object
 * @param args Single Python argument
 */
inline auto py_call_one(PyObject* callable, PyObject* arg) noexcept
{
  return py_object{PyObject_CallOneArg(callable, arg)};
}
#endif  // PY_VERSION_HEX < NPYGL_PY_VERSION(3, 9, 0)

/**
 * Return a UTF-8 encoded string from a Python Unicode (string) object.
 *
 * On error the string is empty and a Python exception is set.
 *
 * @param obj Python object
 */
inline std::string py_utf8_string(PyObject* obj)
{
  // decode as UTF-8 string (use size since UTF-8 can contain NULL)
  Py_ssize_t size;
  auto data = PyUnicode_AsUTF8AndSize(obj, &size);
  // return empty on error
  if (!data)
    return {};
  // note: can't use ternary expression; no deducable common type
  return {data, static_cast<std::size_t>(size)};
}

/**
 * Return a UTF-8 encoded string view from a Python Unicode (string) object.
 *
 * On error, string view's `data()` is `nullptr` and a Python exception is set.
 *
 * @note Generally the string view ctor used here is `noexcept` for most
 *  implementations (although not required by the standard).
 *
 * @param obj Python object
 */
inline std::string_view py_utf8_view(PyObject* obj) noexcept
{
  // decode as UTF-8 string (use size since UTF-8 can contain NULL)
  Py_ssize_t size;
  auto data = PyUnicode_AsUTF8AndSize(obj, &size);
  // return view with nullptr data() on error
  if (!data)
    return {};
  // note: can't use ternary expression; no deducable common type
  return {data, static_cast<std::size_t>(size)};
}

/**
 * Print the Python object to the given file.
 *
 * @param f File to print to
 * @param obj Python object to print
 * @param flags Print flags, e.g. 0 for `repr()`, `Py_PRINT_RAW` for `str()`
 * @returns `true` on success, `false` on failure (exception is set)
 */
inline bool py_print(FILE* f, PyObject* obj, int flags = 0) noexcept
{
  return !PyObject_Print(obj, f, flags);
}

/**
 * Print the Python object to standard output.
 *
 * @param obj Python object to print
 * @param flags Print flags, e.g. 0 for `repr()`, `Py_PRINT_RAW` for `str()`
 * @returns `true` on success, `false` on failure (exception is set)
 */
inline bool py_print(PyObject* obj, int flags = 0) noexcept
{
  return py_print(stdout, obj, flags);
}

/**
 * Return the string representation of the Python object as with `repr()`.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object
 */
inline auto py_repr(PyObject* obj) noexcept
{
  return py_object{PyObject_Repr(obj)};
}

/**
 * Return the string representation of the Python object as with `str()`.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object
 */
inline auto py_str(PyObject* obj) noexcept
{
  return py_object{PyObject_Str(obj)};
}

/**
 * Write the `repr()` of the Python object to an output stream.
 *
 * On error `false` is returned and a Python exception is set.
 *
 * @param out Output stream
 * @param obj Python object to stream
 * @returns `true` on success, `false` on error
 */
inline bool py_repr(std::ostream& out, PyObject* obj)
{
  // get repr() of obj
  auto repr = py_repr(obj);
  if (!repr)
    return false;
  // get string [view] from repr
  auto view = py_utf8_view(repr);
  if (view.empty())
    return false;
  // stream
  out << view;
  return true;
}

}  // namespace npygl

/**
 * Stream the string representation of the Python object as with `repr()`.
 *
 * On error the Python exception trace is printed with `PyErr_Print`. Note that
 * this will also clear the error indicator so use with care.
 *
 * @note In order for ADL to work this is defined in the top-level namespace.
 *
 * @param out Output stream
 * @param obj Python object to stream
 */
inline auto& operator<<(std::ostream& out, PyObject* obj)
{
  if (!npygl::py_repr(out, obj))
    PyErr_Print();
  return out;
}

namespace npygl {

/**
 * Stream the string representation of the Python object as with `repr()`.
 *
 * On error the Python exception trace is printed with `PyErr_Print`. Note that
 * this will also clear the error indicator so use with care.
 *
 * @param out Output stream
 * @param obj Python object to stream
 */
inline auto& operator<<(std::ostream& out, const py_object& obj)
{
  return out << obj.ref();
}

}  // namespace npygl

#endif  // NPYGL_PY_HELPERS_HH_
