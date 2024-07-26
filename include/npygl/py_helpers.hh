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

namespace npygl {

/**
 * Python object ownership class.
 *
 * This ensures that a single reference to a `PyObject*` is held onto.
 */
class py_object {
public:
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
  operator PyObject* const noexcept
  {
    return ref_;
  }

  /**
   * Release ownership of the Python object.
   *
   * @note If you do not correctly manage the reference count for the returned
   *  object pointer the Python interpreter may end up leaking memory.
   */
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

}  // namespace npygl

#endif  // NPYGL_PY_HELPERS_HH_
