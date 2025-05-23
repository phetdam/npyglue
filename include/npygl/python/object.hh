/**
 * @file python/object.hh
 * @author Derek Huang
 * @brief C++ header for Python object ownership
 */

#ifndef NPYGL_PYTHON_OBJECT_HH_
#define NPYGL_PYTHON_OBJECT_HH_

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "npygl/common.h"
#include "npygl/warnings.h"

namespace npygl {

///////////////////////////////////////////////////////////////////////////////
// Python capsules for C++ objects                                           //
///////////////////////////////////////////////////////////////////////////////

/**
 * Capsule name used for capsules that are nonzero `cc_capsule_view` objects.
 */
inline constexpr const char* cc_capsule_name = "npygl C++ capsule";

/**
 * Lightweight view object for a `PyCapsule` holding a C++ object.
 *
 * The `PyCapsule` itself owns the buffer that `obj` is pointing to.
 */
class cc_capsule_view {
public:
  /**
   * Default ctor.
   *
   * Constructs an invalid capsule view.
   */
  cc_capsule_view() noexcept = default;

  /**
   * Ctor.
   *
   * Create a capsule view from a Python object.
   *
   * If the object is not a `PyCapsule` whose name matches `cc_capsule_name`,
   * e.g. it does not follow the `cc_capsule_view` protocol, the view will be
   * invalid and a Python exception is set. Otherwise, the view is valid.
   *
   * @param obj Python object
   */
  cc_capsule_view(PyObject* obj) noexcept
  {
    // check if capsule and if following the protocol (matches name). we also
    // set a Python exception so an invalid view corresponds to a Python error
    if (!PyCapsule_IsValid(obj, cc_capsule_name)) {
      // note: could throw an exception in very rare conditions
      std::stringstream ss;
      ss << NPYGL_PRETTY_FUNCTION_NAME <<
        ": attempted to create a capsule view from an incompatible object";
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
      return;
    }
    // capsule is valid, so populate
    obj_ = PyCapsule_GetPointer(obj, cc_capsule_name);
    if (!obj_)
      return;
    // context is not nullptr if following protocol
    info_ = (decltype(info_)) PyCapsule_GetContext(obj);
    if (!info_) {
      obj_ = nullptr;  // indicate error
      return;
    }
  }

  /**
   * Return the untyped pointer to the C++ object.
   *
   * If the capsule view is invalid this is `nullptr`.
   */
  auto obj() const noexcept { return obj_; }

  /**
   * Return a pointer to the `std::type_info` object.
   *
   * If the capsule view is invalid this is `nullptr`.
   */
  auto info() const noexcept { return info_; }

  /**
   * Return the typed pointer to the C++ object.
   *
   * @note You *must* know what the C++ type is before dereferencing.
   *
   * @tparam T type
   */
  template <typename T>
  auto as() const noexcept
  {
    return reinterpret_cast<T*>(obj_);
  }

  /**
   * Implicitly convert to bool to indicate validity.
   *
   * @note Due to ctor protection we can skip checking `info_`. If `obj_` is
   *  `nullptr` than `info_` will also be `nullptr`.
   */
  operator bool() const noexcept
  {
    // silence MSVC C4800 warning
    return !!obj_;
  }

  /**
   * Check that the capsule view has a particular C++ type.
   *
   * If the view is invalid this also returns `false`.
   *
   * @tparam T type
   */
  template <typename T>
  bool is() const noexcept
  {
    if (!info_)
      return false;
    // note: C++ standard allows for info_ != &typeid(T)
    return *info_ == typeid(T);
  }

private:
  void* obj_{};
  const std::type_info* info_{};
};

/**
 * Function template for a Python capsule destructor for a C++ object.
 *
 * This is the default destructor used via the `create(T&&...)` template.
 *
 * On error a Python exception is set.
 *
 * @note Cannot make this `noexcept` since `PyCapsule_Destructor` is not
 *  `noexcept` under C++17 semantics. We might cast this later.
 */
template <typename T>
void cc_capsule_dtor(PyObject* capsule) noexcept
{
  // get capsule data pointer
  // TODO: maybe we should use PyCapsule_GetName to be more generic later
  auto data = PyCapsule_GetPointer(capsule, cc_capsule_name);
  if (!data)
    return;
  // manually destroy the object + free buffer
  ((T*) data)->~T();
  std::free(data);
}

///////////////////////////////////////////////////////////////////////////////
// Owning/creating Python objects/capsules                                   //
///////////////////////////////////////////////////////////////////////////////

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
   * Ctor.
   *
   * Creates a Python object from a float.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param value Float value
   */
  py_object(float value) noexcept : py_object{PyFloat_FromDouble(value)} {}

  /**
   * Ctor.
   *
   * Creates a Python object from a double.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param value Double value
   */
  py_object(double value) noexcept : py_object{PyFloat_FromDouble(value)} {}

  /**
   * Ctor.
   *
   * Creates a Python object from a boolean value.
   *
   * @param value Boolean value
   */
  py_object(bool value) noexcept : py_object{PyBool_FromLong(value)} {}

  /**
   * Ctor.
   *
   * Creates a Python object from a signed int.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param value Signed integer value
   */
  py_object(int value) noexcept : py_object{PyLong_FromLong(value)} {}

  /**
   * Ctor.
   *
   * Creates a Python object from an unsigned int.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param value Unsigned integer value
   */
  py_object(unsigned int value) noexcept
    : py_object{PyLong_FromUnsignedLong(value)}
  {}

  /**
   * Ctor.
   *
   * Creates a Python object from a `Py_complex` struct.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param value Complex number value
   */
  py_object(Py_complex value) noexcept
    : py_object{PyComplex_FromCComplex(value)}
  {}

  /**
   * Ctor.
   *
   * Creates a Python object from an array of `PyObject*`. If there is more
   * than one object in the array, a tuple of Python objects is created.
   *
   * This ctor overload that takes an index sequence is an advanced usage that
   * allows building a tuple from select members of the `PyObject*[N]` array,
   * even repeating a couple members from the array.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   * @tparam Is... Sequence of array indices within 0 to N - 1 inclusive
   *
   * @param objs Array of Python objects
   * @param seq Index sequence indicating which array objects are used
   */
  template <std::size_t N, std::size_t... Is>
  py_object(PyObject* (&objs)[N], std::index_sequence<Is...> seq) noexcept
    : py_object{create(objs, seq)}
  {}

  /**
   * Ctor.
   *
   * This is a convenience overload for working with a `py_object[N]`. It
   * provides the same semantics as the overload taking a `PyObject*[N]`.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   * @tparam Is... Sequence of array indices within 0 to N - 1 inclusive
   *
   * @param objs Array of objects
   * @param seq Index sequence indicating which array objects are used
   */
  template <std::size_t N, std::size_t... Is>
  py_object(py_object (&objs)[N], std::index_sequence<Is...> seq) noexcept
    : py_object{create(objs, seq)}
  {}

  /**
   * Ctor.
   *
   * Creates a Python object from an array of `PyObject*`. If there is more
   * than one object in the array, a tuple of Python objects is created.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   *
   * @param objs Array of Python objects
   */
  template <std::size_t N>
  py_object(PyObject* (&objs)[N]) noexcept : py_object{create(objs)} {}

  /**
   * Ctor.
   *
   * This is a convenience overload for working with a `py_object[N]`. It
   * provides the same semantics as the overload taking a `PyObject*[N]`.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   *
   * @param objs Array of objects
   */
  template <std::size_t N>
  py_object(py_object (&objs)[N]) noexcept : py_object{create(objs)} {}

  /**
   * Dtor.
   */
  ~py_object()
  {
    Py_XDECREF(ref_);
  }

  /**
   * Create a Python object from an array of `PyObject*`.
   *
   * A tuple of Python objects is returned if array size is greater than 1.
   * This function is for advanced usage, allowing building a tuple from select
   * members of a `PyObject*[N]` array, even repeating members.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @note May want to allow some kind of run-time toggle to enforce whether or
   *  or not a tuple should always be created (even if array is size 1).
   *
   * @tparam N Number of objects
   * @tparam Is... Sequence of array indices within 0 to N - 1 inclusive
   *
   * @param objs Array of Python objects
   * @param seq Index sequence indicating which array objects are used
   */
  template <std::size_t N, std::size_t... Is>
  static auto create(
    PyObject* (&objs)[N],
    std::index_sequence<Is...> NPYGL_UNUSED(seq)) noexcept
  {
    // number of indices must be nonzero
    static_assert(sizeof...(Is), "at least one index must be provided");
    // ensure none of the indices are outside of the array
    // note: parentheses around Is < N are unnecessary but are just for clarity
    static_assert(
      std::conjunction_v<std::bool_constant<(Is < N)>...>,
      "indices must only index within the provided array"
    );
    // build value. note we use sizeof...(Is) since it may not be exactly N
    return py_object{Py_BuildValue(py_object_format<sizeof...(Is)>, objs[Is]...)};
  }

  /**
   * Create a Python object from an array of `py_object`.
   *
   * This is a convenience overload for working with a `py_object[N]`. It
   * provides the same semantics as the overload taking a `PyObject*[N]`.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   * @tparam Is... Sequence of array indices within 0 to N - 1 inclusive
   *
   * @param objs Array of objects
   * @param seq Index sequence indicating which array objects are used
   */
  template <std::size_t N, std::size_t... Is>
  static auto create(
    py_object (&objs)[N], std::index_sequence<Is...> seq) noexcept
  {
    PyObject* refs[] = {objs[Is].ref()...};
    return create(refs, seq);
  }

  /**
   * Create a Python object from an array of `PyObject*`.
   *
   * A tuple of Python objects is returned if array size is greater than 1.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   *
   * @param objs Array of Python objects
   */
  template <std::size_t N>
  static auto create(PyObject* (&objs)[N]) noexcept
  {
    return create(objs, std::make_index_sequence<N>{});
  }

  /**
   * Create a Python object from an array of `py_object`.
   *
   * This is a convenience overload for working with a `py_object[N]`.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @tparam N Number of objects
   *
   * @param objs Array of objects
   */
  template <std::size_t N>
  static auto create(py_object (&objs)[N]) noexcept
  {
    return create(objs, std::make_index_sequence<N>{});
  }

  /**
   * Create a named Python capsule object.
   *
   * See the Python 3 documentation on capsules for more details.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param data Pointer to arbitrary data
   * @param name Capsule object name with lifetime >= that of the capsule
   * @param dtor Capsule destructor (must be `noexcept`)
   */
  static auto create(
    void* data, const char* name, PyCapsule_Destructor dtor = nullptr) noexcept
  {
    // silence C5039
    // TODO: maybe create a noexcept equivalent to PyCapsule_Destructor in
    // order to force user-defined dtors to be noexcept
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(5039)
    return py_object{PyCapsule_New(data, name, dtor)};
NPYGL_MSVC_WARNING_POP()
  }

  /**
   * Create an unnamed Python capsule object.
   *
   * See the Python 3 documentation on capsules for more details.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @param data Pointer to arbitrary data
   * @param dtor Capsule destructor (must be `noexcept`)
   */
  static auto create(void* data, PyCapsule_Destructor dtor = nullptr) noexcept
  {
    return create(data, nullptr, dtor);
  }

  /**
   * Create an unnamed Python capsule object from a C++ object.
   *
   * This uses placement new to copy or move the C++ object into a buffer. If
   * no destructor is provided, the default destructor is used.
   *
   * On error, the created object is empty and a Python exception is set.
   *
   * @note The use of `std::enable_if_t` here restricts overload selection to
   *  rvalues only. Without it, at least with GCC, template deduction results
   *  in a reference type and attempting to decay results in template recusion
   *  that easily exceeds the default recursion depth.
   *
   * @todo May want to refine `noexcept` specification to depend on T ctors.
   *
   * @tparam T type
   *
   * @param obj Rvalue reference to C++ object
   * @param dtor Capsule destructor
   */
  template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
  static py_object create(
    T&& obj, PyCapsule_Destructor dtor = cc_capsule_dtor<T>) noexcept
  {
    // placement buffer
    auto buf = std::malloc(sizeof(T));
    if (!buf) {
      // note: could throw an exception in very rare conditions
      std::stringstream ss;
      ss << NPYGL_PRETTY_FUNCTION_NAME << ": cannot allocate buffer";
      PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
      return {};
    }
    // place object via copy/move into buffer + create capsule
    auto new_obj = new(buf) T{std::forward<T>(obj)};
    auto capsule = create(new_obj, cc_capsule_name, dtor);
    // lambda for calling ~T() and cleaning up the placement buffer on error
    auto cleanup = [new_obj, buf]
    {
      new_obj->~T();
      std::free(buf);
    };
    // note: need to manually call ~T() due to placement new usage
    if (!capsule) {
      cleanup();
      return {};
    }
    // set the capsule context to the type_info pointer
    // note: type_info storage duration is static so no need to free
    if (PyCapsule_SetContext(capsule, (void*) &typeid(T))) {
      cleanup();
      return {};
    }
    return capsule;
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

}  // namespace npygl

#endif  // NPYGL_PYTHON_OBJECT_HH_
