/**
 * @file python.hh
 * @author Derek Huang
 * @brief C++ header for Python C API helpers
 * @copyright MIT License
 */

#ifndef NPYGL_PYTHON_HH_
#define NPYGL_PYTHON_HH_

// as mentioned in Python C API docs, this goes before standard headers
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "npygl/common.h"
#include "npygl/features.h"
#include "npygl/warnings.h"

namespace npygl {

///////////////////////////////////////////////////////////////////////////////
// Python version macros                                                     //
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Python argument parsing                                                   //
///////////////////////////////////////////////////////////////////////////////

namespace detail {

/**
 * Traits type for providing a compile-time `PyObject*[N]` format string.
 *
 * We need to provide an empty base since a partial specialization is used.
 *
 * @tparam T type
 */
template <typename T>
struct py_object_format_type {};

/**
 * Partial specialization for getting an index sequence's parameter pack.
 *
 * This uses the `O` format character to build the format string.
 *
 * We expand the parameter pack to construct a null-terminated character array
 * that can be used as a `PyArg_Parse*` or `Py_BuildValue` format string.
 *
 * @tparam Is... Indices from the `std::index_sequence<Is...>`
 */
template <std::size_t... Is>
struct py_object_format_type<std::index_sequence<Is...>> {
  // Is - Is is used to involve the index parameter pack
  static constexpr const char value[] = {('O' + (Is - Is))..., '\0'};
};

}  // namespace detail

/**
 * Traits type for providing a compile-time `PyObject*[N]` format string.
 *
 * @tparam N Number of `PyObject*` to format with `O`
 */
template <std::size_t N>
struct py_object_format_type
  : detail::py_object_format_type<std::make_index_sequence<N>> {
  static_assert(N, "N must be nonzero for a valid format string");
};

/**
 * Compile-time `PyObject*[N]` format string.
 *
 * @tparam N Number of `PyObject*` to format with `O`
 */
template <std::size_t N>
inline constexpr const char* py_object_format = py_object_format_type<N>::value;

/**
 * Parse Python arguments into an array of `PyObject` pointers.
 *
 * This uses the generic `"O"` conversion specifier with `PyArg_ParseTuple`.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam N Number of expected Python arguments
 * @tparam Is... Sequence of unique array indices within 0 to N - 1 inclusive
 *
 * @param args Python arguments
 * @param objs Array of `PyObject*` to convert to
 * @param seq Index sequence indicating which elements of `objs` are populated
 * @returns `true` on success, `false` on error
 */
template <std::size_t N, std::size_t... Is>
bool parse_args(
  PyObject* args,
  PyObject* (&objs)[N],
  std::index_sequence<Is...> NPYGL_UNUSED(seq)) noexcept
{
  // number of indices must be less than or equal to array size
  static_assert(sizeof...(Is), "at least one index must be provided");
  static_assert(sizeof...(Is) <= N, "index count cannot exceed array size");
  // ensure none of the indices are outside of the array
  // note: parentheses around Is < N are unnecessary but are just for clarity
  static_assert(
    std::conjunction_v<std::bool_constant<(Is < N)>...>,
    "indices must only index within the provided array"
  );
  // parse args. we need to use sizeof...(Is) since it may not equal N
  return !!PyArg_ParseTuple(args, py_object_format<sizeof...(Is)>, &objs[Is]...);
}

/**
 * Parse Python arguments into an array of `PyObject` pointers.
 *
 * This uses the generic `"O"` conversion specifier with `PyArg_ParseTuple`.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam N Number of expected Python arguments
 *
 * @param args Python arguments
 * @param objs Array of `PyObject*` to convert to
 * @returns `true` on success, `false` on error
 */
template <std::size_t N>
inline bool parse_args(PyObject* args, PyObject* (&objs)[N]) noexcept
{
  return parse_args(args, objs, std::make_index_sequence<N>{});
}

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

///////////////////////////////////////////////////////////////////////////////
// Python interpreter initialization                                         //
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Python module import                                                      //
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Python error querying/raising                                             //
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Python object helpers                                                     //
///////////////////////////////////////////////////////////////////////////////

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

/**
 * Call the Python object with no arguments.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @param callable Callable Python object
 */
inline auto py_call(PyObject* callable) noexcept
{
  return py_object{
#if PY_VERSION_HEX >= NPYGL_PY_VERSION(3, 9, 0)
    PyObject_CallNoArgs(callable)
#else
    PyObject_CallObject(callable, nullptr);
#endif  // !PY_VERSION_HEX < NPYGL_PY_VERSION(3, 9, 0)
  };
}

// TODO: use Py_BuildValue("(O)", arg) with PyObject_CallObject for < 3.9.0
#if PY_VERSION_HEX >= NPYGL_PY_VERSION(3, 9, 0)
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

///////////////////////////////////////////////////////////////////////////////
// Python Unicode object helpers                                             //
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// Python object printing                                                    //
///////////////////////////////////////////////////////////////////////////////

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
 * Return the string representation of the Python type object as with `repr()`.
 *
 * On error the returned `py_object` is empty and a Python exception is set.
 *
 * @note This overload removes the need for a `PyObject*` cast when using the
 *  `Py_TYPE()` macro or a function returning a `PyTypeObject*`.
 *
 * @param obj Python type object
 */
inline auto py_repr(PyTypeObject* obj) noexcept
{
  return py_repr((PyObject*) obj);
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

///////////////////////////////////////////////////////////////////////////////
// Python dicstring helpers
///////////////////////////////////////////////////////////////////////////////

/**
 * Macro indicating the end marker of a docstring Argument Clinic signature.
 *
 * See https://devguide.python.org/development-tools/clinic/ for details on the
 * Argument Clinic, especially the "How to override the generated signature"
 * section for an example of what the post-processed docstring looks like.
 */
#define NPYGL_CLINIC_MARKER "--\n\n"

/**
 * Macro for starting the parameters section of the NumPy docstring.
 */
#define NPYGL_NPYDOC_PARAMETERS "Parameters\n----------\n"

/**
 * Macro for starting the returns section of the NumPy docstring.
 */
#define NPYGL_NPYDOC_RETURNS "Returns\n-------\n"

}  // namespace npygl

#endif  // NPYGL_PYTHON_HH_
