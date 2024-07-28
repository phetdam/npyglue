/**
 * @file npy_helpers.hh
 * @author Derek Huang
 * @brief C++ header for NumPy C API helpers
 * @copyright MIT License
 */

#ifndef NPYGL_NPY_HELPERS_HH_
#define NPYGL_NPY_HELPERS_HH_

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <utility>

// ensure clean against MumPy C API deprecations
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "npygl/common.h"
#include "npygl/features.h"
// TODO: change to npygl/py_object.hh
#include "npygl/py_helpers.hh"

#if NPYGL_HAS_CC_20
#include <span>
#endif  // !NPYGL_HAS_CC_20

namespace npygl {

/**
 * Traits class to map the C/C++ types to NumPy type numbers.
 *
 * @tparam T type
 */
template <typename T>
struct npy_type_traits {};

/**
 * Define a NumPy traits specialization for a type.
 *
 * @param npy_type `NPY_TYPES` enum value
 * @param c_type C/C++ type name
 */
#define NPYGL_NPY_TRAITS_SPEC(npy_type, c_type) \
  template <> \
  struct npy_type_traits<c_type> { \
    using type = c_type; \
    static constexpr auto typenum = npy_type; \
    static constexpr const char* name = NPYGL_STRINGIFY(npy_type); \
  }

// NumPy traits specializations
NPYGL_NPY_TRAITS_SPEC(NPY_DOUBLE, double);
NPYGL_NPY_TRAITS_SPEC(NPY_FLOAT, float);
NPYGL_NPY_TRAITS_SPEC(NPY_INT, int);
NPYGL_NPY_TRAITS_SPEC(NPY_UINT, unsigned int);
NPYGL_NPY_TRAITS_SPEC(NPY_LONG, long);
NPYGL_NPY_TRAITS_SPEC(NPY_ULONG, unsigned long);

/**
 * Helper to get the NumPy type number from a C/C++ type.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr auto npy_typenum = npy_type_traits<T>::typenum;

/**
 * Helper to get the NumPy type number string name from a C/C++ type.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr auto npy_typename = npy_type_traits<T>::name;

/**
 * Check if a NumPy array's data type matches the given type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_type(PyArrayObject* arr)
{
  return npy_typenum<T> == PyArray_TYPE(arr);
}

/**
 * Check if a NumPy array is aligned and is of the desired type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_aligned(PyArrayObject* arr)
{
  return is_type<T>(arr) && PyArray_ISALIGNED(arr);
}

/**
 * Check if a NumPy array is writeable and is of the desired type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_writeable(PyArrayObject* arr)
{
  return is_type<T>(arr) && PyArray_ISWRITEABLE(arr);
}

/**
 * Check if a NumPy array is aligned, writeable, and of the desired type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_behaved(PyArrayObject* arr)
{
  return is_type<T>(arr) && PyArray_ISBEHAVED(arr);
}

/**
 * Check if a Python object is a NumPy array of the desired type.
 *
 * @tparam T Desired NumPy array data type
 *
 * @param obj Python object
 */
template <typename T>
inline bool is_ndarray(PyObject* obj)
{
  return PyArray_Check(obj) && is_type<T>(reinterpret_cast<PyArrayObject*>(obj));
}

/**
 * Create a NumPy array of a specifed type from an existing Python object.
 *
 * @param obj Python object to create NumPy array from
 * @param type NumPy type number, e.g. `NPY_DOUBLE`
 * @param flags NumPy array requirement flags, default `NPY_ARRAY_DEFAULT`
 * @returns `py_object` owning the created array
 */
inline auto make_ndarray(PyObject* obj, int type, int flags = NPY_ARRAY_DEFAULT)
{
  return py_object{PyArray_FROM_OTF(obj, type, flags)};
}

/**
 * Create a NumPy array of a specifed type from an existing Python object.
 *
 * @tparam T NumPy array data buffer C type
 *
 * @param obj Python object to create NumPy array from
 * @param flags NumPy array requirement flags, default `NPY_ARRAY_DEFAULT`
 * @returns `py_object` owning the created array
 */
template <typename T>
inline auto make_ndarray(PyObject* obj, int flags = NPY_ARRAY_DEFAULT)
{
  return make_ndarray(obj, npy_typenum<T>, flags);
}

/**
 * Lightweight flat view of a NumPy array.
 *
 * Holds only the data pointer, size, and flags of the NumPy array. If there is
 * no need to know the shape or strides of the NumPy array, use this class.
 */
template <typename T>
class ndarray_flat_view {
public:
  using value_type = T;

  /**
   * Ctor.
   *
   * @note Array must already be of correct type.
   *
   * @param arr NumPy array
   */
  ndarray_flat_view(PyArrayObject* arr) noexcept
    : data_{static_cast<T*>(PyArray_DATA(arr))},
      size_{PyArray_SIZE(arr)},
      flags_{PyArray_FLAGS(arr)}
  {}

  /**
   * Return pointer to the NumPy array's data buffer.
   */
  auto data() const noexcept { return data_; }

  /**
   * Return number of elements in the NumPy array.
   */
  auto size() const noexcept { return size_; }

  /**
   * Return the NumPy array flags.
   */
  auto flags() const noexcept { return flags_; }

  /**
   * Return reference to the `i`th data element without bounds checking.
   */
  auto& operator[](std::size_t i) noexcept
  {
    return data_[i];
  }

private:
  T* data_;
  std::size_t size_;
  int flags_;
};

}  // namespace npygl

#endif  // NPYGL_NPY_HELPERS_HH_
