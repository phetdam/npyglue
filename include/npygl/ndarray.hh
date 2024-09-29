/**
 * @file ndarray.hh
 * @author Derek Huang
 * @brief C++ header for NumPy C API helpers
 * @copyright MIT License
 */

#ifndef NPYGL_NDARRAY_HH_
#define NPYGL_NDARRAY_HH_

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <complex>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

// NumPy C API visibility controls. the NumPy C API function pointer table is
// static by default so PY_ARRAY_UNIQUE_SYMBOL results in it being declared as
// non-static. defining NO_IMPORT_ARRAY results in the table being declared as
// extern so that there is no symbol duplication.
//
// therefore, to use the NumPy C API from an extension module/program that is
// composed of multiple translation units, define the following as follows:
//
// NPYGL_NUMPY_API_EXPORT
//  Defined before including `ndarray.hh` for only one translation unit
//
// NPYGL_NUMPY_API_IMPORT
//  Defined before including `ndarray.hh` for all other translation units
//
// if sharing NumPy C API, need to define new extern name
#if defined(NPYGL_NUMPY_API_EXPORT) || defined(NPYGL_NUMPY_API_IMPORT)
#define PY_ARRAY_UNIQUE_SYMBOL npygl_numpy_api
#endif  // !defined(NPYGL_NUMPY_API_EXPORT) && !defined(NPYGL_NUMPY_API_IMPORT)
// if importing, further define NO_IMPORT_ARRAY to declare table as extern
#ifdef NPYGL_NUMPY_API_IMPORT
#define NO_IMPORT_ARRAY
#endif  // NPYGL_NUMPY_API_IMPORT
// cannot have both defined
#if defined(NPYGL_NUMPY_API_EXPORT) && defined(NPYGL_NUMPY_API_IMPORT)
#error "ndarray.hh: define only one of NPYGL_NUMPY_API_(EXPORT|IMPORT)"
#endif  // !defined(NPYGL_NUMPY_API_EXPORT) || defined(NPYGL_NUMPY_API_IMPORT)

// ensure clean against NumPy C API deprecations
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif  // NPY_NO_DEPRECATED_API
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include "npygl/common.h"
#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/python.hh"
#include "npygl/range_views.hh"

// C++20
#if NPYGL_HAS_CC_20
#include <span>
#endif  // !NPYGL_HAS_CC_20

// Eigen 3 if desired
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

// Armadillo if desired
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)

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
 * @note `constexpr` does not imply inline for static data members until C++17.
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
NPYGL_NPY_TRAITS_SPEC(NPY_CFLOAT, std::complex<float>);
NPYGL_NPY_TRAITS_SPEC(NPY_CDOUBLE, std::complex<double>);

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
 * Helper to check if a type has NumPy type traits.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct has_npy_type_traits : std::false_type {};

/**
 * True specialization checking if a type has NumPy type traits.
 *
 * We discard cv-qualifiers since only the non-qualified type matters.
 *
 * @tparam T type
 */
template <typename T>
struct has_npy_type_traits<
  T, std::void_t<npy_type_traits<std::remove_cv_t<T>>> > : std::true_type {};

/**
 * Boolean helper for checking if a type has NumPy type traits.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool has_npy_type_traits_v = has_npy_type_traits<T>::value;

/**
 * Helper to get a comma-separated list of NumPy type names from C/C++ types.
 *
 * @tparam T C/C++ type to get NumPy type name for
 * @tparam Ts... Other C/C++ types to get NumPy type names for
 */
template <typename T, typename... Ts>
auto npy_typename_list()
{
  // no other types
  if constexpr (!sizeof...(Ts))
    return npy_typename<T>;
  // else we need separator and to recurse
  else
    return npy_typename<T> + std::string{", "} + npy_typename_list<Ts...>();
}

#if NPYGL_HAS_CC_20
/**
 * Concept for a C/C++ type that can be a possible NumPy data type.
 *
 * @tparam T type
 */
template <typename T>
concept npy_type = has_npy_type_traits_v<T>;
#endif  // !NPYGL_HAS_CC_20

/**
 * Import the NumPy C API and make it available for use.
 *
 * @todo Consider adding a default `py_instance` imported by default.
 *
 * @param python Python instance to ensure calling when Python is initialized
 * @returns `true` on success, `false` on error with Python exception set
 */
inline bool npy_api_import(const py_instance& /*python*/) noexcept
{
  return !PyArray_ImportNumPyAPI();
}

}  // namespace npygl

/**
 * Stream the string representation of the NumPy array as with `repr()`.
 *
 * On error the Python exception trace is printed with `PyErr_Print`. Note that
 * this will also clear the error indicator so use with care.
 *
 * @note In order for ADL to work this is defined in the top-level namespace.
 *
 * @param out Output stream
 * @param arr NumPy array to stream
 */
inline auto& operator<<(std::ostream& out, PyArrayObject* arr)
{
  return out << reinterpret_cast<PyObject*>(arr);
}

namespace npygl {

/**
 * Check if a Python object is a NumPy array.
 *
 * @param obj Python object
 */
inline bool is_ndarray(PyObject* obj) noexcept
{
  // double negation to force bool conversion and silence MSVC C4800
  return !!PyArray_Check(obj);
}

/**
 * Check if a NumPy array's data type matches the given type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_type(PyArrayObject* arr) noexcept
{
  return npy_typenum<T> == PyArray_TYPE(arr);
}

/**
 * Check if a Python object is a NumPy array of the desired type.
 *
 * @note If calling other NumPy array checking functions then it is more
 *  efficient to call the non-templated `is_ndarray()` by itself.
 *
 * @tparam T Desired NumPy array data type
 *
 * @param obj Python object
 */
template <typename T>
inline bool is_ndarray(PyObject* obj) noexcept
{
  return PyArray_Check(obj) && is_type<T>(reinterpret_cast<PyArrayObject*>(obj));
}

/**
 * Check if a NumPy array is aligned and is of the desired type.
 *
 * @tparam T Desired data type
 *
 * @param arr NumPy array
 */
template <typename T>
inline bool is_aligned(PyArrayObject* arr) noexcept
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
inline bool is_writeable(PyArrayObject* arr) noexcept
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
inline bool is_behaved(PyArrayObject* arr) noexcept
{
  return is_type<T>(arr) && PyArray_ISBEHAVED(arr);
}

/**
 * Create a NumPy array of a specifed type from an existing Python object.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object to create NumPy array from
 * @param type NumPy type number, e.g. `NPY_DOUBLE`
 * @param flags NumPy array requirement flags, e.g. `NPY_ARRAY_DEFAULT`
 * @returns `py_object` owning the created array
 */
inline auto make_ndarray(PyObject* obj, int type, int flags) noexcept
{
  return py_object{PyArray_FROM_OTF(obj, type, flags)};
}

/**
 * Create a NumPy array of a specifed type from an existing Python object.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @tparam T NumPy array data buffer C type
 *
 * @param obj Python object to create NumPy array from
 * @param flags NumPy array requirement flags, e.g. `NPY_ARRAY_DEFAULT`
 * @returns `py_object` owning the created array
 */
template <typename T>
inline auto make_ndarray(PyObject* obj, int flags) noexcept
{
  return make_ndarray(obj, npy_typenum<T>, flags);
}

/**
 * Create a new NumPy array of a specifed type from an existing Python object.
 *
 * The default array creation flags, `NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY`,
 * ensure that a new aligned, row-major order new NumPy array is returned.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @param obj Python object to create NumPy array from
 * @returns `py_object` owning the created array
 */
template <typename T>
inline auto make_ndarray(PyObject* obj) noexcept
{
  return make_ndarray<T>(obj, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY);
}

/**
 * Forward declaration for a NumPy array builder.
 *
 * This builder class is intended to facilitate creation of NumPy arrays from
 * existing Python capsule objects via the `cc_capsule_view`.
 */
template <typename... Ts>
struct ndarray_capsule_builder;

/**
 * Traits class for the NumPy array capsule builder.
 *
 * This is useful for retrieving the `ndarray_capsule_builder` target type
 * without performing a member access (which requires a complete) type.
 *
 * @tparam T type
 */
template <typename T>
struct ndarray_capsule_builder_traits {};

/**
 * Partial specialization for builders with a single type.
 *
 * @tparam T Builder object type
 */
template <typename T>
struct ndarray_capsule_builder_traits<ndarray_capsule_builder<T>> {
  using object_type = T;
};

/**
 * Partial specialization for builders with a tuple of types.
 *
 * We don't allow targeting multiple types at once.
 *
 * @tparam Ts... Tuple types
 */
template <typename... Ts>
struct ndarray_capsule_builder_traits<
  ndarray_capsule_builder<std::tuple<Ts...>> > {};

/**
 * NumPy array builder CRTP base class.
 *
 * This provides a safe, end-user invocable `operator()` that performs the
 * necessary type checking before calling the unsafe `operator()`.
 *
 * We use CRTP here as a form of static polymorphism.
 *
 * @tparam
 */
template <typename Builder>
struct ndarray_capsule_builder_base {
  /**
   * Create a NumPy array backed by the C++ object managed by a Python capsule.
   *
   * @note We take the `py_object` by rvalue reference to semantically indicate
   *  that the Python object is to be consumed as reference will be stolen.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @param cap Python capsule following the `cc_capsule_view` protocol
   */
  py_object operator()(py_object&& cap) const noexcept
  {
    using T = typename ndarray_capsule_builder_traits<Builder>::object_type;
    // get capsule view
    cc_capsule_view view{cap};
    if (!view)
      return {};
    // check if type is correct
    if (!view.is<T>()) {
      // note: technically does not provide noexcept guarantee
      std::stringstream ss;
      ss << NPYGL_PRETTY_FUNCTION_NAME << ": capsule backed by " <<
        type_name(view.info()) << " is not supported";
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
      return {};
    }
    // create NumPy array from the capsule
    return static_cast<const Builder&>(*this)(std::move(cap), view.as<T>());
  }
};

/**
 * NumPy array builder for building from a capsule holding a C++ vector.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 */
template <typename T, typename A>
struct ndarray_capsule_builder<std::vector<T, A>>
  : ndarray_capsule_builder_base<ndarray_capsule_builder<std::vector<T, A>>> {
  using object_type = std::vector<T, A>;

  /**
   * Create a 1D NumPy array with C memory layout backed by a C++ vector.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_vec Pointer to C++ vector retrieved from the capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_vec) const noexcept
  {
    // create new 1D NumPy array from the vector's data buffer
    // note: cast to suppress C4365 from MSVC. no array needed since 1D
    auto dim = static_cast<npy_intp>(cap_vec->size());
    py_object ar{
      PyArray_SimpleNewFromData(1, &dim, npy_typenum<T>, cap_vec->data())
    };
    if (!ar)
      return {};
    // set base object so NumPy array owns the capsule
    if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), cap.release()) < 0)
      return {};
    // success
    return ar;
  }
};

// enable if we have Eigen 3 unless NPYGL_NO_EIGEN3 is defined
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
/**
 * NumPy array builder for building from a capsule holding an Eigen matrix.
 *
 * @tparam T Element type
 * @tparam R Number of rows
 * @tparam C Number of columns
 * @tparam O Matrix options
 * @tparam RMax Max number of rows
 * @tparam CMax max number of columns
 */
template <typename T, int R, int C, int O, int RMax, int RMin>
struct ndarray_capsule_builder<Eigen::Matrix<T, R, C, O, RMax, RMin>>
  : ndarray_capsule_builder_base<
      ndarray_capsule_builder<Eigen::Matrix<T, R, C, O, RMax, RMin>> > {
  using object_type = Eigen::Matrix<T, R, C, O, RMax, RMin>;

  /**
   * Create a 2D NumPy array backed by an Eigen3 matrix object.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_mat Pointer to C++ Eigen3 matrix retrieved from the capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_mat) const noexcept
  {
    // create dims
    npy_intp dims[2];
    dims[0] = cap_mat->rows();
    dims[1] = cap_mat->cols();
    // data order. Eigen matrices are column major by default
    constexpr auto order = []
    {
      if constexpr (O & Eigen::StorageOptions::RowMajor)
        return NPY_ARRAY_C_CONTIGUOUS;
      else
        return NPY_ARRAY_F_CONTIGUOUS;
    }();
    // create new 2D NumPy array from the matrix data buffer
    py_object ar{
      PyArray_New(
        &PyArray_Type,               // subtype
        sizeof dims / sizeof *dims,  // nd
        dims,                        // dims
        npy_typenum<T>,              // type_num
        nullptr,                     // strides
        cap_mat->data(),             // data
        0,                           // itemsize (ignored)
        order | NPY_ARRAY_BEHAVED,   // flags (Eigen matrix buffers are aligned)
        nullptr                      // obj (ignored)
      )
    };
    if (!ar)
      return {};
    // set base object so NumPy array owns the capsule
    if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), cap.release()) < 0)
      return {};
    return ar;
  }
};
#endif  // !NPYGL_HAS_EIGEN3 || defined(NPYGL_NO_EIGEN3)

/**
 * NumPy array builder for working with a number of capsule types.
 *
 * This is useful when you have a tuple of supported capsule C++ object types
 * you want to check against at runtime to create the NumPy array.
 *
 * @note Does not inherit `ndarray_capsule_builder_base` because we need a
 *  `operator()` that will loop and check all the specified types.
 *
 * @tparam Ts... C++ object types
 */
template <typename... Ts>
struct ndarray_capsule_builder<std::tuple<Ts...>>
  : ndarray_capsule_builder<Ts>... {
  /**
   * Create a NumPy array backed by the C++ object managed by a Python capsule.
   *
   * The capsule type will be checked against each of the specified types.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @param cap Python capsule following the `cc_capsule_view` protocol
   */
  py_object operator()(py_object&& cap) const noexcept
  {
    // get capsule view
    cc_capsule_view view{cap};
    if (!view)
      return {};
    // check if type is correct + create NumPy array if correct
    py_object ar;
    (
      [&, this]
      {
        using self_type = ndarray_capsule_builder<Ts>;
        // wrong type so continue
        if (!view.is<Ts>())
          return true;
        // correct type so create NumPy array
        ar = static_cast<const self_type&>(*this)(std::move(cap), view.as<Ts>());
        return false;
      }()
      &&
      ...
    );
    // if empty, set error
    if (!ar) {
       // note: technically does not provide noexcept guarantee
      std::stringstream ss;
      // TODO: need utility to get joined type list from pack of types
      ss << NPYGL_PRETTY_FUNCTION_NAME << ": capsule backed by " <<
        type_name(view.info()) << " is not supported";
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
    }
    // return (empty on error)
    return ar;
  }
};

/**
 * Global builder for creating a NumPy array from a Python capsule.
 *
 * This provides a functional interface to the `ndarray_capsule_builder`.
 *
 * @tparam Ts... Target C++ types
 */
template <typename... Ts>
inline constexpr ndarray_capsule_builder<Ts...> make_ndarray_from_capsule;

/**
 * Create a 1D NumPy array with C memory layout backed by a C++ vector.
 *
 * The backing Python object is a `PyCapsule` following the `cc_capsule_view`
 * protocol that manages the moved C++ vector. Upon creation of the NumPy array
 * its array flags will be equivalent to `NPY_ARRAY_DEFAULT`.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @todo May consider creating a `make_ndarray(T&&)` template with the same
 *  logic that delegates to `make_ndarray_from_capsule<T>`. Documentation will
 *  instead be moved to the relevant `ndarray_capsule_builder` functions.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 *
 * @param vec Vector to consume
 */
template <typename T, typename A>
py_object make_ndarray(std::vector<T, A>&& vec) noexcept
{
  using V = std::vector<T, A>;
  // create capsule from vector
  auto capsule = py_object::create(std::move(vec));
  if (!capsule)
    return {};
  // get capsule view (should not error)
  cc_capsule_view view{capsule};
  if (!view)
    return {};
  // create new 1D NumPy array from the vector's data buffer
  return make_ndarray_from_capsule<V>(std::move(capsule), view.as<V>());
}

// enable if we have Eigen 3 unless NPYGL_NO_EIGEN3 is defined
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
/**
 * Create a 2D NumPy array backed by an Eigen3 matrix object.
 *
 * The backing Python object is a `PyCapsule` following the `cc_capsule_view`
 * protocol that manages a moved `Eigen::Matrix<...>` object. Upon creation of
 * the NumPy array its flags will have `NPY_ARRAY_BEHAVED` and then one of
 * `NPY_ARRAY_C_CONTIGUOUS` or `NPY_ARRAY_F_CONTIGUOUS`.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @todo May consider creating a `make_ndarray(T&&)` template with the same
 *  logic that delegates to `make_ndarray_from_capsule<T>`. Documentation will
 *  instead be moved to the relevant `ndarray_capsule_builder` functions.
 *
 * @tparam T Element type
 * @tparam R Number of rows
 * @tparam C Number of columns
 * @tparam O Matrix options
 * @tparam RMax Max number of rows
 * @tparam CMax max number of columns
 *
 * @param mat Matrix to consume
 */
template <typename T, int R, int C, int O, int RMax, int RMin>
py_object make_ndarray(Eigen::Matrix<T, R, C, O, RMax, RMin>&& mat) noexcept
{
  using M = Eigen::Matrix<T, R, C, O, RMax, RMin>;
  // create capsule
  auto capsule = py_object::create(std::move(mat));
  if (!capsule)
    return {};
  // capsule view (should not error)
  cc_capsule_view view{capsule};
  if (!view)
    return {};
  // create Eigen3 matrix from capsule
  return make_ndarray_from_capsule<M>(std::move(capsule), view.as<M>());
}
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)

// enable if we have Armadillo unless NPYGL_NO_ARMADILLO is defined
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
/**
 * Create a 2D NumPy array backed by an Armadillo matrix object.
 *
 * The backing Python object is a `PyCapsule` following the `cc_capsule_view`
 * protocol that manages a moved `arma::Mat<T>` object. Upon creation of the
 * NumPy array its flags will have `NPY_ARRAY_FARRAY`.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @note If moving from an `arma::Row<T>` or an `arma::Col<T>` the resulting
 *  NumPy array is still 2D but one dimension will be 1.
 *
 * @todo May consider creating a `make_ndarray(T&&)` template with the same
 *  logic that delegates to `make_ndarray_from_capsule<T>`. Documentation will
 *  instead be moved to the relevant `ndarray_capsule_builder` functions.
 *
 * @tparam T Element type
 *
 * @param mat Matrix to consume
 */
template <typename T>
py_object make_ndarray(arma::Mat<T>&& mat) noexcept
{
  using M = arma::Mat<T>;
  // create capsule
  auto capsule = py_object::create(std::move(mat));
  if (!capsule)
    return {};
  // capsule view (should not error)
  cc_capsule_view view{capsule};
  if (!view)
    return {};
  // pointer to managed matrix + create dims
  auto cap_mat = view.as<M>();
  npy_intp dims[2];
  // cast to silence C4365 warning
  dims[0] = static_cast<npy_intp>(cap_mat->n_rows);
  dims[1] = static_cast<npy_intp>(cap_mat->n_cols);
  // create new 2D NumPy array from the matrix data buffer
  py_object ar{
    PyArray_New(
      &PyArray_Type,               // subtype
      sizeof dims / sizeof *dims,  // nd
      dims,                        // dims
      npy_typenum<T>,              // type_num
      nullptr,                     // strides
      cap_mat->memptr(),           // data
      0,                           // itemsize (ignored)
      NPY_ARRAY_FARRAY,            // flags (Armadillo buffers are aligned)
      nullptr                      // obj (ignored)
    )
  };
  if (!ar)
    return {};
  // set base object so NumPy array owns the capsule
  if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), capsule.release()) < 0)
    return {};
  return ar;
}

/**
 * Create a 3D NumPy array backed by an Armadillo cube object.
 *
 * @note Due to how the memory is laid out in an Armadillo cube the resulting
 *  shape of the NumPy array is `(n_rows, n_cols, n_slices)`. This is not a
 *  true 3D array or tensor; the Armadillo docs call the cube a pseudo-tensor.
 *
 * The backing Python object is a `PyCapsule` following the `cc_capsule_view`
 * protocol that manages a moved `arma::Cube<T>` object. Upon creation of the
 * NumPy array its flags will have `NPY_ARRAY_FARRAY`.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * @tparam T Element type
 *
 * @param cube Cube to consume
 */
template <typename T>
py_object make_ndarray(arma::Cube<T>&& cube) noexcept
{
  using C = arma::Cube<T>;
  // create capsule
  auto capsule = py_object::create(std::move(cube));
  if (!capsule)
    return {};
  // capsule view (should not error)
  cc_capsule_view view{capsule};
  if (!view)
    return {};
  // pointer to managed cube + create dims
  auto cap_cube = view.as<C>();
  npy_intp dims[3];
  // cast to silence C4365 warning
  dims[0] = static_cast<npy_intp>(cap_cube->n_rows);
  dims[1] = static_cast<npy_intp>(cap_cube->n_cols);
  dims[2] = static_cast<npy_intp>(cap_cube->n_slices);
  //
  // if we want to interpret the arma::Cube<T> data as a tensor-like 3D array
  // of the shape (n_slices, n_rows, n_cols), e.g. with dims as follows:
  //
  // dims[0] = static_cast<npy_intp>(cap_cube->n_slices);
  // dims[1] = static_cast<npy_intp>(cap_cube->n_rows);
  // dims[2] = static_cast<npy_intp>(cap_cube->n_cols);
  //
  // we will then need to define a strides (in bytes) array as follows:
  //
  // npy_intp strides[3];
  // strides[0] = sizeof(T) * dims[1] * dims[2];
  // strides[1] = sizeof(T);
  // strides[2] = sizeof(T) * dims[1];
  //
  // this results in a NumPy array that is neither C nor Fortran contiguous so
  // flags passed to PyArray_New are NPY_ARRAY_BEHAVED. essentially we will use
  // strides to reinterpret a Fortran-contiguous (n_rows, n_cols, n_slices) 3D
  // array as a (n_slices, n_rows, n_cols) tensor-like 3D array.
  //
  // create new 3D NumPy array from the cube data buffer
  py_object ar{
    PyArray_New(
      &PyArray_Type,               // subtype
      sizeof dims / sizeof *dims,  // nd
      dims,                        // dims
      npy_typenum<T>,              // type_num
      nullptr,                     // strides
      cap_cube->memptr(),          // data
      0,                           // itemsize (ignored)
      NPY_ARRAY_FARRAY,            // flags (Armadillo buffers are aligned)
      nullptr                      // obj (ignored)
    )
  };
  if (!ar)
    return {};
  // set base object so NumPy array owns the capsule
  if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), capsule.release()) < 0)
    return {};
  return ar;
}
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)

/**
 * Lightweight flat view of a NumPy array.
 *
 * Holds only the NumPy array data pointer and size. If there is no need to
 * know the shape or strides of the NumPy array, use this class.
 *
 * Data buffer must be known to be aligned and writable if need be before use.
 *
 * @tparam T Data type
 */
template <typename T>
class ndarray_flat_view : public flat_view<T> {
public:
  using value_type = T;

  /**
   * Default ctor.
   *
   * Constructs a view with no data.
   */
  ndarray_flat_view() noexcept = default;

  /**
   * Ctor.
   *
   * @note Array must already be of correct type.
   *
   * @param arr NumPy array
   */
  ndarray_flat_view(PyArrayObject* arr) noexcept
    : flat_view<T>{
        static_cast<T*>(PyArray_DATA(arr)),
        // static_cast avoids C2397 narrowing error with MSVC since uniform
        // initialization prohibits implicit narrowing conversions
        static_cast<std::size_t>(PyArray_SIZE(arr))
      }
  {}
};

/**
 * Lightweight matrix view of a NumPy array.
 *
 * Data buffer must be known to be aligned and writeable if need be before use
 * and in the proper ordering (e.g. row vs. column major).
 *
 * @tparam T Data type
 * @tparam R Element ordering
 */
template <typename T, element_order R = element_order::c>
class ndarray_matrix_view : public matrix_view<T, R> {
public:
  using value_type = T;

  /**
   * Default ctor.
   *
   * Constructs a view with no data.
   */
  ndarray_matrix_view() noexcept = default;

  /**
   * Ctor.
   *
   * @note Array must already be of correct type and layout.
   *
   * @param arr NumPy array
   */
  ndarray_matrix_view(PyArrayObject* arr) noexcept
    : matrix_view<T, R>{
        static_cast<T*>(PyArray_DATA(arr)),
        // avoid C2397 narrowing error with MSVC
        static_cast<std::size_t>(PyArray_DIM(arr, 0)),
        static_cast<std::size_t>(PyArray_DIM(arr, 1))
      }
  {}
};

/**
 * Return the NumPy include directory as a static string.
 *
 * This calls `numpy.get_include` underneath so Python must be running.
 *
 * @returns NumPy include directory. On error, empty with Python exception set
 */
inline const auto& npy_get_include()
{
  static auto dir = []() -> std::string
  {
    // import NumPy module
    auto np = py_import("numpy");
    if (!np)
      return {};
    // retrieve get_include member
    auto np_get_include = py_getattr(np, "get_include");
    if (!np_get_include)
      return {};
    // invoke to get include directory
    auto py_inc_dir = py_call(np_get_include);
    if (!py_inc_dir)
      return {};
    // translate to C++ string
    auto inc_dir = py_utf8_string(py_inc_dir);
    if (inc_dir.empty())
      return {};
    // success
    return inc_dir;
  }();
  return dir;
}

/**
 * Return the NumPy include directory as a static path object.
 *
 * This calls `numpy.get_include` underneath so Python must be running.
 *
 * @returns NumPy include path. On error, empty with Python exception set
 */
inline const auto& npy_include_dir()
{
  static std::filesystem::path path{npy_get_include()};
  return path;
}

#if NPYGL_HAS_CC_20
/**
 * Create a `std::span` of the specified type from the NumPy array.
 *
 * Data buffer must be known to be aligned and writable if need be before use.
 *
 * @tparam T Data type
 *
 * @param arr NumPy array
 */
template <npy_type T>
inline std::span<T> make_span(PyArrayObject* arr) noexcept
{
  return {
    static_cast<T*>(PyArray_DATA(arr)),
    // static_cast avoids C2397 narrowing error with MSVC
    static_cast<std::size_t>(PyArray_SIZE(arr))
  };
}
#endif  // !NPYGL_HAS_CC_20

}  // namespace npygl

#endif  // NPYGL_NDARRAY_HH_
