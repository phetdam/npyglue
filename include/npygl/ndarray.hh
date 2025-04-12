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
#include "npygl/type_traits.hh"
#include "npygl/warnings.h"

// C++20
#if NPYGL_HAS_CC_20
#include <span>
#endif  // !NPYGL_HAS_CC_20

// Armadillo if desired
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
// Eigen 3 if desired
#if NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3 && !defined(NPYGL_NO_EIGEN3)
// PyTorch C++ (LibTorch) if desired
#if NPYGL_HAS_LIBTORCH && !defined(NPYGL_NO_LIBTORCH)
#include <torch/torch.h>
#endif  // NPYGL_HAS_LIBTORCH && !defined(NPYGL_NO_LIBTORCH)

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
  T,
  std::void_t<typename npy_type_traits<std::remove_cv_t<T>>::type> >
  : std::true_type {};

/**
 * Boolean helper for checking if a type has NumPy type traits.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool has_npy_type_traits_v = has_npy_type_traits<T>::value;

/**
 * Functor to return a comma-separated NumPy type name list from C/C++ types.
 *
 * @tparam Ts... C/C++ types to get NumPy type names for
 */
template <typename... Ts>
struct npy_typename_lister {};

/**
 * Partial specialization when there is only one type.
 *
 * @tparam T C/C++ type to get NumPy type name for
 */
template <typename T>
struct npy_typename_lister<T> {
  std::string operator()() const
  {
    return npy_typename<T>;
  }
};

/**
 * Partial specialization for multiple types.
 *
 * @tparam T C/C++ type to get NumPy type name for
 * @tparam Ts... Other C/C++ types to get NumPy type names for
 */
template <typename T, typename... Ts>
struct npy_typename_lister<T, Ts...> {
  auto operator()() const
  {
    return npy_typename<T> + std::string{", "} + npy_typename_lister<Ts...>{}();
  }
};

/**
 * Partial specialization for a tuple of types.
 *
 * @tparam Ts... C/C++ types to get NumPy type names for
 */
template <typename... Ts>
struct npy_typename_lister<std::tuple<Ts...>> : npy_typename_lister<Ts...> {};

/**
 * Global lister to get the NumPy type name list from C/C++ types.
 *
 * This provides a functional interface to the `npy_typename_lister<Ts...>`.
 *
 * @tparam Ts... C/C++ types to get NumPy type names for
 */
template <typename... Ts>
inline constexpr npy_typename_lister<Ts...> npy_typename_list;

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
 * @deprecated This function may be removed due to its low utility.
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
 * @deprecated This function may be removed due to its low utility.
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
 * @deprecated This function may be removed due to its low utility.
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
 * @deprecated This function may be removed due to its low utility.
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
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct ndarray_capsule_builder {};

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
 * Global builder for creating a NumPy array from a Python capsule.
 *
 * This provides a functional interface to the `ndarray_capsule_builder<Ts...>`.
 *
 * @tparam Ts... Target C++ types
 */
template <typename... Ts>
inline constexpr ndarray_capsule_builder<Ts...> make_ndarray_from_capsule;

/**
 * Create a NumPy array from an appropriate C++ object.
 *
 * On error the `py_object` is empty and a Python exception is set.
 *
 * Construction involves using `py_object::create()` to create a `PyCapsule`
 * from the C++ object and then creating a NumPy array backed by this capsule
 * via the appropriate `make_ndarray_from_capsule` partial specialization.
 *
 * For details on how the NumPy array is constructed for a particular type see
 * the appropriate `ndarray_capsule_builder<...>` documentation comments.
 *
 * @tparam T C++ object interpretable as a multidimensional array
 *
 * @param obj C++ object to create a NumPy array from
 */
template <typename T, typename = std::enable_if_t<!std::is_reference_v<T>>>
py_object make_ndarray(T&& obj) noexcept
{
  // create capsule
  auto capsule = py_object::create(std::move(obj));
  if (!capsule)
    return {};
  // capsule view (should not error)
  cc_capsule_view view{capsule};
  if (!view)
    return {};
  // create NumPy array from capsule using builder
  return make_ndarray_from_capsule<T>(std::move(capsule), view.as<T>());
}

/**
 * Traits type to indicate if a NumPy array can be created from a C++ type.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct can_make_ndarray : std::false_type {};

/**
 * True specialization for C++ types `make_ndarray(T&&)` accepts.
 *
 * @note Since `make_ndarray(T&&)` does not restrict the template type we rely
 *  on the invokability of the builder specialization.
 *
 * @tparam T type
 */
template <typename T>
struct can_make_ndarray<
  T,
  std::void_t<
    decltype(
      make_ndarray_from_capsule<T>(std::declval<py_object>(), std::declval<T*>())
    )> >
  : std::true_type {};

/**
 * Helper to indicate if a NumPy arary can be created from a C++ type.
 *
 * @tparam T type
 */
template <typename T>
inline constexpr bool can_make_ndarray_v = can_make_ndarray<T>::value;

/**
 * NumPy array builder CRTP base class.
 *
 * This provides a safe, end-user invocable `operator()` that performs the
 * necessary type checking before calling the unsafe `operator()`.
 *
 * We use CRTP here as a form of static polymorphism.
 *
 * @tparam `ndarray_capsule_builder<T>` specialization
 */
template <typename Builder>
struct ndarray_capsule_builder_base {
  /**
   * Create a NumPy array backed by the C++ object managed by a Python capsule.
   *
   * @note We take the `py_object` by rvalue reference to semantically indicate
   *  that the Python object is to be consumed as a reference will be stolen.
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
 * NumPy array builder for a capsule holding a C++ vector.
 *
 * @tparam T Element type
 * @tparam A Allocator type
 */
template <typename T, typename A>
struct ndarray_capsule_builder<
  std::vector<T, A>, std::enable_if_t<has_npy_type_traits_v<T>> >
  : ndarray_capsule_builder_base<ndarray_capsule_builder<std::vector<T, A>>> {
  using object_type = std::vector<T, A>;

  /**
   * Create a 1D NumPy array with C memory layout backed by a C++ vector.
   *
   * The capsule-backed NumPy array will have flags `NPY_ARRAY_DEFAULT`.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @note This overload does not validate the capsule or the object pointer.
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
 * NumPy array builder a capsule holding an Eigen matrix.
 *
 * @tparam T Element type
 * @tparam R Number of rows
 * @tparam C Number of columns
 * @tparam O Matrix options
 * @tparam RMax Max number of rows
 * @tparam CMax max number of columns
 */
template <typename T, int R, int C, int O, int RMax, int RMin>
struct ndarray_capsule_builder<
  Eigen::Matrix<T, R, C, O, RMax, RMin>,
  std::enable_if_t<has_npy_type_traits_v<T>> >
  : ndarray_capsule_builder_base<
      ndarray_capsule_builder<Eigen::Matrix<T, R, C, O, RMax, RMin>> > {
  using object_type = Eigen::Matrix<T, R, C, O, RMax, RMin>;

  /**
   * Create a 2D NumPy array backed by an Eigen3 matrix object.
   *
   * The capsule-backed NumPy array will have flags `NPY_ARRAY_BEHAVED` and one
   * of `NPY_ARRAY_C_CONTIGUOUS` or `NPY_ARRAY_F_CONTIGUOUS`.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @note This overload does not validate the capsule or the object pointer.
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

// enable if we have Armadillo unless NPYGL_NO_ARMADILLO is defined
#if NPYGL_HAS_ARMADILLO && !defined(NPYGL_NO_ARMADILLO)
/**
 * NumPy array builder for a capsule holding an Armadillo matrix.
 *
 * @tparam T Element type
 */
template <typename T>
struct ndarray_capsule_builder<
  arma::Mat<T>, std::enable_if_t<has_npy_type_traits_v<T>> >
  : ndarray_capsule_builder_base<ndarray_capsule_builder<arma::Mat<T>>> {
  using object_type = arma::Mat<T>;

  /**
   * Create a 2D NumPy array backed by an Armadillo matrix object.
   *
   * The capsule-backed NumPy array will have flags `NPY_ARRAY_FARRAY`.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * @note If moving from an `arma::Row<T>` or an `arma::Col<T>` the resulting
   *  NumPy array is still 2D but one dimension will be 1.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_mat Pointer to C++ Armadillo matrix retrieved from the capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_mat) const noexcept
  {
    // create dims
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
    if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), cap.release()) < 0)
      return {};
    return ar;
  }
};

/**
 * NumPy array builder for a capsule holding an Armadillo column vector.
 *
 * We simply subclass because the `arma::Col<T>` is a subclass of the
 * `arma::Mat<T>` and so `ndarray_capsule_builder<arma::Mat<T>>::operator()`
 * logic can be applied to the `arma::Col<T>` to produce a NumPy column vector.
 *
 * @tparam T Element type
 */
template <typename T>
struct ndarray_capsule_builder<arma::Col<T>>
  : ndarray_capsule_builder<arma::Mat<T>> {};

/**
 * NumPy array builder for a capsule holding an Armadillo column vector.
 *
 * We have a specific partial specialization for the `arma::Row<T>` so that the
 * created NumPy array is a row vector, e.g. shape is `(n,)`, not `(1, n)`.
 *
 * @tparam T Element type
 */
template <typename T>
struct ndarray_capsule_builder<
  arma::Row<T>, std::enable_if_t<has_npy_type_traits_v<T>> >
  : ndarray_capsule_builder_base<ndarray_capsule_builder<arma::Row<T>>> {
  using object_type = arma::Row<T>;

  /**
   * Create a 1D NumPy array backed by an Armadillo row vector.
   *
   * The capsule-backed NumPy array will have flags `NPY_ARRAY_FARRAY`.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_mat Pointer to C++ Armadillo row vector retrieved from capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_vec) const noexcept
  {
    // only one dimension + cast to silence C4365 warning
    auto dims = static_cast<npy_intp>(cap_vec->n_cols);
    // create new 1D NumPy array from the data buffer
    py_object ar{
      PyArray_New(
        &PyArray_Type,               // subtype
        1,                           // nd
        &dims,                       // dims
        npy_typenum<T>,              // type_num
        nullptr,                     // strides
        cap_vec->memptr(),           // data
        0,                           // itemsize (ignored)
        NPY_ARRAY_FARRAY,            // flags (Armadillo buffers are aligned)
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

/**
 * NumPy array builder for a capsule holding an Armadillo cube.
 *
 * @tparam T Element type
 */
template <typename T>
struct ndarray_capsule_builder<
  arma::Cube<T>, std::enable_if_t<has_npy_type_traits_v<T>> >
  : ndarray_capsule_builder_base<ndarray_capsule_builder<arma::Cube<T>>> {
  using object_type = arma::Cube<T>;

  /**
   * Create a 3D NumPy array backed by an Armadillo cube object.
   *
   * The capsule-backed NumPy array will have flags `NPY_ARRAY_FARRAY`.
   *
   * On error the `py_object` is empty and a Python exception is set.
   *
   * @note Due to how the memory is laid out in an Armadillo cube the resulting
   *  shape of the NumPy array is `(n_rows, n_cols, n_slices)`. This is not a
   *  true 3D array or tensor; the Armadillo docs call the cube a pseudo-tensor.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_cube Pointer to C++ Armadillo cube retrieved from the capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_cube) const noexcept
  {
    // create dims
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
    // this results in a NumPy array that is neither C nor Fortran contiguous
    // so flags passed to PyArray_New are NPY_ARRAY_BEHAVED. essentially we use
    // strides to reinterpret a Fortran-contiguous (n_rows, n_cols, n_slices)
    // 3D array as a (n_slices, n_rows, n_cols) tensor-like 3D array.
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
    if (PyArray_SetBaseObject(ar.as<PyArrayObject>(), cap.release()) < 0)
      return {};
    return ar;
  }
};
#endif  // !NPYGL_HAS_ARMADILLO || defined(NPYGL_NO_ARMADILLO)

// enable if we have PyTorch C++ headers unless NPYGL_NO_LIBTORCH is defined
#if NPYGL_HAS_LIBTORCH && !defined(NPYGL_NO_LIBTORCH)
/**
 * Get the corresponding NumPy type value from a PyTorch data type.
 *
 * On error `NPY_NOTYPE` is returned (not a valid type).
 *
 * @param type PyTorch data type
 */
constexpr NPY_TYPES npy_type(at::ScalarType type) noexcept
{
  switch (type) {
    case at::ScalarType::Bool:
      return NPY_BOOL;
    case at::ScalarType::Char:
      return NPY_BYTE;
    case at::ScalarType::Short:
      return NPY_SHORT;
    case at::ScalarType::Int:
      return NPY_INT;
    case at::ScalarType::Long:
      return NPY_LONGLONG;
    case at::ScalarType::Byte:
      return NPY_UBYTE;
    case at::ScalarType::UInt16:
      return NPY_USHORT;
    case at::ScalarType::UInt32:
      return NPY_UINT;
    case at::ScalarType::UInt64:
      return NPY_ULONGLONG;
    case at::ScalarType::Half:
      return NPY_HALF;
    case at::ScalarType::Float:
      return NPY_FLOAT;
    case at::ScalarType::Double:
      return NPY_DOUBLE;
    case at::ScalarType::ComplexFloat:
      return NPY_CFLOAT;
    case at::ScalarType::ComplexDouble:
      return NPY_CDOUBLE;
    default:
      return NPY_NOTYPE;
  }
}

/**
 * NumPy array builder for a capsule holding a PyTorch tensor.
 *
 * The PyTorch tensor's type is given at runtime so this class is non-template.
 *
 * @note PyTorch provides an API for creating custom operators that should use
 *  pybind11 to wrap the native ``Tensor`` back into Python. This class is more
 *  of a curiosity than for practical usage unless you really want to convert a
 *  PyTorch tensor object into a plain NumPy array.
 */
template <>
struct ndarray_capsule_builder<torch::Tensor>
  : ndarray_capsule_builder_base<ndarray_capsule_builder<torch::Tensor>> {
private:
  /**
   * Validate and convert a PyTorch dtype to the NumPy type enum value.
   *
   * On error `NPY_NOTYPE` is returned and a Python exception is set.
   *
   * @param dtype PyTorch dtype
   */
  auto convert_type(caffe2::TypeMeta dtype) const
  {
    // note: we can use C10_UNLIKELY here since this branch is highly unlikely
    if (C10_UNLIKELY(!dtype.isScalarType())) {
      // note: technically does not provide noexcept guarantee
      std::stringstream ss;
      ss << NPYGL_PRETTY_FUNCTION_NAME <<
        ": torch::Tensor cannot have non-scalar dtype " << dtype.name();
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
      return NPY_NOTYPE;
    }
    // attempt to convert PyTorch scalar type to NumPy type
    // note: toScalarType should not fail since valid tensor types are scalar
    auto stype = dtype.toScalarType();
    auto type = npy_type(stype);
    // unsupported PyTorch scalar type
    if (type == NPY_NOTYPE) {
      std::stringstream ss;
      ss << NPYGL_PRETTY_FUNCTION_NAME << ": torch::Tensor scalar dtype " <<
        stype << " is unsupported";
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
      return NPY_NOTYPE;
    }
    // success
    return type;
  }

  /**
   * Retrieve the dimensions and strides of the tensor in a pair.
   *
   * If integral type (`int64_t`) used to index the Torch tensor via `sizes()`,
   * an `IntArrayRef`, is type-accessible as an `npy_intp`, we directly alias
   * the data of the `IntArrayRef`. Otherwise, to satisfy the strict aliasing
   * rule, we copy and return the values as a `std::vector<npy_intp>`.
   *
   * @param ten Tensor to get dimensions + strides for
   */
  auto retrieve_sizes(const torch::Tensor& ten) const
  {
    // number of dimensions in tensor (unused if sizeof conditions match)
    [[maybe_unused]] auto n_dim = ten.dim();
    // get dimensions from tensor
    auto dims = [&ten, n_dim]
    {
      // direct alias if type-accessible
      if constexpr (is_accessible_as_v<decltype(ten.size(0)), npy_intp>)
        return ten.sizes();
      // otherwise we are forced to copy
      else {
        // note: could we do a small vector optimization? for LP64 systems
        // however this is a discarded code path as sizeof(npy_intp) will be
        // equal to sizeof(int64_t), the element size of IntArrayRef
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4365)  // signed/unsigned mismatch
        std::vector<npy_intp> dims(n_dim);
        for (decltype(n_dim) i = 0; i < n_dim; i++)
          dims[i] = ten.size(i);
        return dims;
NPYGL_MSVC_WARNING_POP()
      }
    }();
    // get strides from tensor
    auto strides = [&ten, n_dim]
    {
      // direct alias if type-accessible
      if constexpr (is_accessible_as_v<decltype(ten.stride(0)), npy_intp>)
        return ten.strides();
      // otherwise we are forced to copy
      else {
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4365)  // signed/unsigned mismatch
        std::vector<npy_intp> strides(n_dim);
        for (decltype(n_dim) i = 0; i < n_dim; i++)
          strides[i] = ten.stride(i);
        return strides;
      }
NPYGL_MSVC_WARNING_POP()
    }();
    // return
    return std::make_pair(std::move(dims), std::move(strides));
  }

public:
  using object_type = torch::Tensor;

  /**
   * Create a NumPy array backed by a PyTorch tensor.
   *
   * @note This overload does not validate the capsule or the object pointer.
   *
   * @param cap Python capsule following `cc_capsule_view` protocol
   * @param cap_cube Pointer to C++ PyTorch tensor retrieved from the capsule
   */
  py_object operator()(py_object&& cap, object_type* cap_ten) const noexcept
  {
    // NumPy arrays can only be backed by CPU memory. if the tensor is not a
    // CPU tensor, e.g. on GPU, we need to convert it to a CPU tensor. we just
    // call make_ndarray recursively with a new CPU tensor
    if (!cap_ten->is_cpu())
      return make_ndarray(cap_ten->cpu());
    // get tensor type as NumPy type
    auto type = convert_type(cap_ten->dtype());
    // ensure strided (NumPy only works with strided, dense representations)
    if (cap_ten->layout() != c10::Layout::Strided) {
      PyErr_SetString(
        PyExc_ValueError,
        "only strided (dense) torch::Tensor can be converted to a NumPy array"
      );
      return {};
    }
    // get tensor dimensions and strides
    auto [dims, strides] = retrieve_sizes(*cap_ten);
    // create new NumPy array from tensor
    py_object ar{
      PyArray_New(
        &PyArray_Type,                // subtype
NPYGL_MSVC_WARNING_PUSH()
NPYGL_MSVC_WARNING_DISABLE(4244)  // possible loss of data due to narrowing
        cap_ten->dim(),               // nd
NPYGL_MSVC_WARNING_POP()
        // may be direct alias or copy depending on type accessibility
        dims.data(),                  // dims
        type,                         // type_num
        // may be direct alias or copy depending on type accessibility
        strides.data(),               // strides
        cap_ten->mutable_data_ptr(),  // data
        0,                            // itemsize (ignored)
        // FIXME: if possible we should specify NPY_ARRAY_CARRAY or
        // NPY_ARRAY_FARRAY depending on the strides
        NPY_ARRAY_BEHAVED,            // flags (aligned, unknown ordering)
        nullptr                       // obj (ignored)
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
#endif  // NPYGL_HAS_LIBTORCH && !defined(NPYGL_NO_LIBTORCH)

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
      ss << NPYGL_PRETTY_FUNCTION_NAME << ": capsule backed by " <<
        type_name(view.info()) << " is not one of supported " <<
        type_name_list<Ts...>();
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
    }
    // return (empty on error)
    return ar;
  }
};

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
