/**
 * @file ndarray.i
 * @author Derek Huang
 * @brief npyglue SWIG NumPy array helpers
 * @copyright MIT License
 */

// ensure SWIG is running in C++ mode
#ifndef __cplusplus
#error "SWIG C++ processing must be enabled with -c++"
#endif  // __cplusplus

// ensure SWIG is running to generate Python wrappers
#ifndef SWIGPYTHON
#error "Python is the only supported target language"
#endif  // SWIGPYTHON

%{
#include <cstddef>
#include <sstream>

#include <npygl/ndarray.hh>      // includes <numpy/ndarrayobject.h>
#include <npygl/python.hh>
#include <npygl/range_views.hh>
#include <npygl/warnings.h>

namespace npygl {
namespace {

/**
 * Convert the `element_order` data layout enum to a NumPy data layout flag.
 *
 * @todo Consider promoting this to `ndarray.hh` later.
 *
 * @param order Element order
 */
constexpr auto npy_layout(element_order order) noexcept
{
  switch (order) {
  case element_order::f:
    return NPY_ARRAY_F_CONTIGUOUS;
  default:
    return NPY_ARRAY_C_CONTIGUOUS;
  }
}

/**
 * Return NumPy input array flags given the specified data ordering.
 *
 * This function returns either `NPY_ARRAY_IN_ARRAY` or `NPY_ARRAY_IN_FARRAY`.
 *
 * @todo Consider promoting this to `ndarray.hh` later.
 *
 * @param order Element order
 */
constexpr auto npy_in_flags(element_order order) noexcept
{
  return npy_layout(order) | NPY_ARRAY_ALIGNED;
}

// GCC complains about check_dims() not being used
NPYGL_GNU_WARNING_PUSH()
NPYGL_GNU_WARNING_DISABLE(unused-function)
/**
 * Check NumPy array dimensions and set a Python exception on error.
 *
 * This function returns `false` if the NumPy array doesn't have exactly the
 * number of dimensions specified and will set a Python runtime error.
 *
 * @param arr NumPy array
 * @param ndim Number of required NumPy array dimensions
 */
bool check_dims(PyArrayObject* arr, std::size_t ndim)
{
  // dimensions match
  // note: cast to suppress C2397 and -Wnarrowing
  if (ndim == static_cast<std::size_t>(PyArray_NDIM(arr)))
    return true;
  // otherwise set exception
  std::stringstream ss;
  ss << "NumPy array shape " << ndarray_dims_view{arr} << " has " <<
    PyArray_NDIM(arr) << " != required " << ndim << " dimensions";
  PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
  return false;
}
NPYGL_GNU_WARNING_POP()

}  // namespace
}  // namespace npygl
%}

// forward declaration of the ndarray_flat_view template in the npygl namespace
// so that SWIG can properly match typemaps against unqualified use of the type
// name inside a namespace in processed C++ code without us needing to %include
// the actual header. ndarray.hh is too complicated for SWIG and would require
// that #ifndef SWIG ... #endif blocks be sprinkled throughout.
namespace npygl {

template <typename T>
class ndarray_flat_view;

// note: declaration of element_order to avoid including range_views.hh. we
// also don't want to expose the actual definition to SWIG; if we do, then SWIG
// will wrap it and generate constants in the Python module
enum class element_order : int;

// TODO: may have to drop use of non-type type parameter since SWIG doesn't
// seem to cope well with having a scoped enum non-type type parameter
template <typename T, element_order R>
class ndarray_2d_view;

}  // namespace npygl

/**
 * Typemap macro for converting Python input into a new read-only NumPy array.
 *
 * This macro simplifies creation of `npygl::ndarray_flat_view<T>` typemaps
 * for all the types that have `npy_type_traits` specializations.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_FLAT_VIEW_IN_TYPEMAP(type)
%typemap(in) npygl::ndarray_flat_view<const type> AR_IN (npygl::py_object in) {
  // attempt to create new input array (C order, avoid copy if possible)
  in = npygl::make_ndarray<type>($input, NPY_ARRAY_IN_ARRAY);
  if (!in)
    SWIG_fail;
  // create view
  $1 = in.as<PyArrayObject>();
}
%enddef  // NPYGL_FLAT_VIEW_IN_TYPEMAP(type)

/**
 * Typemap application macro for applying a flat view in typemap.
 *
 * This macro is used to apply the typemap to a specific type + name pair.
 *
 * @param type C/C++ view class element type
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAP(type, name)
%apply npygl::ndarray_flat_view<const type> AR_IN {
  npygl::ndarray_flat_view<const type> name
};
%enddef  // NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all flat view in typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAPS(type)
%apply npygl::ndarray_flat_view<const type> AR_IN {
  npygl::ndarray_flat_view<const type>
};
%enddef  // NPYGL_APPLY_FLAT_VIEW_IN_TYPEMAPS(type)

/**
 * Typemap macro for converting Python input into a new NumPy array to modify.
 *
 * This macro simplifies creation of `npygl::ndarray_flat_view<T>` typemaps
 * for all the types that have `npy_type_traits` specializations.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_FLAT_VIEW_INOUT_TYPEMAP(type)
/**
 * Typemap converting Python input into new NumPy array to modify.
 *
 * This is intended to be applied to a view through which changes are made.
 */
%typemap(in) npygl::ndarray_flat_view<type> AR_INOUT (npygl::py_object res) {
  // attempt to create new inout array view to modify (writes back if copy)
  res = npygl::make_ndarray<type>($input, NPY_ARRAY_INOUT_ARRAY);
  if (!res)
    SWIG_fail;
  // create view
  $1 = res.as<PyArrayObject>();
}
%enddef  // NPYGL_FLAT_VIEW_INOUT_TYPEMAP(type)

/**
 * Typemap application macro for applying a flat view in/out typemap.
 *
 * This macro is used to apply the typemap to a specific type + name pair.
 *
 * @param type C/C++ view class element type
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAP(type, name)
%apply npygl::ndarray_flat_view<type> AR_INOUT {
  npygl::ndarray_flat_view<type> name
};
%enddef  // NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all flat view in/out typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAPS(type)
%apply npygl::ndarray_flat_view<type> AR_INOUT {
  npygl::ndarray_flat_view<type>
};
%enddef  // NPYGL_APPLY_FLAT_VIEW_INOUT_TYPEMAPS(type)

/**
 * Typemap clearing macro for a single flat view typemap.
 *
 * @param type C/C++ view class element type
 * @param name Parameter name typemap was applied to
 */
%define NPYGL_CLEAR_FLAT_VIEW_TYPEMAP(type, name)
%clear npygl::ndarray_flat_view<type> name;
%enddef   // NPYGL_CLEAR_FLAT_VIEW_TYPEMAP(type, name)

/**
 * Typemap clearing macro for the flat view typemaps.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(type)
%clear npygl::ndarray_flat_view<type>;
%enddef  // NPYGL_CLEAR_FLAT_VIEW_TYPEMAPS(type)

/**
 * Typemap macro for converting returned C++ objects into new NumPy arrays.
 *
 * @note If the `type` is a template instantiation you will need to wrap the
 *  type in parentheses, e.g. with `(std::vector<T, std::allocator<T>>)`.
 *
 * @param type C/C++ object compatible with `npygl::make_ndarray`
 */
%define NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(type)
%typemap(out) type {
  $result = npygl::make_ndarray(std::move($1)).release();
}
%enddef  // NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(type, allocator)

/**
 * Typemap clearing macro for a C++ object being returned as a NumPy array.
 *
 * @note If the `type` is a template instantiation you will need to wrap the
 *  type in parentheses, e.g. with `(std::vector<T, std::allocator<T>>)`.
 *
 * @param type C/C++ object compatible with `npygl::make_ndarray`
 */
%define NPYGL_CLEAR_NDARRAY_OUT_TYPEMAP(type)
%typemap(out) type;
%enddef  // NPYGL_CLEAR_NDARRAY_OUT_TYPEMAP(type, allocator)

/**
 * Typemap macro for converting Python input into a new read-only NumPy array.
 *
 * This macro simplifies creation of `std::span<T>` typemaps for relevant types.
 *
 * Typically the type is a const-qualified type since the view is read-only.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_STD_SPAN_IN_TYPEMAP(type)
%typemap(in) std::span<const type> AR_IN (npygl::py_object in) {
  // attempt to create new input array (C order, avoid copy if possible)
  in = npygl::make_ndarray<type>($input, NPY_ARRAY_IN_ARRAY);
  if (!in)
    SWIG_fail;
  // create STL span
  $1 = npygl::make_span<const type>(in.as<PyArrayObject>());
}
%enddef  // NPYGL_STD_SPAN_IN_TYPEMAP(type)

/**
 * Typemap application macro for applying a STL span in typemap.
 *
 * This macro is used to apply the typemap to a specific type + name pair.
 *
 * @param type C/C++ view class element type
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_APPLY_STD_SPAN_IN_TYPEMAP(type, name)
%apply std::span<const type> AR_IN { std::span<const type> name };
%enddef  // NPYGL_APPLY_STD_SPAN_IN_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all STL span in typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_APPLY_STD_SPAN_IN_TYPEMAPS(type)
%apply std::span<const type> AR_IN { std::span<const type> };
%enddef  // NPYGL_APPLY_STD_SPAN_IN_TYPEMAPS(type)

/**
 * Typemap macro for converting Python input into a new NumPy array to modify.
 *
 * This macro simplifies creation of `std::span<T>` typemaps for relevant types.
 *
 * @note Requires C++20 compiler to be used on generated code.
 *
 * @param type C/C++ type with `npy_type_traits` specialization
 */
%define NPYGL_STD_SPAN_INOUT_TYPEMAP(type)
/**
 * Typemap converting Python input into new NumPy array to modify.
 *
 * This is intended to be applied to a view through which changes are made.
 */
%typemap(in) std::span<type> AR_INOUT (npygl::py_object res) {
  // attempt to create new inout array view to modify (writes back if copy)
  res = npygl::make_ndarray<type>($input, NPY_ARRAY_INOUT_ARRAY);
  if (!res)
    SWIG_fail;
  // create STL span
  $1 = npygl::make_span<type>(res.as<PyArrayObject>());
}
%enddef  // NPYGL_STD_SPAN_INOUT_TYPEMAP(type)

/**
 * Typemap application macro for applying a STL span in/out typemap.
 *
 * This macro is used to apply the typemap to a specific type + name pair.
 *
 * @param type C/C++ type with `npy_type_traits` specialization
 * @param name Parameter name to apply typemape to
 */
%define NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAP(type, name)
%apply std::span<type> AR_INOUT { std::span<type> name };
%enddef  // NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all STL span in/out typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ type with `npy_type_traits` specialization
 */
%define NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAPS(type)
%apply std::span<type> AR_INOUT { std::span<type> };
%enddef  // NPYGL_APPLY_STD_SPAN_INOUT_TYPEMAPS(type)

/**
 * Typemap clearing macro for a single STL span typemap.
 *
 * @param type C/C++ type with `npy_type_traits` specialization
 * @param name Parameter name typemap was applied to
 */
%define NPYGL_CLEAR_STD_SPAN_TYPEMAP(type, name)
%clear std::span<type> name;
%enddef  // NPYGL_CLEAR_STD_SPAN_TYPEMAP(type, name)

/**
 * Typemap clearing macro for the STL span typemaps.
 *
 * @param type C/C++ type with `npy_type_traits` specialization
 */
%define NPYGL_CLEAR_STD_SPAN_TYPEMAPS(type)
%clear std::span<type>;
%enddef  // NPYGL_CLEAR_STD_SPAN_TYPEMAPS(type)

/**
 * Typemap macro for converting Python input into a read-nly 2D NumPy array.
 *
 * Unlike the `ndarray_flat_view<T>` typemaps this *requires* that the created
 * NumPy array has *exactly* 2 dimensions and is of a particular data order.
 *
 * This macro simplifies creation of `npygl::ndarray_2d_view<T>` typemaps for
 * all the types that have `npy_type_traits` specializations.
 *
 * Typically the type is a const-qualified type since the view is read-only.
 *
 * @param type C/C++ view class element type
 * @param order Element ordering enum value
 */
%define NPYGL_2D_VIEW_IN_TYPEMAP(type, order)
%typemap(in) npygl::ndarray_2d_view<const type, order> AR_IN
  (npygl::py_object in) {
  // attempt to create new input array (avoid copy if possible)
  in = npygl::make_ndarray<type>($input, npygl::npy_in_flags(order));
  if (!in)
    SWIG_fail;
  // get array pointer + check dimensions
  auto arr = in.as<PyArrayObject>();
  if (!npygl::check_dims(arr, 2u))
    SWIG_fail;
  // create view
  $1 = in.as<PyArrayObject>();
}
%enddef  // NPYGL_2D_VIEW_IN_TYPEMAP(type)

/**
 * Typemap application macro for applying a 2D view in typemap.
 *
 * This macro is used to apply the typemap to a specific type + name pair.
 *
 * @param type C/C++ view class element type
 * @param order Element ordering enum value
 * @param name Parameter name to apply typemap to
 */
%define NPYGL_APPLY_2D_VIEW_IN_TYPEMAP(type, order, name)
%apply npygl::ndarray_2d_view<const type, order> AR_IN {
  npygl::ndarray_2d_view<const type, order> name
};
%enddef  // NPYGL_APPLY_2D_VIEW_IN_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all 2D view in typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ view class element type
 * @param order Element ordering enum value
 */
%define NPYGL_APPLY_2D_VIEW_IN_TYPEMAPS(type, order)
%apply npygl::ndarray_2d_view<const type, order> AR_IN {
  npygl::ndarray_2d_view<const type, order>
};
%enddef  // NPYGL_APPLY_2D_VIEW_IN_TYPEMAPS(type)

// supported in flat view typemaps
NPYGL_FLAT_VIEW_IN_TYPEMAP(double)
NPYGL_FLAT_VIEW_IN_TYPEMAP(float)
NPYGL_FLAT_VIEW_IN_TYPEMAP(int)
NPYGL_FLAT_VIEW_IN_TYPEMAP(unsigned int)
NPYGL_FLAT_VIEW_IN_TYPEMAP(long)
NPYGL_FLAT_VIEW_IN_TYPEMAP(unsigned long)

// supported in C++20 STL span typemaps
NPYGL_STD_SPAN_IN_TYPEMAP(double)
NPYGL_STD_SPAN_IN_TYPEMAP(float)
NPYGL_STD_SPAN_IN_TYPEMAP(int)
NPYGL_STD_SPAN_IN_TYPEMAP(unsigned int)
NPYGL_STD_SPAN_IN_TYPEMAP(long)
NPYGL_STD_SPAN_IN_TYPEMAP(unsigned long)

// supported in + out flat view typemaps
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(double)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(float)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(int)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(unsigned int)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(long)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(unsigned long)

// supported in + out C++20 STL span typemaps
NPYGL_STD_SPAN_INOUT_TYPEMAP(double)
NPYGL_STD_SPAN_INOUT_TYPEMAP(float)
NPYGL_STD_SPAN_INOUT_TYPEMAP(int)
NPYGL_STD_SPAN_INOUT_TYPEMAP(unsigned int)
NPYGL_STD_SPAN_INOUT_TYPEMAP(long)
NPYGL_STD_SPAN_INOUT_TYPEMAP(unsigned long)

// supported in 2D view typemaps
NPYGL_2D_VIEW_IN_TYPEMAP(double, npygl::element_order::c)
NPYGL_2D_VIEW_IN_TYPEMAP(double, npygl::element_order::f)
NPYGL_2D_VIEW_IN_TYPEMAP(float, npygl::element_order::c)
NPYGL_2D_VIEW_IN_TYPEMAP(float, npygl::element_order::f)
// TODO: add other NumPy integral types

/**
 * Macro for starting the parameters section of the NumPy docstring.
 */
%define NPYGL_NPYDOC_PARAMETERS
"Parameters\n"
"----------\n"
%enddef  // NPYGL_NPYDOC_PARAMETERS

/**
 * Macro for starting the returns section of the NumPy docstring.
 */
%define NPYGL_NPYDOC_RETURNS
"Returns\n"
"-------\n"
%enddef  // NPYGL_NPYDOC_RETURNS
