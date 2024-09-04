/**
 * @file numpy.i
 * @author Derek Huang
 * @brief SWIG interface file for npyglue NumPy array helpers
 * @copyright MIT License
 */

// ensure SWIG is running in C++ mode
#ifndef __cplusplus
#error "numpy.i: SWIG C++ processing must be enabled with -c++"
#endif  // __cplusplus

// ensure SWIG is running to generate Python wrappers
#ifndef SWIGPYTHON
#error "numpy.i: can only be used with Python as target language"
#endif  // SWIGPYTHON

%{
#include <exception>
#include <stdexcept>

#include <npygl/npy_helpers.hh>  // includes <numpy/ndarrayobject.h>
#include <npygl/py_helpers.hh>
%}

// forward declaration of the ndarray_flat_view template in the npygl namespace
// so that SWIG can properly match typemaps against unqualified use of the type
// name inside a namespace in processed C++ code without us needing to %include
// the actual header. npy_helpers.hh is too complicated for SWIG and would
// require that #ifndef SWIG ... #endif blocks be sprinkled throughout.
namespace npygl {

template <typename T>
class ndarray_flat_view;

}  // namespace npygl

/**
 * Typemap macro for converting Python input into a new read-only NumPy array.
 *
 * This macro simplifies creation of `npygl::ndarray_flat_view<T>` typemaps
 * for all the types that have `npy_type_traits` specializations.
 *
 * Typically the type is a const-qualified type since the view is read-only.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_FLAT_VIEW_IN_TYPEMAP(type)
%typemap(in) npygl::ndarray_flat_view<type> AR_IN (npygl::py_object in) {
  // attempt to create new input array (C order, avoid copy if possible)
  in = npygl::make_ndarray<type>($input, NPY_ARRAY_IN_ARRAY);
  if (!in)
    SWIG_fail;
  // create view
  $1 = npygl::ndarray_flat_view<type>{in.as<PyArrayObject>()};
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
%apply npygl::ndarray_flat_view<type> AR_IN {
  npygl::ndarray_flat_view<type> name
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
%apply npygl::ndarray_flat_view<type> AR_IN { npygl::ndarray_flat_view<type> };
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
  // attempt to create new output array to modify through view
  res = npygl::make_ndarray<type>($input);
  if (!res)
    SWIG_fail;
  // create view
  $1 = npygl::ndarray_flat_view<type>{res.as<PyArrayObject>()};
}

/**
 * Typemap releasing modified NumPy array back to Python.
 */
%typemap(argout) npygl::ndarray_flat_view<type> AR_INOUT {
  // release value back to Python
  $result = res$argnum.release();
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
%apply npygl::ndarray_flat_view<type> AR_INOUT { npygl::ndarray_flat_view<type> };
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
 * Typemap macro for converting Python input into a new read-only NumPy array.
 *
 * This macro simplifies creation of `std::span<T>` typemaps for relevant types.
 *
 * Typically the type is a const-qualified type since the view is read-only.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_STD_SPAN_IN_TYPEMAP(type)
%typemap(in) std::span<type> AR_IN (npygl::py_object in) {
  // attempt to create new input array (C order, avoid copy if possible)
  in = npygl::make_ndarray<type>($input, NPY_ARRAY_IN_ARRAY);
  if (!in)
    SWIG_fail;
  // create STL span
  $1 = npygl::make_span<type>(in.as<PyArrayObject>());
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
%apply std::span<type> AR_IN { std::span<type> name };
%enddef  // NPYGL_APPLY_STD_SPAN_IN_TYPEMAP(type, name)

/**
 * Typemap application macro for applying all STL span in typemaps.
 *
 * This macro applies the typemap to every occurrence of the type.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_APPLY_STD_SPAN_IN_TYPEMAPS(type)
%apply std::span<type> AR_IN { std::span<type> };
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
  // attempt to create new output array to modify through view
  res = npygl::make_ndarray<type>($input);
  if (!res)
    SWIG_fail;
  // create STL span
  $1 = npygl::make_span<type>(res.as<PyArrayObject>());
}

/**
 * Typemap releasing modified NumPy array back to Python.
 */
%typemap(argout) std::span<type> AR_INOUT {
  // release value back to Python
  $result = res$argnum.release();
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
