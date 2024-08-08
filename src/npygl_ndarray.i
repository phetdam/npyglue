/**
 * @file npygl_ndarray.i
 * @author Derek Huang
 * @brief SWIG interface file for npyglue NumPy array helpers
 * @copyright MIT License
 */

%{
#include <npygl/npy_helpers.hh>  // includes <numpy/ndarrayobject.h>
#include <npygl/py_helpers.hh>
%}

/**
 * Typemap macro for converting Python input into a new NumPy array to modify.
 *
 * This macro simplifies creation of `npygl::ndarray_flat_view<T>` typemaps
 * for all the the types that have `npy_type_traits` specializations.
 *
 * @param type C/C++ view class element type
 */
%define NPYGL_FLAT_VIEW_INOUT_TYPEMAP(type)
/**
 * Typemap converting Python input into new NumPy array to modify.
 *
 * This is intended to be applied to a view through which changes are made.
 */
%typemap(in) npygl::ndarray_flat_view<type> AR_INOUT (npygl::py_object res)
{
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
%enddef

// supported int + out flat view typemaps
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(double)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(float)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(int)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(unsigned int)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(long)
NPYGL_FLAT_VIEW_INOUT_TYPEMAP(unsigned long)
