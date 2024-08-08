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
 * Typemap converting Python input into new double NumPy array to modify.
 */
%typemap(in) npygl::ndarray_flat_view<double> AR_INOUT (npygl::py_object res)
{
  // attempt to create new output array to modify through view
  res = npygl::make_ndarray<double>($input);
  if (!res)
    SWIG_fail;
  // create view
  $1 = npygl::ndarray_flat_view<double>{res.as<PyArrayObject>()};
}

/**
 * Typemap releasing modified double NumPy array back to Python.
 */
%typemap(argout) npygl::ndarray_flat_view<double> AR_INOUT {
  // release value back to Python
  $result = res$argnum.release();
}

/**
 * Typemap converting Python input into new float NumPy array to modify.
 */
%typemap(in) npygl::ndarray_flat_view<float> AR_INOUT (npygl::py_object res)
{
  // attempt to create new output array to modify through view
  res = npygl::make_ndarray<float>($input);
  if (!res)
    SWIG_fail;
  // create view
  $1 = npygl::ndarray_flat_view<float>{res.as<PyArrayObject>()};
}

/**
 * Typemap releasing modified double NumPy array back to Python.
 */
%typemap(argout) npygl::ndarray_flat_view<float> AR_INOUT {
  // release value back to Python
  $result = res$argnum.release();
}
