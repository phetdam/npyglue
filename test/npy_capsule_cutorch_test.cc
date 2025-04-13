/**
 * @file npy_capsule_cutorch_test.cc
 * @author Derek Huang
 * @brief C++ program to test NumPy arrays backed by Torch GPU tensors
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <iostream>
#include <typeinfo>  // for well-formed typeid usage
#include <utility>

#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>

#include "npygl/demangle.hh"
#include "npygl/ndarray.hh"  // includes <numpy/ndarrayobject.h>
#include "npygl/python.hh"

// TODO: share the same testing infrastructure with npy_capsule_test.cc

int main()
{
  // initialize Python + import NumPy API
  npygl::npy_api_import(npygl::py_init());
  npygl::py_error_exit();
  // generator for reproducibility
  // note: no mutex acquisition not needed if single thread
  auto gen = at::make_generator<at::CUDAGeneratorImpl>();
  // create GPU tensor + print
  auto ten = torch::randn({6, 4}, gen, torch::kCUDA);
  std::cout << "-- " << npygl::type_name(typeid(decltype(ten))) << '\n';
  std::cout << ten << std::endl;
  // now move GPU tensor into NumPy array. this will cause the GPU tensor
  // contents to be *copied* back to the CPU for NumPy access
  auto arr = npygl::make_ndarray(std::move(ten));
  npygl::py_error_exit();
  // print CPU tensor created from GPU tensor
  std::cout << "-- " << Py_TYPE(arr) << '\n' << arr << std::endl;
  // get the base capsule throw a view
  npygl::cc_capsule_view ten_view{PyArray_BASE(arr.as<PyArrayObject>())};
  npygl::py_error_exit();
  std::cout << "-- " << npygl::type_name(*ten_view.info()) << '\n';
  std::cout << *ten_view.as<torch::Tensor>() << std::endl;
  return EXIT_SUCCESS;
}
