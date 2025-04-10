/**
 * @file py_capsule_cutorch_test.cc
 * @author Derek Huabg
 * @brief C++ program to test PyCapsule creation from GPU-backed Torch tensors
 * @copyright MIT License
 */

#include <cstdlib>
#include <iostream>
#include <typeinfo>  // for well-formed typeid usage
#include <utility>

#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // for at::CUDAGeneratorImpl
#include <torch/torch.h>

#include "npygl/demangle.hh"
#include "npygl/ostream.hh"
#include "npygl/python.hh"

int main()
{
  // init Python + print version
  npygl::py_init();
  std::cout << Py_GetVersion() << std::endl;
  // create CUDA generator + tensor
  auto gen = at::make_generator<at::CUDAGeneratorImpl>();
  auto ten = torch::randn({3, 4, 5}, gen, torch::kCUDA);
  // note: following is copied from capsule_test in py_capsule_test.cc
  // wrapped stream
  npygl::ostream_wrapper stream{std::cout};
  // create capsule from the moved object
  auto cap = npygl::py_object::create(std::move(ten));
  npygl::py_error_exit();
  // take capsule view
  npygl::cc_capsule_view view{cap};
  npygl::py_error_exit();
  // print the type name
  const auto& val = *view.as<decltype(ten)>();
  stream << "-- " << npygl::type_name(view.info()) << '\n';
  // write the object to the output stream + flush
  stream << val << std::endl;
  // create copy via addition + print
  auto val2 = val + val;
  stream << "-- " << npygl::type_name(typeid(val2)) << '\n';
  stream << val2 << std::endl;
  return EXIT_SUCCESS;
}
