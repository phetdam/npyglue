/**
 * @file llvm_Demangle_test.cc
 * @author Derek Huang
 * @brief C++ program testing LLVM's demangler on `type_traits_test_driver`
 * @copyright MIT License
 *
 * @file See npygl::testing::type_traits_test_driver for some historical
 *  background on why `llvm::itaniumDemangle` was introduced.
 */

#include <cstdlib>
#include <iostream>
#include <typeinfo>  // for well-formed typeid() usage

// note: usually this should not be defined by hand in a source file
#define NPYGL_USE_LLVM_DEMANGLE
#include "npygl/demangle.hh"
#include "npygl/testing/type_traits_test_driver.hh"

int main()
{
  using driver_type = npygl::testing::type_traits_test_driver;
  std::cout << npygl::type_name(typeid(driver_type)) << std::endl;
  return EXIT_SUCCESS;
}
