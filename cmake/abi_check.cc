/**
 * @file abi_check.cc
 * @author Derek Huang
 * @brief C++ source for checking if a compiler follows the Itanium C++ ABI
 * @copyright MIT License
 */

#include <cxxabi.h>

#include <cstdlib>

// note: sources used with CMake's try_compile are required to define main
int main()
{
  return EXIT_SUCCESS;
}
