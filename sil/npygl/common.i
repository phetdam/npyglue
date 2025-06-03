/**
 * @file common.i
 * @author Derek Huang
 * @brief npyglue SWIG common helpers
 * @copyright MIT License
 */

// ensure SWIG is running in C++ mode
#ifndef __cplusplus
#error SWIG C++ processing must be enabled with -c++
#endif  // __cplusplus

// ensure SWIG is running to generate Python wrappers
#ifndef SWIGPYTHON
#error Python is the only supported target language
#endif  // SWIGPYTHON
