/**
 * @file features.h
 * @author Derek Huang
 * @brief C/C++ header for compiler/platform feature detection
 * @copyright MIT License
 */

#ifndef NPYGL_FEATURES_H_
#define NPYGL_FEATURES_H_

#include "npygl/common.h"

// C++ standards support
#ifdef NPYGL_CPLUSPLUS
// C++11
#if NPYGL_CPLUSPLUS >= 201103L
#define NPYGL_HAS_CC_11 1
#else
#define NPYGL_HAS_CC_11 0
#endif  // NPYGL_CPLUSPLUS < 201103L
// C++14
#if NPYGL_CPLUSPLUS >= 201402L
#define NPYGL_HAS_CC_14 1
#endif  // NPYGL_CPLUSPLUS < 201402L
// C++17
#if NPYGL_CPLUSPLUS >= 201703L
#define NPYGL_HAS_CC_17 1
#endif  // NPYGL_CPLUSPLUS < 201703L
// C++20
#if NPYGL_CPLUSPLUS >= 202002L
#define NPYGL_HAS_CC_20 1
#endif  // NPYGL_CPLUSPLUS < 202002L
#endif  // NPYGL_CPLUSPLUS

// no C++ standards support
#ifndef NPYGL_HAS_CC_11
#define NPYGL_HAS_CC_11 0
#endif  // NPYGL_HAS_CC_11
#ifndef NPYGL_HAS_CC_14
#define NPYGL_HAS_CC_14 0
#endif  // NPYGL_HAS_CC_14
#ifndef NPYGL_HAS_CC_17
#define NPYGL_HAS_CC_17 0
#endif  // NPYGL_HAS_CC_17
#ifndef NPYGL_HAS_CC_20
#define NPYGL_HAS_CC_20 0
#endif  // NPYGL_HAS_CC_20

// 1 or 0 macro for Windows
#if defined(_WIN32)
#define NPYGL_WIN32 1
#else
#define NPYGL_WIN32 0
#endif  // !defined(_WIN32)

// TODO: add C standards support

// compiling with RTTI or not
// TODO: C++ capsule functionality requires RTTI; should we enforce?
#if defined(_CPPRTTI) || defined(__GXX_RTTI)  // MSVC, GCC-like
#define NPYGL_HAS_RTTI 1
#else
#define NPYGL_HAS_RTTI 0
#endif  // !defined(_CPPRTTI) && !defined(__GXX_RTTI)

// check if __has_include is available
#if defined(__has_include)
#define NPYGL_HAS_INCLUDE_CHECK 1
#else
#define NPYGL_HAS_INCLUDE_CHECK 0
#endif  // !defined(__has_include)

// check if the main NumPy array object header is available
#if NPYGL_HAS_INCLUDE_CHECK
#if __has_include(<numpy/ndarrayobject.h>)
#define NPYGL_HAS_NUMPY 1
#endif  // __has_include(<numpy/ndarrayobject.h>)
#endif  // NPYGL_HAS_INCLUDE_CHECK

#ifndef NPYGL_HAS_NUMPY
#define NPYGL_HAS_NUMPY 0
#endif  // NPYGL_HAS_NUMPY

// check if we have Eigen 3 available by looking for its signature file
#if NPYGL_HAS_INCLUDE_CHECK
#if __has_include(<signature_of_eigen3_matrix_library>)
#define NPYGL_HAS_EIGEN3 1
#endif  // __has_include(<signature_of_eigen3_matrix_library>)
#endif  // NPYGL_HAS_INCLUDE_CHECK

#ifndef NPYGL_HAS_EIGEN3
#define NPYGL_HAS_EIGEN3 0
#endif  // NPYGL_HAS_EIGEN3

// check if compiler follows the Itanium C++ ABI
#if NPYGL_HAS_INCLUDE_CHECK
#if __has_include(<cxxabi.h>)
#define NPYGL_HAS_ITANIUM_ABI 1
#endif  // __has_include(<cxxabi.h>)
#endif  // NPYGL_HAS_INCLUDE_CHECK

#ifndef NPYGL_HAS_ITANIUM_ABI
#define NPYGL_HAS_ITANIUM_ABI 0
#endif  // NPYGL_CC_ABI_ITANIUM

// check if we have Armadillo available
#if NPYGL_HAS_INCLUDE_CHECK
#if __has_include(<armadillo>)
#define NPYGL_HAS_ARMADILLO 1
#endif  // __has_include(<armadillo>)
#endif  // NPYGL_HAS_INCLUDE_CHECK

#ifndef NPYGL_HAS_ARMADILLO
#define NPYGL_HAS_ARMADILLO 0
#endif  // NPYGL_HAS_ARMADILLO

#endif  // NPYGL_FEATURES_H_
