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

// TODO: add C standards support

#endif  // NPYGL_FEATURES_H_
