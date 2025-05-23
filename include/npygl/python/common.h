/**
 * @file python/common.h
 * @author Derek Huang
 * @brief C/C++ header for common shared Python macros
 * @copyright MIT License
 */

#ifndef NPYGL_PYTHON_COMMON_H_
#define NPYGL_PYTHON_COMMON_H_

/**
 * Create a Python hex version number from the given components.
 *
 * @param major Python major version, e.g. the 3 in 3.4.1a2
 * @param minor Python minor version, e.g. the 4 in 3.4.1a2
 * @param micro Python micro version, e.g. the 1 in 3.4.1a2
 * @param level Python release level, e.g. `PY_RELEASE_LEVEL_FINAL`, which
 *  expands to `0xF`. This is the a in 3.4.1a2. Can be `0xB`, `0xC`, `0xF`.
 * @param serial Python release serial, e.g. the 2 in 3.4.1a2
 */
#define NPYGL_PY_VERSION_EX(major, minor, micro, level, serial) \
  (((major) << 24) | ((minor) << 16) | ((micro) << 8) | (level << 4) | (serial))

/**
 * Create a Python release hex version number.
 *
 * The release level is final (0xF) with the final release serial of zero.
 *
 * @param major Python major version, e.g. the 3 in 3.4.1
 * @param minor Python minor version, e.g. the 4 in 3.4.1
 * @param micro Python micro version, e.g. the 1 in 3.4.1
 */
#define NPYGL_PY_VERSION(major, minor, micro) \
  NPYGL_PY_VERSION_EX(major, minor, micro, 0xF, 0)

#endif  // NPYGL_PYTHON_COMMON_H_
