cmake_minimum_required(VERSION 3.20)

##
# FindSYCL.cmake
#
# npyglue find module for locating SYCL development artifacts.
#
# This find module assumes a oneAPI-like distribution. Compared to the PyTorch
# FindSYCLToolkit.cmake find module, this one is more concise and not as ugly.
#
# CMPLR_ROOT will be checked first and then SYCL_ROOT CMake and environment
# variables in the typical find_package fashion.
#
# The SYCL_FOUND variable will set to TRUE or FALSE if SYCL development
# artifacts were correctly found or not. On success, the following helper
# variables are defined by the find module:
#
#   SYCL_COMPILER               Path to the SYCL compiler (icx)
#   SYCL_ROOT_DIR               Root directory of the SYCL installation
#   SYCL_INCLUDE_DIR            Include directory for SYCL/OpenCL headers
#   SYCL_LIBRARY                SYCL main [import] library
#
# On Windows, a successful completion defines additional variables:
#
#   SYCL_LIBRARY_DEBUG          SYCL debug import library
#   SYCL_LIBRARY_DLL            SYCL library DLL
#   SYCL_LIBRARY_DLL_DEBUG      SYCL debug library DLL
#
# For all platforms, the following IMPORTED library target is also defined:
#
#   SYCL:sycl
#
# A required call to FindOpenCL is also done internally in this find module.
#

include(FindPackageHandleStandardArgs)

# look for the SYCL compiler
find_program(
    SYCL_COMPILER
    NAMES icx  # dpcpp is another name but we stick with Intel-style name
    # CMPLR_ROOT has highest priority
    HINTS ENV CMPLR_ROOT "${SYCL_ROOT}" ENV SYCL_ROOT
    PATH_SUFFIXES bin
    NO_CACHE
)
# if not found, exit early
if(NOT SYCL_COMPILER)
    find_package_handle_standard_args(SYCL REQUIRED_VARS SYCL_COMPILER)
    return()
endif()

# get the compiler version
execute_process(
    COMMAND ${SYCL_COMPILER} --version
    TIMEOUT 5
    RESULT_VARIABLE _sycl_version_res
    OUTPUT_VARIABLE _sycl_version_out
    ERROR_VARIABLE _sycl_version_err
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
# if failed, exit
if(_sycl_version_res)
    find_package_handle_standard_args(
        SYCL
        REQUIRED_VARS SYCL_VERSION
        REASON_FAILURE_MESSAGE "error running icx --version")
    return()
endif()
# get regex from first line
string(
    REGEX REPLACE
    # we want the major.minor.patch version
    "Intel.+Compiler[ ]+([0-9.]+).+" "\\1" SYCL_VERSION "${_sycl_version_out}"
)

# get the SYCL installation root
cmake_path(GET SYCL_COMPILER PARENT_PATH SYCL_ROOT_DIR)
cmake_path(GET SYCL_ROOT_DIR PARENT_PATH SYCL_ROOT_DIR)

# look for include directory by looking for main SYCL header file
find_path(
    SYCL_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    HINTS "${SYCL_ROOT_DIR}"
    PATH_SUFFIXES include
    NO_CACHE
    NO_DEFAULT_PATH
)
# look for the SYCL library
find_library(
    SYCL_LIBRARY
    NAMES sycl sycl7 sycl8  # more names are accepted
    HINTS "${SYCL_ROOT_DIR}"
    PATH_SUFFIXES lib
    NO_CACHE
    NO_DEFAULT_PATH
)

# on Windows, also find the SYCL debug library + DLLs
if(WIN32)
    # find SYCL debug import library
    find_library(
        SYCL_LIBRARY_DEBUG
        NAMES sycld sycl7d sycl8d
        HINTS "${SYCL_ROOT_DIR}"
        PATH_SUFFIXES lib
        NO_CACHE
        NO_DEFAULT_PATH
    )
    # find SYCL release DLL
    find_file(
        SYCL_LIBRARY_DLL
        # note: possible for sycl.lib and sycl8.dll to be paired up
        NAMES sycl.dll sycl7.dll sycl8.dll
        HINTS "${SYCL_ROOT_DIR}"
        PATH_SUFFIXES bin
        NO_CACHE
        NO_DEFAULT_PATH
    )
    # find SYCL debug DLL
    find_file(
        SYCL_LIBRARY_DLL_DEBUG
        NAMES sycld.dll sycl7d.dll sycl8d.dll
        HINTS "${SYCL_ROOT_DIR}"
        PATH_SUFFIXES bin
        NO_CACHE
        NO_DEFAULT_PATH
    )
endif()

# find OpenCL using the specific root (required)
set(OpenCL_ROOT "${SYCL_ROOT_DIR}")
find_package(OpenCL REQUIRED)
unset(OpenCL_ROOT)

# set up imported target. on non-Windows systems we ignore static/shared
if(WIN32)
    add_library(SYCL::sycl SHARED IMPORTED)
    set_target_properties(
        SYCL::sycl PROPERTIES
        IMPORTED_IMPLIB "${SYCL_LIBRARY}"
        IMPORTED_IMPLIB_DEBUG "${SYCL_LIBRARY_DEBUG}"
        IMPORTED_LOCATION "${SYCL_LIBRARY_DLL}"
        IMPORTED_LOCATION_DEBUG "${SYCL_LIBRARY_DLL_DEBUG}"
    )
else()
    add_library(SYCL::sycl UNKNOWN IMPORTED)
    set_target_properties(
        SYCL::sycl PROPERTIES
        IMPORTED_LOCATION "${SYCL_LIBRARY}"
    )
endif()
# set include directories
target_include_directories(SYCL::sycl INTERFACE "${SYCL_INCLUDE_DIR}")
# ensure OpenCL is part of link interface
target_link_libraries(SYCL::sycl INTERFACE OpenCL::OpenCL)

# required variables to check. on Windows also include DLLs and debug library.
# SYCL_COMPILER and SYCL_ROOT_DIR are valid already. we put SYCL_LIBRARY first
# so it shows up the find_package_handle_standard_args success message
set(_sycl_required_vars SYCL_LIBRARY SYCL_INCLUDE_DIR)
if(WIN32)
    list(
        APPEND _sycl_required_vars
        SYCL_LIBRARY_DEBUG SYCL_LIBRARY_DLL SYCL_LIBRARY_DLL_DEBUG
    )
endif()

# final check of all variables including version
find_package_handle_standard_args(
    SYCL
    REQUIRED_VARS ${_sycl_required_vars}
    VERSION_VAR SYCL_VERSION
    HANDLE_VERSION_RANGE
)
