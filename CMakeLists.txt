cmake_minimum_required(VERSION 3.20)

project(
    npyglue
    VERSION 0.1.0
    DESCRIPTION "A header-only C++ library simplifying Python/C++ interop"
    HOMEPAGE_URL https://github.com/phetdam/npyglue
    LANGUAGES C CXX
)

# C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set CMake module path and import CMake helpers
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(npygl_doxygen_add_docs)
include(npygl_get_build_info)
include(npygl_llvm_library_target)
include(npygl_swig_include)
include(npygl_target_helpers)

# options
option(NPYGL_BUILD_TESTS "Build ${PROJECT_NAME} tests" ON)
option(NPYGL_PYTHON_USE_RELEASE_CRT "MSVC: Python targets use release CRT" ON)
option(
    NPYGL_TORCH_PRECOMPILE
    "Precompile torch/torch.h for faster incremental builds" OFF
)
option(NPYGL_BUILD_RELEASE "Build an official release version" OFF)

# find Git
find_package(Git)
if(Git_FOUND)
    message(STATUS "Git version: ${GIT_VERSION_STRING}")
else()
    message(STATUS "Git version: None")
endif()

# check if generator is multi-config
get_property(NPYGL_MULTI_CONFIG_GENERATOR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NPYGL_MULTI_CONFIG_GENERATOR)
    message(STATUS "Generator config: Multi")
else()
    message(STATUS "Generator config: Single")
endif()

# if single-config, set CMAKE_BUILD_TYPE if not set
if(NPYGL_MULTI_CONFIG_GENERATOR)
    message(STATUS "Build config: Multi")
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build config: ${CMAKE_BUILD_TYPE}")
endif()

# get and print build info
# note: empty string if official release
npygl_get_build_info(NPYGL_BUILD_INFO IS_RELEASE ${NPYGL_BUILD_RELEASE})
if(NPYGL_BUILD_RELEASE)
    message(STATUS "Build info: Release")
else()
    message(STATUS "Build info: ${NPYGL_BUILD_INFO}")
endif()

# flat output directory with per-config subdirectories
# TODO: keep flat for single-config or not?
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# check if compiler follows the Itanium ABI by seeing if cxxabi.h is available
try_compile(
    NPYGL_ITANIUM_ABI ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/abi_check.cc
)
if(NPYGL_ITANIUM_ABI)
    message(STATUS "cxxabi.h found: Yes")
else()
    message(STATUS "cxxabi.h found: No")
endif()

# check if we have C++20 available
if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(NPYGL_HAS_CC_20 ON)
    message(STATUS "C++20 support: Yes")
else()
    set(NPYGL_HAS_CC_20 OFF)
    message(STATUS "C++20 support: No")
endif()

# check if CUDA is available. if so, enable and set standards
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "CUDA version: None")
endif()

# find Python 3. not tested with Python 2
find_package(Python3 COMPONENTS Development Interpreter NumPy)
if(Python3_Interpreter_FOUND)
    message(STATUS "Python version: ${Python3_VERSION}")
    message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")
else()
    message(STATUS "Python version: None")
    message(STATUS "Python interpreter: None")
endif()
if(Python3_Development_FOUND)
    message(STATUS "Python development: Yes")
else()
    message(STATUS "Python development: No")
endif()
if(Python3_NumPy_FOUND)
    message(STATUS "NumPy version: ${Python3_NumPy_VERSION}")
else()
    message(STATUS "NumPy version: None")
endif()

# find PyTorch if Python was found
if(Python3_FOUND)
    # check if PyTorch is installed
    execute_process(
        COMMAND pip show torch
        RESULT_VARIABLE NPYGL_SHOW_TORCH
        OUTPUT_QUIET ERROR_QUIET
    )
    # failed, no PyTorch
    if(NPYGL_SHOW_TORCH)
        set(NPYGL_PYTORCH_FOUND OFF)
    else()
        set(NPYGL_PYTORCH_FOUND ON)
    endif()
    # clean up
    unset(NPYGL_SHOW_TORCH)
endif()

# if PyTorch available, get some info
if(NPYGL_PYTORCH_FOUND)
    # get PyTorch config info
    execute_process(
        COMMAND python torch_info.py
        RESULT_VARIABLE NPYGL_TORCH_INFO_RESULT
        OUTPUT_VARIABLE NPYGL_TORCH_INFO_OUTPUT
    )
    # failed
    if(NPYGL_TORCH_INFO_RESULT)
        message(FATAL_ERROR "Failed to get PyTorch configuration info")
    endif()
    # turn string output into list
    string(
        REGEX REPLACE "\n" ";"
        NPYGL_TORCH_INFO_OUTPUT ${NPYGL_TORCH_INFO_OUTPUT}
    )
    # get version, CMake prefix path, if CUDA is available
    list(GET NPYGL_TORCH_INFO_OUTPUT 0 NPYGL_PYTORCH_VERSION)
    list(GET NPYGL_TORCH_INFO_OUTPUT 1 NPYGL_PYTORCH_CMAKE_PREFIX_PATH)
    list(GET NPYGL_TORCH_INFO_OUTPUT 2 NPYGL_PYTORCH_CUDA_VERSION)
    # PyTorch does not have CUDA
    if(NPYGL_PYTORCH_CUDA_VERSION STREQUAL "None")
        set(NPYGL_PYTORCH_HAS_CUDA OFF)
        unset(NPYGL_PYTORCH_CUDA_VERSION)
    # PyTorch has CUDA
    else()
        set(NPYGL_PYTORCH_HAS_CUDA ON)
    endif()
    # update CMake prefix path
    set(CMAKE_PREFIX_PATH ${NPYGL_PYTORCH_CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
    # clean up
    unset(NPYGL_TORCH_INFO_RESULT)
    unset(NPYGL_TORCH_INFO_OUTPUT)
    # print version info
    message(STATUS "PyTorch version: ${NPYGL_PYTORCH_VERSION}")
else()
    message(STATUS "PyTorch version: None")
endif()

# does PyTorch have CUDA?
# note: may not be the same as FindCUDAToolkit value
if(NPYGL_PYTORCH_HAS_CUDA)
    message(STATUS "PyTorch CUDA version: ${NPYGL_PYTORCH_CUDA_VERSION}")
else()
    message(STATUS "PyTorch CUDA version: None")
endif()

# find Torch C++ libraries (CMake module path already updated)
if(NPYGL_PYTORCH_FOUND)
    # for PyTorch 2.5.0 torch/share/cmake/Caffe2/public/cuda.cmake hardcodes a
    # search location for NVTX, which became a header-only library and for CUDA
    # 12+ no longer ships the library binary. therefore, CUDA::nvToolsExt is
    # not defined, so for PyTorch 2.5.0 GPU versions using CUDA 12+ NVTX is not
    # properly located and using Torch C++ will cause CMake configuration to
    # fail. our workaround is to manually define CUDA::nvToolsExt as a library
    # target that use the NVTX include directories if possible
    if(
        NPYGL_PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND
        NPYGL_PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
    )
        message(STATUS "PyTorch NVTX headers workaround: Yes")
        # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
        if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            # ensure that PyTorch is told to use NVTX3 headers
            target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
            target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
        endif()
    else()
        message(STATUS "PyTorch NVTX headers workaround: No")
    endif()
    find_package(Torch CONFIG)
    # Torch headers cause a lot of warnings to be emitted for MSVC so we define
    # a custom Torch target that disables some of the annoying warnings. in
    # general you might want to leave these on but when they show up for
    # *every* program that uses Torch headers this really becomes a PITA
    add_library(npyglue::Torch INTERFACE IMPORTED)
    if(MSVC)
        # note: to ensure each of the C++ flags makes it to CUDA, we have to
        # prepend each with -Xcompiler=. it is a bit clunky unfortunately
        # note: some warnings were first emitted during CUDA compilation. when
        # /Wall is enabled, we see C4242, C4365, C4582, C4583, C4619, C4800,
        # C5219, C5246, C5262 all start to crop up
        target_compile_options(
            npyglue::Torch INTERFACE
            # C4100: unreferenced formal parameter (from c10/core/SymNodeImpl.h)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4100>
            # C4127: conditional expression constant (from c10/util/Half.h)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4127>
            # C4244: possible loss of data in conversion (from
            # c10/util/BFloat16-inl.h, Float8_e4m3fn[uz]-inl.h, etc.)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244>
            # C4251: class member needs dll-interface (c10/util/Exception.h, etc.)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4251>
            # C4267: conversion from size_t to int (various Torch headers)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4267
            # C4275: non-dll interface class used as base (c10/util/Exception.h)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4275>
            # C4702: unreachable code (from c10/util/irange.h)
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4702
        )
        # NOMINMAX must be defined for MSVC since Torch headers use std::min +
        # std::max inline which of course results in a compilation nightmare
        target_compile_definitions(npyglue::Torch INTERFACE NOMINMAX)
    endif()
    # note: sadly, /external:W0 doesn't work as broadly as desired with MSVC
    target_include_directories(
        npyglue::Torch
        SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS}
    )
    # precompile the torch/torch.h header (decrease in incremental build time
    # by nearly half for targets that include torch/torch.h but the initial PCH
    # generation will double this build time just once)
    if(NPYGL_TORCH_PRECOMPILE)
        target_precompile_headers(npyglue::Torch INTERFACE <torch/torch.h>)
    endif()
    target_link_libraries(npyglue::Torch INTERFACE ${TORCH_LIBRARIES})
endif()
if(TORCH_FOUND)
    message(STATUS "PyTorch C++: Yes")
    if(NPYGL_TORCH_PRECOMPILE)
        message(STATUS "PyTorch PCH: Yes")
    else()
        message(STATUS "PyTorch PCH: No")
    endif()
else()
    message(STATUS "PyTorch C++: No")
    message(STATUS "PyTorch PCH: None")
endif()

# find Google Test
find_package(GTest 1.10)
if(GTest_FOUND)
    message(STATUS "Google Test version: ${GTest_VERSION}")
else()
    message(STATUS "Google Test version: None")
endif()

# find SWIG + ensure it supports Python
find_package(SWIG 4.0.2 COMPONENTS python)
if(SWIG_FOUND)
    include(UseSWIG)
    message(STATUS "SWIG version: ${SWIG_VERSION}")
    # global SWIG options
    set(CMAKE_SWIG_FLAGS -Wall)
else()
    message(STATUS "SWIG version: None")
endif()

# find Eigen 3 (need config mode typically)
find_package(Eigen3 3.4.0 CONFIG)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
else()
    message(STATUS "Eigen3 version: None")
endif()

# find Armadillo. on Windows, we use our own script, as we want to use
# Armadillo in header-only mode without requiring BLAS/LAPACK/ARPACK/HDF5
set(NPYGL_ARMADILLO_MINVER 10.8.2)
if(WIN32)
    # search paths for Armadillo
    set(
        _arma_paths
        ENV Armadillo_ROOT ENV Armadillo_DIR
        "$ENV{ProgramFiles}/Armadillo/include"  # used by FindArmadillo
    )
    # use Armadillo_DIR and Armadillo_ROOT if possible
    foreach(_prefix Armadillo_DIR Armadillo_ROOT)
        if(${_prefix})
            list(PREPEND _arma_paths ${${_prefix}})
        endif()
    endforeach()
    # find Armadillo header using search paths
    find_file(
        _arma_header
        NAMES armadillo
        PATHS ${_arma_paths}
        PATH_SUFFIXES include
        NO_CACHE
    )
    # if header exists, grab version info
    if(NOT _arma_header)
        message(STATUS "Could NOT find Armadillo header armadillo")
    else()
        # back up to get include directory
        cmake_path(GET _arma_header PARENT_PATH ARMADILLO_INCLUDE_DIR)
        # CMake sources say old Armadillo versions didn't have
        # armadillo_bits/arma_version.hpp so default to 0.0.0
        set(ARMADILLO_VERSION_MAJOR 0)
        set(ARMADILLO_VERSION_MINOR 0)
        set(ARMADILLO_VERSION_PATCH 0)
        set(ARMADILLO_VERSION_NAME "EARLY RELEASE")
        # if arma_version.hpp is available then we use it
        set(
            ARMA_VERSION_HPP_PATH
            "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp"
        )
        if(EXISTS "${ARMA_VERSION_HPP_PATH}")
            # read lines containing ARMA_VERSION_* variables
            file(
                STRINGS "${ARMA_VERSION_HPP_PATH}" ARMA_VERSION_HPP_LINES
                REGEX "#define ARMA_VERSION_[A-Z]+"
            )
            # get major, minor, patch versions
            string(
                REGEX REPLACE
                ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1"
                ARMADILLO_VERSION_MAJOR "${ARMA_VERSION_HPP_LINES}"
            )
            string(
                REGEX REPLACE
                ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1"
                ARMADILLO_VERSION_MINOR "${ARMA_VERSION_HPP_LINES}"
            )
            string(
                REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1"
                ARMADILLO_VERSION_PATCH "${ARMA_VERSION_HPP_LINES}"
            )
            # as noted in FindArmadillo.cmake version name has more than one
            # space between the macro and the string literal. we also want to
            # drop the quotes around the string literal
            string(
                REGEX REPLACE
                ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1"
                ARMADILLO_VERSION_NAME "${ARMA_VERSION_HPP_LINES}"
            )
        endif()
        # set version string
        set(
            ARMADILLO_VERSION_STRING
            ${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}
        )
    endif()
    unset(_arma_header)
    unset(_arma_paths)
    # set result variables like FindArmadillo does as appropriate
    if(ARMADILLO_VERSION_STRING VERSION_GREATER_EQUAL NPYGL_ARMADILLO_MINVER)
        set(Armadillo_FOUND TRUE)
        set(ARMADILLO_FOUND TRUE)
        set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
    else()
        message(
            STATUS
            "Could NOT find Armadillo >= ${NPYGL_ARMADILLO_MINVER} "
"(found: ${ARMADILLO_VERSION_STRING})"
        )
    endif()
else()
    find_package(Armadillo ${NPYGL_ARMADILLO_MINVER})
endif()
# note: module is kind of old and does not define Armadillo_VERSION
if(Armadillo_FOUND)
    message(STATUS "Armadillo version: ${ARMADILLO_VERSION_STRING}")
    # convenience interface target to manage Armadillo include
    add_library(npyglue::Armadillo INTERFACE IMPORTED)
    # note: we always use Armadillo in header-only mode, e.g. by defining
    # ARMA_DONT_USE_WRAPPER, as a convenience when compiling on Windows. this
    # saves use from needing debug and release CRT versions of the wrapper
    target_compile_definitions(npyglue::Armadillo INTERFACE ARMA_DONT_USE_WRAPPER)
    target_include_directories(
        npyglue::Armadillo
        SYSTEM INTERFACE ${ARMADILLO_INCLUDE_DIRS}
    )
else()
    message(STATUS "Armadillo version: None")
endif()

# find LLVM. we can use the demangler since some older (even with GCC 11.3)
# versions of abi::__cxa_demangle chokes on long type identifiers. the issue is
# that typically LLVM code is compiled with -fno-exceptions
find_package(LLVM 14.0.0)
# note: LLVM_FOUND works but not documented in the actual LLVM CMake docs at
# https://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
if(LLVM_FOUND)
    # note: for LLVM 14.0.0, LLVM_VERSION should also work
    message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
    # add target for LLVMDemangle library
    npygl_llvm_library_target(demangle)
else()
    message(STATUS "LLVM version: None")
endif()

# find Doxygen
find_package(Doxygen 1.10)
if(DOXYGEN_FOUND)
    message(STATUS "Doxygen version: ${DOXYGEN_VERSION}")
else()
    message(STATUS "Doxygen version: None")
endif()

# find Pygments (for syntax highlighting languages Doxygen doesn't support)
if(Python3_FOUND)
    execute_process(
        COMMAND pip show pygments
        RESULT_VARIABLE _pygments_res
        OUTPUT_VARIABLE _pygments_out
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # zero exit code
    if(NOT _pygments_res)
        set(NPYGL_HAS_PYGMENTS TRUE)
    endif()
    unset(_pygments_res)
endif()
# if not defined, set to FALSE
if(NOT DEFINED NPYGL_HAS_PYGMENTS)
    set(NPYGL_HAS_PYGMENTS FALSE)
endif()
# if Pygments is available, get the version
if(NPYGL_HAS_PYGMENTS)
    # convert _pygments_out to list
    string(REPLACE "\n" ";" NPYGL_PYGMENTS_VERSION "${_pygments_out}")
    unset(_pygments_out)
    # keep only the Version: element + remove to get version
    list(FILTER NPYGL_PYGMENTS_VERSION INCLUDE REGEX "Version\:[ ]+[0-9]+.+")
    string(
        REGEX REPLACE "Version\:[ ]+" ""
        NPYGL_PYGMENTS_VERSION "${NPYGL_PYGMENTS_VERSION}"
    )
    # print status
    message(STATUS "Pygments version: ${NPYGL_PYGMENTS_VERSION}")
else()
    message(STATUS "Pygments version: None")
endif()

# compile flags. we do this after any FetchContent calls so that we don't
# interfere with any compile flags that dependency targets require and we use
# the COMPILE_LANGUAGE generator expression so we don't pass C++ flags to CUDA.
# some of the warnings are CUDA-specific as the CUDA headers are a bit messy
if(MSVC)
    add_compile_options(
        # note: nvcc doesn't provide a way to mark include directories as
        # external so we live with a lower warning level to avoid drowning
        $<$<COMPILE_LANGUAGE:C,CXX>:/Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        # C4061: enum value not explicitly handled by case label
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4061
        # C4324: struct was padded due to alignment specifier (from
        # Thrust header thrust/detail/alignment.h)
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4324>
        # C4514: unreferenced inline function has been removed
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4514
        # C4623: default ctor was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4623
        # C4625: copy ctor was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4625
        # C4626: assignment operator was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4626
        # C4710: function not inlined
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4710
        # C4711: function selected for automatic inline expansion
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4711
        # C4820: padding added after struct member
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd4820
        # C5026: move ctor was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd5026
        # C5027: move assignment operator was implicitly defined as deleted
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd5027
        # C5045: compiler will insert Spectre mitigation if using /QSpectre
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>/wd5045
    )
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=>-Wall)
endif()

# project include directory
set(NPYGL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# set version-related variables
# note: should use NPYGLUE_ prefix but NPYGL_ has already been established
set(NPYGL_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(NPYGL_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(NPYGL_VERSION_PATCH ${PROJECT_VERSION_PATCH})
# non-official releases have extra build info
set(NPYGL_VERSION ${PROJECT_VERSION})
if(NOT NPYGL_BUILD_RELEASE)
    string(APPEND NPYGL_VERSION "-${NPYGL_BUILD_INFO}")
endif()
message(STATUS "Build version: ${NPYGL_VERSION}")

# generate version header. only replace @VAR@ and keep LF newlines
set(NPYGL_VERSION_HEADER include/npygl/version.h)
# variables consumed:
#
#   NPYGL_VERSION_MAJOR
#   NPYGL_VERSION_MINOR
#   NPYGL_VERSION_PATCH
#   NPYGL_VERSION
#   NPYGL_BUILD_INFO
#
configure_file(${NPYGL_VERSION_HEADER}.in ${NPYGL_VERSION_HEADER} @ONLY LF)
message(STATUS "Generated ${NPYGL_VERSION_HEADER}")

# npyglue: C++ library facilitating C++/Python NumPy array interop
add_library(npyglue INTERFACE)
add_library(npyglue::npyglue ALIAS npyglue)
target_include_directories(
    npyglue INTERFACE
    $<BUILD_INTERFACE:${NPYGL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# install rule for headers (including generated)
install(
    DIRECTORY ${NPYGL_INCLUDE_DIR}/npygl ${CMAKE_CURRENT_BINARY_DIR}/include/npygl
    TYPE INCLUDE
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN testing EXCLUDE  #  no testing headers
)

# interface targets for disabling third-party header inclusion. by default, the
# features.h header allows conditional inclusion of other library headers, but
# using these targets is helpful for explicitly disabling inclusion if desired,
# e.g. to speed up compile time when not using one of the libraries' headers.

# npyglue_no_armadillo: disable inclusion of Armadillo headers
add_library(npyglue_no_armadillo INTERFACE)
add_library(npyglue::no_armadillo ALIAS npyglue_no_armadillo)
target_compile_definitions(npyglue_no_armadillo INTERFACE NPYGL_NO_ARMADILLO)
# allow exported target to be used the same as the alias target
set_target_properties(npyglue_no_armadillo PROPERTIES EXPORT_NAME no_armadillo)

# npyglue_no_eigen3: disable inclusion of Eigen3 headers
add_library(npyglue_no_eigen3 INTERFACE)
add_library(npyglue::no_eigen3 ALIAS npyglue_no_eigen3)
target_compile_definitions(npyglue_no_eigen3 INTERFACE NPYGL_NO_EIGEN3)
set_target_properties(npyglue_no_eigen3 PROPERTIES EXPORT_NAME no_eigen3)

# npyglue_no_torch: disable inclusion of LibTorch headers
add_library(npyglue_no_torch INTERFACE)
add_library(npyglue::no_torch ALIAS npyglue_no_torch)
target_compile_definitions(npyglue_no_torch INTERFACE NPYGL_NO_TORCH)
set_target_properties(npyglue_no_torch PROPERTIES EXPORT_NAME no_torch)

# npyglue_use_llvm_demangle: enable use of LLVMDemangle in demangle.hh
add_library(npyglue_use_llvm_demangle INTERFACE)
add_library(npyglue::use_llvm_demangle ALIAS npyglue_use_llvm_demangle)
target_compile_definitions(
    npyglue_use_llvm_demangle INTERFACE
    NPYGL_USE_LLVM_DEMANGLE
)
set_target_properties(
    npyglue_use_llvm_demangle PROPERTIES
    EXPORT_NAME use_llvm_demangle
)

# location of SIL SWIG interface files in build tree
set(NPYGL_SIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sil)
# prefix-relative directory for SIL SWIG interface files on install
set(NPYGL_SIL_INSTALL_DIR share/${PROJECT_NAME}/sil)

# npyglue_sil: SWIG interface files for C++/Python NumPy array interop
# this is *not* to be used via target_link_libraries but rather via the
# npygl_swig_include CMake function to populate the SWIG_* compile properties
add_library(npyglue_sil INTERFACE)
add_library(npyglue::SIL ALIAS npyglue_sil)
# TODO: not sure if swig_add_library likes generator expressions here
target_include_directories(
    npyglue_sil INTERFACE
    $<BUILD_INTERFACE:${NPYGL_SIL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${NPYGL_SIL_INSTALL_DIR}>
)
# install rule for the SWIG interface files
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sil/npygl
    DESTINATION ${NPYGL_SIL_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.i"
)

# TODO: add install rule for npygl_swig_include

# prefix-relative directory for CMake files on installation
set(NPYGL_CMAKE_INSTALL_DIR lib/cmake/${PROJECT_NAME})
# name of the CMake target exports script
set(NPYGL_CMAKE_TARGETS_FILE ${PROJECT_NAME}-targets.cmake)

# install rules for all interface targets + export
install(
    TARGETS
        npyglue
        npyglue_no_armadillo
        npyglue_no_eigen3
        npyglue_no_torch
        npyglue_use_llvm_demangle
    EXPORT npyglue_exports
)
install(
    EXPORT npyglue_exports
    DESTINATION ${NPYGL_CMAKE_INSTALL_DIR}
    NAMESPACE ${PROJECT_NAME}::
    FILE ${NPYGL_CMAKE_TARGETS_FILE}
)

# list of CMake modules to provide to downstream consumers
set(NPYGL_EXPORTED_CMAKE_MODULES npygl_swig_include)
# convert list of modules into file names + includes for CMake config script
foreach(module ${NPYGL_EXPORTED_CMAKE_MODULES})
    list(
        APPEND _exported_modules
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${module}.cmake
    )
    list(
        APPEND NPYGL_INCLUDE_EXPORTED_CMAKE_MODULES
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/${module}.cmake\")\n"
    )
endforeach()
# install rule for any CMake modules to provide to downstream consumers
install(FILES ${_exported_modules} DESTINATION ${NPYGL_CMAKE_INSTALL_DIR})
unset(_exported_modules)

# enable testing with CTest
include(CTest)

# build tests
# TODO: consider changing the semantics of NPYGL_HAS_<A> to also include
# whether or not NPYGL_NO_<A> is defined or not, e.g. it NPYGL_HAS_<A> is 1
# only when both the relevant header is available and no NPYGL_NO_<A> defined
if(NPYGL_BUILD_TESTS)
    add_subdirectory(test)
else()
    message(STATUS "Skipping ${PROJECT_NAME} test directory")
endif()

# add Doxygen documentation target
if(DOXYGEN_FOUND AND NPYGL_HAS_PYGMENTS)
    # Pygments stylesheet for selected code block highlighting
    set(pygments_stylesheet doc/pygments-style.css)
    # target for generating the Pygments CSS stylesheet. we don't use
    # add_custom_command because if someone updates their Pygments version and
    # then rebuilds, the CSS stylesheet won't be regenerated
    add_custom_target(
        npyglue_pygstylegen
        COMMAND python tools/pygstylegen.py -o ${pygments_stylesheet}
        BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/${pygments_stylesheet}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating ${pygments_stylesheet}"
    )
    # TODO: rewrite README.rst as Markdown for use with Doxygen
    # add Doxygen target with presets
    npygl_doxygen_add_docs(
        ${PROJECT_NAME}_doc
        SOURCES ${NPYGL_INCLUDE_DIR} doc/npyglue-sil.md
        COMMENT "Generating ${PROJECT_NAME} HTML documentation"
        VERBOSE
        # Doxygen configuration
        HTML_EXTRA_STYLESHEET doc/npyglue.css ${pygments_stylesheet}
        PREDEFINED __cplusplus
        # Doxygen layout file used to customize HTML page layout
        LAYOUT_FILE doc/layout.xml
    )
    # Doxygen target depends on Pygments stylesheet generation
    add_dependencies(${PROJECT_NAME}_doc npyglue_pygstylegen)
else()
    message(STATUS "Skipping ${PROJECT_NAME} documentation generation")
endif()

# name of the CMake package config script + version config script
set(NPYGL_CMAKE_CONFIG_FILE ${PROJECT_NAME}-config.cmake)
set(NPYGL_CMAKE_VERSION_CONFIG_FILE ${PROJECT_NAME}-config-version.cmake)

include(CMakePackageConfigHelpers)

# configure package config file + version resolution script
#
# variables consumed:
#
#   PROJECT_NAME
#   NPYGL_CMAKE_TARGETS_FILE
#   NPYGL_INCLUDE_EXPORTED_CMAKE_MODULES
#
configure_package_config_file(
    cmake/${NPYGL_CMAKE_CONFIG_FILE}.in ${NPYGL_CMAKE_CONFIG_FILE}
    INSTALL_DESTINATION ${NPYGL_CMAKE_INSTALL_DIR}
)
# note: PROJECT_VERSION is used as the version
write_basic_package_version_file(
    ${NPYGL_CMAKE_VERSION_CONFIG_FILE}
    COMPATIBILITY AnyNewerVersion
    ARCH_INDEPENDENT
)
# install rules for the package and version config scripts
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${NPYGL_CMAKE_CONFIG_FILE}
        ${CMAKE_CURRENT_BINARY_DIR}/${NPYGL_CMAKE_VERSION_CONFIG_FILE}
    DESTINATION ${NPYGL_CMAKE_INSTALL_DIR}
)
