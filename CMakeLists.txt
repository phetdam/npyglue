cmake_minimum_required(VERSION 3.20)

project(
    npyglue
    VERSION 0.0.1
    DESCRIPTION "C++ glue code simplifying Python/C++ interop."
    HOMEPAGE_URL https://github.com/phetdam/npyglue
    LANGUAGES C CXX
)

# C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(BUILD_TESTS "Build ${PROJECT_NAME} tests" ON)
option(PY_MSVC_ALWAYS_RELEASE "MSVC: Python targets use release runtimes" ON)
option(
    NPYGL_TORCH_PRECOMPILE
    "Precompile torch/torch.h for faster incremental builds" OFF
)

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

# flat output directory with per-config subdirectories
# TODO: keep flat for single-config or not?
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# check if we have C++20 available
if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(NPYGL_HAS_CC_20 ON)
    message(STATUS "C++20 support: Yes")
else()
    set(NPYGL_HAS_CC_20 OFF)
    message(STATUS "C++20 support: No")
endif()

# check if CUDA is available
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
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
        RESULT_VARIABLE SAND_SHOW_TORCH
        OUTPUT_QUIET ERROR_QUIET
    )
    # failed, no PyTorch
    if(SAND_SHOW_TORCH)
        set(SAND_PYTORCH_FOUND OFF)
    else()
        set(SAND_PYTORCH_FOUND ON)
    endif()
    # clean up
    unset(SAND_SHOW_TORCH)
endif()

# if PyTorch available, get some info
if(SAND_PYTORCH_FOUND)
    # get PyTorch config info
    execute_process(
        COMMAND python torch_info.py
        RESULT_VARIABLE SAND_TORCH_INFO_RESULT
        OUTPUT_VARIABLE SAND_TORCH_INFO_OUTPUT
    )
    # failed
    if(SAND_TORCH_INFO_RESULT)
        message(FATAL_ERROR "Failed to get PyTorch configuration info")
    endif()
    # turn string output into list
    string(REGEX REPLACE "\n" ";" SAND_TORCH_INFO_OUTPUT ${SAND_TORCH_INFO_OUTPUT})
    # get version, CMake prefix path, if CUDA is available
    list(GET SAND_TORCH_INFO_OUTPUT 0 SAND_PYTORCH_VERSION)
    list(GET SAND_TORCH_INFO_OUTPUT 1 SAND_PYTORCH_CMAKE_PREFIX_PATH)
    list(GET SAND_TORCH_INFO_OUTPUT 2 SAND_PYTORCH_CUDA_VERSION)
    # PyTorch does not have CUDA
    if(SAND_PYTORCH_CUDA_VERSION STREQUAL "None")
        set(SAND_PYTORCH_HAS_CUDA OFF)
        unset(SAND_PYTORCH_CUDA_VERSION)
    # PyTorch has CUDA
    else()
        set(SAND_PYTORCH_HAS_CUDA ON)
    endif()
    # update CMake prefix path
    set(CMAKE_PREFIX_PATH ${SAND_PYTORCH_CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH})
    # clean up
    unset(SAND_TORCH_INFO_RESULT)
    unset(SAND_TORCH_INFO_OUTPUT)
    # print version info
    message(STATUS "PyTorch version: ${SAND_PYTORCH_VERSION}")
else()
    message(STATUS "PyTorch version: None")
endif()

# does PyTorch have CUDA?
# note: may not be the same as FindCUDAToolkit value
if(SAND_PYTORCH_HAS_CUDA)
    message(STATUS "PyTorch CUDA version: ${SAND_PYTORCH_CUDA_VERSION}")
else()
    message(STATUS "PyTorch CUDA version: None")
endif()

# find Torch C++ libraries (CMake module path already updated)
if(SAND_PYTORCH_FOUND)
    # for PyTorch 2.5.0 torch/share/cmake/Caffe2/public/cuda.cmake hardcodes a
    # search location for NVTX, which became a header-only library and for CUDA
    # 12+ no longer ships the library binary. therefore, CUDA::nvToolsExt is
    # not defined, so for PyTorch 2.5.0 GPU versions using CUDA 12+ NVTX is not
    # properly located and using Torch C++ will cause CMake configuration to
    # fail. our workaround is to manually define CUDA::nvToolsExt as a library
    # target that use the NVTX include directories if possible
    if(
        SAND_PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND
        SAND_PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
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
    add_library(npygl::external::torch INTERFACE IMPORTED)
    if(MSVC)
        target_compile_options(
            npygl::external::torch INTERFACE
            # C4267: conversion from size_t to int (various Torch headers)
            /wd4267
            # C4702: unreachable code (from c10/util/irange.h)
            /wd4702
        )
        # NOMINMAX must be defined for MSVC since Torch headers use std::min +
        # std::max inline which of course results in a compilation nightmare
        target_compile_definitions(npygl::external::torch INTERFACE NOMINMAX)
    endif()
    # note: sadly, /external:W0 doesn't work as broadly as desired with MSVC
    target_include_directories(
        npygl::external::torch
        SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS}
    )
    # precompile the torch/torch.h header (decrease in incremental build time
    # by nearly half for targets that include torch/torch.h but the initial PCH
    # generation will double this build time just once)
    if(NPYGL_TORCH_PRECOMPILE)
        target_precompile_headers(npygl::external::torch INTERFACE <torch/torch.h>)
    endif()
    target_link_libraries(npygl::external::torch INTERFACE ${TORCH_LIBRARIES})
endif()
if(TORCH_FOUND)
    message(STATUS "PyTorch C++: Yes")
else()
    message(STATUS "PyTorch C++: No")
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
    # find Armadillo include directory like in FindArmadillo.cmake
    find_path(
        ARMADILLO_INCLUDE_DIR
        NAMES armadillo
        PATHS "$ENV{ProgramFiles}/Armadillo/include"
    )
    # include directory exists so grab version info
    if(ARMADILLO_INCLUDE_DIR)
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
    # need to manually add ARMADILLO_INCLUDE_DIRS if non-system install
    include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
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
    # add LLVM include directories + resolve LLVM demangle library
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    llvm_map_components_to_libnames(NPYGL_LLVM_DEMANGLE_LIB demangle)
else()
    message(STATUS "LLVM version: None")
endif()

# compile flags. we do this after any FetchContent calls so that we don't
# interfere with any compile flags that dependency targets require
if(MSVC)
    add_compile_options(
        /Wall
        # C4061: enum value not explicitly handled by case label
        /wd4061
        # C4514: unreferenced inline function has been removed
        /wd4514
        # C4623: default ctor was implicitly defined as deleted
        /wd4623
        # C4625: copy ctor was implicitly defined as deleted
        /wd4625
        # C4626: assignment operator was implicitly defined as deleted
        /wd4626
        # C4711: function selected for automatic inline expansion
        /wd4711
        # C4820: padding added after struct member
        /wd4820
        # C5026: move ctor was implicitly defined as deleted
        /wd5026
        # C5027: move assignment operator was implicitly defined as deleted
        /wd5027
        # C5045: compiler will insert Spectre mitigation if using /QSpectre
        /wd5045
    )
else()
    add_compile_options(-Wall)
endif()

# set CMake module path and import target helpers
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(npygl_target_helpers)

# project include directory
set(NPYGL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${NPYGL_INCLUDE_DIR})

# generate version header. only replace @VAR@ and keep LF newlines
set(NPYGL_VERSION_HEADER ${NPYGL_INCLUDE_DIR}/npygl/version.h)
cmake_path(GET NPYGL_VERSION_HEADER FILENAME NPYGL_VERSION_HEADER_NAME)
configure_file(${NPYGL_VERSION_HEADER}.in ${NPYGL_VERSION_HEADER} @ONLY LF)
message(STATUS "Generated ${NPYGL_VERSION_HEADER_NAME}")

# enable testing with CTest
include(CTest)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
