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

# compile flags. we do this after any FetchContent calls so that we don't
# interfere with any compile flags that dependency targets require
if(MSVC)
    add_compile_options(
        /Wall
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
