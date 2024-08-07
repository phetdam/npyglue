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

# compile flags
if(MSVC)
    add_compile_options(
        /Wall
        # C4514: unreferenced inline function has been removed
        /wd4514
        # C4625: copy ctor was implicitly defined as deleted
        /wd4625
        # C4626: assignment operator was implicitly defined as deleted
        /wd4626
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
