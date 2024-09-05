cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

add_subdirectory(npygl_test)

# py_helpers_test: Python helpers tests
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_helpers_test
        SOURCES py_helpers_test.cc
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    add_test(NAME py_helpers_test COMMAND py_helpers_test)
endif()

# py_capsule_test: Python capsule test
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_capsule_test
        SOURCES py_capsule_test.cc
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    add_test(NAME py_capsule_test COMMAND py_capsule_test)
endif()

# npy_helpers_test: NumPy Python helpers tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # C++17
    npygl_add_py3_executable(
        TARGET npy_helpers_test
        SOURCES npy_helpers_test.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    add_test(NAME npy_helpers_test COMMAND npy_helpers_test)
    # C++20
    if(NPYGL_HAS_CC_20)
        npygl_add_py3_executable(
            TARGET npy_helpers_test_cc20
            SOURCES npy_helpers_test.cc
            LIBRARIES Python3::NumPy
            USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
        )
        npygl_require_cc_std(TARGET npy_helpers_test_cc20 CC_STD 20)
        add_test(NAME npy_helpers_test_cc20 COMMAND npy_helpers_test_cc20)
    endif()
endif()

# npy_capsule_test: NumPy test using capsules to back arrays
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_capsule_test
        SOURCES npy_capsule_test.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # enable Eigen features if available
    if(Eigen3_FOUND)
        target_link_libraries(npy_capsule_test PRIVATE Eigen3::Eigen)
    endif()
    add_test(NAME npy_capsule_test COMMAND npy_capsule_test)
endif()

# npy_views_test: testing views with NumPy
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_views_test
        SOURCES npy_views_test.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    add_test(NAME npy_views_test COMMAND npy_views_test)
endif()

# pyhelpers: Python helpers test module
if(Python3_Development_FOUND)
    npygl_add_py3_extension(
        TARGET pyhelpers
        SOURCES pyhelpers.cc
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # TODO: add test script
endif()

# pymath: Python math functions module
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # C++17
    npygl_add_py3_extension(
        TARGET pymath
        SOURCES pymath.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # unittest test script
    add_test(
        NAME pymath_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pymath_test.py
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
    )
    # C++20
    if(NPYGL_HAS_CC_20)
        npygl_add_py3_extension(
            TARGET pymath_cc20
            SOURCES pymath.cc
            LIBRARIES Python3::NumPy
            USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
        )
        npygl_require_cc_std(TARGET pymath_cc20 CC_STD 20)
        # unittest test script
        add_test(
            NAME pymath_cc20_test
            COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pymath_test.py -std cc20
            WORKING_DIRECTORY
                ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        )
    endif()
endif()

# helper variable to determine if we can build SWIG + NumPy extension module
if(SWIG_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
    set(NPYGL_SWIG_NPY_AVAILABLE ON)
else()
    set(NPYGL_SWIG_NPY_AVAILABLE OFF)
endif()

# pymath_swig: SWIG Python math functions module
if(NPYGL_SWIG_NPY_AVAILABLE)
    npygl_add_swig_py3_module(
        TARGET pymath_swig
        INTERFACE pymath_swig.i
        SWIG_CC ON
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # unittest test script
    add_test(
        NAME pymath_swig_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pymath_test.py -f swig
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
    )
    # C++20
    if(NPYGL_HAS_CC_20)
        npygl_add_swig_py3_module(
            TARGET pymath_swig_cc20
            INTERFACE pymath_swig.i
            SWIG_CC ON
            # enable SWIG to see the C++20 declarations in math.hh
            SWIG_DEFINES NPYGL_SWIG_CC_20
            LIBRARIES Python3::NumPy
            USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
        )
        npygl_require_cc_std(TARGET pymath_swig_cc20 CC_STD 20)
        # unittest test script
        add_test(
            NAME pymath_swig_cc20_test
            COMMAND
                python ${CMAKE_CURRENT_SOURCE_DIR}/pymath_test.py -f swig -std cc20
            WORKING_DIRECTORY
                ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        )
    endif()
endif()
