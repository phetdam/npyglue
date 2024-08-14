cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

add_subdirectory(npygl_test)

# py_helpers_test: Python helpers tests
if(Python3_Development_FOUND)
    add_executable(py_helpers_test py_helpers_test.cc)
    target_link_libraries(py_helpers_test PRIVATE Python3::Python)
    # usually no Python debug runtime library
    if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
        set_target_properties(
            py_helpers_test PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    add_test(NAME py_helpers_test COMMAND py_helpers_test)
endif()

# npy_helpers_test: NumPy Python helpers tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # C++17
    add_executable(npy_helpers_test npy_helpers_test.cc)
    target_link_libraries(
        npy_helpers_test PRIVATE
        Python3::NumPy Python3::Python
    )
    # usually no Python debug runtime library
    if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
        set_target_properties(
            npy_helpers_test PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    add_test(NAME npy_helpers_test COMMAND npy_helpers_test)
    # C++20
    if(NPYGL_HAS_CC_20)
        add_executable(npy_helpers_test_cc20 npy_helpers_test.cc)
        set_target_properties(
            npy_helpers_test_cc20 PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
        target_link_libraries(
            npy_helpers_test_cc20 PRIVATE
            Python3::NumPy Python3::Python
        )
        # usually no Python debug runtime library
        if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
            set_target_properties(
                npy_helpers_test_cc20 PROPERTIES
                MSVC_RUNTIME_LIBRARY MultiThreadedDLL
            )
        endif()
        add_test(NAME npy_helpers_test_cc20 COMMAND npy_helpers_test_cc20)
    endif()
endif()

# npygl_math: SWIG Python math functions module
if(SWIG_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # enable C++ mode
    set_source_files_properties(npygl_math.i PROPERTIES CPLUSPLUS ON)
    swig_add_library(
        npygl_math
        TYPE MODULE
        LANGUAGE python
        # generated output artifacts path. for multi-config generators we need
        # to include the extra per-config subdirectory manually
        OUTPUT_DIR
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        # generated wrapper source output path
        OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES npygl_math.i
    )
    # compile for Python 3
    set_target_properties(npygl_math PROPERTIES SWIG_COMPILE_OPTIONS -py3)
    # usually no Python debug runtime library
    if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
        set_target_properties(
            npygl_math PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    target_link_libraries(npygl_math PRIVATE Python3::NumPy Python3::Python)
    # unittest test script
    add_test(
        NAME npygl_math_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/npygl_math_test.py
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
    )
endif()
