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

# helper variable to determine if we can build SWIG + NumPy extension module
if(SWIG_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
    set(NPYGL_SWIG_NPY_AVAILABLE ON)
else()
    set(NPYGL_SWIG_NPY_AVAILABLE OFF)
endif()

# pymath_swig: SWIG Python math functions module
# TODO: add CMake module with npygl_add_swig_py3_module() function to make the
# common SWIG tweaks, e.g. MSVC warning suppression, output dir, SWIG <4.1 and
# below use of -py3, debug/release Python runtime library, less tedious
if(NPYGL_SWIG_NPY_AVAILABLE)
    # enable C++ mode
    set_source_files_properties(pymath_swig.i PROPERTIES CPLUSPLUS ON)
    swig_add_library(
        pymath_swig
        TYPE MODULE
        LANGUAGE python
        # generated output artifacts path. for multi-config generators we need
        # to include the extra per-config subdirectory manually
        OUTPUT_DIR
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        # generated wrapper source output path
        # note: we could use OUTPUT_DIR/gensrc for this to avoid WSL/Windows
        # builds sometimes (?) recompiling the SWIG file
        OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
        SOURCES pymath_swig.i
    )
    # silence some MSVC warnings
    if(MSVC)
        target_compile_options(
            pymath_swig PRIVATE
            # C4365: signed/unsigned mismatch during type conversion
            /wd4365
            # C4668: macro not defined, replacing with 0 (__GNUC__ not defined)
            /wd4668
        )
    endif()
    # compile for Python 3. from SWIG 4.1 onwards however we need to use
    # %feature("python:annotations", "c") directive instead
    # FIXME: SWIG 4.1+ will not have the C++ type annotations
    if(SWIG_VERSION VERSION_LESS 4.1)
        set_target_properties(pymath_swig PROPERTIES SWIG_COMPILE_OPTIONS -py3)
    endif()
    # usually no Python debug runtime library
    if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
        set_target_properties(
            pymath_swig PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    target_link_libraries(pymath_swig PRIVATE Python3::NumPy Python3::Python)
    # unittest test script
    add_test(
        NAME pymath_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pymath_test.py
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
    )
endif()
