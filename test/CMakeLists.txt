cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

add_subdirectory(npygl_test)

# note: we always use Armadillo in header-only mode, e.g. by defining
# ARMA_DONT_USE_WRAPPER, as a convenience when compiling on Windows. this saves
# use from needing a debug and release C runtime version of the wrapper
add_compile_definitions(ARMA_DONT_USE_WRAPPER)

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
    # enable Eigen3 tests if available
    if(Eigen3_FOUND)
        target_link_libraries(py_capsule_test PRIVATE Eigen3::Eigen)
    endif()
    # enable PyTorch tests if available
    # note: still uses -D_GLIBCXX_USE_CXX11_ABI=0 for manylinux wheels in 2024
    if(TORCH_FOUND)
        target_link_libraries(py_capsule_test PRIVATE npygl::external::torch)
        npygl_copy_torch_dlls(py_capsule_test)
    endif()
    add_test(NAME py_capsule_test COMMAND py_capsule_test)
endif()

# py_format_test: Python compile-time format strings test
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_format_test
        SOURCES py_format_test.cc
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    add_test(NAME py_format_test COMMAND py_format_test)
endif()

# npy_helpers_test: NumPy Python helpers tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # reduce compile time since we don't need Eigen3/Armadillo
    set_property(
        SOURCE npy_helpers_test.cc PROPERTY COMPILE_DEFINITIONS
        NPYGL_NO_ARMADILLO NPYGL_NO_EIGEN3
    )
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
    # not using Armadillo currently (may change later)
    target_compile_definitions(npy_helpers_test PRIVATE NPYGL_NO_ARMADILLO)
    # enable Eigen features if available
    if(Eigen3_FOUND)
        target_link_libraries(npy_capsule_test PRIVATE Eigen3::Eigen)
    endif()
    add_test(NAME npy_capsule_test COMMAND npy_capsule_test)
endif()

# npy_traits_test: NumPy traits class tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_traits_test
        SOURCES npy_traits_test.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    if(Eigen3_FOUND)
        target_link_libraries(npy_traits_test PRIVATE Eigen3::Eigen)
    endif()
    add_test(NAME npy_traits_test COMMAND npy_traits_test)
endif()

# npy_views_test: testing views with NumPy
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_views_test
        SOURCES npy_views_test.cc
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # reduce compile time since we don't need Eigen3/Armadillo
    target_compile_definitions(
        npy_views_test PRIVATE
        NPYGL_NO_ARMADILLO NPYGL_NO_EIGEN3
    )
    add_test(NAME npy_views_test COMMAND npy_views_test)
endif()

# pyhelpers: Python helpers test module
# TODO: consider allowing conditional compilation without NumPy, in which case
# there will be a separate Python test script run for the NumPy tests
if(Python3_Development_FOUND)
    npygl_add_py3_extension(
        TARGET pyhelpers
        SOURCES pyhelpers.cc
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    if(Python3_NumPy_FOUND)
        target_link_libraries(pyhelpers PRIVATE Python3::NumPy)
    endif()
    if(Eigen3_FOUND)
        target_link_libraries(pyhelpers PRIVATE Eigen3::Eigen)
    endif()
    # test script for non-NumPy tests
    add_test(
        NAME pyhelpers_test
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pyhelpers_test.py
        WORKING_DIRECTORY
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
    )
    # test script for NumPy tests
    if(Python3_NumPy_FOUND)
        add_test(
            NAME pyhelpers_npy_test
            COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/pyhelpers_npy_test.py
            WORKING_DIRECTORY
                ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        )
    endif()
endif()

# pymath: Python math functions module
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # not using Armadillo/Eigen3 currently (may change later)
    set_property(
        SOURCE pymath.cc PROPERTY COMPILE_DEFINITIONS
        NPYGL_NO_ARMADILLO NPYGL_NO_EIGEN3
    )
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

# pymath_swig: SWIG Python math functions module
if(SWIG_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_swig_py3_module(
        TARGET pymath_swig
        INTERFACE pymath_swig.i
        SWIG_CC ON
        LIBRARIES Python3::NumPy
        USE_RELEASE_CRT ${PY_MSVC_ALWAYS_RELEASE}
    )
    # not using Armadillo/Eigen3 currently (may change later)
    target_compile_definitions(
        pymath_swig PRIVATE
        NPYGL_NO_ARMADILLO NPYGL_NO_EIGEN3
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
        # not using Armadillo/Eigen3 currently (may change later)
        target_compile_definitions(
            pymath_swig_cc20 PRIVATE
            NPYGL_NO_ARMADILLO NPYGL_NO_EIGEN3
        )
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

# type_traits_test: C++ type traits test
add_executable(type_traits_test type_traits_test.cc)
add_test(NAME type_traits_test COMMAND type_traits_test)

# termcolors: program to print the SGR values and 8-bit terminal colors
add_executable(termcolors termcolors.cc)
if(WIN32)
    target_link_libraries(termcolors PRIVATE kernel32)
endif()
# smoke test. not super useful as we need to visually inspect for correctness
add_test(NAME termcolors COMMAND termcolors)

# llvm_demangle_test: test demangling of driver_type using LLVM
# note: only built when not on Windows since std::type_info::name() is not
# mangled in the MSVC ABI. checking compiler ABI is the direct solution
if(NOT WIN32 AND LLVM_FOUND)
    add_executable(llvm_demangle_test type_traits_test.cc)
    # opt-in is required since we need to link against LLVMDemangle
    target_compile_definitions(llvm_demangle_test PRIVATE NPYGL_USE_LLVM_DEMANGLE)
    target_link_libraries(llvm_demangle_test PRIVATE ${NPYGL_LLVM_DEMANGLE_LIB})
    add_test(NAME llvm_demangle_test COMMAND llvm_demangle_test)
endif()

# torch_test: test torch.hh LibTorch helpers
if(TORCH_FOUND)
    add_executable(torch_test torch_test.cc)
    # Torch C++ DLLs are linked against release C runtime
    npygl_use_release_crt(torch_test ON)
    target_link_libraries(torch_test PRIVATE npygl::external::torch)
    if(Eigen3_FOUND)
        target_link_libraries(torch_test PRIVATE Eigen3::Eigen)
    endif()
    # TODO: post-build copy is kind of slow (and runs more often than expected)
    # so maybe it's better to have a pre-build target that does the copy
    npygl_copy_torch_dlls(torch_test)
    add_test(NAME torch_test COMMAND torch_test)
endif()
