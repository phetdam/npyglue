cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

add_subdirectory(npygl_test)

# py_helpers_test: Python helpers tests
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_helpers_test
        SOURCES py_helpers_test.cc
        LIBRARIES npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    add_test(NAME py_helpers_test COMMAND py_helpers_test)
endif()

# py_capsule_test: Python capsule test
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_capsule_test
        SOURCES py_capsule_test.cc
        LIBRARIES npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    # enable Armadillo tests if available
    if(Armadillo_FOUND)
        target_link_libraries(py_capsule_test PRIVATE npyglue::Armadillo)
    endif()
    # enable Eigen3 tests if available
    if(Eigen3_FOUND)
        target_link_libraries(py_capsule_test PRIVATE Eigen3::Eigen)
    endif()
    # enable PyTorch tests if available
    # note: still uses -D_GLIBCXX_USE_CXX11_ABI=0 for manylinux wheels in 2024
    if(TORCH_FOUND)
        target_link_libraries(py_capsule_test PRIVATE npyglue::Torch)
        npygl_copy_torch_dlls(py_capsule_test)
    endif()
    add_test(NAME py_capsule_test COMMAND py_capsule_test)
endif()

# py_capsule_cutorch_test: Python capsule test with CUDA Torch tensors
if(CUDAToolkit_FOUND AND Python3_Development_FOUND AND TORCH_FOUND)
    npygl_add_py3_executable(
        TARGET py_capsule_cutorch_test
        SOURCES py_capsule_cutorch_test.cc
        LIBRARIES npyglue npyglue::Torch
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    npygl_copy_torch_dlls(py_capsule_cutorch_test)
    add_test(NAME py_capsule_cutorch_test COMMAND py_capsule_cutorch_test)
endif()

# py_format_test: Python compile-time format strings test
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_format_test
        SOURCES py_format_test.cc
        LIBRARIES npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    add_test(NAME py_format_test COMMAND py_format_test)
endif()

# py_convert_test: test as<> conversion specializations
if(Python3_Development_FOUND)
    npygl_add_py3_executable(
        TARGET py_convert_test
        SOURCES py_convert_test.cc
        LIBRARIES npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    add_test(NAME py_convert_test COMMAND py_convert_test)
endif()

# npy_helpers_test: NumPy Python helpers tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    # libraries used by npy_helpers_test. we don't need Eigen3 and Armadillo so
    # we specify the interface targets that disable including their headers
    set(
        npy_helpers_test_libs
        Python3::NumPy npyglue npyglue_no_armadillo npyglue_no_eigen3
    )
    # C++17
    npygl_add_py3_executable(
        TARGET npy_helpers_test
        SOURCES npy_helpers_test.cc
        LIBRARIES ${npy_helpers_test_libs}
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    add_test(NAME npy_helpers_test COMMAND npy_helpers_test)
    # C++20
    if(NPYGL_HAS_CC_20)
        npygl_add_py3_executable(
            TARGET npy_helpers_test_cc20
            SOURCES npy_helpers_test.cc
            LIBRARIES ${npy_helpers_test_libs}
            USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
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
        LIBRARIES Python3::NumPy npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    # enable Armadillo features if available
    if(Armadillo_FOUND)
        target_link_libraries(npy_capsule_test PRIVATE npyglue::Armadillo)
    endif()
    # enable Eigen features if available
    if(Eigen3_FOUND)
        target_link_libraries(npy_capsule_test PRIVATE Eigen3::Eigen)
    endif()
    # enable LibTorch features if available
    if(TORCH_FOUND)
        npygl_use_release_crt(npy_capsule_test ON)
        target_link_libraries(npy_capsule_test PRIVATE npyglue::Torch)
        npygl_copy_torch_dlls(npy_capsule_test)
    endif()
    add_test(NAME npy_capsule_test COMMAND npy_capsule_test)
endif()

# npy_traits_test: NumPy traits class tests
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_traits_test
        SOURCES npy_traits_test.cc
        LIBRARIES Python3::NumPy npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    if(Armadillo_FOUND)
        target_link_libraries(npy_traits_test PRIVATE npyglue::Armadillo)
    endif()
    if(Eigen3_FOUND)
        target_link_libraries(npy_traits_test PRIVATE Eigen3::Eigen)
    endif()
    # note: don't really need to specify release C runtime for Windows and to
    # copy the Torch DLLs because we only need the torch::Tensor type
    # definition but we keep everything as usual for consistency
    if(TORCH_FOUND)
        npygl_use_release_crt(npy_traits_test ON)
        target_link_libraries(npy_traits_test PRIVATE npyglue::Torch)
        npygl_copy_torch_dlls(npy_traits_test)
    endif()
    add_test(NAME npy_traits_test COMMAND npy_traits_test)
endif()

# npy_views_test: testing views with NumPy
if(Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_py3_executable(
        TARGET npy_views_test
        SOURCES npy_views_test.cc
        # reduce compile time since we don't need Eigen3/Armadillo
        LIBRARIES Python3::NumPy npyglue npyglue_no_armadillo npyglue_no_eigen3
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    add_test(NAME npy_views_test COMMAND npy_views_test)
endif()

# pyhelpers: Python helpers test module
if(Python3_Development_FOUND)
    npygl_add_py3_extension(
        TARGET pyhelpers
        SOURCES pyhelpers.cc
        LIBRARIES npyglue
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
    )
    if(Python3_NumPy_FOUND)
        target_link_libraries(pyhelpers PRIVATE Python3::NumPy)
    endif()
    if(Armadillo_FOUND)
        target_link_libraries(pyhelpers PRIVATE npyglue::Armadillo)
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
    # libraries used by pymath. we don't need Eigen3 and Armadillo so we
    # specify the interface targets that disable including their headers
    set(
        pymath_libs
        Python3::NumPy npyglue npyglue_no_armadillo npyglue_no_eigen3
    )
    # C++17
    npygl_add_py3_extension(
        TARGET pymath
        SOURCES pymath.cc
        LIBRARIES ${pymath_libs}
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
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
            LIBRARIES ${pymath_libs}
            USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
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
    # FIXME: currently skipping until we determine how to determine constraints
    # on input/output shapes. all input shapes are allowed, but for functions
    # that return e.g. std::vector<T>, the resulting NumPy array is 1D
    set_tests_properties(
        pymath_test
        pymath_cc20_test PROPERTIES
        SKIP_RETURN_CODE 1
    )
endif()

# pymath_swig: SWIG Python math functions module
if(SWIG_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
    npygl_add_swig_py3_module(
        TARGET pymath_swig
        INTERFACE pymath_swig.i
        SWIG_CXX ON
        # not using Armadillo/Eigen3 currently (may change later) so we
        # explicitly disable inclusion of those headers for faster compile
        LIBRARIES Python3::NumPy npyglue npyglue_no_armadillo npyglue_no_eigen3
        USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
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
            SWIG_CXX ON
            # enable SWIG to see the C++20 declarations in math.hh
            SWIG_DEFINES NPYGL_SWIG_CC_20
            LIBRARIES
                Python3::NumPy
                npyglue
                # not using Armadillo/Eigen3 currently (may change later)
                npyglue_no_armadillo
                npyglue_no_eigen3
            USE_RELEASE_CRT ${NPYGL_PYTHON_USE_RELEASE_CRT}
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
    # FIXME: currently skipping until we determine how to determine constraints
    # on input/output shapes. all input shapes are allowed, but for functions
    # that return e.g. std::vector<T>, the resulting NumPy array is 1D
    set_tests_properties(
        pymath_swig_test
        pymath_swig_cc20_test PROPERTIES
        SKIP_RETURN_CODE 1
    )
endif()

# string_test: C++ string helpers test
add_executable(string_test string_test.cc)
target_link_libraries(string_test PRIVATE npyglue)
add_test(NAME string_test COMMAND string_test)

# type_traits_test: C++ type traits tests
add_executable(type_traits_test type_traits_test.cc)
target_link_libraries(type_traits_test PRIVATE npyglue)
add_test(NAME type_traits_test COMMAND type_traits_test)

# traits_checker_filter_test: traits_checker test filtering experiments
add_executable(traits_checker_filter_test traits_checker_filter_test.cc)
target_link_libraries(traits_checker_filter_test PRIVATE npyglue)
# note: this works but is too visually noisy
# npygl_traits_checker_add_tests(traits_checker_filter_test)
add_test(NAME traits_checker_filter_test COMMAND traits_checker_filter_test)
# tests for -h, --help
add_test(
    NAME "traits_checker_filter_test -h"
    COMMAND traits_checker_filter_test -h
)
add_test(
    NAME "traits_checker_filter_test --help"
    COMMAND traits_checker_filter_test --help
)
set_tests_properties(
    "traits_checker_filter_test -h"
    "traits_checker_filter_test --help" PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage: \\<progname\\> \\[-h\\]"
)
# tests for -v, --version
add_test(
    NAME "traits_checker_filter_test -v"
    COMMAND traits_checker_filter_test -v
)
add_test(
    NAME "traits_checker_filter_test --version"
    COMMAND traits_checker_filter_test --version
)
set_tests_properties(
    "traits_checker_filter_test -v"
    "traits_checker_filter_test --version" PROPERTIES
    PASS_REGULAR_EXPRESSION "${NPYGL_VERSION}"
)
# tests for -l, --list-tests
add_test(
    NAME "traits_checker_filter_test -l"
    COMMAND traits_checker_filter_test -l
)
add_test(
    NAME "traits_checker_filter_test --list-tests"
    COMMAND traits_checker_filter_test --list-tests
)
# note: can't test for -t, --test-name as demangling varies across platforms
# except for testing the error behavior as shown below
add_test(
    NAME "traits_checker_filter_test -t unknown"
    COMMAND traits_checker_filter_test -t unknown
)
add_test(
    NAME "traits_checker_filter_test --test-name unknown"
    COMMAND traits_checker_filter_test --test-name unknown
)
set_tests_properties(
    "traits_checker_filter_test -t unknown"
    "traits_checker_filter_test --test-name unknown" PROPERTIES
    PASS_REGULAR_EXPRESSION "Error: No test with name"
)
# tests for -T, --test-pattern
add_test(
    NAME "traits_checker_filter_test -T static_size"
    COMMAND traits_checker_filter_test -T static_size
)
add_test(
    NAME "traits_checker_filter_test --test-pattern static_size"
    COMMAND traits_checker_filter_test --test-pattern static_size
)
# test for --print-driver-type
add_test(
    NAME "traits_checker_filter_test --print-driver-type"
    COMMAND traits_checker_filter_test --print-driver-type
)

# ostream_test: C++ ostream extensions test
add_executable(ostream_test ostream_test.cc)
target_link_libraries(ostream_test PRIVATE npyglue)
add_test(NAME ostream_test COMMAND ostream_test)
add_test(
    NAME ostream_test_with_concurrent
    COMMAND ostream_test --with-concurrent
)
# brief tests of the CLI options
add_test(NAME ostream_test_h COMMAND ostream_test -h)
add_test(NAME ostream_test_help COMMAND ostream_test --help)
add_test(NAME ostream_test_list_tests COMMAND ostream_test --list-tests)
add_test(
    NAME ostream_test_list_all_tests
    COMMAND ostream_test --list-tests --with-concurrent
)
add_test(NAME ostream_test_bad_option COMMAND ostream_test --list-tests ssss)
set_tests_properties(
    ostream_test_bad_option PROPERTIES
    PASS_REGULAR_EXPRESSION "Unknown argument"
)

# traits_checker_test: C++ traits_checker.hh functionality tests
# add_executable(traits_checker_test traits_checker_test.cc)
# add_test(NAME traits_checker_test COMMAND traits_checker_test)

# termcolors: program to print the SGR values and 8-bit terminal colors
add_executable(termcolors termcolors.cc)
target_link_libraries(termcolors PRIVATE npyglue)
if(WIN32)
    target_link_libraries(termcolors PRIVATE kernel32)
endif()
# smoke test. not super useful as we need to visually inspect for correctness
add_test(NAME termcolors COMMAND termcolors)

# llvm_demangle_test: test demangling of type_traits_test_driver using LLVM
# note: only applies to compilers that follow the Itanium C++ ABI. nowadays
# this is all the major compilers except for MSVC
if(NPYGL_ITANIUM_ABI AND LLVM_FOUND)
    # note: NPYGL_USE_LLVM_DEMANGLE already defined in source file. typically,
    # one would just add this to the compiler's macro definitions or for CMake,
    # use the npyglue::use_llvm_demangle INTERFACE target
    add_executable(llvm_demangle_test llvm_demangle_test.cc)
    target_link_libraries(
        llvm_demangle_test PRIVATE
        npyglue npyglue::llvm_demangle
    )
    add_test(NAME llvm_demangle_test COMMAND llvm_demangle_test)
    # traits_checker_driver and the last traits in type_traits_test_driver
    # should be part of the printed type name
    set_tests_properties(
        llvm_demangle_test PROPERTIES
        PASS_REGULAR_EXPRESSION "traits_checker_driver<.+is_iterable,.+>"
    )
endif()

# torch_test: test torch.hh LibTorch helpers
if(TORCH_FOUND)
    add_executable(torch_test torch_test.cc)
    target_link_libraries(torch_test PRIVATE npyglue)
    # Torch C++ DLLs are linked against release C runtime
    npygl_use_release_crt(torch_test ON)
    target_link_libraries(torch_test PRIVATE npyglue::Torch)
    npygl_copy_torch_dlls(torch_test)
    if(Armadillo_FOUND)
        target_link_libraries(torch_test PRIVATE npyglue::Armadillo)
    endif()
    if(Eigen3_FOUND)
        target_link_libraries(torch_test PRIVATE Eigen3::Eigen)
    endif()
    add_test(NAME torch_test COMMAND torch_test)
endif()

# torch_traits_test: test LibTorch type traits
if(TORCH_FOUND)
    add_executable(torch_traits_test torch_traits_test.cc)
    npygl_use_release_crt(torch_traits_test ON)
    target_link_libraries(torch_traits_test PRIVATE npyglue::Torch npyglue)
    npygl_copy_torch_dlls(torch_traits_test)
    if(Armadillo_FOUND)
        target_link_libraries(torch_traits_test PRIVATE npyglue::Armadillo)
    endif()
    if(Eigen3_FOUND)
        target_link_libraries(torch_traits_test PRIVATE Eigen3::Eigen)
    endif()
    add_test(NAME torch_traits_test COMMAND torch_traits_test)
endif()

# ctti_test: npygl::type_name<T> unit tests
add_executable(ctti_test ctti_test.cc)
target_link_libraries(ctti_test PRIVATE npyglue)
add_test(NAME ctti_test COMMAND ctti_test)
# skip if not supported
set_tests_properties(ctti_test PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED")

# ctti_soak_test: npygl::type_name<T> soak test with type_traits_test_driver
add_executable(ctti_soak_test ctti_soak_test.cc)
target_link_libraries(ctti_soak_test PRIVATE npyglue)
add_test(NAME ctti_soak_test COMMAND ctti_soak_test)
# like llvm_demangle_test traits_checker_driver and the last traits in
# type_traits_test_driver should be part of the type name printed
set_tests_properties(
    ctti_soak_test PROPERTIES
    PASS_REGULAR_EXPRESSION "traits_checker_driver<.+is_iterable,.+>"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
)

# npy_intp_alias_test: test when npy_intp can alias different types
if(Python3_NumPy_FOUND AND TORCH_FOUND)
    npygl_add_py3_executable(
        TARGET npy_intp_alias_test
        SOURCES npy_intp_alias_test.cc
        # note: the included Torch header requires some externally-defined
        # symbols and thus needs to link against the Torch libraries.
        # Python3::NumPy also transitively links against the Python 3 runtime
        LIBRARIES Python3::NumPy npyglue npyglue::Torch
        # using release CRT unconditionally to match with Torch DLLs
        USE_RELEASE_CRT ON
    )
    # still need to copy Torch DLLs for completeness
    npygl_copy_torch_dlls(npy_intp_alias_test)
    add_test(NAME npy_intp_alias_test COMMAND npy_intp_alias_test)
    # skips when sizeof(npt_intp) != sizeof(torch_dim_type)
    set_tests_properties(
        npy_intp_alias_test PROPERTIES
        SKIP_REGULAR_EXPRESSION "SKIPPED"
    )
endif()

# npy_capsule_cutorch_test: NumPy test using Torch GPU tensors to back arrays
if(CUDAToolkit_FOUND AND Python3_NumPy_FOUND AND TORCH_FOUND)
    npygl_add_py3_executable(
        TARGET npy_capsule_cutorch_test
        SOURCES npy_capsule_cutorch_test.cc
        LIBRARIES Python3::NumPy npyglue npyglue::Torch
        USE_RELEASE_CRT ON
    )
    npygl_copy_torch_dlls(npy_capsule_cutorch_test)
    add_test(NAME npy_capsule_cutorch_test COMMAND npy_capsule_cutorch_test)
endif()
