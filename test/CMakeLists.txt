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
    # note: uses swig_add_library which does not properly handle conditional
    # compilation of SWIG interfaces for Visual Studio generators
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
    # custom implementation specialized for Python C++ SWIG wrappers
    # TODO: replace npygl_add_swig_py3_module implementation with cleaned up
    # version of the following. this is significant cleaner in comparison
    if(NPYGL_HAS_CC_20)
        # target name
        set(PM_SWIG_CC20_TARGET pymath_swig_cc20)
        # input file
        set(PM_SWIG_CC20_INPUT pymath_swig.i)
        # extra SWIG definitions
        set(PM_SWIG_CC20_DEFINES NPYGL_SWIG_CC_20)
        list(TRANSFORM PM_SWIG_CC20_DEFINES PREPEND -D)
        # intermediate output directory
        set(PM_SWIG_CC20_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR}/${PM_SWIG_CC20_TARGET})
        # make directory if it doesn't exist
        file(MAKE_DIRECTORY ${PM_SWIG_CC20_OUTDIR})
        # deps file
        set(PM_SWIG_CC20_DEPS ${PM_SWIG_CC20_OUTDIR}/${PM_SWIG_CC20_INPUT}.d)
        # output file name + path
        set(PM_SWIG_CC20_OUTNAME ${PM_SWIG_CC20_TARGET}PYTHON_wrap.cxx)
        set(PM_SWIG_CC20_OUTFILE ${PM_SWIG_CC20_OUTDIR}/${PM_SWIG_CC20_OUTNAME})
        # generate list of interface dependencies
        execute_process(
            COMMAND
                ${SWIG_EXECUTABLE} -python -c++ -I${NPYGL_INCLUDE_DIR} -MM -MF
                    ${PM_SWIG_CC20_DEPS} -o ${PM_SWIG_CC20_OUTFILE}
                    ${PM_SWIG_CC20_INPUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE NPYGL_PM_SWIG_CC20_DEPS_RESULT
        )
        # if failed, error
        if(NPYGL_PM_SWIG_CC20_DEPS_RESULT)
            message(WARNING "Unable to generate deps for ${PM_SWIG_CC20_TARGET}")
        endif()
        # read file
        file(STRINGS ${PM_SWIG_CC20_DEPS} NPYGL_PM_SWIG_CC20_DEPS)
        # transform the deps. for Windows, we replace all \ with /
        list(TRANSFORM NPYGL_PM_SWIG_CC20_DEPS STRIP)
        list(TRANSFORM NPYGL_PM_SWIG_CC20_DEPS REPLACE " \\$" "")
        list(TRANSFORM NPYGL_PM_SWIG_CC20_DEPS STRIP)
        list(TRANSFORM NPYGL_PM_SWIG_CC20_DEPS REPLACE ":$" "")
        list(TRANSFORM NPYGL_PM_SWIG_CC20_DEPS REPLACE "\\\\" "/")
        # pop first element to remove output file
        list(POP_FRONT NPYGL_PM_SWIG_CC20_DEPS)
        # development debugging
        # message(STATUS "NPYGL_PM_SWIG_CC20_DEPS: ${NPYGL_PM_SWIG_CC20_DEPS}")
        # message(STATUS "PM_SWIG_CC20_OUTFILE: ${PM_SWIG_CC20_OUTFILE}")
        # custom command to build SWIG outfile when deps change
        add_custom_command(
            OUTPUT ${PM_SWIG_CC20_OUTFILE}
            COMMAND
                ${SWIG_EXECUTABLE} -python -c++ -I${NPYGL_INCLUDE_DIR}
                    ${PM_SWIG_CC20_DEFINES} -o ${PM_SWIG_CC20_OUTFILE} -outdir
                    ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
                    ${PM_SWIG_CC20_INPUT}
            DEPENDS ${NPYGL_PM_SWIG_CC20_DEPS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating SWIG Python wrapper ${PM_SWIG_CC20_OUTNAME}"
            COMMAND_EXPAND_LISTS
        )
        # create SWIG module
        add_library(${PM_SWIG_CC20_TARGET} MODULE ${PM_SWIG_CC20_OUTFILE})
        npygl_require_cc_std(TARGET ${PM_SWIG_CC20_TARGET} CC_STD 20)
        # library linking
        target_link_libraries(
            ${PM_SWIG_CC20_TARGET} PRIVATE
            Python3::Python Python3::NumPy
        )
        # follow convention of prefixing with underscore
        set_target_properties(
            ${PM_SWIG_CC20_TARGET} PROPERTIES
            OUTPUT_NAME _${PM_SWIG_CC20_TARGET}
        )
        # no lib prefix on UNIX systems and Windows has .pyd suffix
        if(UNIX)
            set_target_properties(${PM_SWIG_CC20_TARGET} PROPERTIES PREFIX "")
        endif()
        if(WIN32)
            set_target_properties(${PM_SWIG_CC20_TARGET} PROPERTIES SUFFIX ".pyd")
        endif()
        # silence some MSVC warnings we can't do anything about
        if(MSVC)
            target_compile_options(
                ${PM_SWIG_CC20_TARGET} PRIVATE
                # C4365: signed/unsigned mismatch during type conversion
                /wd4365
                # C4668: macro not defined, replacing with 0 (__GNUC__ not defined)
                /wd4668
            )
        endif()
        # use release VC++ C runtime if specified
        if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
            set_target_properties(
                ${PM_SWIG_CC20_TARGET} PROPERTIES
                MSVC_RUNTIME_LIBRARY MultiThreadedDLL
            )
        endif()
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
