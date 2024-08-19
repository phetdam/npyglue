cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##
# Require that a target is compiled under a particular C++ standard.
#
# Arguments:
#   TARGET target
#       Name of the target being compiled
#   CC_STD std
#       C++ standard to compile under, e.g. a value for CXX_STANDARD like 20
#
function(npygl_require_cc_std)
    # target name and C++ standard
    set(SINGLE_VALUE_ARGS TARGET CC_STD)
    cmake_parse_arguments(HOST "" "${SINGLE_VALUE_ARGS}" "" ${ARGV})
    # set target properties (standard is required)
    set_target_properties(
        ${HOST_TARGET} PROPERTIES
        CXX_STANDARD ${HOST_CC_STD}
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

##
# Add a C/C++ executable that embeds the Python 3 interpreter.
#
# This function provides some convenience logic, e.g. ensuring that we are
# linking against the right Python 3 runtime library.
#
# Arguments:
#   TARGET target
#       Name of the executable target
#   SOURCES source1...
#       Target sources needed for compilation
#   LIBRARIES library1...
#       Additional library targets besides Python3::Python needed for linking
#   USE_RELEASE_CRT (ON|OFF)
#       Indicate whether or not the dynamic release C runtime should be used,
#       which ensures that on Windows, the release Python runtime library is
#       used. This can be passed a condition and has no effect when not
#       compiling for Windows. Generally should always be ON.
#
function(npygl_add_py3_executable)
    # target name + whether or not to use release C runtime on Windows
    set(SINGLE_VALUE_ARGS TARGET USE_RELEASE_CRT)
    # source list + libraries to link against
    set(MULTI_VALUE_ARGS SOURCES LIBRARIES)
    # parse arguments
    cmake_parse_arguments(
        HOST
        "" "${SINGLE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGV}
    )
    # add executable and link libraries
    add_executable(${HOST_TARGET} ${HOST_SOURCES})
    target_link_libraries(
        ${HOST_TARGET} PRIVATE
        Python3::Python ${HOST_LIBRARIES}
    )
    # usually no Python debug runtime library
    if(MSVC AND HOST_USE_RELEASE_CRT)
        set_target_properties(
            ${HOST_TARGET} PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
endfunction()

##
# Add a Python C/C++ extension module.
#
# Arguments:
#   TARGET target
#       Name of the Python extension module target
#   SOURCES source1...
#       Target sources needed for compilation
#   LIBRARIES library1...
#       Additional library targets besides Python3::Python needed for linking
#   USE_RELEASE_CRT (ON|OFF)
#       Indicate whether or not the dynamic release C runtime should be used,
#       which ensures that on Windows, the release Python runtime library is
#       used. This can be passed a condition and has no effect when not
#       compiling for Windows. Generally should always be ON.
#
function(npygl_add_py3_extension)
    # target name + whether or not to use release C runtime on Windows
    set(SINGLE_VALUE_ARGS TARGET USE_RELEASE_CRT)
    # source list + libraries to link against
    set(MULTI_VALUE_ARGS SOURCES LIBRARIES)
    # parse arguments
    cmake_parse_arguments(
        HOST
        "" "${SINGLE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGV}
    )
    # add module, ensure no library prefix, .pyd suffix on Windows
    add_library(${HOST_TARGET} MODULE ${HOST_SOURCES})
    # technically, could encase in a if(UNIX) ... endif() block
    set_target_properties(${HOST_TARGET} PROPERTIES PREFIX "")
    if(WIN32)
        set_target_properties(${HOST_TARGET} PROPERTIES SUFFIX ".pyd")
    endif()
    target_link_libraries(
        ${HOST_TARGET} PRIVATE
        Python3::Python ${HOST_LIBRARIES}
    )
    # usually no Python debug runtime library
    if(MSVC AND HOST_USE_RELEASE_CRT)
        set_target_properties(
            ${HOST_TARGET} PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
endfunction()

##
# Add a SWIG-generated Python C/C++ extension module.
#
# Note:
#
# This is still experimental as there were issues identified with
# swig_add_library when attempting to conditionally compile pymath_swig.i for
# use in two different swig_add_library calls for Visual Studio generators.
#
# We are considering using a custom add_custom_command for SWIG compilation.
#
# Arguments:
#   TARGET target
#       Name of the SWIG-generated Python extension module target
#   INTERFACE
#       SWIG interface input file for wrapper generation
#   SWIG_CC (ON|OFF)
#       Enabled/disable SWIG C++ mode (defaults to OFF)
#   SWIG_DEFINES macro1...
#       SWIG macros to define when running SWIG
#   SWIG_OPTIONS option1...
#       Additional options to pass to SWIG
#   SOURCES source1...
#       Additional C/C++ sources needed for compilation
#   LIBRARIES library1...
#       Additional library targets besides Python3::Python needed for linking
#   USE_RELEASE_CRT (ON|OFF)
#       Indicate whether or not the dynamic release C runtime should be used,
#       which ensures that on Windows, the release Python runtime library is
#       used. This can be passed a condition and has no effect when not
#       compiling for Windows. Generally should always be ON.
#
function(npygl_add_swig_py3_module)
    # target name, SWIG source, SWIG C++ mode, use release C runtime on Windows
    set(SINGLE_VALUE_ARGS TARGET INTERFACE SWIG_CC USE_RELEASE_CRT)
    # source list + libraries to link against
    set(MULTI_VALUE_ARGS SOURCES LIBRARIES SWIG_DEFINES SWIG_OPTIONS)
    # parse arguments
    cmake_parse_arguments(
        HOST
        "" "${SINGLE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGV}
    )
    # enable SWIG C++ mode
    if(HOST_SWIG_CC)
        set_source_files_properties(${HOST_INTERFACE} PROPERTIES CPLUSPLUS ON)
    endif()
    swig_add_library(
        ${HOST_TARGET}
        TYPE MODULE
        LANGUAGE python
        # generated output artifacts path. for multi-config generators we need
        # to include the extra per-config subdirectory manually
        OUTPUT_DIR
            ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
        # generated wrapper source output path
        # note: we could use OUTPUT_DIR/gensrc for this to avoid WSL/Windows
        # builds sometimes (?) recompiling the SWIG file
        OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_TARGET}
        SOURCES ${HOST_INTERFACE} ${HOST_SOURCES}
    )
    # add SWIG compile definitions and options
    if(HOST_SWIG_DEFINES)
        set_property(
            TARGET ${HOST_TARGET} PROPERTY
            SWIG_COMPILE_DEFINITIONS ${HOST_SWIG_DEFINES}
        )
    endif()
    if(HOST_SWIG_OPTIONS)
        set_property(
            TARGET ${HOST_TARGET} PROPERTY
            SWIG_COMPILE_OPTIONS ${HOST_SWIG_OPTIONS}
        )
    endif()
    # silence some MSVC warnings we can't do anything about
    if(MSVC)
        target_compile_options(
            ${HOST_TARGET} PRIVATE
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
        set_property(
            TARGET ${HOST_TARGET} APPEND PROPERTY
            SWIG_COMPILE_OPTIONS -py3
        )
    endif()
    # usually no Python debug runtime library
    if(MSVC AND PY_MSVC_ALWAYS_RELEASE)
        set_target_properties(
            ${HOST_TARGET} PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    target_link_libraries(
        ${HOST_TARGET} PRIVATE
        Python3::Python ${HOST_LIBRARIES}
    )
endfunction()
