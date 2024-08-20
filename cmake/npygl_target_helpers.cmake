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
# This is implemented using add_custom_command and regenerates the SWIG wrapper
# code whenever the file's dependencies are detected to change. Therefore, it
# actually is an improvement over the UseSWIG swig_add_library command and does
# not result in any changes to source file properties.
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
#   SWIG_INCLUDE_DIRS dir1...
#       Additional C/C++/SWIG include directories to pass to SWIG
#   USE_TARGET_NAME (ON|OFF))
#       Indicate whether CMake target name should be used as the module name.
#       This can be convenient to map the same source to different modules.
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
    # target name, SWIG source, SWIG C++ mode, uese CMake target name as SWIG
    # module name, use release C runtime on Windows
    set(
        SINGLE_VALUE_ARGS
        TARGET INTERFACE SWIG_CC USE_TARGET_NAME USE_RELEASE_CRT
    )
    # source list + libraries to link against and include directories
    set(
        MULTI_VALUE_ARGS
        SOURCES LIBRARIES SWIG_DEFINES SWIG_OPTIONS SWIG_INCLUDE_DIRS
    )
    # parse arguments
    cmake_parse_arguments(
        HOST
        "" "${SINGLE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGV}
    )
    # prepend -D to SWIG defines if any
    if(HOST_SWIG_DEFINES)
        list(TRANSFORM HOST_SWIG_DEFINES PREPEND -D)
    endif()
    # intermediate output directory + make if it does not exist
    set(OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_TARGET})
    file(MAKE_DIRECTORY ${OUTFILE_DIR})
    # dependencies file for the target
    set(DEPS_FILE ${OUTFILE_DIR}/${HOST_INTERFACE}.d)
    # C/C++ wrapper output file name + path following swig_add_library format
    if(HOST_SWIG_CC)
        set(OUTFILE_NAME ${HOST_TARGET}PYTHON_wrap.cxx)
    else()
        set(OUTFILE_NAME ${HOST_TARGET}PYTHON_wrap.c)
    endif()
    set(OUTFILE_PATH ${OUTFILE_DIR}/${OUTFILE_NAME})
    # base SWIG compile options
    set(SWIG_BASE_OPTIONS -python)
    # compile for Python 3. from SWIG 4.1 onwards however we need to use
    # %feature("python:annotations", "c") directive instead
    # FIXME: SWIG 4.1+ will not have the C++ type annotations
    # note: probably don't need it when running deps command either
    if(SWIG_VERSION VERSION_LESS 4.1)
        set(SWIG_BASE_OPTIONS ${SWIG_BASE_OPTIONS} -py3)
    endif()
    # enable SWIG C++ mode
    if(HOST_SWIG_CC)
        set(SWIG_BASE_OPTIONS ${SWIG_BASE_OPTIONS} -c++)
    endif()
    # user include directory added by default
    set(SWIG_BASE_OPTIONS ${SWIG_BASE_OPTIONS} -I${NPYGL_INCLUDE_DIR})
    if(HOST_SWIG_INCLUDE_DIRS)
        set(SWIG_BASE_OPTIONS ${SWIG_BASE_OPTIONS} ${HOST_SWIG_INCLUDE_DIRS})
    endif()
    # use target name as module name
    # note: probably don't need it when running deps command
    if(HOST_USE_TARGET_NAME)
        set(SWIG_BASE_OPTIONS ${SWIG_BASE_OPTIONS} -module ${HOST_TARGET})
    endif()
    # generate list of interface dependencies
    execute_process(
        COMMAND
            ${SWIG_EXECUTABLE} ${SWIG_BASE_OPTIONS} -MM -MF ${DEPS_FILE}
                -o ${OUTFILE_PATH} ${HOST_INTERFACE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE DEPS_RESULT
    )
    # warn on error
    if(DEPS_RESULT)
        message(WARNING "Unable to generated dependencies for ${HOST_TARGET}")
    endif()
    # read dependencies file lines as strings
    file(STRINGS ${DEPS_FILE} DEPS_LIST)
    # transform the deps list. for Windows, replace all \ with /. need to strip
    # whitespace a couple times as we remove extra characters
    list(TRANSFORM DEPS_LIST STRIP)
    list(TRANSFORM DEPS_LIST REPLACE " \\$" "")
    list(TRANSFORM DEPS_LIST STRIP)
    list(TRANSFORM DEPS_LIST REPLACE ":$" "")
    list(TRANSFORM DEPS_LIST REPLACE "\\\\" "/")
    # pop first element to remove wrapper output file itself
    list(POP_FRONT DEPS_LIST)
    # custom command to build SWIG wrapper files when deps change. the target
    # language files are copied to build directory
    add_custom_command(
        OUTPUT ${OUTFILE_PATH}
        COMMAND
            ${SWIG_EXECUTABLE} ${SWIG_BASE_OPTIONS} ${HOST_SWIG_DEFINES}
                ${HOST_SWIG_OPTIONS} -o ${OUTFILE_PATH} -outdir
                ${CMAKE_BINARY_DIR}$<${NPYGL_MULTI_CONFIG_GENERATOR}:/$<CONFIG>>
                ${HOST_INTERFACE}
        DEPENDS ${DEPS_LIST}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating SWIG Python wrapper ${OUTFILE_NAME}"
        COMMAND_EXPAND_LISTS
    )
    # create SWIG module
    add_library(${HOST_TARGET} MODULE ${OUTFILE_PATH} ${HOST_SOURCES})
    target_link_libraries(${HOST_TARGET} PRIVATE Python3::Python ${HOST_LIBRARIES})
    # following convention of prefixing with underscore and ensure no lib prefix
    set_target_properties(${HOST_TARGET} PROPERTIES OUTPUT_NAME _${HOST_TARGET})
    set_target_properties(${HOST_TARGET} PROPERTIES PREFIX "")
    # on Windows, extension is .pyd
    if(WIN32)
        set_target_properties(${HOST_TARGET} PROPERTIES SUFFIX ".pyd")
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
    # use release VC++ C runtime if specified
    if(MSVC AND HOST_USE_RELEASE_CRT)
        set_target_properties(
            ${HOST_TARGET} PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
endfunction()
