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
#   LIBRARIES library...
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
#   LIBRARIES library...
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
