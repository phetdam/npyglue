cmake_minimum_required(VERSION 3.15)

##
# npygl_get_build_info.cmake
#
# Determine the build info of the project from the current Git hash.
#
# If Git is not available "DEV" will simply be used.
#

##
# Write the current Git build hash into the given variable of "DEV" if no Git.
#
# If marked as official release the build info will be the empty string.
#
# Arguments:
#   var                         Variable to write Git build has or "DEV" into
#   [IS_RELEASE is_release]     TRUE if an official release (info will be empty)
#
function(npygl_get_build_info var)
    cmake_parse_arguments(ARG "" "IS_RELEASE" "" ${ARGN})
    # official release so empty
    if(ARG_IS_RELEASE)
        set(${var} "" PARENT_SCOPE)
        return()
    endif()
    # Git not found
    if(NOT Git_FOUND)
        set(${var} DEV PARENT_SCOPE)
        return()
    endif()
    # otherwise, use git rev-parse --short HEAD
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        RESULT_VARIABLE git_res
        ERROR_VARIABLE git_err
        OUTPUT_VARIABLE git_out
        ERROR_STRIP_TRAILING_WHITESPACE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # failed
    if(git_res)
        message(FATAL_ERROR "git rev-parse --short failed:\n${git_err}")
    endif()
    # otherwise, use as build info
    set(${var} ${git_out} PARENT_SCOPE)
endfunction()
