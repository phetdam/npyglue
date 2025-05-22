cmake_minimum_required(VERSION 3.15)

##
# npygl_doxygen_add_docs.cmake
#
# Provides a function to for conditonally setting up a Doxygen doc target.
#
# By using a function variables are scoped before calling doxygen_add_docs.
#

##
# Sets a DOXYGEN_* variable in the current scope if it is not already defined.
#
# ArgumentsL
#   option      Doxygen option name, e.g. DISABLE_INDEX
#   args...     Argument values for that option
#
function(npygl_doxygen_default option)
    if(NOT DEFINED DOXYGEN_${option})
        set(DOXYGEN_${option} ${ARGN} PARENT_SCOPE)
    endif()
endfunction()

##
# Create the Doxygen documentation target for the project.
#
# This sets some DOXYGEN_* variables by default to better follow the preferred
# conventions for the project, but since any overrides are applied after
# defaults are set, the presets can be overridden as desired.
#
# The following Doxygen configuration options are supported:
#
#   QUIET                   default: YES
#   WARN_IF_UNDOCUMENTED    default: NO
#   JAVADOC_AUTOBRIEF       default: YES
#   DISABLE_INDEX           default: NO
#   GENERATE_TREEVIEW       default: NO
#   PROJECT_NUMBER          default: ${NPYGL_VERSION}
#   HTML_EXTRA_STYLESHEET
#   HTML_DYNAMIC_SECTIONS   default: YES
#   STRIP_FROM_INC_PATH     default: ${NPYGL_INCLUDE_DIR}
#   PREDEFINED
#   LAYOUT_FILE
#
# Arguments:
#   target                      Name for the Doxygen documentation target
#   SOURCES file_or_dir...      Files and directories for Doxygen to process
#   [COMMENT comment]           Message to print while running Doxygen
#   [VERBOSE]                   Run verbosely, printing the Doxygen config
#                               options used. This is helpful for debugging
#   [[OPTION value[...]]...]    Doxygen options and values
#
function(npygl_doxygen_add_docs target)
    # single-value options
    set(
        _single_value_opts
        # Doxygen options
        QUIET
        WARN_IF_UNDOCUMENTED
        JAVADOC_AUTOBRIEF
        DISABLE_INDEX
        GENERATE_TREEVIEW
        PROJECT_NUMBER
        HTML_DYNAMIC_SECTIONS
        LAYOUT_FILE
        # non-Doxygen options
        COMMENT
    )
    # multi-value options
    set(
        _multi_value_opts
        # Doxygen options
        HTML_EXTRA_STYLESHEET
        STRIP_FROM_INC_PATH
        PREDEFINED
        # non-Doxygen options
        SOURCES
    )
    # parse arguments
    cmake_parse_arguments(
        DOXYGEN
        "VERBOSE" "${_single_value_opts}" "${_multi_value_opts}" ${ARGN}
    )
    # sources required
    if(NOT DEFINED DOXYGEN_SOURCES)
        message(FATAL_ERROR "No SOURCES provided for Doxygen")
    endif()
    # register Doxygen defaults (sets variables if not defined)
    npygl_doxygen_default(QUIET YES)
    npygl_doxygen_default(WARN_IF_UNDOCUMENTED NO)
    npygl_doxygen_default(JAVADOC_AUTOBRIEF YES)
    # for Doxygen >= 1.13.0 compatibility since defaults were changed to YES
    npygl_doxygen_default(DISABLE_INDEX NO)
    npygl_doxygen_default(GENERATE_TREEVIEW NO)
    # uses NPYGL_VERSION instead of PROJECT_VERSION
    npygl_doxygen_default(PROJECT_NUMBER ${NPYGL_VERSION})
    npygl_doxygen_default(HTML_DYNAMIC_SECTIONS YES)
    npygl_doxygen_default(STRIP_FROM_INC_PATH ${NPYGL_INCLUDE_DIR})
    # register non-Doxygen defaults (sets variables if not defined)
    npygl_doxygen_default(COMMENT "Running Doxygen for ${target}")
    # if verbose, print DOXYGEN_* variables
    if(DOXYGEN_VERBOSE)
        message(STATUS "Configuring Doxygen target ${target} with options:")
        foreach(_var ${_single_value_opts} ${_multi_value_opts})
            message(STATUS "  ${_var} = ${DOXYGEN_${_var}}")
        endforeach()
    endif()
    # add Doxygen target
    doxygen_add_docs(${target} ${DOXYGEN_SOURCES} COMMENT ${DOXYGEN_COMMENT})
endfunction()
