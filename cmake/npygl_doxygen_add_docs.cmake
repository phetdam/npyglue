cmake_minimum_required(VERSION 3.20)

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


##
# Run pyginject on the specified Markdown files for Pygments code highlighting.
#
# Doxygen will not provide its default syntax highlighting for Markdown code
# blocks for languages that it does not understand, so we use the pyginject.py
# Python script to replace Markdown triple-backtick code blocks that are
# surrounded with the following tags with Pygments formatted HTML. E.g.
#
# <!-- pygmentize: on -->
#
# ... code blocks ...
#
# <!-- pygmentize: off -->
#
# Each Markdown input file input.md will be converted to input-pygmentized.md
# that contains the formatted HTML (if any) and the resulting file names are
# saved in a list variable for easy integration with npygl_doxygen_add_docs.
#
# Arguments:
#
#   TARGET target               Name of top-level target driving execution
#   OUTPUTS_LIST out_list       List of the generated file names
#   SOURCES input...            List of Markdown input files to process
#
function(npygl_pygmentize)
    cmake_parse_arguments(ARG "" "TARGET;OUTPUTS_LIST" "SOURCES" ${ARGN})
    # must be nonempty
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "TARGET is required")
    endif()
    if(NOT ARG_OUTPUTS_LIST)
        message(FATAL_ERROR "OUTPUTS_LIST is required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "SOURCES requires at least one input file")
    endif()
    # .md extension is required
    foreach(_source ${ARG_SOURCES})
        string(LENGTH "${_source}" name_len)
        # name must be at least 4 characters (.md is 3 characters)
        if(name_len LESS 4)
            message(STATUS "SOURCES input ${_source} file name is invalid")
        endif()
        # substring to get the .md extension
        math(EXPR ext_index "${name_len} - 3")
        string(SUBSTRING "${_source}" ${ext_index} -1 _source_ext)
        # if not Markdown, error
        if(NOT _source_ext STREQUAL ".md")
            message(FATAL_ERROR "SOURCES input ${_source} is not a Markdown file")
        endif()
    endforeach()
    # top-level custom target + output list
    add_custom_target(
        ${ARG_TARGET}
        COMMENT "Running pyginject on Markdown files"
    )
    set(${ARG_OUTPUTS_LIST} "")
    # add custom command targets for each file
    foreach(_source ${ARG_SOURCES})
        # create valid target name by replacing special chars with _ + prefix
        string(REPLACE "/" "_" _source_target "${_source}")
        string(REPLACE "\\" "_" _source_target "${_source_target}")
        string(REPLACE "\." "_" _source_target "${_source_target}")
        string(REPLACE "-" "_" _source_target "${_source_target}")
        string(PREPEND _source_target "npyglue_")
        # create output name from source (need absolute path for BYPRODUCTS)
        string(REGEX REPLACE "\.md$" "-pygmentized\.md" _output "${_source}")
        cmake_path(ABSOLUTE_PATH _output OUTPUT_VARIABLE _output_path )
        # add target for the file
        add_custom_target(
            ${_source_target}
            COMMAND python
                    tools/pyginject.py -i ${_source} -o ${_output} -w doxygen
            BYPRODUCTS ${_output_path}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating ${_output}"
        )
        # hang dependency on main target + update output list
        add_dependencies(${ARG_TARGET} ${_source_target})
        list(APPEND ${ARG_OUTPUTS_LIST} ${_output})
    endforeach()
    # propagate output list to parent scope
    set(${ARG_OUTPUTS_LIST} ${${ARG_OUTPUTS_LIST}} PARENT_SCOPE)
endfunction()
