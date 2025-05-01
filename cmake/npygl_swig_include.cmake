cmake_minimum_required(VERSION 3.15)

##
# npygl_swig_include.cmake
#
# Propagates CMake INTERFACE target properties to SWIG target properties.
#
# This allows one to create INTERFACE targets that use the INTERFACE include
# directories, compile definitions, and compile options properties and then
# propagate these to the SWIG_* properties consumed by swig_add_library.
#

##
# Propagate INTERFACE properties as SWIG properties to the module target.
#
# The following INTERFACE properties will be consumed:
#
#   INTERFACE_INCLUDE_DIRECTORIES
#   INTERFACE_COMPILE_DEFNIITIONS
#   INTERFACE_COMPILE_OPTIONS
#
# These will be appended to the corresponding SWIG_* properties for the target.
#
# Arguments:
#   target              Module target created by swig_add_library
#   TARGETS tgts...     INTERFACE targets to consume properties from
#
function(npygl_swig_include target)
    cmake_parse_arguments(ARG "" "" "TARGETS" ${ARGN})
    # must have at least one target
    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "TARGETS requires at least one argument")
    endif()
    # iterate through targets
    foreach(tgt ${ARG_TARGETS})
        # for each property suffix
        foreach(prop INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
            # retrieve the INTERFACE_${suffix} property
            get_target_property(prop_values ${tgt} INTERFACE_${prop})
            # skip if not found
            if(NOT prop_values)
                continue()
            endif()
            # otherwise, append values to the SWIG_${suffix} property
            set_property(
                TARGET ${target} APPEND
                PROPERTY SWIG_${suffix} ${prop_values}
            )
        endforeach()
    endforeach()
endfunction()
