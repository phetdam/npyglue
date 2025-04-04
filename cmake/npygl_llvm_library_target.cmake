cmake_minimum_required(VERSION 3.15)

##
# npygl_llvm_library_target.cmake
#
# Provides a function to create an LLVM target from the given library name.
#

##
# Create a target for the specified LLVM component library.
#
# The created INTERFACE imported target will be npyglue::llvm_<component>. It
# is assumed that LLVM was found with find_package.
#
# Arguments:
#   component       LLVM component to create target from
#
function(npygl_llvm_library_target component)
    # resolve LLVM component
    llvm_map_components_to_libnames(libname ${component})
    # add target + LLVM include directories
    add_library(npyglue::llvm_${component} INTERFACE IMPORTED)
    target_include_directories(
        npyglue::llvm_${component}
        SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS}
    )
    target_link_libraries(npyglue::llvm_${component} INTERFACE ${libname})
endfunction()
