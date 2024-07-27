cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# py_helpers_test: Python helpers tests
if(BUILD_TESTS AND Python3_Development_FOUND)
    add_executable(py_helpers_test py_helpers_test.cc)
    target_link_libraries(py_helpers_test PRIVATE Python3::Python)
    # usually no Python debug runtime library
    if(MSVC)
        set_target_properties(
            py_helpers_test PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
    endif()
    add_test(NAME py_helpers_test COMMAND py_helpers_test)
endif()
