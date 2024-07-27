cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# py_helpers_test: Python helpers tests
if(Python3_Development_FOUND)
    add_executable(py_helpers_test py_helpers_test.cc)
    target_link_libraries(py_helpers_test PRIVATE Python3::Python)
endif()
