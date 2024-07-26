cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# npygl_py_helpers_test: Python helpers tests
if(Python3_Development_FOUND)
    add_executable(npygl_py_helpers_test npygl_py_helpers_test.cc)
    target_link_libraries(npygl_py_helpers_test PRIVATE Python3::Python)
endif()
