cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# npygl_test: C++ Google Test unit test runner
if(GTest_FOUND)
    add_executable(npygl_test range_views_test.cc)
    target_link_libraries(npygl_test PRIVATE GTest::gtest GTest::gtest_main)
endif()
