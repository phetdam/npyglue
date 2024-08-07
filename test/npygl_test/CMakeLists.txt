cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# npygl_test: C++ Google Test unit test runner
if(GTest_FOUND)
    add_executable(npygl_test range_views_test.cc)
    target_link_libraries(npygl_test PRIVATE GTest::gtest GTest::gtest_main)
    # add as one test instead of having each test as an individual CTest test
    add_test(NAME npygl_test COMMAND npygl_test)
endif()
