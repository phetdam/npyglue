cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# npygl_test: C++ Google Test unit test runner
if(GTest_FOUND)
    # list of sources
    set(NPYGL_TEST_SOURCES range_views_test.cc)
    # C++17
    add_executable(npygl_test ${NPYGL_TEST_SOURCES})
    target_link_libraries(npygl_test PRIVATE GTest::gtest GTest::gtest_main)
    # add as one test instead of having each test as an individual CTest test
    add_test(NAME npygl_test COMMAND npygl_test)
    # C++20
    if(NPYGL_HAS_CC_20)
        add_executable(npygl_test_cc20 ${NPYGL_TEST_SOURCES})
        set_target_properties(
            npygl_test_cc20 PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
        target_link_libraries(
            npygl_test_cc20 PRIVATE
            GTest::gtest GTest::gtest_main
        )
        # for now, better to just run the tests in a single-threaded batch
        add_test(NAME npygl_test_cc20 COMMAND npygl_test_cc20)
    endif()
endif()
