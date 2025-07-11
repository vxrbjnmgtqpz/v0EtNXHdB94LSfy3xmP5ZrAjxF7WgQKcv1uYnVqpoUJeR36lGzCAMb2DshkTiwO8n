# Tests for PNBTR+JELLIE DSP

project(PNBTR_JELLIE_Tests)

# Find testing framework (prefer Google Test)
find_package(GTest QUIET)

if(GTest_FOUND)
    # Unit tests
    add_subdirectory(unit_tests)
    
    # Integration tests  
    add_subdirectory(integration_tests)
    
    # Performance tests
    add_subdirectory(performance_tests)
    
    message(STATUS "Tests configured with Google Test")
else()
    # Simple test executables without framework
    add_executable(simple_tests
        simple_tests.cpp
    )
    
    target_link_libraries(simple_tests
        pnbtr_jellie_core
        ${COMMON_LIBRARIES}
    )
    
    message(STATUS "Tests configured with simple test runner")
    message(STATUS "  Install Google Test for full test suite")
endif()

# Test data
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_output)

message(STATUS "Test suite configured:")
message(STATUS "  Framework: ${GTest_FOUND}")
message(STATUS "  Test data: test_data/")
message(STATUS "  Test output: test_output/") 