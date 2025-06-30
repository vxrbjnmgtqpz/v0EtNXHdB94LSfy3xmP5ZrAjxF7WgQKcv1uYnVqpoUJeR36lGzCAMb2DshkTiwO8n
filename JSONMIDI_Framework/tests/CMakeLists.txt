cmake_minimum_required(VERSION 3.20)

# Individual test executables
add_executable(test_basic_messages test_basic_messages.cpp)
add_executable(test_validation test_validation.cpp)
add_executable(test_performance test_performance.cpp)
add_executable(test_round_trip test_round_trip.cpp)

target_link_libraries(test_basic_messages
    PRIVATE
        jsonmidi_framework
        nlohmann_json::nlohmann_json
)

target_link_libraries(test_validation
    PRIVATE
        jsonmidi_framework
        nlohmann_json::nlohmann_json
)

target_link_libraries(test_performance
    PRIVATE
        jsonmidi_framework
        nlohmann_json::nlohmann_json
)

target_link_libraries(test_round_trip
    PRIVATE
        jsonmidi_framework
        nlohmann_json::nlohmann_json
)

# Phase 1.2 Performance Test
add_executable(test_phase12_performance 
    test_phase12_performance.cpp
)

target_link_libraries(test_phase12_performance
    PRIVATE 
    jsonmidi_framework
    gtest 
    gtest_main
    Threads::Threads
)

# Add individual tests
add_test(NAME basic_messages_test COMMAND test_basic_messages --test-basic)
add_test(NAME validation_test COMMAND test_validation --test-validation)
add_test(NAME performance_test COMMAND test_performance --test-performance)
add_test(NAME round_trip_test COMMAND test_round_trip --test-round-trip)
add_test(NAME Phase12Performance COMMAND test_phase12_performance)

# Set test properties
set_tests_properties(basic_messages_test PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All basic message tests passed"
)

set_tests_properties(validation_test PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "All validation tests passed"
)

set_tests_properties(performance_test PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "Performance targets met"
)

set_tests_properties(round_trip_test PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All round-trip tests passed"
)
