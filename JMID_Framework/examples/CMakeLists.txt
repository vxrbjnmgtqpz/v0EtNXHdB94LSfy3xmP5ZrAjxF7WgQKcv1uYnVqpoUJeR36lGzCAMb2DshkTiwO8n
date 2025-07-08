# Simple example placeholder
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example jmid_framework nlohmann_json::nlohmann_json)

# Phase 1.2 Demo
add_executable(phase12_demo 
    phase12_demo.cpp
)

target_link_libraries(phase12_demo
    PRIVATE 
    jmid_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Live Demo - Real-world testing application  
add_executable(live_demo live_demo.cpp)
target_link_libraries(live_demo
    PRIVATE 
    jmid_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Simple Demo - Working demonstration
add_executable(simple_demo simple_demo.cpp)
target_link_libraries(simple_demo
    PRIVATE 
    jmid_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# SIMD Parser Test - Phase 4 performance validation
add_executable(simd_parser_test simd_parser_test.cpp)
target_link_libraries(simd_parser_test
    PRIVATE 
    jmid_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Performance Validation Test - Phase 5 end-to-end validation
add_executable(performance_validation_test performance_validation_test.cpp)
target_link_libraries(performance_validation_test
    PRIVATE 
    jmid_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)
