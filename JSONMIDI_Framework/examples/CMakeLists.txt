# Simple example placeholder
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example jsonmidi_framework nlohmann_json::nlohmann_json)

# Phase 1.2 Demo
add_executable(phase12_demo 
    phase12_demo.cpp
)

target_link_libraries(phase12_demo
    PRIVATE 
    jsonmidi_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Live Demo - Real-world testing application  
add_executable(live_demo live_demo.cpp)
target_link_libraries(live_demo
    PRIVATE 
    jsonmidi_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Simple Demo - Working demonstration
add_executable(simple_demo simple_demo.cpp)
target_link_libraries(simple_demo
    PRIVATE 
    jsonmidi_framework
    nlohmann_json::nlohmann_json
    Threads::Threads
)
