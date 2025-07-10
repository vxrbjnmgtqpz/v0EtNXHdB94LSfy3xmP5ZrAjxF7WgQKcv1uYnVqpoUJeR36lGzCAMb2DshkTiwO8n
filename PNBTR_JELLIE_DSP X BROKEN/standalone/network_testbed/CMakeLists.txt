# Network Testbed - Test JDAT UDP streaming and packet loss handling
# Simulates JELLIE 8-channel network transmission

add_executable(jdat_network_test
    main.cpp
)

target_include_directories(jdat_network_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../JDAT_Framework/include
)

target_link_libraries(jdat_network_test PRIVATE
    pnbtr_jellie_core
)

# Set C++ standard
set_target_properties(jdat_network_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

message(STATUS "Network Testbed configured:")
message(STATUS "  Executable: jdat_network_test")
message(STATUS "  Features: UDP streaming, packet loss simulation, ADAT redundancy")
message(STATUS "  Purpose: Test JDAT network reliability and recovery") 