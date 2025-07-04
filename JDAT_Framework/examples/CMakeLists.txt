# Examples for JELLIE Framework

# Basic JELLIE demo
add_executable(basic_jellie_demo basic_jellie_demo.cpp)
target_link_libraries(basic_jellie_demo 
    PRIVATE 
        JELLIE
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# Studio monitoring demo
add_executable(studio_monitoring studio_monitoring.cpp)
target_link_libraries(studio_monitoring 
    PRIVATE 
        JELLIE
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# Multicast session demo
add_executable(multicast_session multicast_session.cpp)
target_link_libraries(multicast_session 
    PRIVATE 
        JELLIE
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# 192k ADAT simulation demo
add_executable(adat_192k_demo adat_192k_demo.cpp)
target_link_libraries(adat_192k_demo 
    PRIVATE 
        JELLIE
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# Set target properties
set_target_properties(basic_jellie_demo adat_192k_demo 
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

# Install examples (optional)
if(INSTALL_EXAMPLES)
    install(TARGETS basic_jellie_demo adat_192k_demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    )
endif() 