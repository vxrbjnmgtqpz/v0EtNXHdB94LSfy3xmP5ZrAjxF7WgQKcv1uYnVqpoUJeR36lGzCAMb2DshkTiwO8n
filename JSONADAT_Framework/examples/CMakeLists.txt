# Examples for JSONADAT Framework

# Basic JELLIE demo
add_executable(basic_jellie_demo basic_jellie_demo.cpp)
target_link_libraries(basic_jellie_demo 
    PRIVATE 
        jsonadat_framework
        nlohmann_json::nlohmann_json
        ${PLATFORM_LIBS}
)

# 192k ADAT simulation demo
add_executable(adat_192k_demo adat_192k_demo.cpp)
target_link_libraries(adat_192k_demo 
    PRIVATE 
        jsonadat_framework
        nlohmann_json::nlohmann_json
        ${PLATFORM_LIBS}
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