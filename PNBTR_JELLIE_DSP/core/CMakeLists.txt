# Core DSP Engine Library
# Provides the foundation for PNBTR and JELLIE integration

# For now, create a header-only library until we implement the full engine
add_library(pnbtr_jellie_core INTERFACE)

target_include_directories(pnbtr_jellie_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# Link system libraries
if(APPLE)
    target_link_libraries(pnbtr_jellie_core INTERFACE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
elseif(UNIX)
    target_link_libraries(pnbtr_jellie_core INTERFACE
        pthread
    )
endif()

message(STATUS "Core DSP Engine configured:")
message(STATUS "  Library: pnbtr_jellie_core")
message(STATUS "  Features: PNBTR integration, Audio processing, Network handling") 