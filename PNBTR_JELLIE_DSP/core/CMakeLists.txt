# Core DSP Engine Library
# Provides the foundation for PNBTR and JELLIE integration with roadmap implementations

# Create the core library with roadmap components
add_library(pnbtr_jellie_core
    network_simulator.cpp
    signal_transmission.cpp
    comprehensive_logging.cpp
    training_preparation.cpp
    MetalBridge.mm
    CoreAudioBridge.mm
)

target_include_directories(pnbtr_jellie_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# Link system libraries
if(APPLE)
    target_link_libraries(pnbtr_jellie_core PUBLIC
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
elseif(UNIX)
    target_link_libraries(pnbtr_jellie_core PUBLIC
        pthread
    )
endif()

message(STATUS "Core DSP Engine configured:")
message(STATUS "  Library: pnbtr_jellie_core")
message(STATUS "  Components: Network Simulator, Signal Transmission, Logging, Training Prep")
message(STATUS "  Features: PNBTR integration, GPU Native processing, ML Training pipeline") 