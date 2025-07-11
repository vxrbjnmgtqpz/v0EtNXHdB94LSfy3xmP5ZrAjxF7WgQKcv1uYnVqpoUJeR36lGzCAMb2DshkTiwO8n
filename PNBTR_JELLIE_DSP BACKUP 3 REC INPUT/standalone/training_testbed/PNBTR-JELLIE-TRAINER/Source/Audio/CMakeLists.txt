# ==============================================================================
# AudioEngine Standalone Test Build (Without JUCE)
# 
# This builds the AudioEngine test program to demonstrate that our
# game engine architecture is ready for real audio I/O once JUCE
# build issues are resolved.
# ==============================================================================

cmake_minimum_required(VERSION 3.16)
project(AudioEngineTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create standalone test executable
add_executable(AudioEngineTest
    AudioEngineTest.cpp
    AudioEngine.cpp
    # Add only non-JUCE dependent sources
    ../DSP/PNBTRTrainer.cpp
    ../Metrics/TrainingMetrics.cpp
    ../Network/PacketLossSimulator.cpp
)

# Include directories
target_include_directories(AudioEngineTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../DSP
    ${CMAKE_CURRENT_SOURCE_DIR}/../Metrics
    ${CMAKE_CURRENT_SOURCE_DIR}/../Network
)

# Compiler flags for optimization and warnings
target_compile_options(AudioEngineTest PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(AudioEngineTest PRIVATE Threads::Threads)

# Platform-specific libraries
if(APPLE)
    target_link_libraries(AudioEngineTest PRIVATE
        "-framework CoreFoundation"
        "-framework CoreAudio"
        "-framework AudioUnit"
    )
elseif(WIN32)
    target_link_libraries(AudioEngineTest PRIVATE
        winmm
        dsound
    )
elseif(UNIX)
    target_link_libraries(AudioEngineTest PRIVATE
        asound
        pthread
    )
endif()

# Define macro to bypass JUCE dependencies
target_compile_definitions(AudioEngineTest PRIVATE
    STANDALONE_TEST=1
    NO_JUCE_DEPENDENCIES=1
)

# Build target for testing
add_custom_target(test_audio_engine
    COMMAND AudioEngineTest
    DEPENDS AudioEngineTest
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running AudioEngine standalone test"
)

message(STATUS "AudioEngine test configured - run 'make AudioEngineTest && ./AudioEngineTest'")
message(STATUS "This tests the complete audio pipeline without JUCE dependencies") 