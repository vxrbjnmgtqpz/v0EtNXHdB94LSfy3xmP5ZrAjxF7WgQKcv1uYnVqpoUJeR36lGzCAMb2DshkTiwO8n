cmake_minimum_required(VERSION 3.15)

project(MIDILink VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

# Add JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

# Create the main application target
juce_add_gui_app(MIDILink
    PRODUCT_NAME "MIDILink"
    COMPANY_NAME "MIDIp2p"
    BUNDLE_ID "com.midip2p.midilink"
    VERSION "1.0.0"
    MICROPHONE_PERMISSION_ENABLED TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
)

# Add source files
target_sources(MIDILink PRIVATE
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/TransportController.cpp
    Source/NetworkConnectionPanel.cpp
    Source/MIDITestingPanel.cpp
    Source/PerformanceMonitorPanel.cpp
    Source/ClockSyncPanel.cpp
)

# Add header files
target_include_directories(MIDILink PRIVATE Source)

# Link JUCE modules
target_link_libraries(MIDILink PRIVATE
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
)

# Compiler definitions
target_compile_definitions(MIDILink PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:MIDILink,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:MIDILink,JUCE_VERSION>"
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

# Create ProjectInfo.h for project metadata
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/ProjectInfo.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Source/ProjectInfo.h"
)

target_include_directories(MIDILink PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/Source")
