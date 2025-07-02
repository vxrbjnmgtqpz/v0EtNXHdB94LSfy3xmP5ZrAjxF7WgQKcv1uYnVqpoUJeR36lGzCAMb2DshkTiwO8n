cmake_minimum_required(VERSION 3.15)

project(TOASTer VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

# Add our JSONMIDI Framework
add_subdirectory(../JSONMIDI_Framework jsonmidi_framework)

# Add JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

# Create the main application target
juce_add_gui_app(TOASTer
    PRODUCT_NAME "TOASTer"
    COMPANY_NAME "JAMNet"
    BUNDLE_ID "com.jamnet.toaster"
    VERSION "1.0.0"
    MICROPHONE_PERMISSION_ENABLED TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
)

# Add source files
target_sources(TOASTer PRIVATE
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/TransportController.cpp
    Source/NetworkConnectionPanel.cpp
    Source/MIDITestingPanel.cpp
    Source/MIDIManager.cpp
    Source/PerformanceMonitorPanel.cpp
    Source/ClockSyncPanel.cpp
    Source/JSONMIDIIntegrationPanel.cpp
)

# Add header files
target_include_directories(TOASTer PRIVATE 
    Source
    ../JSONMIDI_Framework/include
)

# Link JUCE modules and our framework
target_link_libraries(TOASTer PRIVATE
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
    jsonmidi_framework
    nlohmann_json::nlohmann_json
)

# Compiler definitions
target_compile_definitions(TOASTer PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:TOASTer,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:TOASTer,JUCE_VERSION>"
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

# Create ProjectInfo.h for project metadata
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/ProjectInfo.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Source/ProjectInfo.h"
)

target_include_directories(TOASTer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/Source")
