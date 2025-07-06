cmake_minimum_required(VERSION 3.15)

project(TOASTer VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

# Add JUCE
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
)
FetchContent_MakeAvailable(JUCE)

# Add nlohmann JSON library (for basic JSON handling)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Create the main application target
juce_add_gui_app(TOASTer
    PRODUCT_NAME "TOASTer"
    COMPANY_NAME "JAMNet"
    BUNDLE_ID "com.jamnet.toaster"
    VERSION "1.0.0"
)

# Disable code signing for development
set_target_properties(TOASTer PROPERTIES
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED NO
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED NO
)

# Add simplified source files (without heavy framework dependencies)
target_sources(TOASTer PRIVATE
    Source/Main.cpp
    Source/MainComponent.cpp
    Source/BasicTransportPanel.cpp
)

# Add header files  
target_include_directories(TOASTer PRIVATE 
    Source
)

# Link only essential JUCE modules
target_link_libraries(TOASTer PRIVATE
    juce::juce_gui_basics
    juce::juce_audio_basics
    juce::juce_events
    juce::juce_core
)

# Add macOS-specific frameworks for basic functionality
if(APPLE)
    target_link_libraries(TOASTer PRIVATE
        "-framework Foundation"
        "-framework CoreAudio"
        "-framework CoreMIDI"
    )
endif()

# Compiler definitions
target_compile_definitions(TOASTer PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:TOASTer,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:TOASTer,JUCE_VERSION>"
    JUCE_DISPLAY_SPLASH_SCREEN=0
)
