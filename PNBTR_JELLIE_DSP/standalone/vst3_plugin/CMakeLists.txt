# PNBTR+JELLIE VST3 Plugin - Real GUI Application
cmake_minimum_required(VERSION 3.15)
project(PnbtrJellieGUI)

# Include directories
include_directories(include)

# Core engine library
add_library(pnbtr_jellie_core INTERFACE)
target_include_directories(pnbtr_jellie_core INTERFACE include)

# Real GUI Application with native macOS interface
add_executable(pnbtr_jellie_gui_app
    src/PnbtrJellieGUIApp_Fixed.mm
    src/PnbtrJellieEngine.cpp
)

# Set C++17 standard
set_property(TARGET pnbtr_jellie_gui_app PROPERTY CXX_STANDARD 17)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pnbtr_jellie_gui_app Threads::Threads)

# macOS specific frameworks for GUI
if(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_library(APPKIT_FRAMEWORK AppKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREAUDIO_FRAMEWORK CoreAudio)
    find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox)
    find_library(AVFOUNDATION_FRAMEWORK AVFoundation)
    
    target_link_libraries(pnbtr_jellie_gui_app 
        ${COCOA_FRAMEWORK}
        ${APPKIT_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREAUDIO_FRAMEWORK}
        ${AUDIOTOOLBOX_FRAMEWORK}
        ${AVFOUNDATION_FRAMEWORK}
    )
    
    # Set up as macOS app bundle with microphone permission
    set_target_properties(pnbtr_jellie_gui_app PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "PNBTR+JELLIE Real-time Audio Processing"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.jamnet.pnbtr-jellie-audio"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
        XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# Compiler flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(pnbtr_jellie_gui_app PRIVATE -O2 -march=native -Wno-deprecated-declarations)
endif() 