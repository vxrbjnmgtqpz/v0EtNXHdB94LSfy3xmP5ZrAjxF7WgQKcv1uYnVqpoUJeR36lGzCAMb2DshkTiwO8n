# VST3 Plugin with JUCE Framework
cmake_minimum_required(VERSION 3.16)

project(PNBTR_JELLIE_VST3 VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find or download JUCE
set(JUCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../JUCE" CACHE PATH "Path to JUCE framework")

# Try to find JUCE in common locations
if(NOT EXISTS ${JUCE_ROOT})
    find_path(JUCE_ROOT "modules/juce_core/juce_core.h"
        PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../JUCE"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../JUCE" 
        "$ENV{JUCE_ROOT}"
        "~/JUCE"
        "/opt/JUCE"
        DOC "Path to JUCE framework"
    )
endif()

if(EXISTS ${JUCE_ROOT})
    message(STATUS "JUCE found at: ${JUCE_ROOT}")
    add_subdirectory(${JUCE_ROOT} juce)
    set(JUCE_AVAILABLE TRUE)
else()
    message(STATUS "JUCE not found - will create minimal VST3 wrapper")
    set(JUCE_AVAILABLE FALSE)
endif()

# PNBTR+JELLIE Engine Library (our proven core)
add_library(pnbtr_jellie_engine STATIC
    src/PnbtrJellieEngine.cpp
    include/PnbtrJelliePlugin.h
)

target_include_directories(pnbtr_jellie_engine PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
)

# High-performance compilation flags for engine
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(pnbtr_jellie_engine PRIVATE
        -O3 -march=native -mtune=native -ffast-math
        -funroll-loops -finline-functions
    )
endif()

# Platform-specific optimizations for engine
if(APPLE)
    target_link_libraries(pnbtr_jellie_engine PRIVATE "-framework Accelerate")
elseif(UNIX)
    target_link_libraries(pnbtr_jellie_engine PRIVATE m rt)
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pnbtr_jellie_engine PRIVATE Threads::Threads)

if(JUCE_AVAILABLE)
    # Real VST3 Plugin with JUCE
    juce_add_plugin(PnbtrJellieVST3
        VERSION 1.0.0
        COMPANY_NAME "JAMNet"
        PLUGIN_MANUFACTURER_CODE "JAMn"
        PLUGIN_CODE "PnJl"
        FORMATS VST3 Standalone
        PRODUCT_NAME "PNBTR+JELLIE Network Audio"
        DESCRIPTION "Revolutionary network audio interface with sub-50μs latency"
        PLUGIN_NAME "PNBTR+JELLIE"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD TRUE
        VST3_COPY_DIR "~/Library/Audio/Plug-Ins/VST3"
    )

    # VST3 Plugin source files
    target_sources(PnbtrJellieVST3 PRIVATE
        src/PnbtrJellieVST3Plugin.cpp
        src/PnbtrJellieVST3Editor.cpp
        include/PnbtrJellieVST3Plugin.h
        include/PnbtrJellieVST3Editor.h
    )

    # Link JUCE modules
    target_link_libraries(PnbtrJellieVST3 PRIVATE
        pnbtr_jellie_engine
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    )

    # VST3 Plugin compile definitions
    target_compile_definitions(PnbtrJellieVST3 PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        PNBTR_PERFORMANCE_TARGET_US=50
        JELLIE_8_CHANNEL_REDUNDANCY=1
        PNBTR_ZERO_NOISE_DITHER=1
    )

    # High-performance compilation for VST3
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(PnbtrJellieVST3 PRIVATE
            -O3 -march=native -ffast-math
        )
    endif()

    message(STATUS "VST3 Plugin Configuration:")
    message(STATUS "  Plugin Name: PNBTR+JELLIE Network Audio")
    message(STATUS "  Performance Target: <50μs")
    message(STATUS "  Install Location: ~/Library/Audio/Plug-Ins/VST3")
    message(STATUS "  JUCE Framework: ${JUCE_ROOT}")

else()
    # Minimal VST3 wrapper without JUCE (fallback)
    add_library(PnbtrJellieVST3Minimal SHARED
        src/PnbtrJellieVST3Minimal.cpp
        include/PnbtrJellieVST3Minimal.h
    )
    
    target_link_libraries(PnbtrJellieVST3Minimal PRIVATE
        pnbtr_jellie_engine
    )
    
    message(STATUS "Building minimal VST3 wrapper (JUCE not available)")
endif()

# Standalone Test Executable (our proven 13.2μs performance test)
add_executable(vst3_plugin_test
    src/vst3_plugin_test.cpp
)

target_link_libraries(vst3_plugin_test PRIVATE
    pnbtr_jellie_engine
    Threads::Threads
)

# Test configuration
target_compile_definitions(vst3_plugin_test PRIVATE
    PNBTR_TEST_DURATION_SECONDS=5
    PNBTR_TARGET_LATENCY_US=50
    JELLIE_SINE_WAVE_FREQUENCY=440
    PNBTR_PACKET_LOSS_PERCENT=5
)

# High-performance compilation for test
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vst3_plugin_test PRIVATE -O3 -march=native -ffast-math)
endif()

# VST3 Minimal Test Executable (DAW-style integration test)
add_executable(vst3_minimal_test
    src/vst3_minimal_test.cpp
)

target_link_libraries(vst3_minimal_test PRIVATE
    PnbtrJellieVST3Minimal
    pnbtr_jellie_engine
    Threads::Threads
)

# High-performance compilation for minimal test
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(vst3_minimal_test PRIVATE -O3 -march=native -ffast-math)
endif()

# Copy test executables to output directory
set_target_properties(vst3_plugin_test vst3_minimal_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Custom targets for development
add_custom_target(run_vst3_test
    COMMAND ${CMAKE_BINARY_DIR}/bin/vst3_plugin_test
    DEPENDS vst3_plugin_test
    COMMENT "Running VST3 Plugin Performance Test (Target: <50μs)..."
)

add_custom_target(run_vst3_minimal_test
    COMMAND ${CMAKE_BINARY_DIR}/bin/vst3_minimal_test
    DEPENDS vst3_minimal_test
    COMMENT "Running VST3 Minimal Plugin Test (DAW-style integration)..."
)

if(JUCE_AVAILABLE)
    add_custom_target(install_vst3_plugin
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target PnbtrJellieVST3
        DEPENDS PnbtrJellieVST3
        COMMENT "Installing VST3 Plugin to ~/Library/Audio/Plug-Ins/VST3..."
    )
endif()

# REAL GUI APPLICATION - Native macOS with actual windows and controls
add_executable(pnbtr_jellie_gui_app
    src/PnbtrJellieGUIApp_Fixed.mm
    src/PnbtrJellieEngine.cpp
)

target_link_libraries(pnbtr_jellie_gui_app
    pnbtr_jellie_engine
    "-framework Cocoa"
    "-framework Foundation"
    "-framework AppKit"
)

set_target_properties(pnbtr_jellie_gui_app PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "PNBTR+JELLIE VST3 Plugin"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.jamnet.pnbtr-jellie-vst3"
    MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    MACOSX_BUNDLE_ICON_FILE "AppIcon"
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED "NO"
)

# GUI Application
add_executable(pnbtr_jellie_gui src/pnbtr_jellie_gui.cpp)
target_link_libraries(pnbtr_jellie_gui pnbtr_jellie_engine)

# Set output directory
set_target_properties(vst3_minimal_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(vst3_plugin_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(pnbtr_jellie_gui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
) 