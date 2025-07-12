# ==============================================================================
# PNBTR+JELLIE Training Testbed
# GPU-native training harness with Metal compute shaders
# ==============================================================================

cmake_minimum_required(VERSION 3.15)
project(PnbtrJellieTrainer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "" FORCE)

# ==============================================================================
# JUCE Configuration (simplified)
# ==============================================================================

# Add JUCE using FetchContent (matching TOASTer approach)
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG origin/master
)
FetchContent_MakeAvailable(JUCE)

# ==============================================================================
# JAM Framework v2 Integration (for TOAST protocol)
# ==============================================================================

# Add JAM Framework v2 as subdirectory (ENABLED FOR REAL TOAST INTEGRATION)
set(JAM_FRAMEWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../JAM_Framework_v2")
if(EXISTS "${JAM_FRAMEWORK_DIR}/CMakeLists.txt")
    add_subdirectory("${JAM_FRAMEWORK_DIR}" jam_framework_v2 EXCLUDE_FROM_ALL)
    message(STATUS "JAM Framework v2 found and added - REAL TOAST PROTOCOL ACTIVE")
    set(JAM_FRAMEWORK_AVAILABLE TRUE)
else()
    message(STATUS "JAM Framework v2 NOT FOUND - using TOAST stubs")
    set(JAM_FRAMEWORK_AVAILABLE FALSE)
endif()

# ==============================================================================
# Project Structure
# ==============================================================================

# Create the main application target
juce_add_gui_app(PnbtrJellieTrainer
    PRODUCT_NAME "PNBTR+JELLIE Training Testbed"
    VERSION "1.0.0"
    COMPANY_NAME "JAMNet"
    BUNDLE_ID "com.jamnet.pnbtrjellietrainer"
)

# Source files (only .cpp files, no headers)
target_sources(PnbtrJellieTrainer PRIVATE
    # Main application
    Source/Main.cpp

    # GUI components (existing - DO NOT MODIFY)
    Source/GUI/MainComponent.cpp

    # NEW schematic GUI components (added per roadmap)
    Source/GUI/ProfessionalTransportController.cpp
    Source/GUI/TitleComponent.cpp
    Source/GUI/OscilloscopeRow.cpp
    Source/GUI/TOASTNetworkOscilloscope.cpp
    Source/GUI/WaveformAnalysisRow.cpp
    Source/GUI/SpectralAudioTrack.cpp
    Source/GUI/AudioThumbnailComponent.cpp
    Source/GUI/LogStatusComponent.cpp
    Source/GUI/ControlsRow.cpp
    Source/GUI/OscilloscopeComponent.cpp
    Source/GUI/MetricsDashboard.cpp
    Source/GUI/SchematicMainWindow.cpp
    Source/GUI/SchematicLauncher.cpp

    # Core systems
    Source/Core/SessionManager.cpp


    # DSP (add PNBTRTrainer.cpp for linker)
    Source/DSP/PNBTRTrainer.cpp
    Source/Metrics/TrainingMetrics.cpp
    Source/Network/PacketLossSimulator.cpp

    # GPU integration
    Source/GPU/MetalBridge.mm
    Source/GPU/CoreAudioBridge.mm
    Source/GPU/GPUCommandFencePool.mm
    Source/GPU/BiquadCoefficients.cpp
    
    # Core synchronization
    Source/Core/FrameSyncCoordinator.cpp
    
    # Debug and diagnostic systems
    Source/Core/FrameStateTracker.cpp
    Source/GPU/GPUDebugHelper.mm
    Source/GPU/MetalBridgeDebugIntegration.mm
)


# Set include directory (prioritize our Source directory and JUCE modules)
target_include_directories(PnbtrJellieTrainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JUCE_SOURCE_DIR}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/juce-src/modules
)

# Add JAM Framework v2 includes if available
# 🔧 TEMPORARILY DISABLED: JAM Framework includes for GPU pipeline testing
# if(JAM_FRAMEWORK_AVAILABLE)
#     target_include_directories(PnbtrJellieTrainer PRIVATE
#         "${JAM_FRAMEWORK_DIR}/include"
#         "${JAM_FRAMEWORK_DIR}/src"
#     )
# endif()

# Link required JUCE modules
target_link_libraries(PnbtrJellieTrainer PRIVATE
    juce::juce_gui_basics
    juce::juce_audio_devices
    juce::juce_audio_basics
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# Link JAM Framework v2 if available
# 🔧 TEMPORARILY DISABLED: JAM Framework integration for GPU pipeline testing
# if(JAM_FRAMEWORK_AVAILABLE)
#     target_link_libraries(PnbtrJellieTrainer PRIVATE jam_framework_v2)
# endif()

# ==============================================================================
# Metal Integration
# ==============================================================================

# Link macOS frameworks
if(APPLE)
    target_link_libraries(PnbtrJellieTrainer PRIVATE
        "-framework CoreAudio"
        "-framework AudioUnit"
        "-framework AVFoundation"
        "-framework Cocoa"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()

# Enable Objective-C++ for Metal integration
set_source_files_properties(
    Source/GPU/MetalAudioTrack.mm
    Source/GUI/MetalDrawableTrack.mm
    Source/GPU/MetalBridge.mm
    Source/GPU/CoreAudioBridge.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# Ensure CoreAudioBridge.mm is compiled as Objective-C++
set_source_files_properties(Source/GPU/CoreAudioBridge.mm PROPERTIES LANGUAGE OBJCXX)

# ==============================================================================
# Metal Shader Compilation (simplified)
# ==============================================================================

# Custom command to compile Metal shaders from shaders/ directory
if(APPLE)
    # Define all 8 GPU pipeline shaders
    set(SHADER_SOURCES
        shaders/AudioInputCaptureShader.metal
        shaders/AudioInputGateShader.metal
        shaders/DJSpectralAnalysisShader.metal
        shaders/RecordArmVisualShader.metal
        shaders/JELLIEPreprocessShader.metal
        shaders/NetworkSimulationShader.metal
        shaders/PNBTRReconstructionShader.metal
        shaders/MetricsComputeShader.metal
    )

    set(AIR_FILES)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air)
    
    add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${AIR_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
            COMMENT "Compiling Metal shader: ${SHADER_NAME}"
        )
        
        list(APPEND AIR_FILES ${AIR_FILE})
    endforeach()
    
    # Create combined Metal library from all 8 shaders
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
        COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
        DEPENDS ${AIR_FILES}
        COMMENT "Linking complete Metal library with 8 GPU pipeline shaders"
    )
    
    # Add Metal shader compilation as dependency
    add_custom_target(CompileMetalShaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
    add_dependencies(PnbtrJellieTrainer CompileMetalShaders)
endif()