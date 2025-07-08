cmake_minimum_required(VERSION 3.20)

project(PNBTR_JELLIE_DSP VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(APPLE)
    set(PLATFORM_MACOS TRUE)
    enable_language(OBJCXX)
elseif(WIN32)
    set(PLATFORM_WINDOWS TRUE)
elseif(UNIX)
    set(PLATFORM_LINUX TRUE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add common warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../JDAT_Framework/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../PNBTR_Framework/include
)

# Link directories
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../JDAT_Framework/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../PNBTR_Framework/lib
)

# Find required packages
find_package(Threads REQUIRED)

# Audio libraries
if(PLATFORM_MACOS)
    find_library(COREAUDIO_LIBRARY CoreAudio)
    find_library(AUDIOUNIT_LIBRARY AudioUnit)
    find_library(AUDIOTOOLBOX_LIBRARY AudioToolbox)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    set(PLATFORM_LIBRARIES 
        ${COREAUDIO_LIBRARY} 
        ${AUDIOUNIT_LIBRARY} 
        ${AUDIOTOOLBOX_LIBRARY}
        ${METAL_LIBRARY}
        ${METALKIT_LIBRARY}
    )
elseif(PLATFORM_WINDOWS)
    set(PLATFORM_LIBRARIES winmm dsound)
elseif(PLATFORM_LINUX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ALSA REQUIRED alsa)
    set(PLATFORM_LIBRARIES ${ALSA_LIBRARIES})
endif()

# Common libraries for all platforms
set(COMMON_LIBRARIES
    Threads::Threads
    ${PLATFORM_LIBRARIES}
)

# JDAT Framework libraries
set(JDAT_LIBRARIES
    jdat_core
    jellie_encoder
    jellie_decoder
    audio_buffer_manager
)

# PNBTR Framework libraries  
set(PNBTR_LIBRARIES
    pnbtr_framework
    pnbtr_engine
    pnbtr_gpu
)

# Core DSP library
add_subdirectory(core)

# Standalone testbeds
add_subdirectory(standalone)

# VST3 plugin (only if VST3 SDK is available)
find_path(VST3_SDK_PATH "pluginterfaces/vst/ivstcomponent.h"
    PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../vst3sdk"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../vst3sdk"
    "$ENV{VST3_SDK_PATH}"
    DOC "Path to VST3 SDK"
)

if(VST3_SDK_PATH)
    message(STATUS "VST3 SDK found at: ${VST3_SDK_PATH}")
    add_subdirectory(vst3_plugin)
else()
    message(STATUS "VST3 SDK not found - skipping plugin build")
    message(STATUS "To build VST3 plugin, set VST3_SDK_PATH or place SDK in ../vst3sdk")
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_subdirectory(docs)
endif()

# Installation
install(TARGETS 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Package configuration
set(CPACK_PACKAGE_NAME "PNBTR_JELLIE_DSP")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Revolutionary Audio Processing with Neural Intelligence")
set(CPACK_PACKAGE_VENDOR "JAMNet")

if(PLATFORM_MACOS)
    set(CPACK_GENERATOR "DragNDrop")
elseif(PLATFORM_WINDOWS)
    set(CPACK_GENERATOR "NSIS")
elseif(PLATFORM_LINUX)
    set(CPACK_GENERATOR "DEB;RPM")
endif()

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "PNBTR+JELLIE DSP Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  VST3 SDK: ${VST3_SDK_PATH}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  Core DSP Engine: YES")
message(STATUS "  Standalone Testbeds: YES")
message(STATUS "  VST3 Plugin: ${VST3_SDK_PATH}")
message(STATUS "  Tests: YES")
message(STATUS "  Documentation: ${DOXYGEN_FOUND}")
message(STATUS "") 