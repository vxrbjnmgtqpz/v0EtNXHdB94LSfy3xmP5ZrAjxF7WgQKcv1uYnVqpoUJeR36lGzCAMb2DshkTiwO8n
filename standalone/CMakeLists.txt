cmake_minimum_required(VERSION 3.16)

project(PNBTR_JELLIE_DSP_Standalone VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# PNBTR+JELLIE DSP Standalone Components

# Audio Testbed
add_subdirectory(audio_testbed)

# Network Testbed
add_subdirectory(network_testbed)

# VST3 Plugin Standalone Test
add_subdirectory(vst3_plugin) 