cmake_minimum_required(VERSION 3.20)
project(PNBTRFramework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)

# Platform-specific GPU backends
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    set(GPU_LIBS ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${METALPERFORMANCESHADERS_FRAMEWORK})
    add_definitions(-DPNBTR_METAL_BACKEND)
else()
    set(GPU_LIBS Vulkan::Vulkan)
    add_definitions(-DPNBTR_VULKAN_BACKEND)
endif()

# PNBTR Framework core library
add_library(pnbtr_framework STATIC
    src/pnbtr_engine.cpp
    src/pnbtr_learning.cpp
    src/pnbtr_interface.cpp
    src/pnbtr_gpu.cpp
    src/pnbtr_prediction.cpp
    src/pnbtr_reconstruction.cpp
    src/pnbtr_statistics.cpp
)

target_include_directories(pnbtr_framework
    PUBLIC include
    PRIVATE src
)

target_link_libraries(pnbtr_framework
    ${GPU_LIBS}
)

# GPU shader compilation
if(APPLE)
    # Metal shader compilation
    file(GLOB METAL_SHADERS "shaders/*.metal")
    foreach(SHADER ${METAL_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.metallib
            COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.air
            COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.air -o ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.metallib
            DEPENDS ${SHADER}
        )
        list(APPEND COMPILED_SHADERS ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.metallib)
    endforeach()
else()
    # Vulkan GLSL compilation
    file(GLOB GLSL_SHADERS "shaders/*.glsl")
    foreach(SHADER ${GLSL_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} -o ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv
            DEPENDS ${SHADER}
        )
        list(APPEND COMPILED_SHADERS ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
    endforeach()
endif()

# Shader compilation target
add_custom_target(pnbtr_shaders ALL DEPENDS ${COMPILED_SHADERS})
add_dependencies(pnbtr_framework pnbtr_shaders)

# Performance optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(pnbtr_framework PRIVATE
        -O3
        -march=native
        -flto
        -ffast-math
    )
endif()

# Examples
add_subdirectory(examples)

# Tests
add_subdirectory(tests)

# Install
install(TARGETS pnbtr_framework
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/shaders/
    DESTINATION share/pnbtr/shaders
)
