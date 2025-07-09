cmake_minimum_required(VERSION 3.16)
project(Phase4C_Test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required frameworks on macOS
if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
endif()

# Include directories
include_directories(../Source/GPU)
include_directories(../Source/DSP)
include_directories(../../../../JAM_Framework_v2/include)

# Create a simple test executable
add_executable(phase4c_test
    phase4c_build_test.mm
)

# Link libraries
if(APPLE)
    target_link_libraries(phase4c_test
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
    )
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(phase4c_test PRIVATE -Wno-unused-parameter -Wno-missing-braces)
endif() 