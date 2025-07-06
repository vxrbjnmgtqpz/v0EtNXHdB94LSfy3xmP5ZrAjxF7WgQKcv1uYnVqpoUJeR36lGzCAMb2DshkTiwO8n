cmake_minimum_required(VERSION 3.20)
project(JAMFramework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(UV REQUIRED libuv)

# Platform-specific GPU backends
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    set(GPU_LIBS ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
    add_definitions(-DJAM_METAL_BACKEND)
else()
    set(GPU_LIBS Vulkan::Vulkan)
    add_definitions(-DJAM_VULKAN_BACKEND)
endif()

# JAM Framework core library
add_library(jam_framework STATIC
    src/jam_framework.cpp
    src/jam_parser.cpp
    src/jam_multicast.cpp
    src/jam_gpu.cpp
    src/jam_toast.cpp
    src/jam_session.cpp
    src/jam_compression.cpp
    src/jam_network.cpp
)

target_include_directories(jam_framework
    PUBLIC include
    PRIVATE src
)

target_link_libraries(jam_framework
    ${GPU_LIBS}
    ${UV_LIBRARIES}
)

target_compile_definitions(jam_framework PRIVATE
    ${UV_CFLAGS_OTHER}
)

# Performance optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(jam_framework PRIVATE
        -O3
        -march=native
        -flto
    )
endif()

# Examples
add_subdirectory(examples)

# Tests
add_subdirectory(tests)

# Install
install(TARGETS jam_framework
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
