cmake_minimum_required(VERSION 3.20)
project(JAMFrameworkV2 VERSION 2.0.0 LANGUAGES CXX)

# JAM Framework v2: Pure UDP GPU Architecture
# NO TCP/HTTP dependencies - UDP multicast only

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(JAM_BUILD_TESTS "Build JAM Framework tests" OFF)   # Disable tests for now
option(JAM_BUILD_EXAMPLES "Build JAM Framework examples" OFF) # Disable examples for now
option(JAM_GPU_BACKEND "GPU backend (metal, vulkan, opengl)" "auto")

# Platform detection for GPU backend
if(JAM_GPU_BACKEND STREQUAL "auto")
    if(APPLE)
        set(JAM_GPU_BACKEND "metal")
    elseif(UNIX AND NOT APPLE)
        set(JAM_GPU_BACKEND "vulkan")
    else()
        set(JAM_GPU_BACKEND "opengl")
    endif()
endif()

# Force Metal on macOS for GPU-native architecture
if(APPLE AND NOT JAM_GPU_BACKEND)
    set(JAM_GPU_BACKEND "metal")
endif()

message(STATUS "JAM GPU Backend: ${JAM_GPU_BACKEND}")

# Compiler flags for performance
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
endif()

# Find dependencies (NO TCP/HTTP libraries)
find_package(Threads REQUIRED)

# GPU backend dependencies
if(JAM_GPU_BACKEND STREQUAL "metal")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(GPU_LIBRARIES ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    add_compile_definitions(JAM_GPU_METAL=1)
elseif(JAM_GPU_BACKEND STREQUAL "vulkan")
    find_package(Vulkan REQUIRED)
    set(GPU_LIBRARIES Vulkan::Vulkan)
    add_compile_definitions(JAM_GPU_VULKAN=1)
elseif(JAM_GPU_BACKEND STREQUAL "opengl")
    find_package(OpenGL REQUIRED)
    set(GPU_LIBRARIES OpenGL::GL)
    add_compile_definitions(JAM_GPU_OPENGL=1)
endif()

# Core JAM Framework library with GPU-Native infrastructure
set(JAM_SOURCES
    # GPU-Native Core (Phase 2 Implementation)
    src/gpu_native/gpu_shared_timeline.cpp
    
    # GPU-Native JMID Framework (Phase 3 Implementation)
    src/jmid_gpu/gpu_jmid_framework.cpp
    
    # GPU-Native JDAT Framework (Phase 3 Implementation)
    src/jdat_gpu/gpu_jdat_framework.cpp
    
    # GPU-Native JVID Framework (Phase 3 Implementation)
    src/jvid_gpu/gpu_jvid_framework.cpp
    
    # Legacy Core (being transformed)
    src/core/jam_core.cpp
    src/core/message_router.cpp
    src/core/network_state_detector.cpp
    src/core/device_discovery.cpp
    src/toast/toast_v2.cpp
    src/toast/multicast.cpp
    src/toast/discovery.cpp
    src/jsonl/jsonl_parser.cpp
    src/jsonl/jsonl_generator.cpp
    src/jsonl/compact_format.cpp
    src/gpu/compute_pipeline.cpp
    src/gpu/gpu_manager.cpp
    src/gpu/memory_mapper.cpp
)

# Platform-specific GPU timebase implementation
if(JAM_GPU_BACKEND STREQUAL "metal")
    list(APPEND JAM_SOURCES src/gpu_native/gpu_timebase.mm)
elseif(JAM_GPU_BACKEND STREQUAL "vulkan")
    list(APPEND JAM_SOURCES src/gpu_native/gpu_timebase.cpp)
endif()

# GPU and UDP transport disabled for initial build
# TODO: Re-enable when implementations are complete
# src/core/udp_transport.cpp
# src/core/gpu_backend.cpp

# Platform-specific GPU implementation
if(JAM_GPU_BACKEND STREQUAL "metal")
    list(APPEND JAM_SOURCES src/gpu/metal_backend.mm)
elseif(JAM_GPU_BACKEND STREQUAL "vulkan")
    list(APPEND JAM_SOURCES src/gpu/vulkan_backend.cpp)
elseif(JAM_GPU_BACKEND STREQUAL "opengl")
    list(APPEND JAM_SOURCES src/gpu/opengl_backend.cpp)
endif()

# Create JAM Framework library
add_library(jam_framework_v2 ${JAM_SOURCES})

target_include_directories(jam_framework_v2 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        src
)

target_link_libraries(jam_framework_v2 
    PUBLIC 
        Threads::Threads
        ${GPU_LIBRARIES}
)

# Compile definitions
target_compile_definitions(jam_framework_v2 
    PUBLIC
        JAM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        JAM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        JAM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        GPU_NATIVE_ENABLED=1  # Enable GPU-Native architecture
    PRIVATE
        JAM_BUILDING_LIBRARY=1
)

# Shader compilation for GPU-Native timing shaders
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(TIMING_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/timing)
set(COMPILED_SHADER_DIR ${CMAKE_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${COMPILED_SHADER_DIR})

# Find shader compiler based on backend
if(JAM_GPU_BACKEND STREQUAL "metal")
    find_program(METAL_COMPILER xcrun)
    if(METAL_COMPILER)
        set(SHADER_COMPILE_COMMAND ${METAL_COMPILER} -sdk macosx metal)
        set(SHADER_EXTENSION ".metallib")
        
        # Compile GPU-Native timing shaders
        set(TIMING_SHADERS
            ${TIMING_SHADER_DIR}/master_timebase.metal
        )
        
        # Compile GPU-Native audio processing shaders
        set(AUDIO_SHADERS
            ${SHADER_DIR}/audio/gpu_audio_processing.metal
        )
        
        # Compile GPU-Native video processing shaders
        set(VIDEO_SHADERS
            ${SHADER_DIR}/video/gpu_video_processing.metal
        )
        
        set(ALL_METAL_SHADERS ${TIMING_SHADERS} ${AUDIO_SHADERS} ${VIDEO_SHADERS})
        
        foreach(SHADER_FILE ${ALL_METAL_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
            set(AIR_FILE ${COMPILED_SHADER_DIR}/${SHADER_NAME}.air)
            set(COMPILED_SHADER ${COMPILED_SHADER_DIR}/${SHADER_NAME}${SHADER_EXTENSION})
            
            add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND ${METAL_COMPILER} -sdk macosx metal -c ${SHADER_FILE} -o ${AIR_FILE}
                DEPENDS ${SHADER_FILE}
                COMMENT "Compiling Metal shader ${SHADER_NAME} to AIR"
            )
            
            add_custom_command(
                OUTPUT ${COMPILED_SHADER}
                COMMAND ${METAL_COMPILER} -sdk macosx metallib ${AIR_FILE} -o ${COMPILED_SHADER}
                DEPENDS ${AIR_FILE}
                COMMENT "Creating Metal library ${SHADER_NAME}"
            )
            
            list(APPEND COMPILED_TIMING_SHADERS ${COMPILED_SHADER})
        endforeach()
    endif()
elseif(JAM_GPU_BACKEND STREQUAL "vulkan")
    find_program(GLSLC_COMPILER glslc)
    if(GLSLC_COMPILER)
        set(SHADER_COMPILE_COMMAND ${GLSLC_COMPILER})
        set(SHADER_EXTENSION ".spv")
        
        # Compile GPU-Native timing shaders
        set(TIMING_SHADERS
            ${TIMING_SHADER_DIR}/master_timebase.glsl
        )
        
        foreach(SHADER_FILE ${TIMING_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
            set(COMPILED_SHADER ${COMPILED_SHADER_DIR}/${SHADER_NAME}${SHADER_EXTENSION})
            
            add_custom_command(
                OUTPUT ${COMPILED_SHADER}
                COMMAND ${GLSLC_COMPILER} ${SHADER_FILE} -o ${COMPILED_SHADER}
                DEPENDS ${SHADER_FILE}
                COMMENT "Compiling GLSL shader ${SHADER_NAME}"
            )
            
            list(APPEND COMPILED_TIMING_SHADERS ${COMPILED_SHADER})
        endforeach()
    endif()
endif()

# Compile shaders (GPU-native timing shaders only for now)
if(SHADER_COMPILE_COMMAND)
    # Skip legacy shaders (.comp) that are incompatible with Metal
    # file(GLOB SHADER_FILES "${SHADER_DIR}/*.comp")
    
    # for(SHADER_FILE ${SHADER_FILES})
    #     get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    #     set(COMPILED_SHADER ${COMPILED_SHADER_DIR}/${SHADER_NAME}${SHADER_EXTENSION})
    #     
    #     add_custom_command(
    #         OUTPUT ${COMPILED_SHADER}
    #         COMMAND ${SHADER_COMPILE_COMMAND} ${SHADER_FILE} -o ${COMPILED_SHADER}
    #         DEPENDS ${SHADER_FILE}
    #         COMMENT "Compiling shader ${SHADER_NAME}"
    #     )
    #     
    #     list(APPEND COMPILED_SHADERS ${COMPILED_SHADER})
    # endforeach()
    
    # Add timing shaders to compiled shaders list
    if(COMPILED_TIMING_SHADERS)
        list(APPEND COMPILED_SHADERS ${COMPILED_TIMING_SHADERS})
    endif()
    
    if(COMPILED_SHADERS)
        add_custom_target(jam_shaders ALL DEPENDS ${COMPILED_SHADERS})
        add_dependencies(jam_framework_v2 jam_shaders)
    endif()
endif()

# Examples
if(JAM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(JAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS jam_framework_v2
    EXPORT JAMFrameworkTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${COMPILED_SHADER_DIR}/
    DESTINATION share/jamframework/shaders
    FILES_MATCHING PATTERN "*${SHADER_EXTENSION}"
)

# Export configuration
install(EXPORT JAMFrameworkTargets
    FILE JAMFrameworkTargets.cmake
    NAMESPACE JAM::
    DESTINATION lib/cmake/JAMFramework
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    JAMFrameworkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/JAMFrameworkConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/JAMFrameworkConfig.cmake
    INSTALL_DESTINATION lib/cmake/JAMFramework
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/JAMFrameworkConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/JAMFrameworkConfigVersion.cmake
    DESTINATION lib/cmake/JAMFramework
)

# Print configuration summary
message(STATUS "")
message(STATUS "JAM Framework v2 Configuration:")
message(STATUS "  Architecture: GPU-NATIVE (Phase 2 Implementation)")
message(STATUS "  GPU Backend: ${JAM_GPU_BACKEND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build Tests: ${JAM_BUILD_TESTS}")
message(STATUS "  Build Examples: ${JAM_BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "🚀 GPU-NATIVE ARCHITECTURE ENABLED 🚀")
message(STATUS "  ✅ GPU Timebase: Master timing controller")
message(STATUS "  ✅ GPU Shared Timeline: Memory-mapped event scheduling")
message(STATUS "  ✅ GPU Compute Shaders: Sample-accurate timing")
message(STATUS "")
message(STATUS "UDP-NATIVE ARCHITECTURE - NO TCP/HTTP DEPENDENCIES")
message(STATUS "Ready to revolutionize multimedia streaming!")
message(STATUS "")
