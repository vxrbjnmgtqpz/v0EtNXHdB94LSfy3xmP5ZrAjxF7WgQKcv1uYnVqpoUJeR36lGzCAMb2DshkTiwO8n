cmake_minimum_required(VERSION 3.16)
project(JSONVID_Framework VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configure build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic")

# Ultra-low latency specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops -finline-functions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fno-exceptions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=0") # Disable security checks for performance
endif()

# Platform specific optimizations
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.15")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Find required dependencies
find_package(Threads REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json 3.9.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, will download during build")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Optional dependencies
option(JSONVID_ENABLE_BENCHMARKS "Enable benchmark builds" ON)
option(JSONVID_ENABLE_TESTS "Enable test builds" ON)
option(JSONVID_ENABLE_EXAMPLES "Enable example builds" ON)
option(JSONVID_ENABLE_GPU "Enable GPU acceleration" OFF)
option(JSONVID_ENABLE_SIMD "Enable SIMD optimizations" ON)

# GPU acceleration support
if(JSONVID_ENABLE_GPU)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        add_compile_definitions(JSONVID_GPU_ENABLED)
        message(STATUS "CUDA found: GPU acceleration enabled")
    else()
        message(STATUS "CUDA not found: GPU acceleration disabled")
        set(JSONVID_ENABLE_GPU OFF)
    endif()
endif()

# SIMD optimizations
if(JSONVID_ENABLE_SIMD)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        add_compile_definitions(JSONVID_AVX2_ENABLED)
        message(STATUS "AVX2 support enabled")
    elseif(COMPILER_SUPPORTS_SSE42)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
        add_compile_definitions(JSONVID_SSE42_ENABLED)
        message(STATUS "SSE4.2 support enabled")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../JSONMIDI_Framework/include  # For TOAST transport
)

# Source files
set(JSONVID_SOURCES
    src/JSONVIDMessage.cpp
    src/JAMCamEncoder.cpp
    src/JAMCamDecoder.cpp
    src/VideoBufferManager.cpp
    src/FramePredictor.cpp
    src/VideoDownscaler.cpp
    src/SchemaValidator.cpp
)

# Header files
set(JSONVID_HEADERS
    include/JSONVIDMessage.h
    include/JAMCamEncoder.h
    include/JAMCamDecoder.h
    include/VideoBufferManager.h
    include/FramePredictor.h
    include/VideoDownscaler.h
    include/LockFreeQueue.h
)

# Create JSONVID framework library
add_library(jsonvid_framework STATIC ${JSONVID_SOURCES} ${JSONVID_HEADERS})

# Link dependencies
target_link_libraries(jsonvid_framework 
    PUBLIC 
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# GPU libraries
if(JSONVID_ENABLE_GPU AND CUDA_FOUND)
    target_link_libraries(jsonvid_framework PUBLIC ${CUDA_LIBRARIES})
endif()

# Platform specific libraries
if(APPLE)
    find_library(CORE_VIDEO CoreVideo)
    find_library(CORE_MEDIA CoreMedia)
    find_library(AVFOUNDATION AVFoundation)
    target_link_libraries(jsonvid_framework PUBLIC 
        ${CORE_VIDEO}
        ${CORE_MEDIA} 
        ${AVFOUNDATION}
    )
elseif(UNIX)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(V4L2 libv4l2)
        if(V4L2_FOUND)
            target_link_libraries(jsonvid_framework PUBLIC ${V4L2_LIBRARIES})
            target_include_directories(jsonvid_framework PUBLIC ${V4L2_INCLUDE_DIRS})
            add_compile_definitions(JSONVID_V4L2_ENABLED)
        endif()
    endif()
endif()

# Set target properties
set_target_properties(jsonvid_framework PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PUBLIC_HEADER "${JSONVID_HEADERS}"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Add compile definitions
target_compile_definitions(jsonvid_framework 
    PUBLIC
        JSONVID_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        JSONVID_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        JSONVID_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        JSONVID_BUILDING_LIBRARY
)

# Installation configuration
include(GNUInstallDirs)
install(TARGETS jsonvid_framework
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jsonvid
)

# Install schemas
install(FILES schemas/jsonvid-message.schema.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jsonvid/schemas
)

# CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/jsonvid_frameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jsonvid_frameworkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonvid_frameworkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonvid_framework
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/jsonvid_frameworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonvid_frameworkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonvid_framework
)

# Add subdirectories
if(JSONVID_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(JSONVID_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(JSONVID_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Custom targets for development
add_custom_target(jsonvid_format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include 
            -name "*.cpp" -o -name "*.h" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(jsonvid_lint
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include 
            -name "*.cpp" -o -name "*.h" | xargs clang-tidy --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
    COMMENT "Running static analysis"
)

# Performance profiling target
add_custom_target(jsonvid_profile
    COMMAND valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes
            $<TARGET_FILE:jamcam_basic_demo>
    DEPENDS jamcam_basic_demo
    COMMENT "Running performance profiling"
)

# Documentation target
find_package(Doxygen QUIET)
if(Doxygen_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_PROJECT_NAME "JSONVID Framework")
    set(DOXYGEN_PROJECT_BRIEF "Ultra-low latency JSON video streaming framework")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    
    doxygen_add_docs(jsonvid_docs
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        COMMENT "Generating JSONVID documentation"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "JSONVID Framework Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Examples: ${JSONVID_ENABLE_EXAMPLES}")
message(STATUS "  Tests: ${JSONVID_ENABLE_TESTS}")
message(STATUS "  Benchmarks: ${JSONVID_ENABLE_BENCHMARKS}")
message(STATUS "  GPU acceleration: ${JSONVID_ENABLE_GPU}")
message(STATUS "  SIMD optimizations: ${JSONVID_ENABLE_SIMD}")
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "  SIMD level: AVX2")
elseif(COMPILER_SUPPORTS_SSE42)
    message(STATUS "  SIMD level: SSE4.2")
else()
    message(STATUS "  SIMD level: None")
endif()
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "") 