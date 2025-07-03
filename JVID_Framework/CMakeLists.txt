cmake_minimum_required(VERSION 3.16)
project(JVID_Framework VERSION 1.0.0 LANGUAGES CXX)

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
option(JVID_ENABLE_BENCHMARKS "Enable benchmark builds" ON)
option(JVID_ENABLE_TESTS "Enable test builds" ON)
option(JVID_ENABLE_EXAMPLES "Enable example builds" ON)
option(JVID_ENABLE_GPU "Enable GPU acceleration" OFF)
option(JVID_ENABLE_SIMD "Enable SIMD optimizations" ON)

# GPU acceleration support
if(JVID_ENABLE_GPU)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        add_compile_definitions(JVID_GPU_ENABLED)
        message(STATUS "CUDA found: GPU acceleration enabled")
    else()
        message(STATUS "CUDA not found: GPU acceleration disabled")
        set(JVID_ENABLE_GPU OFF)
    endif()
endif()

# SIMD optimizations
if(JVID_ENABLE_SIMD)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        add_compile_definitions(JVID_AVX2_ENABLED)
        message(STATUS "AVX2 support enabled")
    elseif(COMPILER_SUPPORTS_SSE42)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
        add_compile_definitions(JVID_SSE42_ENABLED)
        message(STATUS "SSE4.2 support enabled")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../JMID_Framework/include  # For TOAST transport
)

# Source files
set(JVID_SOURCES
    src/JVIDMessage.cpp
    src/JAMCamEncoder.cpp
    src/JAMCamDecoder.cpp
    src/VideoBufferManager.cpp
    src/FramePredictor.cpp
    src/VideoDownscaler.cpp
    src/SchemaValidator.cpp
)

# Header files
set(JVID_HEADERS
    include/JVIDMessage.h
    include/JAMCamEncoder.h
    include/JAMCamDecoder.h
    include/VideoBufferManager.h
    include/FramePredictor.h
    include/VideoDownscaler.h
    include/LockFreeQueue.h
)

# Create JVID framework library
add_library(jvid_framework STATIC ${JVID_SOURCES} ${JVID_HEADERS})

# Link dependencies
target_link_libraries(jvid_framework 
    PUBLIC 
        nlohmann_json::nlohmann_json
        Threads::Threads
)

# GPU libraries
if(JVID_ENABLE_GPU AND CUDA_FOUND)
    target_link_libraries(jvid_framework PUBLIC ${CUDA_LIBRARIES})
endif()

# Platform specific libraries
if(APPLE)
    find_library(CORE_VIDEO CoreVideo)
    find_library(CORE_MEDIA CoreMedia)
    find_library(AVFOUNDATION AVFoundation)
    target_link_libraries(jvid_framework PUBLIC 
        ${CORE_VIDEO}
        ${CORE_MEDIA} 
        ${AVFOUNDATION}
    )
elseif(UNIX)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(V4L2 libv4l2)
        if(V4L2_FOUND)
            target_link_libraries(jvid_framework PUBLIC ${V4L2_LIBRARIES})
            target_include_directories(jvid_framework PUBLIC ${V4L2_INCLUDE_DIRS})
            add_compile_definitions(JVID_V4L2_ENABLED)
        endif()
    endif()
endif()

# Set target properties
set_target_properties(jvid_framework PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PUBLIC_HEADER "${JVID_HEADERS}"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Add compile definitions
target_compile_definitions(jvid_framework 
    PUBLIC
        JVID_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        JVID_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        JVID_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        JVID_BUILDING_LIBRARY
)

# Installation configuration
include(GNUInstallDirs)
install(TARGETS jvid_framework
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jvid
)

# Install schemas
install(FILES schemas/jvid-message.schema.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jvid/schemas
)

# CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/jvid_frameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jvid_frameworkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/jvid_frameworkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jvid_framework
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/jvid_frameworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/jvid_frameworkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jvid_framework
)

# Add subdirectories
if(JVID_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(JVID_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(JVID_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Custom targets for development
add_custom_target(jvid_format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include 
            -name "*.cpp" -o -name "*.h" | xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(jvid_lint
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include 
            -name "*.cpp" -o -name "*.h" | xargs clang-tidy --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
    COMMENT "Running static analysis"
)

# Performance profiling target
add_custom_target(jvid_profile
    COMMAND valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes
            $<TARGET_FILE:jamcam_basic_demo>
    DEPENDS jamcam_basic_demo
    COMMENT "Running performance profiling"
)

# Documentation target
find_package(Doxygen QUIET)
if(Doxygen_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_PROJECT_NAME "JVID Framework")
    set(DOXYGEN_PROJECT_BRIEF "Ultra-low latency JSON video streaming framework")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    
    doxygen_add_docs(jvid_docs
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        COMMENT "Generating JVID documentation"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "JVID Framework Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Examples: ${JVID_ENABLE_EXAMPLES}")
message(STATUS "  Tests: ${JVID_ENABLE_TESTS}")
message(STATUS "  Benchmarks: ${JVID_ENABLE_BENCHMARKS}")
message(STATUS "  GPU acceleration: ${JVID_ENABLE_GPU}")
message(STATUS "  SIMD optimizations: ${JVID_ENABLE_SIMD}")
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "  SIMD level: AVX2")
elseif(COMPILER_SUPPORTS_SSE42)
    message(STATUS "  SIMD level: SSE4.2")
else()
    message(STATUS "  SIMD level: None")
endif()
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "") 