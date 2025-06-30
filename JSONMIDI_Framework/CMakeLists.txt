cmake_minimum_required(VERSION 3.20)

project(JSONMIDIFramework 
    VERSION 1.0.0
    DESCRIPTION "JSONMIDI Framework with TOAST Transport Layer"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(PkgConfig REQUIRED)

# Check for nlohmann/json (JSON library)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json not found, will use FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Check for simdjson (high-performance JSON parser)
find_package(simdjson QUIET)
if(NOT simdjson_FOUND)
    message(STATUS "simdjson not found, will use FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        simdjson
        GIT_REPOSITORY https://github.com/simdjson/simdjson.git
        GIT_TAG v3.7.0
    )
    FetchContent_MakeAvailable(simdjson)
endif()

# Check for nlohmann/json-schema (JSON schema validation)
find_package(nlohmann_json_schema_validator QUIET)
if(NOT nlohmann_json_schema_validator_FOUND)
    message(STATUS "nlohmann_json_schema_validator not found, will use FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json_schema_validator
        GIT_REPOSITORY https://github.com/pboettch/json-schema-validator.git
        GIT_TAG 2.3.0
    )
    FetchContent_MakeAvailable(nlohmann_json_schema_validator)
endif()

# Threading support
find_package(Threads REQUIRED)

# Platform-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
    find_library(COREAUDIO_FRAMEWORK CoreAudio)
    find_library(COREMIDI_FRAMEWORK CoreMIDI)
    set(PLATFORM_LIBS ${COREAUDIO_FRAMEWORK} ${COREMIDI_FRAMEWORK})
elseif(WIN32)
    set(PLATFORM_LIBS winmm ws2_32)
elseif(UNIX)
    pkg_check_modules(ALSA REQUIRED alsa)
    set(PLATFORM_LIBS ${ALSA_LIBRARIES})
endif()

# Compiler-specific settings
if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -fsanitize=address -fsanitize=undefined)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Source files
set(JSONMIDI_SOURCES
    src/JSONMIDIMessage.cpp
    src/VersionNegotiator.cpp
    src/JSONMIDIParser.cpp
    src/TOASTTransport.cpp
    src/BassoonParser.cpp
    src/SchemaValidator.cpp
    src/MessageFactory.cpp
    src/PerformanceProfiler.cpp
    src/ConnectionManager.cpp
    src/ProtocolHandler.cpp
    src/SessionManager.cpp
)

# Header files
set(JSONMIDI_HEADERS
    include/JSONMIDIMessage.h
    include/VersionNegotiator.h
    include/JSONMIDIParser.h
    include/TOASTTransport.h
    include/LockFreeQueue.h
)

# Create the main library
add_library(jsonmidi_framework STATIC ${JSONMIDI_SOURCES} ${JSONMIDI_HEADERS})

# Link libraries
target_link_libraries(jsonmidi_framework
    PRIVATE
        nlohmann_json::nlohmann_json
        simdjson::simdjson
        Threads::Threads
        ${PLATFORM_LIBS}
)

# Set target properties
set_target_properties(jsonmidi_framework PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${JSONMIDI_HEADERS}"
)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Performance benchmarks
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install configuration - disabled for Phase 1.1
# Will be enabled in future phases
if(FALSE)
include(GNUInstallDirs)

install(TARGETS jsonmidi_framework
    EXPORT jsonmidi_frameworkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jsonmidi
)

install(DIRECTORY schemas/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jsonmidi/schemas
    FILES_MATCHING PATTERN "*.json"
)

# Export targets for find_package
install(EXPORT jsonmidi_frameworkTargets
    FILE jsonmidi_frameworkTargets.cmake
    NAMESPACE jsonmidi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonmidi_framework
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jsonmidi_frameworkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonmidi_frameworkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonmidi_framework
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/jsonmidi_frameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/jsonmidi_frameworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonmidi_frameworkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonmidi_framework
)

# Print build information
message(STATUS "")
message(STATUS "JSONMIDIFramework Build Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
endif() # End install configuration disable
message(STATUS "")
