cmake_minimum_required(VERSION 3.20)

project(JSONADATFramework 
    VERSION 1.0.0
    DESCRIPTION "JSONADAT Framework with TOAST Transport Layer for Audio Streaming"
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
    find_library(AUDIOUNIT_FRAMEWORK AudioUnit)
    find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox)
    set(PLATFORM_LIBS ${COREAUDIO_FRAMEWORK} ${AUDIOUNIT_FRAMEWORK} ${AUDIOTOOLBOX_FRAMEWORK})
elseif(WIN32)
    set(PLATFORM_LIBS winmm ws2_32 dsound)
elseif(UNIX)
    pkg_check_modules(ALSA REQUIRED alsa)
    pkg_check_modules(JACK jack)
    set(PLATFORM_LIBS ${ALSA_LIBRARIES} ${JACK_LIBRARIES})
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
set(JSONADAT_SOURCES
    src/JSONADATMessage.cpp
    src/JSONADATParser.cpp
    src/TOASTTransport.cpp
    src/ClockDriftArbiter.cpp
    src/AudioBufferManager.cpp
    src/WaveformPredictor.cpp
    src/SchemaValidator.cpp
    src/MessageFactory.cpp
    src/PerformanceProfiler.cpp
    src/ConnectionManager.cpp
    src/ProtocolHandler.cpp
    src/SessionManager.cpp
    src/JELLIEEncoder.cpp
    src/JELLIEDecoder.cpp
    src/ADATSimulator.cpp
)

# Header files
set(JSONADAT_HEADERS
    include/JSONADATMessage.h
    include/JSONADATParser.h
    include/TOASTTransport.h
    include/ClockDriftArbiter.h
    include/AudioBufferManager.h
    include/WaveformPredictor.h
    include/LockFreeQueue.h
    include/MessageFactory.h
    include/JELLIEEncoder.h
    include/JELLIEDecoder.h
    include/ADATSimulator.h
    include/JSONADATTestLibrary.h
)

# Create the main library
add_library(jsonadat_framework STATIC ${JSONADAT_SOURCES} ${JSONADAT_HEADERS})

# Link libraries
target_link_libraries(jsonadat_framework
    PRIVATE
        nlohmann_json::nlohmann_json
        simdjson::simdjson
        Threads::Threads
        ${PLATFORM_LIBS}
)

# Set target properties
set_target_properties(jsonadat_framework PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${JSONADAT_HEADERS}"
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

# Install configuration - disabled for initial phase
# Will be enabled in future phases
if(FALSE)
include(GNUInstallDirs)

install(TARGETS jsonadat_framework
    EXPORT jsonadat_frameworkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jsonadat
)

install(DIRECTORY schemas/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jsonadat/schemas
    FILES_MATCHING PATTERN "*.json"
)

# Export targets for find_package
install(EXPORT jsonadat_frameworkTargets
    FILE jsonadat_frameworkTargets.cmake
    NAMESPACE jsonadat::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonadat_framework
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jsonadat_frameworkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonadat_frameworkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonadat_framework
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/jsonadat_frameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/jsonadat_frameworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/jsonadat_frameworkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jsonadat_framework
)
endif() 