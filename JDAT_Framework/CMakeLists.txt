cmake_minimum_required(VERSION 3.20)

project(JELLIEFramework 
    VERSION 1.0.0
    DESCRIPTION "JELLIE: JAM Embedded Low-Latency Instrument Encoding Framework"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /O2)
endif()

# Options
option(ENABLE_GPU "Enable GPU acceleration" OFF)
option(ENABLE_TESTS "Enable testing" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)

# Find required packages
find_package(Threads REQUIRED)

# JSON library
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

# GPU support
if(ENABLE_GPU)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        add_compile_definitions(JELLIE_GPU_ENABLED)
        message(STATUS "GPU acceleration enabled with Vulkan")
    else()
        message(WARNING "GPU acceleration requested but Vulkan not found")
        set(ENABLE_GPU OFF)
    endif()
endif()

# Include directories
include_directories(include)

# Source files
set(JELLIE_SOURCES
    src/JELLIEEncoder.cpp
    src/JELLIEDecoder.cpp
    src/AudioBufferManager.cpp
    src/ADATSimulator.cpp
    src/WaveformPredictor.cpp
)

# Create main JELLIE library
add_library(JELLIE STATIC ${JELLIE_SOURCES})
target_link_libraries(JELLIE 
    nlohmann_json::nlohmann_json
    Threads::Threads
)

if(ENABLE_GPU AND Vulkan_FOUND)
    target_link_libraries(JELLIE Vulkan::Vulkan)
endif()

# Examples
add_subdirectory(examples)

# Tests
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Install targets
install(TARGETS JELLIE
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY shaders/
    DESTINATION share/jellie/shaders
    FILES_MATCHING PATTERN "*.comp"
)

# Summary
message(STATUS "")
message(STATUS "JELLIE Framework Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  GPU acceleration: ${ENABLE_GPU}")
message(STATUS "  Testing: ${ENABLE_TESTS}")
message(STATUS "  Benchmarks: ${ENABLE_BENCHMARKS}")
message(STATUS "")
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
set(JDAT_SOURCES
    src/JDATMessage.cpp
    src/JDATParser.cpp
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
set(JDAT_HEADERS
    include/JDATMessage.h
    include/JDATParser.h
    include/TOASTTransport.h
    include/ClockDriftArbiter.h
    include/AudioBufferManager.h
    include/WaveformPredictor.h
    include/LockFreeQueue.h
    include/MessageFactory.h
    include/JELLIEEncoder.h
    include/JELLIEDecoder.h
    include/ADATSimulator.h
    include/JDATTestLibrary.h
)

# Create the main library
add_library(jdat_framework STATIC ${JDAT_SOURCES} ${JDAT_HEADERS})

# Link libraries
target_link_libraries(jdat_framework
    PRIVATE
        nlohmann_json::nlohmann_json
        simdjson::simdjson
        Threads::Threads
        ${PLATFORM_LIBS}
)

# Set target properties
set_target_properties(jdat_framework PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${JDAT_HEADERS}"
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

install(TARGETS jdat_framework
    EXPORT jdat_frameworkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jdat
)

install(DIRECTORY schemas/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/jdat/schemas
    FILES_MATCHING PATTERN "*.json"
)

# Export targets for find_package
install(EXPORT jdat_frameworkTargets
    FILE jdat_frameworkTargets.cmake
    NAMESPACE jdat::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jdat_framework
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/jdat_frameworkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/jdat_frameworkConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jdat_framework
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/jdat_frameworkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/jdat_frameworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/jdat_frameworkConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jdat_framework
)
endif() 