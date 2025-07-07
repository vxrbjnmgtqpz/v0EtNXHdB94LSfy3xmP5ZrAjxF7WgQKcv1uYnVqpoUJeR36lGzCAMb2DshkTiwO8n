cmake_minimum_required(VERSION 3.20)

# Examples for JAM Framework v2
project(JAMExamples)

# Set include directories for all examples
include_directories(${CMAKE_SOURCE_DIR}/include)

# Cross-platform audio engine test (Phase E implementation)
add_executable(cross_platform_audio_test cross_platform_audio_test.cpp)
target_link_libraries(cross_platform_audio_test jam_framework_v2)
target_include_directories(cross_platform_audio_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# GPU engine creation test
add_executable(gpu_test gpu_test.cpp)
target_link_libraries(gpu_test jam_framework_v2)
target_include_directories(gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Step-by-step test
add_executable(step_test step_test.cpp)
target_link_libraries(step_test jam_framework_v2)
target_include_directories(step_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Audio backend test
add_executable(audio_backend_test audio_backend_test.cpp)
target_link_libraries(audio_backend_test jam_framework_v2)
target_include_directories(audio_backend_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Basic UDP transport examples - Fixed library name
add_executable(basic_sender basic_sender.cpp)
target_link_libraries(basic_sender jam_framework_v2)
target_include_directories(basic_sender PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(basic_receiver basic_receiver.cpp)
target_link_libraries(basic_receiver jam_framework_v2)
target_include_directories(basic_receiver PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install examples
install(TARGETS cross_platform_audio_test gpu_test step_test audio_backend_test basic_sender basic_receiver
    RUNTIME DESTINATION bin
)
