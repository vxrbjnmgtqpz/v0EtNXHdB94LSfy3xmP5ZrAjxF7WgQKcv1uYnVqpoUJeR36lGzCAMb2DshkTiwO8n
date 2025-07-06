cmake_minimum_required(VERSION 3.20)

# Examples for JAM Framework v2
project(JAMExamples)

# Cross-platform audio engine test (Phase E implementation)
add_executable(cross_platform_audio_test cross_platform_audio_test.cpp)
target_link_libraries(cross_platform_audio_test jam_framework_v2)

# GPU engine creation test
add_executable(gpu_test gpu_test.cpp)
target_link_libraries(gpu_test jam_framework_v2)

# Step-by-step test
add_executable(step_test step_test.cpp)
target_link_libraries(step_test jam_framework_v2)

# Audio backend test
add_executable(audio_backend_test audio_backend_test.cpp)
target_link_libraries(audio_backend_test jam_framework_v2)

# Basic UDP transport examples
add_executable(basic_sender basic_sender.cpp)
target_link_libraries(basic_sender jamframework)

add_executable(basic_receiver basic_receiver.cpp)
target_link_libraries(basic_receiver jamframework)

# Install examples
install(TARGETS cross_platform_audio_test gpu_test basic_sender basic_receiver
    RUNTIME DESTINATION bin
)
