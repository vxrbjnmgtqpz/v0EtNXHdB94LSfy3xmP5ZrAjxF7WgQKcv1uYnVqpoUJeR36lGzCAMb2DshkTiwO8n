# Audio Testbed - Test PNBTR dither replacement and audio quality
# Validates revolutionary zero-noise processing

add_executable(pnbtr_audio_test
    main.cpp
    audio_testbed.cpp
    audio_quality_analyzer.cpp
)

target_include_directories(pnbtr_audio_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../JDAT_Framework/include
    ${PROJECT_SOURCE_DIR}/../PNBTR_Framework/include
)

target_link_libraries(pnbtr_audio_test PRIVATE
    pnbtr_jellie_core
)

# Set C++ standard
set_target_properties(pnbtr_audio_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Copy test audio files to build directory
file(GLOB AUDIO_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test_audio/*")
if(AUDIO_TEST_FILES)
    file(COPY ${AUDIO_TEST_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test_audio/)
endif()

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reports)

# Installation
install(TARGETS pnbtr_audio_test
    RUNTIME DESTINATION bin/testbeds
)

message(STATUS "Audio Testbed configured:")
message(STATUS "  Executable: pnbtr_audio_test")
message(STATUS "  Features: PNBTR dither replacement, A/B comparison, quality analysis")
message(STATUS "  Output: Audio files and quality reports") 