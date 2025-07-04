cmake_minimum_required(VERSION 3.20)

# Basic audio streaming example
add_executable(jam_audio_example
    audio_streaming_example.cpp
)

target_link_libraries(jam_audio_example
    jam_framework
)

# MIDI burst logic example
add_executable(jam_midi_example
    midi_burst_example.cpp
)

target_link_libraries(jam_midi_example
    jam_framework
)

# Video streaming example
add_executable(jam_video_example
    video_streaming_example.cpp
)

target_link_libraries(jam_video_example
    jam_framework
)

# Complete session example
add_executable(jam_session_example
    complete_session_example.cpp
)

target_link_libraries(jam_session_example
    jam_framework
)
