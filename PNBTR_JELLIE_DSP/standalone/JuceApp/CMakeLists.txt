cmake_minimum_required(VERSION 3.15)
project(PnbtrJellieJuceApp)

set(CMAKE_CXX_STANDARD 17)

# Add JUCE using FetchContent, like in the TOASTer project
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG origin/master # Using master for latest updates
)
FetchContent_MakeAvailable(JUCE)


# Add the application
juce_add_gui_app(PnbtrJellieJuceApp
    VERSION "1.0.0"
    PRODUCT_NAME "PNBTR JELLIE JUCE App"
    SOURCE_FILES
        Source/Main.cpp
)

target_compile_options(PnbtrJellieJuceApp PRIVATE
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# Add include directories for our frameworks
target_include_directories(PnbtrJellieJuceApp PRIVATE
    ../../core
    ../../include
)


target_link_libraries(PnbtrJellieJuceApp
    PRIVATE
        pnbtr_jellie_core
        juce::juce_standalone_app
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        "-framework Cocoa"
        "-framework CoreAudio"
        "-framework CoreMIDI"
        "-framework AudioToolbox"
        "-framework Metal"
) 