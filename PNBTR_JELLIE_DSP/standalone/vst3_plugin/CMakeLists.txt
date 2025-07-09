# PNBTR+JELLIE VST3 Plugin - FULL GUI Application
cmake_minimum_required(VERSION 3.15)
project(PnbtrJellieGUI)

# Feature flags - enable real processing by default
option(USE_REAL_PROCESSING "Enable real audio processing instead of placeholder data" ON)
option(DISABLE_PLACEHOLDER_DATA "Disable all placeholder/fake data generation" ON)

# Include directories
include_directories(include)

# Define compile flags based on options
if(USE_REAL_PROCESSING)
    add_definitions(-DUSE_REAL_PROCESSING=1)
endif()

if(DISABLE_PLACEHOLDER_DATA)
    add_definitions(-DDISABLE_PLACEHOLDER_DATA=1)
endif()

# FULL COMPLEX GUI APPLICATION (the one you actually want!)
add_executable(pnbtr_jellie_gui_app MACOSX_BUNDLE
    src/PnbtrJellieGUIApp_Fixed.mm
)

# Link with required frameworks
target_link_libraries(pnbtr_jellie_gui_app
    "-framework Cocoa"
    "-framework CoreAudio" 
    "-framework AudioUnit"
    "-framework QuartzCore"
)

# Set bundle properties
set_target_properties(pnbtr_jellie_gui_app PROPERTIES
    BUNDLE TRUE
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.jamnet.pnbtr.jellie.full"
    MACOSX_BUNDLE_BUNDLE_NAME "PNBTR+JELLIE Training Testbed"
    MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    MACOSX_BUNDLE_INFO_STRING "JAMNet PNBTR+JELLIE Training Testbed - FULL VERSION"
)

# Set C++17 standard
set_property(TARGET pnbtr_jellie_gui_app PROPERTY CXX_STANDARD 17)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pnbtr_jellie_gui_app Threads::Threads)

# macOS specific frameworks for GUI
if(APPLE)
    target_link_libraries(pnbtr_jellie_gui_app
        "-framework CoreAudio"
        "-framework AudioUnit"
        "-framework AudioToolbox"
        "-framework Cocoa"
        "-framework Foundation"
        "-framework CoreFoundation"
        "-framework AVFoundation"
    )
    
    # Enable Objective-C++
    set_source_files_properties(src/PnbtrJellieGUIApp_Fixed.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()

# Compiler flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(pnbtr_jellie_gui_app PRIVATE -O2 -march=native -Wno-deprecated-declarations)
endif()

# Optional: VST3 Plugin (if VST3 SDK is available)
# find_package(VST3SDK QUIET)
# if(VST3SDK_FOUND)
#     add_vst3_plugin(...)
# endif() 