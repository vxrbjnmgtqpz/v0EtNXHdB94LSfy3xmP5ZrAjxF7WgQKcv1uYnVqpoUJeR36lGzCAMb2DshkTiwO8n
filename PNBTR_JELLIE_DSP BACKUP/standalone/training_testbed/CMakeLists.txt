cmake_minimum_required(VERSION 3.20)

project(training_testbed)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Simple Training Testbed using proven core components
add_executable(training_testbed
    main.cpp
    training_testbed.cpp
    simple_gui.mm
)

target_include_directories(training_testbed PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/PNBTR_JELLIE_DSP/core
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/PNBTR_JELLIE_DSP/include
)

target_link_libraries(training_testbed
    pnbtr_jellie_core
)

if(APPLE)
    target_link_libraries(training_testbed
        "-framework CoreAudio"
        "-framework AudioUnit"
        "-framework AudioToolbox"
        "-framework Cocoa"
        "-framework Foundation"
    )
    
    # Enable Objective-C++ for simple GUI
    set_source_files_properties(simple_gui.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()
