cmake_minimum_required(VERSION 3.20)

# Examples for JAM Framework v2
project(JAMExamples)

# Basic UDP transport examples
add_executable(basic_sender basic_sender.cpp)
target_link_libraries(basic_sender jamframework)

add_executable(basic_receiver basic_receiver.cpp)
target_link_libraries(basic_receiver jamframework)

# Install examples
install(TARGETS basic_sender basic_receiver
    RUNTIME DESTINATION bin
)
