cmake_minimum_required(VERSION 3.20)

# PNBTR demonstration example
add_executable(pnbtr_demonstration
    pnbtr_demonstration.cpp
)

target_link_libraries(pnbtr_demonstration
    pnbtr_framework
)

# Dither replacement comparison example
add_executable(dither_replacement_test
    dither_replacement_test.cpp
)

target_link_libraries(dither_replacement_test
    pnbtr_framework
)

# 50ms prediction benchmark
add_executable(prediction_benchmark
    prediction_benchmark.cpp
)

target_link_libraries(prediction_benchmark
    pnbtr_framework
)

# Learning system test
add_executable(learning_system_test
    learning_system_test.cpp
)

target_link_libraries(learning_system_test
    pnbtr_framework
)
