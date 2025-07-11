# Standalone Testbeds for PNBTR+JELLIE DSP

# Audio Testbed - Test PNBTR dither replacement and audio quality
add_subdirectory(audio_testbed)

# Network Testbed - Test JDAT UDP streaming and packet loss handling  
add_subdirectory(network_testbed)

# Training Testbed - Test neural network training and validation
add_subdirectory(training_testbed)

# Future testbeds (directories will be created in later phases)
# add_subdirectory(pnbtr_testbed)
# add_subdirectory(integration_testbed)

# Print testbed summary
message(STATUS "Standalone Testbeds:")
message(STATUS "  Audio Testbed: Validate PNBTR dither replacement")
message(STATUS "  Network Testbed: Test JDAT UDP streaming")
message(STATUS "  PNBTR Testbed: Test neural prediction")
message(STATUS "  Integration Testbed: Test complete pipeline")