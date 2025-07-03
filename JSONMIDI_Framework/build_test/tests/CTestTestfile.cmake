# CMake generated Testfile for 
# Source directory: /Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests
# Build directory: /Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[basic_messages_test]=] "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests/test_basic_messages" "--test-basic")
set_tests_properties([=[basic_messages_test]=] PROPERTIES  PASS_REGULAR_EXPRESSION "All basic message tests passed" TIMEOUT "30" _BACKTRACE_TRIPLES "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;47;add_test;/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;0;")
add_test([=[validation_test]=] "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests/test_validation" "--test-validation")
set_tests_properties([=[validation_test]=] PROPERTIES  PASS_REGULAR_EXPRESSION "All validation tests passed" TIMEOUT "60" _BACKTRACE_TRIPLES "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;48;add_test;/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;0;")
add_test([=[performance_test]=] "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests/test_performance" "--test-performance")
set_tests_properties([=[performance_test]=] PROPERTIES  PASS_REGULAR_EXPRESSION "Performance targets met" TIMEOUT "120" _BACKTRACE_TRIPLES "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;49;add_test;/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;0;")
add_test([=[round_trip_test]=] "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests/test_round_trip" "--test-round-trip")
set_tests_properties([=[round_trip_test]=] PROPERTIES  PASS_REGULAR_EXPRESSION "All round-trip tests passed" TIMEOUT "30" _BACKTRACE_TRIPLES "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;50;add_test;/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;0;")
add_test([=[Phase12Performance]=] "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/tests/test_phase12_performance")
set_tests_properties([=[Phase12Performance]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;51;add_test;/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/tests/CMakeLists.txt;0;")
