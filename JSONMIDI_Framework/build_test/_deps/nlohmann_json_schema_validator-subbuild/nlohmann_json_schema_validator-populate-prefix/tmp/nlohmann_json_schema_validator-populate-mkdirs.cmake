# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-src")
  file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-src")
endif()
file(MAKE_DIRECTORY
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-build"
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix"
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/tmp"
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/src/nlohmann_json_schema_validator-populate-stamp"
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/src"
  "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/src/nlohmann_json_schema_validator-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/src/nlohmann_json_schema_validator-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONMIDI_Framework/build_test/_deps/nlohmann_json_schema_validator-subbuild/nlohmann_json_schema_validator-populate-prefix/src/nlohmann_json_schema_validator-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
