# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-src")
  file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-src")
endif()
file(MAKE_DIRECTORY
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-build"
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix"
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/tmp"
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/src/simdjson-populate-stamp"
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/src"
  "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/src/simdjson-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/src/simdjson-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/timothydowler/Projects/MIDIp2p/JSONADAT_Framework/build_test/_deps/simdjson-subbuild/simdjson-populate-prefix/src/simdjson-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
