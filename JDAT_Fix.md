# JDAT Terminology Update Plan
## Safe Migration from JDAT/JMID/JVID → JDAT/JMID/JVID

**Objective**: Update project terminology to match patent documentation while avoiding Alesis ADAT copyright and ensuring zero build breakage.

## Phase 1: Pre-Migration Testing & Validation ✅

### 1.1 Build System Validation ✅
```bash
# Test current build state before any changes
cd /Users/timothydowler/Projects/MIDIp2p

# Test JMID Framework build ✅
cd JMID_Framework
mkdir -p build_test
cd build_test
cmake ..
make -j4
./tests/test_basic_messages # ✅ PASSED
cd ../..

# Test TOASTer build ✅
cd TOASTer
mkdir -p build_test
cmake -B build_test -S .
cmake --build build_test # ✅ PASSED - TOASTer.app created
cd ..

# Test JDAT Framework build ⚠️
cd JDAT_Framework
# Note: Framework appears to be header-only (no source files)
# Will need special handling during migration
```

### 1.2 Create Backup Checkpoints ✅
```bash
# Create tagged backup before any changes
git add -A
git commit -m "PRE-JDAT-RENAME: Complete working state before terminology update"
git tag v0.8.0-pre-jdat-rename
# Ready to push when needed
```

### 1.3 Dependency Analysis ✅
**Core Framework Status:**
- [x] JMID Framework: ✅ FULLY FUNCTIONAL - Core library builds and tests pass
- [x] TOASTer: ✅ FULLY FUNCTIONAL - macOS app builds successfully
- [x] JDAT Framework: ⚠️ HEADER-ONLY - No source files, only headers exist

**Critical Dependencies:**
- [x] TOASTer depends on JMID Framework via add_subdirectory
- [x] nlohmann/json, simdjson, and json-schema-validator are fetched via CMake
- [x] JUCE framework integrated successfully in TOASTer

## Phase 2: Documentation-Only Updates (Safe) ✅

### 2.1 Markdown Documentation Updates ✅
**Files updated** (zero build impact):
- [x] `README.md` - Updated all JDAT → JDAT references
- [x] `JDAT.md` → renamed to `JDAT.md`
- [x] `JVID+JAMCAM.md` → updated JVID → JVID
- [x] `JMID_Framework/Roadmap.md` - Updated JDAT → JDAT
- [ ] All `*_STATUS_REPORT.md` files (remaining)
- [ ] `networkdiagram.md` - Update patent doc terminology
- [ ] `250703PatentJAMNet.md` - Verify consistency

### 2.2 Code Comment Updates (In Progress)
**Safe string literal updates**:
- [ ] Update `std::cout` messages in examples
- [ ] Update header file comments
- [ ] Update JSON schema descriptions
- [ ] Update error messages and logs

### 2.3 Validation After Phase 2 ✅
```bash
# Builds still work after doc changes ✅
cd JMID_Framework && make -C build_test clean && make -C build_test
./tests/test_basic_messages # ✅ ALL TESTS PASS
```

### 2.4 Backup After Phase 2 ✅
```bash
git add -A
git commit -m "PHASE 2 PARTIAL: Documentation updates - JDAT→JDAT, JVID→JVID, ADAT→4-channel strategy"
```

## Phase 3: Schema & Configuration Updates 🔧

### 3.1 JSON Schema Files
- [ ] `JMID_Framework/schemas/jmid-message.schema.json` → `jmid-message.schema.json`
- [ ] `JDAT_Framework/schemas/jdat-message.schema.json` → `jdat-message.schema.json`  
- [ ] Update schema `"id"` fields from `"jdat"` → `"jdat"`
- [ ] Update schema `"title"` fields appropriately

### 3.2 Example File Updates
- [ ] `JDAT_Framework/examples/basic_jellie_demo.cpp` - Update all references
- [ ] `JDAT_Framework/examples/adat_192k_demo.cpp` - Update to `jdat_192k_demo.cpp`
- [ ] Update all example CMakeLists.txt target names

### 3.3 Test After Schema Changes
```bash
# Validate JSON parsing still works
cd JMID_Framework && ./build_test/examples/basic_example
cd ../JDAT_Framework && ./build_test/examples/basic_jellie_demo
```

## Phase 4: Source Code Symbol Updates 🔄

### 4.1 Namespace Renames (Careful)
**Current → Target**:
- `namespace JMID` → `namespace JMID`
- `namespace jdat` → `namespace jdat`  
- `namespace jvid` → `namespace jvid`

### 4.2 Class Name Updates
**JMID Framework**:
- `JMIDMessage` → `JMIDMessage`
- `JMIDParser` → `JMIDParser`

**JDAT Framework**:
- `JDATMessage` → `JDATMessage`
- `ADATSimulator` → `JDATChannelMapper` (avoid ADAT copyright)

### 4.3 Header File Renames
- `JMIDMessage.h` → `JMIDMessage.h`
- `JMIDParser.h` → `JMIDParser.h`
- `JDATMessage.h` → `JDATMessage.h`
- `ADATSimulator.h` → `JDATChannelMapper.h`

### 4.4 Incremental Testing Strategy
```bash
# Test each framework individually after updates
for framework in JMID JDAT JVID; do
    echo "Testing ${framework} framework..."
    cd ${framework}_Framework
    cmake --build build_test
    ./build_test/tests/test_basic_messages
    cd ..
done
```

## Phase 5: Build System Updates 🏗️

### 5.1 CMakeLists.txt Updates
**Library Target Names**:
- `jmid_framework` → `jmid_framework`
- `jdat_framework` → `jdat_framework`
- `jvid_framework` → `jvid_framework`

**Install Paths**:
- `${CMAKE_INSTALL_INCLUDEDIR}/jmid` → `${CMAKE_INSTALL_INCLUDEDIR}/jmid`
- `${CMAKE_INSTALL_DATADIR}/jdat` → `${CMAKE_INSTALL_DATADIR}/jdat`

### 5.2 TOASTer Integration Updates
- [ ] Update `TOASTer/CMakeLists.txt` target_link_libraries
- [ ] Update include paths in `TOASTer/Source/`
- [ ] Test TOASTer build with new framework names

### 5.3 Cross-Framework Dependencies
- [ ] Update all `find_package()` calls
- [ ] Update all `target_link_libraries()` calls
- [ ] Update all `add_subdirectory()` paths

## Phase 6: Directory Structure Migration 📁

### 6.1 Directory Renames (Final Step)
```bash
# Only after all builds are working
git mv JMID_Framework JMID_Framework
git mv JDAT_Framework JDAT_Framework  
git mv JVID_Framework JVID_Framework

# Update any hardcoded paths
grep -r "JMID_Framework" . --exclude-dir=.git
grep -r "JDAT_Framework" . --exclude-dir=.git
grep -r "JVID_Framework" . --exclude-dir=.git
```

### 6.2 .gitignore Updates
- [ ] Update build directory exclusions
- [ ] Update cache directory exclusions

## Phase 7: Final Validation & Documentation 🎯

### 7.1 Complete Build Test
```bash
# Clean build of entire project
rm -rf */build_test
cd JMID_Framework && cmake -B build && cmake --build build
cd ../JDAT_Framework && cmake -B build && cmake --build build  
cd ../JVID_Framework && cmake -B build && cmake --build build
cd ../TOASTer && cmake -B build && cmake --build build
```

### 7.2 Integration Tests
- [ ] TOASTer GUI launches successfully
- [ ] JMID message parsing works
- [ ] JDAT audio streaming tests pass
- [ ] All example programs execute

### 7.3 Documentation Audit
- [ ] Update all README files with new paths
- [ ] Update build instructions
- [ ] Update developer documentation
- [ ] Verify patent document consistency

### 7.4 Final Backup
```bash
git add -A
git commit -m "COMPLETE: JDAT terminology update - all frameworks renamed and validated"
git tag v0.8.1-jdat-complete
git push origin v0.8.1-jdat-complete
```

## ADAT Copyright Mitigation 🛡️

### Alternative Terminology:
- **"ADAT simulation"** → **"4-channel interleaving strategy"**
- **"ADAT channel mapping"** → **"JDAT channel distribution"**
- **"192k ADAT mode"** → **"192k quad-channel mode"**
- **"ADATSimulator"** → **"JDATChannelMapper"**

### Code Comment Updates:
```cpp
// OLD: "Uses ADAT's 4-channel capability"
// NEW: "Uses 4-channel interleaving for high sample rates"

// OLD: "4-channel interleaving protocol behavior"  
// NEW: "Multi-channel distribution protocol"
```

## Risk Mitigation 🔒

### Rollback Plan:
1. **Git tag checkpoints** at each phase
2. **Parallel build directories** for testing
3. **Incremental validation** after each change
4. **Automated test scripts** to verify functionality

### Success Criteria:
- [ ] All frameworks build without errors
- [ ] All unit tests pass
- [ ] TOASTer launches and connects
- [ ] No performance regression
- [ ] Documentation is consistent
- [ ] Patent terminology matches codebase

---

**Next Steps**: Begin with Phase 1 validation to establish current working state, then proceed incrementally through documentation updates before touching any build-critical code.