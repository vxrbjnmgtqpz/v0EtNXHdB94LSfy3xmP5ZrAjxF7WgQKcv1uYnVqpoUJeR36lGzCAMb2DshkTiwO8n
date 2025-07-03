# JDAT Terminology Update Plan
## Safe Migration from JSONADAT/JSONMIDI/JSONVID → JDAT/JMID/JVID

**Objective**: Update project terminology to match patent documentation while avoiding Alesis ADAT copyright and ensuring zero build breakage.

## Phase 1: Pre-Migration Testing & Validation ✅

### 1.1 Build System Validation
```bash
# Test current build state before any changes
cd /Users/timothydowler/Projects/MIDIp2p

# Test JSONMIDI Framework build
cd JSONMIDI_Framework
mkdir -p build_test
cd build_test
cmake ..
make -j4
cd ../..

# Test TOASTer build  
cd TOASTer
mkdir -p build_test
cmake -B build_test -S .
cmake --build build_test
cd ..

# Test JSONADAT Framework build
cd JSONADAT_Framework
mkdir -p build_test
cmake -B build_test -S .
cmake --build build_test
cd ..
```

### 1.2 Create Backup Checkpoints
```bash
# Create tagged backup before any changes
git add -A
git commit -m "PRE-JDAT-RENAME: Complete working state before terminology update"
git tag v0.8.0-pre-jdat-rename
git push origin v0.8.0-pre-jdat-rename
```

### 1.3 Dependency Analysis
- [ ] Map all CMakeLists.txt cross-references
- [ ] Identify all header include paths
- [ ] Document all namespace usages
- [ ] List all target_link_libraries dependencies

## Phase 2: Documentation-Only Updates (Safe) 📝

### 2.1 Markdown Documentation Updates
**Files to update** (zero build impact):
- [ ] `README.md` - Update all JSONADAT → JDAT references
- [ ] `JSONADAT.md` → rename to `JDAT.md`
- [ ] `JSONVID+JAMCAM.md` → update JSONVID → JVID
- [ ] All `*_STATUS_REPORT.md` files
- [ ] `networkdiagram.md` - Update patent doc terminology
- [ ] `250703PatentJAMNet.md` - Verify consistency

### 2.2 Code Comment Updates  
**Safe string literal updates**:
- [ ] Update `std::cout` messages in examples
- [ ] Update header file comments
- [ ] Update JSON schema descriptions
- [ ] Update error messages and logs

### 2.3 Validation After Phase 2
```bash
# Ensure builds still work after doc changes
cd JSONMIDI_Framework && make -C build_test clean && make -C build_test
cd ../TOASTer && cmake --build build_test --clean-first  
cd ../JSONADAT_Framework && cmake --build build_test --clean-first
```

## Phase 3: Schema & Configuration Updates 🔧

### 3.1 JSON Schema Files
- [ ] `JSONMIDI_Framework/schemas/jsonmidi-message.schema.json` → `jmid-message.schema.json`
- [ ] `JSONADAT_Framework/schemas/jsonadat-message.schema.json` → `jdat-message.schema.json`  
- [ ] Update schema `"id"` fields from `"jsonadat"` → `"jdat"`
- [ ] Update schema `"title"` fields appropriately

### 3.2 Example File Updates
- [ ] `JSONADAT_Framework/examples/basic_jellie_demo.cpp` - Update all references
- [ ] `JSONADAT_Framework/examples/adat_192k_demo.cpp` - Update to `jdat_192k_demo.cpp`
- [ ] Update all example CMakeLists.txt target names

### 3.3 Test After Schema Changes
```bash
# Validate JSON parsing still works
cd JSONMIDI_Framework && ./build_test/examples/basic_example
cd ../JSONADAT_Framework && ./build_test/examples/basic_jellie_demo
```

## Phase 4: Source Code Symbol Updates 🔄

### 4.1 Namespace Renames (Careful)
**Current → Target**:
- `namespace JSONMIDI` → `namespace JMID`
- `namespace jsonadat` → `namespace jdat`  
- `namespace jsonvid` → `namespace jvid`

### 4.2 Class Name Updates
**JSONMIDI Framework**:
- `JSONMIDIMessage` → `JMIDMessage`
- `JSONMIDIParser` → `JMIDParser`

**JSONADAT Framework**:
- `JSONADATMessage` → `JDATMessage`
- `ADATSimulator` → `JDATChannelMapper` (avoid ADAT copyright)

### 4.3 Header File Renames
- `JSONMIDIMessage.h` → `JMIDMessage.h`
- `JSONMIDIParser.h` → `JMIDParser.h`
- `JSONADATMessage.h` → `JDATMessage.h`
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
- `jsonmidi_framework` → `jmid_framework`
- `jsonadat_framework` → `jdat_framework`
- `jsonvid_framework` → `jvid_framework`

**Install Paths**:
- `${CMAKE_INSTALL_INCLUDEDIR}/jsonmidi` → `${CMAKE_INSTALL_INCLUDEDIR}/jmid`
- `${CMAKE_INSTALL_DATADIR}/jsonadat` → `${CMAKE_INSTALL_DATADIR}/jdat`

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
git mv JSONMIDI_Framework JMID_Framework
git mv JSONADAT_Framework JDAT_Framework  
git mv JSONVID_Framework JVID_Framework

# Update any hardcoded paths
grep -r "JSONMIDI_Framework" . --exclude-dir=.git
grep -r "JSONADAT_Framework" . --exclude-dir=.git
grep -r "JSONVID_Framework" . --exclude-dir=.git
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

// OLD: "ADAT protocol behavior"  
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