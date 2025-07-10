# PBJ Documentation Directory

**PNBTR+JELLIE Training Testbed Documentation**

## 📖 **Primary Documentation**

### **COMPREHENSIVE_GPU_AUDIO_DEVELOPMENT_GUIDE.md**

**⭐ MAIN REFERENCE DOCUMENT ⭐**

This is the consolidated, definitive guide that combines:

- Video game engine architecture patterns applied to DAW development
- Complete Metal GPU compute pipeline implementation
- JUCE integration patterns with lessons learned
- Production-ready CMake configuration
- Build system with error handling
- All critical integration rules and best practices

**Use this document for all development work.** It incorporates every lesson learned from the actual implementation process.

---

## 🔧 **Quick Start**

1. **Read the comprehensive guide** for complete understanding
2. **Use the build script**: `./build_pnbtr_jellie.sh` (includes all lessons learned)
3. **Follow the checklist** in the guide for setup verification

---

## 📋 **Document Consolidation History**

**Previously separate documents (now consolidated):**

- ~~VIDEO_GAME_ENGINE_AS_DAW_ARCHITECTURE.md~~ → Merged into comprehensive guide
- ~~GPU_NATIVE_METAL_SHADER_INDEX.md~~ → Merged into comprehensive guide

**Benefits of consolidation:**

- ✅ Single source of truth
- ✅ No duplicate or conflicting information
- ✅ All lessons learned incorporated
- ✅ Complete JUCE + CMake + Metal workflow
- ✅ Production-ready build configuration

---

## 🎯 **Key Lessons Incorporated**

### **Threading & Memory**

- ❌ Never use `std::vector<float>` in audio callbacks
- ❌ Never use atomic operations on structs with strings
- ✅ Always use mutex for complex data structures
- ✅ Always use stack arrays for real-time audio buffers

### **JUCE Integration**

- ❌ Never use deprecated `jmin/jmax` functions
- ❌ Never process DSP on Message thread
- ✅ Always use `std::min/max/clamp`
- ✅ Always use address-of operator for `addAndMakeVisible`
- ✅ Always link `juce::juce_dsp` for FFT functionality

### **Build System**

- ✅ Metal shader compilation before C++ compilation
- ✅ Proper framework linking order (JUCE first, then Metal)
- ✅ Comprehensive build verification
- ✅ Error handling and prerequisite checking

---

## 🚀 **For Developers**

**New to the project?** Start with `COMPREHENSIVE_GPU_AUDIO_DEVELOPMENT_GUIDE.md`

**Building the app?** Use `../build_pnbtr_jellie.sh`

**Having issues?** All known problems and solutions are documented in the comprehensive guide

**This documentation represents the culmination of hands-on development experience and eliminates the trial-and-error development cycle.**
