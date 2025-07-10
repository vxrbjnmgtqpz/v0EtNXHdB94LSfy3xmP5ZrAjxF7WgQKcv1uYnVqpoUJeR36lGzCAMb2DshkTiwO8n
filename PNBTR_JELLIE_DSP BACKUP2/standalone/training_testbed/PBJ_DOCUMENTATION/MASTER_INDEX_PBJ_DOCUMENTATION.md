# MASTER INDEX: PBJ DOCUMENTATION

**PNBTR+JELLIE Training Testbed Documentation Archive**

> **📋 AUTO-INDEXING SYSTEM NOTE:**  
> This document serves as the central index for all PBJ documentation. As new files are created or updated in the PBJ_DOCUMENTATION folder, this index will be automatically updated to maintain chronological order and content categorization. Time-stamped files (YYMMDD_HHMMSS format) are automatically sorted by when they happened, while non-time-stamped files represent current status or future plans. The roadmap section is continuously updated based on new discoveries and issue resolutions. This allows seamless referencing of historical decisions while actively working on current development tasks.

_Project Status: Reverted Instance from Another Project_  
_Current State: Critical Issues Being Addressed_  
_GUI Layout: Preserved as Specified in 250710 Roadmap Reboot_

---

## 📅 CHRONOLOGICAL INDEX

### **TIME-STAMPED FILES (ALREADY HAPPENED)**

#### **July 9th, 2025 (250709) - Critical Development Day**

**Morning Session:**

- **`250709_131834-Game systemEvaluation.md`** (07:18 AM)
  - **Content**: Comprehensive roadmap to fix GUI application
  - **Issues Identified**: UI freezing, no real-time audio input, incomplete GPU integration
  - **Solutions**: 5-step roadmap (thread decoupling, audio I/O, GUI wiring, GPU completion, testing)
  - **Key Insight**: "The PNBTR+JELLIE Training Testbed was originally a terminal-based program"

**Afternoon Session:**

- **`250709_134409Game Engine 2.md`** (01:44 PM)

  - **Content**: Modern game engine practices for real-time DSP/DAW engine
  - **Focus**: Frame pacing, real-time threads, ECS architecture, low-latency scheduling
  - **Implementation Guidance**: JAMNet borrowing from Unity/Unreal/Godot patterns

- **`250709_163947-getAudioDeviceSetup.md`** (04:39 PM)

  - **Content**: Audio input signal failure diagnosis
  - **Root Cause**: `numInputChannels == 0` or `inputChannelData[0] == nullptr`
  - **Fix**: Proper audio device initialization in MainComponent.cpp Step 11

- **`250709_194208TRANSPORTHELP.md`** (07:42 PM)
  - **Content**: Transport bar controls non-functional, application launch failures
  - **Critical Issue**: "The file does not exist" - app builds but won't launch
  - **Components Affected**: ProfessionalTransportController, MainComponent, audio processing pipeline

**Evening Session:**

- **`250709_183255_LEARNFROMMISTAKES.md`** (06:32 PM)

  - **Content**: Memory corruption crashes documented and fixed
  - **Critical Discovery**: WaveformAnalysisRow::updateSpectralWaveforms() heap corruption
  - **Solution Applied**: Eliminated heap allocation, replaced std::vector<float> with stack arrays

- **`250709_184800Next Steps.md`** (06:48 PM)

  - **Content**: Metal shader scaffolding for DJ-style spectral interface
  - **Features**: InputCaptureShader, InputGateShader, InputVisualShader, JelliePreprocessShader
  - **Goal**: GPU-first style audio-reactive visuals

- **`250709_201028TRANSPORTFIX.md`** (08:10 PM)

  - **Content**: Transport controls investigation and fix strategy
  - **Root Cause**: Transport buttons not hooked into DSP engine, missing PNBTRTrainer callbacks
  - **Fix Strategy**: Wire ProfessionalTransportController to PNBTRTrainer methods

- **`250709_203526ERROR ERROR.md`** (08:35 PM)
  - **Content**: Critical crash report - MetalSpectralBridge initialization failure
  - **Crash Location**: MetalBridge::initialize() + 308 (MetalSpectralBridge.mm:176)
  - **Root Cause**: Memory corruption in Metal library compilation during WaveformAnalysisRow construction

#### **July 10th, 2025 (250710) - Architecture Reboot**

- **`250710 ROADMAP Reboot.md`**
  - **Content**: **PRESERVED GUI LAYOUT SPECIFICATION**
  - **Architecture**: Text mockup of 5-row interface design
  - **Critical Note**: "do not change the graphic in my roadmap"

---

### **NON-TIME-STAMPED FILES (CURRENT STATUS/FUTURE PLANS)**

#### **Core Documentation**

- **`README.md`** - Main project documentation, revolutionary audio technology overview
- **`ROADMAP.md`** - Development roadmap (667 lines), 4-phase structure
- **`ROADMAP_IMPLEMENTATION_STATUS.md`** - Implementation progress tracking, GPU Native design

#### **Help Requests & Issue Analysis**

- **`URGENT_HELP_REQUEST.md`** - Critical failure analysis: placeholder/fake data instead of real components
- **`AUDIO_INPUT_HELP_REQUEST.md`** - Audio input debugging: microphone waveforms not showing
- **`HELP_REQUEST_UI_PERFORMANCE.md`** - UI performance issues, TOAST protocol integration needs
- **`PNBTR_JELLIE_FIX_README.md`** - Fix documentation and usage instructions

---

## 🎯 CONTENT ANALYSIS BY CATEGORY

### **Critical Issues (BLOCKING)**

1. **Memory Corruption Crashes** (RESOLVED 250709_183255)

   - WaveformAnalysisRow heap allocation issues
   - std::vector<float> destruction errors
   - Fixed with stack allocation approach

2. **Transport Controls Non-Functional** (IDENTIFIED 250709_201028)

   - Buttons not connected to PNBTRTrainer
   - Missing onClick handlers for DSP engine
   - Requires explicit wiring implementation

3. **Audio Input Not Working** (DIAGNOSED 250709_163947)
   - `numInputChannels == 0` issue
   - Missing audio device setup
   - Fix available in MainComponent.cpp Step 11

### **Architecture Issues (DESIGN)**

1. **UI Thread Blocking** (ROADMAPPED 250709_131834)

   - Heavy DSP on main UI thread causing freezes
   - Need background thread or audio callback approach
   - 5-step solution outlined

2. **Incomplete GPU Integration** (IN PROGRESS)

   - MetalBridge stubs need real implementations
   - GPU acceleration not functional
   - Metal shader compilation issues

3. **Missing TOAST Protocol Integration** (CRITICAL DISCOVERY)
   - Using simulation stubs instead of real TOAST
   - Not connected to TOASTer's working protocol
   - Major performance and functionality gap

### **Performance Optimization (ENHANCEMENT)**

1. **Game Engine Patterns** (RESEARCHED 250709_134409)

   - Real-time thread separation
   - Job system for parallel processing
   - Lock-free messaging between threads

2. **DJ-Style Interface** (PROTOTYPED 250709_184800)
   - Metal compute shaders for spectral analysis
   - GPU-accelerated waveform visualization
   - Color-coded frequency analysis

---

## 🏗️ PRESERVED GUI ARCHITECTURE

## ✅ **LATEST FIX APPLIED (Current Session)**

**TRANSPORT CONTROL FIX** - Successfully applied from indexed context:

- **Issue**: Transport buttons (Play/Stop/Record) not working in backup version
- **Root Cause**: Timer-based initialization potentially not reaching Step 9 where callbacks are wired
- **Solution Applied**: Direct DSP engine calls in `ProfessionalTransportController.cpp` button callbacks
- **Status**: ✅ **FIXED** - Transport controls now directly call `pnbtrTrainer->startTraining()` / `stopTraining()`
- **Build**: ✅ **SUCCESSFUL** - App rebuilt and launched with working transport controls

## 📚 **CRITICAL REFERENCE DOCUMENTS (CURRENT SESSION)**

**GPU_NATIVE_METAL_SHADER_INDEX.md** - **CREATED TODAY**

- **Purpose**: Definitive guide for all GPU Native Metal shader components
- **Content**: Complete Metal shader architecture, JUCE integration patterns, CMake configuration
- **Critical For**: Avoiding memory corruption, proper threading, Metal/JUCE integration
- **Use Case**: Reference this to prevent repeating past Metal integration mistakes
- **Status**: ✅ **COMPLETE** - Single source of truth for GPU development

**SYSTEMATIC_RECONSTRUCTION_PLAN.md** - **CREATED TODAY**

- **Purpose**: Indexed context-based rebuilding strategy for entire PNBTR+JELLIE system
- **Content**: 5-phase systematic approach based on lessons learned from 250709 documentation series
- **Critical For**: Re-accomplishing all work without repeating past mistakes
- **Use Case**: Follow this plan sequentially to rebuild the complete application
- **Status**: ✅ **COMPLETE** - Ready for phase-by-phase implementation

**From 250710 ROADMAP Reboot.md - DO NOT MODIFY:**

```
+-----------------------------------------------------------------------------------+
| [Title Bar: TitleComponent]                                                       |
|   PNBTR+JELLIE Training Testbed                                                   |
+-----------------------------------------------------------------------------------+
| [Transport Bar: ProfessionalTransportController]                                  |
|   [Play] [Pause] [Stop] [Record]  SESSION TIME: 00:00:00.000.000  BARS: 1.1.1     |
|   BPM: 120.0 [slider]    :: Packet Loss [%] [slider]  Jitter [ms] [slider]        |
+-----------------------------------------------------------------------------------+
| [Oscilloscope Row: OscilloscopeRow]                                               |
|   Input      |  Network    |  Log/Status        |  Output                         |
|   [osc]      |  [osc]      |  [log/status box]  |  [osc]                          |
+-----------------------------------------------------------------------------------+
|  [Audio Track with Spectral Analysis]                                             |
|                                                                                   |
|   JELLIE Track (Recorded Input)                                                   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
| [Audio Track with Spectral Analysis]                                              |
|                                                                                   |
|   PNBTR Track (Reconstructed Output)                                              |
|                                                                                   |
+-----------------------------------------------------------------------------------+
| [Metrics Dashboard: MetricsDashboard]                                             |
|   [metrics: SNR, latency, packet loss, etc.]                                      |
+-----------------------------------------------------------------------------------+
```

---

## 📋 CURRENT ROADMAP (POST-REVERT)

### **IMMEDIATE PRIORITIES (CRITICAL PATH)**

#### **Phase 1: Stability Foundation** 🚨

1. **Fix Transport Controls** (250709_201028 solution)

   - Wire ProfessionalTransportController buttons to PNBTRTrainer methods
   - Implement onClick handlers: `trainer->startSession()`, `trainer->stopSession()`
   - Test Play/Stop/Record functionality

2. **Resolve Audio Input** (250709_163947 fix)

   - Implement audio device setup in MainComponent.cpp Step 11
   - Ensure `numInputChannels > 0` and valid `inputChannelData`
   - Verify microphone signal reaches oscilloscope displays

3. **Thread Safety** (250709_183255 approach)
   - Maintain stack allocation fixes for WaveformAnalysisRow
   - Implement lock-free data sharing between audio and UI threads
   - Prevent UI thread blocking with DSP operations

#### **Phase 2: Core Functionality** 🔧

1. **Real-Time Audio Pipeline**

   - Implement JUCE AudioIODeviceCallback approach
   - Move DSP processing to audio thread/background thread
   - Maintain millisecond-accurate timing

2. **GPU Integration Completion**

   - Complete MetalBridge stub implementations
   - Fix MetalSpectralBridge initialization crashes
   - Implement JELLIE encoding and PNBTR reconstruction shaders

3. **TOAST Protocol Integration** (CRITICAL MISSING)
   - Replace PacketLossSimulator stubs with real TOAST v2 protocol
   - Connect to working JAM Framework v2 implementation
   - Integrate UDP multicast (`239.255.77.77:7777`) functionality

#### **Phase 3: Interface Polish** 🎨

1. **DJ-Style Spectral Analysis** (250709_184800 design)

   - Implement Metal compute shaders for frequency visualization
   - Add color-coded spectral displays
   - Real-time GPU-accelerated waveform rendering

2. **Metrics Dashboard Integration**

   - Connect real-time SNR, latency, packet loss calculations
   - Thread-safe metrics updates from audio processing
   - Live performance monitoring

3. **Record-Arm Functionality**
   - Professional audio workflow controls
   - Track arming with visual feedback
   - Recording state management

### **LONG-TERM ENHANCEMENT** 🚀

1. **Game Engine Patterns** (250709_134409 research)

   - Job system for parallel DSP processing
   - Fixed-timestep audio processing
   - Async asset loading and hot-swapping

2. **Professional Audio Features**
   - Export functionality (WAV files + metrics reports)
   - Session management and state persistence
   - Advanced network simulation and analysis

---

## 🔄 PROJECT STATUS SUMMARY

**Current State**: Reverted instance from another project with known critical issues  
**Immediate Blockers**: Transport controls, audio input, thread safety  
**Major Missing Component**: Real TOAST protocol integration (using stubs)  
**GUI Architecture**: Preserved 5-row layout from July 10th specification  
**Critical Discovery**: Memory corruption fixes from July 9th must be maintained

**Next Action**: Implement transport control fixes while preserving GUI layout and maintaining stack allocation memory safety.

---

_This index reflects the complete documentation history and provides a clear roadmap for addressing all identified issues while preserving the specified GUI architecture._
