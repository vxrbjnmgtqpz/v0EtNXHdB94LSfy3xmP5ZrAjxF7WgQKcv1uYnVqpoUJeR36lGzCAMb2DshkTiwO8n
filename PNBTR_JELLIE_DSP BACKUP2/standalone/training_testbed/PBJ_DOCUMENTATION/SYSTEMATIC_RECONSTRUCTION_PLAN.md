# SYSTEMATIC RECONSTRUCTION PLAN

**PNBTR+JELLIE Training Testbed - Indexed Context-Based Rebuilding Strategy**

> **📋 MISSION:**  
> Systematically re-accomplish all PNBTR+JELLIE Training Testbed functionality using lessons learned from indexed context system (250709 documentation series). This plan ensures we don't repeat past mistakes and build a stable, high-performance real-time audio application.

---

## 📊 **INDEXED CONTEXT ANALYSIS**

### **Root Cause Analysis (From Documentation Archive)**

| Issue                        | Root Cause                         | Documentation Source                   | Solution                                        |
| ---------------------------- | ---------------------------------- | -------------------------------------- | ----------------------------------------------- |
| **UI Freezing**              | DSP on Message thread              | 250709_131834-Game systemEvaluation.md | Separate audio thread via AudioIODeviceCallback |
| **Memory Crashes**           | std::vector<float> heap corruption | 250709_183255_LEARNFROMMISTAKES.md     | Stack arrays, bounds checking                   |
| **No Audio Input**           | numInputChannels == 0              | 250709_163947-getAudioDeviceSetup.md   | Force device selection, verify setup            |
| **Transport Non-Functional** | Callbacks not wired to DSP         | Transport fix docs                     | Direct DSP engine calls                         |
| **GPU Stubs**                | MetalBridge placeholders           | 250709_131834-Game systemEvaluation.md | Full Metal compute pipeline                     |

### **Architecture Principles (From 250709_134409Game Engine 2.md)**

- **Frame Pacing**: Fixed-timestep audio (real-time) + variable UI (degradable)
- **Thread Separation**: Audio thread ↔ UI thread via lock-free communication
- **ECS Design**: Modular DSP nodes with clear data flow
- **Hot-Swapping**: Live parameter updates without audio interruption
- **Resource Management**: Pre-allocated buffers, no real-time allocation

---

## 🎯 **PHASE 1: FOUNDATION (CRITICAL PATH)**

### **1.1 Memory-Safe Architecture**

**Priority: CRITICAL - Must complete before any GUI work**

```cpp
// Stack-based audio buffers (NEVER std::vector in real-time)
class SafeAudioBuffers {
    static constexpr size_t MAX_BUFFER_SIZE = 4096;
    float inputBuffer[MAX_BUFFER_SIZE];
    float outputBuffer[MAX_BUFFER_SIZE];
    float processingBuffer[MAX_BUFFER_SIZE];

public:
    void processBlock(const float* input, float* output, int numSamples) {
        const int safeSize = std::min(numSamples, static_cast<int>(MAX_BUFFER_SIZE));
        std::memcpy(inputBuffer, input, safeSize * sizeof(float));
        // Process...
        std::memcpy(output, outputBuffer, safeSize * sizeof(float));
    }
};
```

**Acceptance Criteria:**

- [ ] All audio buffers use stack allocation
- [ ] Bounds checking on every buffer operation
- [ ] Zero heap allocation in audio callback
- [ ] Memory safety unit tests pass

### **1.2 Threading Architecture**

**Priority: CRITICAL - Required for real-time performance**

```cpp
class ThreadSafeAudioEngine : public juce::AudioIODeviceCallback {
public:
    void audioDeviceIOCallback(const float** input, int numInputChannels,
                              float** output, int numOutputChannels,
                              int numSamples) override {
        // CRITICAL: Real-time thread - no locks, no allocation
        if (numInputChannels == 0 || !input[0]) {
            // Handle missing input gracefully
            std::memset(output[0], 0, numSamples * sizeof(float));
            return;
        }

        // Update from atomic parameters
        float currentGain = gainLevel.load();
        float currentLoss = packetLossPercentage.load();

        // Process with Metal GPU pipeline
        processAudioBlock(input, output, numSamples);
    }

    // Thread-safe parameter updates from UI
    void setGainLevel(float gain) { gainLevel.store(gain); }
    void setPacketLoss(float loss) { packetLossPercentage.store(loss); }

private:
    std::atomic<float> gainLevel{1.0f};
    std::atomic<float> packetLossPercentage{2.0f};
    std::atomic<bool> recordingActive{false};
};
```

**Acceptance Criteria:**

- [ ] Audio callback never blocks on locks
- [ ] UI ↔ Audio communication via atomics only
- [ ] No UI freezing during audio processing
- [ ] Real-time thread priority verified

### **1.3 Audio Device Management**

**Priority: HIGH - Required for live input**

Based on **250709_163947-getAudioDeviceSetup.md** analysis:

```cpp
class AudioDeviceManager {
public:
    bool initializeAudioDevice() {
        auto setup = deviceManager.getAudioDeviceSetup();

        // Force input device selection if none chosen
        if (setup.inputDeviceName.isEmpty()) {
            auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
            auto inputs = deviceType->getDeviceNames(true);
            if (!inputs.isEmpty()) {
                setup.inputDeviceName = inputs[0];
                juce::Logger::writeToLog("Auto-selected input: " + setup.inputDeviceName);
            }
        }

        auto result = deviceManager.setAudioDeviceSetup(setup, true);

        // Verify input channels are available
        auto* device = deviceManager.getCurrentAudioDevice();
        if (!device || device->getActiveInputChannels().isEmpty()) {
            juce::Logger::writeToLog("ERROR: No active input channels");
            return false;
        }

        juce::Logger::writeToLog("Audio device initialized successfully");
        return true;
    }
};
```

**Acceptance Criteria:**

- [ ] Audio input device automatically selected
- [ ] numInputChannels > 0 verified
- [ ] Live microphone signal reaching oscilloscopes
- [ ] Audio device errors handled gracefully

---

## 🎛️ **PHASE 2: GPU NATIVE METAL PIPELINE**

### **2.1 MetalBridge Architecture**

**Priority: HIGH - Core performance component**

Following **GPU_NATIVE_METAL_SHADER_INDEX.md**:

```cpp
class MetalBridge {
public:
    static MetalBridge& getInstance() {
        static MetalBridge instance;
        return instance;
    }

    bool initialize() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return false;

        commandQueue = [device newCommandQueue];
        library = [device newDefaultLibrary];

        // Load all shader kernels
        if (!loadAllKernels()) return false;

        // Pre-allocate all GPU buffers
        if (!allocateGPUBuffers()) return false;

        isInitialized = true;
        return true;
    }

    void processAudioBlock(const juce::AudioBuffer<float>& inputBuffer) {
        if (!isInitialized) return;

        // GPU pipeline execution
        captureAudioToGPU(inputBuffer);
        preprocessWithJellie();
        simulateNetworkLoss();
        reconstructWithPNBTR();
        updateSpectralVisuals();
    }

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    std::atomic<bool> isInitialized{false};

    // Pre-allocated GPU buffers
    id<MTLBuffer> audioInputBuffer;
    id<MTLBuffer> audioOutputBuffer;
    id<MTLBuffer> spectrumBuffer;
    id<MTLBuffer> jellieBuffer;
};
```

### **2.2 Metal Shader Implementation**

**Priority: HIGH - Real-time audio processing**

**File Structure:**

```
PNBTR-JELLIE-TRAINER/
├── shaders/
│   ├── InputCaptureShader.metal      # Audio input capture
│   ├── SpectrumAnalysisShader.metal  # Real-time FFT
│   ├── JelliePreprocessShader.metal  # JELLIE encoding
│   ├── NetworkSimulationShader.metal # Packet loss simulation
│   └── PNBTRReconstructionShader.metal # Neural reconstruction
├── Source/
│   ├── GPU/
│   │   ├── MetalBridge.h/.mm
│   │   ├── MetalShaderManager.h/.cpp
│   │   └── GPUBufferManager.h/.cpp
│   └── ...
```

**Acceptance Criteria:**

- [ ] All Metal shaders compile to .metallib
- [ ] GPU pipeline processes audio in <2ms latency
- [ ] Spectral analysis updates at 60fps
- [ ] Neural reconstruction maintains audio quality
- [ ] GPU memory usage optimized

### **2.3 CMake Integration**

**Priority: MEDIUM - Build system reliability**

```cmake
# Metal shader compilation
function(compile_metal_shaders)
    file(GLOB METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.metal")

    foreach(METAL_FILE ${METAL_SOURCES})
        get_filename_component(SHADER_NAME ${METAL_FILE} NAME_WE)
        set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.metallib")

        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND xcrun -sdk macosx metal -c ${METAL_FILE} -o ${OUTPUT_FILE}
            DEPENDS ${METAL_FILE}
            COMMENT "Compiling Metal shader: ${SHADER_NAME}"
        )

        list(APPEND COMPILED_SHADERS ${OUTPUT_FILE})
    endforeach()

    add_custom_target(CompileMetalShaders ALL DEPENDS ${COMPILED_SHADERS})
endfunction()
```

**Acceptance Criteria:**

- [ ] Metal shaders compile before C++ compilation
- [ ] .metallib files bundled in app package
- [ ] Metal framework linked correctly
- [ ] Build system validates shader compilation

---

## 🎵 **PHASE 3: PROFESSIONAL AUDIO WORKFLOW**

### **3.1 Transport Controls**

**Priority: HIGH - Core user interaction**

Based on previous transport fix success:

```cpp
class ProfessionalTransportController {
public:
    void playButtonClicked() {
        if (pnbtrTrainer) {
            pnbtrTrainer->startTraining();
            juce::Logger::writeToLog("[TRANSPORT] Play -> startTraining()");
        }
        if (onPlay) onPlay();
        updateTransportState(TransportState::Playing);
    }

    void stopButtonClicked() {
        if (pnbtrTrainer) {
            pnbtrTrainer->stopTraining();
            juce::Logger::writeToLog("[TRANSPORT] Stop -> stopTraining()");
        }
        if (onStop) onStop();
        updateTransportState(TransportState::Stopped);
    }

    void recordButtonClicked() {
        if (pnbtrTrainer) {
            bool newState = !pnbtrTrainer->isRecording();
            pnbtrTrainer->setRecording(newState);
            juce::Logger::writeToLog("[TRANSPORT] Record -> " + juce::String(newState));
        }
        updateTransportState(TransportState::Recording);
    }
};
```

**Acceptance Criteria:**

- [ ] Play button starts audio processing immediately
- [ ] Stop button halts processing cleanly
- [ ] Record button arms tracks for capture
- [ ] Transport state syncs with audio engine
- [ ] Visual feedback shows current state

### **3.2 Record-Arm System**

**Priority: HIGH - Professional workflow**

```cpp
class RecordArmableTrack : public juce::Component {
public:
    void setRecordArmed(bool armed) {
        isRecordArmed = armed;

        // Update Metal shader uniforms
        if (metalBridge) {
            metalBridge->setTrackRecordArmed(trackIndex, armed);
        }

        // Visual feedback
        armButton.setToggleState(armed, juce::dontSendNotification);
        armButton.setColour(juce::TextButton::buttonColourId,
                           armed ? juce::Colours::red : juce::Colours::darkred);

        repaint();
    }

    void paint(juce::Graphics& g) override {
        if (isRecordArmed) {
            g.setColour(juce::Colours::red);
            g.fillRect(getLocalBounds().removeFromLeft(20));
            g.setColour(juce::Colours::white);
            g.drawText("[REC]", armRect, juce::Justification::centred);
        }
    }

private:
    bool isRecordArmed = false;
    juce::TextButton armButton{"R"};
    juce::Rectangle<int> armRect;
    int trackIndex;
};
```

**Acceptance Criteria:**

- [ ] Each track has independent record arm button
- [ ] Armed tracks show red "[REC]" indicator
- [ ] Record arm state connects to Metal GPU capture
- [ ] Only armed tracks capture to recording buffer
- [ ] Professional DAW-style visual feedback

### **3.3 Real-Time Visualization**

**Priority: MEDIUM - User feedback**

```cpp
class SpectralWaveformComponent : public juce::Component, public juce::Timer {
public:
    void setTrainer(PNBTRTrainer* trainer) {
        pnbtrTrainer = trainer;
        startTimer(16); // 60 FPS - safe update rate
    }

    void timerCallback() override {
        if (!pnbtrTrainer) return;

        // Thread-safe data access
        if (isRecordArmed && pnbtrTrainer->isRecording()) {
            updateFromRecordedBuffer();
        } else {
            updateFromLiveOscilloscope();
        }

        if (dataChanged.load()) {
            repaint();
            dataChanged.store(false);
        }
    }

private:
    void updateFromRecordedBuffer() {
        // Safe buffer access - no heap allocation
        float buffer[4096];
        int samplesRead = pnbtrTrainer->getRecordedSamples(buffer, 4096);
        updateDisplayBuffer(buffer, samplesRead);
    }

    std::atomic<bool> dataChanged{false};
    PNBTRTrainer* pnbtrTrainer = nullptr;
    bool isRecordArmed = false;
};
```

**Acceptance Criteria:**

- [ ] Real-time waveform updates at 60fps
- [ ] Spectral analysis shows frequency content
- [ ] DJ-style color coding for frequency bands
- [ ] No visual glitches or tearing
- [ ] Performance doesn't affect audio processing

---

## 🎛️ **PHASE 4: REAL-TIME DSP PIPELINE**

### **4.1 JELLIE Encoder Integration**

**Priority: HIGH - Core algorithm**

```cpp
class JELLIEEncoder {
public:
    void processBlock(const float* input, int numSamples) {
        // Metal GPU processing
        MetalBridge::getInstance().jellieEncode(input, numSamples);

        // Update metrics
        encodingLatency = measureLatency();
        compressionRatio = calculateCompressionRatio();
    }

    void getEncodedOutput(float* output, int& outputSize) {
        MetalBridge::getInstance().getJellieOutput(output, outputSize);
    }

private:
    std::atomic<float> encodingLatency{0.0f};
    std::atomic<float> compressionRatio{1.0f};
};
```

### **4.2 Network Loss Simulation**

**Priority: HIGH - Training environment**

```cpp
class NetworkSimulator {
public:
    void setPacketLossPercentage(float percentage) {
        packetLossPercentage.store(percentage);
        MetalBridge::getInstance().updateNetworkSimParams(percentage, jitterMs.load());
    }

    void setJitterAmount(float jitterMs) {
        this->jitterMs.store(jitterMs);
        MetalBridge::getInstance().updateNetworkSimParams(packetLossPercentage.load(), jitterMs);
    }

    void processBlock(const float* input, float* output, int numSamples) {
        MetalBridge::getInstance().simulateNetworkLoss(input, output, numSamples);
        updateLossStatistics();
    }

private:
    std::atomic<float> packetLossPercentage{2.0f};
    std::atomic<float> jitterMs{1.0f};
};
```

### **4.3 PNBTR Neural Reconstruction**

**Priority: HIGH - Core algorithm**

```cpp
class PNBTRDecoder {
public:
    void processBlock(const float* lossyInput, float* reconstructedOutput, int numSamples) {
        MetalBridge::getInstance().pnbtrReconstruct(lossyInput, reconstructedOutput, numSamples);

        // Calculate quality metrics
        updateQualityMetrics(reconstructedOutput, numSamples);
    }

    float getCurrentSNR() const { return snrDb.load(); }
    float getReconstructionLatency() const { return latencyMs.load(); }

private:
    std::atomic<float> snrDb{20.0f};
    std::atomic<float> latencyMs{1.5f};
};
```

**Acceptance Criteria:**

- [ ] JELLIE encoding maintains audio quality
- [ ] Network simulation creates realistic packet loss
- [ ] PNBTR reconstruction achieves target SNR
- [ ] End-to-end latency < 5ms
- [ ] Real-time metrics update continuously

---

## 📊 **PHASE 5: TESTING & VALIDATION**

### **5.1 Automated Testing Framework**

**Priority: MEDIUM - Reliability assurance**

```python
# Python GUI automation test
import pyautogui
import subprocess
import time

class PNBTRTestFramework:
    def test_transport_controls(self):
        """Test transport button functionality"""
        # Launch application
        app_process = subprocess.Popen(['./PNBTR+JELLIE Training Testbed.app/Contents/MacOS/PNBTR+JELLIE Training Testbed'])
        time.sleep(3)

        # Test play button
        play_button = pyautogui.locateOnScreen('play_button.png')
        pyautogui.click(play_button)

        # Verify audio processing started
        assert self.check_audio_activity(), "Audio processing not started"

        # Test stop button
        stop_button = pyautogui.locateOnScreen('stop_button.png')
        pyautogui.click(stop_button)

        # Verify audio processing stopped
        assert not self.check_audio_activity(), "Audio processing not stopped"

    def test_record_arm_functionality(self):
        """Test record arm button functionality"""
        # Test each track's record arm button
        for track_index in range(2):  # Input and Reconstructed tracks
            arm_button = pyautogui.locateOnScreen(f'track_{track_index}_arm.png')
            pyautogui.click(arm_button)

            # Verify visual feedback
            assert self.check_record_indicator(track_index), f"Track {track_index} not showing record indicator"
```

### **5.2 Performance Profiling**

**Priority: MEDIUM - Optimization**

```cpp
class PerformanceProfiler {
public:
    void profileAudioCallback(const std::function<void()>& callback) {
        auto start = std::chrono::high_resolution_clock::now();
        callback();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        audioCallbackLatency.store(duration.count());

        // Alert if callback takes too long
        if (duration.count() > MAX_CALLBACK_MICROSECONDS) {
            juce::Logger::writeToLog("WARNING: Audio callback overrun: " +
                                   juce::String(duration.count()) + "μs");
        }
    }

private:
    static constexpr int MAX_CALLBACK_MICROSECONDS = 2000; // 2ms at 48kHz
    std::atomic<int> audioCallbackLatency{0};
};
```

**Acceptance Criteria:**

- [ ] Audio callback never exceeds 2ms
- [ ] GPU processing completes within buffer time
- [ ] Memory usage remains stable over time
- [ ] No audio dropouts or glitches
- [ ] CPU usage optimized for real-time performance

---

## 📋 **IMPLEMENTATION SCHEDULE**

### **Week 1: Foundation (CRITICAL PATH)**

- [ ] **Day 1-2**: Memory-safe architecture implementation
- [ ] **Day 3-4**: Threading architecture and audio device management
- [ ] **Day 5**: Foundation testing and validation

### **Week 2: GPU Pipeline**

- [ ] **Day 1-2**: MetalBridge implementation and shader compilation
- [ ] **Day 3-4**: Metal shader implementation (all 4 core shaders)
- [ ] **Day 5**: GPU pipeline testing and optimization

### **Week 3: Audio Workflow**

- [ ] **Day 1-2**: Transport controls and record-arm system
- [ ] **Day 3-4**: Real-time visualization components
- [ ] **Day 5**: Professional workflow testing

### **Week 4: DSP Pipeline**

- [ ] **Day 1-2**: JELLIE encoder and network simulation
- [ ] **Day 3-4**: PNBTR neural reconstruction
- [ ] **Day 5**: End-to-end pipeline testing

### **Week 5: Validation**

- [ ] **Day 1-2**: Automated testing framework
- [ ] **Day 3-4**: Performance profiling and optimization
- [ ] **Day 5**: Final validation and documentation

---

## 🚨 **CRITICAL SUCCESS FACTORS**

### **Must-Have Requirements**

1. **Zero Heap Allocation** in audio callback thread
2. **Real-time Audio Processing** without dropouts
3. **Thread-Safe Communication** between UI and audio
4. **Professional Transport Controls** that work immediately
5. **GPU-Accelerated DSP Pipeline** for performance

### **Risk Mitigation**

1. **Backup Plan**: CPU fallback if Metal initialization fails
2. **Incremental Testing**: Validate each phase before proceeding
3. **Documentation**: Reference GPU_NATIVE_METAL_SHADER_INDEX.md
4. **Version Control**: Branch for each phase implementation
5. **Performance Monitoring**: Continuous profiling during development

### **Definition of Done**

- [ ] Application starts quickly (<3 seconds)
- [ ] Transport controls respond immediately
- [ ] Record arm buttons function correctly
- [ ] Real-time audio processing without glitches
- [ ] GPU pipeline processes audio efficiently
- [ ] All automated tests pass
- [ ] Documentation updated and complete

---

**This systematic reconstruction plan is based on indexed context analysis from all July 9th documentation. Follow this plan sequentially to avoid repeating past mistakes and achieve a stable, high-performance PNBTR+JELLIE Training Testbed.**
