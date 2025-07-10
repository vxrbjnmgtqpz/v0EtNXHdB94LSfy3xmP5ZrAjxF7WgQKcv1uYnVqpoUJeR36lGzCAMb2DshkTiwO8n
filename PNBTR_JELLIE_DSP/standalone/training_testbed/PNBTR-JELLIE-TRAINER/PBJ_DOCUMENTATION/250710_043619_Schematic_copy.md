+--------------------------------------------------------------------------------------+
|                                PNBTR+JELLIE Training Testbed                         |
+--------------------------------------------------------------------------------------+
|  Input (Mic)   |   Network Sim   |      Log/Status      |   Output (Reconstructed)   |
|--------------- |----------------|----------------------|-----------------------------|
|  [Oscilloscope]| [Oscilloscope] |   [Log Window]       |   [Oscilloscope]            |
|                |                |                      |                             |
|                |                |                      |                             |
|  (1) CoreAudio |  (3) Simulate  |  (6) Log events,     |  (4) PNBTR neural           |
|      input     |      packet    |      errors,         |      reconstruction         |
|      callback  |      loss,     |      metrics         |      (output buffer)        |
|      (mic)     |      jitter    |                      |                             |
|      →         |      →         |                      |                             |
|      (2) JELLIE|      (5)       |                      |                             |
|      encode    |      update    |                      |                             |
|      (48kHz→   |      network   |                      |                             |
|      192kHz,   |      stats     |                      |                             |
|      8ch)      |                |                      |                             |
|                |                |                      |                             |
|  [PNBTR_JELLIE_DSP/standalone/vst3_plugin/src/PnbtrJellieGUIApp_Fixed.mm]           |
+--------------------------------------------------------------------------------------+
|                                Waveform Analysis Row                                 |
|   [Original Waveform Oscilloscope]      [Reconstructed Waveform Oscilloscope]        |
|   (inputBuffer, real mic data)          (reconstructedBuffer, after PNBTR)           |
|   updateOscilloscope(inputBuffer)       updateOscilloscope(reconstructedBuffer)      |
|   [src/oscilloscope/ or GUI class]      [src/oscilloscope/ or GUI class]             |
+--------------------------------------------------------------------------------------+
|                        JUCE Audio Tracks: JELLIE & PNBTR Recordings                  |
|   [JUCE::AudioThumbnail: JELLIE Track]     [JUCE::AudioThumbnail: PNBTR Track]       |
|   (recorded input, .wav)                  (reconstructed output, .wav)               |
|   JUCE::AudioTransportSource,             JUCE::AudioTransportSource,                |
|   JUCE::AudioFormatManager                JUCE::AudioFormatManager                   |
|   [PNBTR_JELLIE_DSP/standalone/juce/]                                              |
+--------------------------------------------------------------------------------------+
|                                Metrics Dashboard Row                                 |
| SNR |  THD  | Latency | Recon Rate | Gap Fill | Quality |  [Progress Bars/Values]    |
| (7) calculateSNR()  calculateTHD()  calculateLatency()  calculateReconstructionRate() |
|     calculateGapFillQuality()  calculateOverallQuality()                             |
|   updateMetricDisplay(metric, value, bar)                                            |
|   [src/metrics/metrics.cpp, .h]                                                      |
+--------------------------------------------------------------------------------------+
| [Start] [Stop] [Export] [Sliders: Packet Loss, Jitter, Gain]                         |
| (8) startAudio()  stopAudio()  exportWAV()                                           |
|     setPacketLoss(), setJitter(), setGain()                                          |
|   [src/gui/controls.cpp, .h]                                                         |
+--------------------------------------------------------------------------------------+

Function/Data Flow Wiring (with explicit file/module references):

1. CoreAudio input callback (PnbtrJellieGUIApp_Fixed.mm) captures real mic data → fills inputBuffer (atomic/thread-safe)
2. JELLIE encoder (src/jellie_encoder.cpp, .h) processes inputBuffer → upsample/quantize → jellieEncoded (vector)
3. Network simulation (src/network_sim.cpp, .h) applies packet loss/jitter to jellieEncoded → networkProcessed
4. PNBTR neural reconstruction (src/pnbtr_recon.cpp, .h) processes networkProcessed → fills reconstructedBuffer
5. Network stats (packet loss, jitter) updated for metrics (atomic counters in src/metrics/metrics.cpp)
6. Log/status window (src/gui/log.cpp, .h) receives events from all major steps (errors, state changes, metrics)
7. Metrics dashboard (src/metrics/metrics.cpp, .h) calculates SNR, THD, latency, etc. from inputBuffer & reconstructedBuffer
8. Control buttons (src/gui/controls.cpp, .h) wire to start/stop audio, export WAV, and adjust simulation parameters
9. JUCE::AudioFormatWriter (standalone/juce/recording.cpp) records inputBuffer and reconstructedBuffer to JELLIE and PNBTR .wav files
10. JUCE::AudioThumbnail (standalone/juce/waveform.cpp) displays waveform of recorded tracks; JUCE::AudioTransportSource enables playback

// Each oscilloscope, metric bar, and JUCE audio track must be updated with real data from the processing pipeline.
// All placeholder/fake data sources must be removed and replaced with real function outputs.

=======================================================================================

# PNBTR+JELLIE TRAINER: DEVELOPMENT ROADMAP (EXPLICIT)

## 0. CMake Structure & Build Process
- [ ] All source files must be listed in PNBTR_JELLIE_DSP/standalone/vst3_plugin/CMakeLists.txt:
      - src/PnbtrJellieGUIApp_Fixed.mm
      - src/jellie_encoder.cpp, .h
      - src/network_sim.cpp, .h
      - src/pnbtr_recon.cpp, .h
      - src/metrics/metrics.cpp, .h
      - src/gui/controls.cpp, .h
      - src/gui/log.cpp, .h
      - standalone/juce/recording.cpp, waveform.cpp
- [ ] CMake options must be set:
      -DUSE_REAL_PROCESSING=ON -DDISABLE_PLACEHOLDER_DATA=ON
- [ ] All required frameworks and libraries must be linked:
      - CoreAudio, AudioUnit, AVFoundation, Cocoa, Foundation, JUCE modules
- [ ] Build steps:
      1. cd PNBTR_JELLIE_DSP/standalone/vst3_plugin
      2. rm -rf build
      3. mkdir -p build && cd build
      4. cmake -DUSE_REAL_PROCESSING=ON -DDISABLE_PLACEHOLDER_DATA=ON ..
      5. make -j$(sysctl -n hw.ncpu)
- [ ] After build, verify:
      - pnbtr_jellie_gui_app.app/Contents/MacOS/pnbtr_jellie_gui_app exists
      - Binary size > 100KB (indicates real code is included)
      - No build errors or missing symbols

## 1. Core Audio Input & Output
- [ ] Implement CoreAudio input callback for real-time mic capture (inputBuffer, atomic)
- [ ] Implement CoreAudio output for playback (reconstructedBuffer, atomic)
- [ ] Ensure thread safety for audio buffers (use std::atomic, std::mutex)
- [ ] File: src/PnbtrJellieGUIApp_Fixed.mm

## 2. JELLIE Encoder
- [ ] Implement 48kHz→192kHz upsampling and 24-bit quantization (src/jellie_encoder.cpp)
- [ ] Distribute samples over 8 JDAT channels
- [ ] Integrate with inputBuffer and output to jellieEncoded
- [ ] All placeholder code must be removed

## 3. Network Simulation
- [ ] Simulate packet loss (2%) and jitter on jellieEncoded (src/network_sim.cpp)
- [ ] Update networkProcessed buffer
- [ ] Expose packet loss/jitter controls to UI (src/gui/controls.cpp)

## 4. PNBTR Neural Reconstruction
- [ ] Implement neural prediction for lost packets (src/pnbtr_recon.cpp)
- [ ] Map networkProcessed back to 48kHz (reconstructedBuffer)
- [ ] Integrate with metrics and oscilloscope

## 5. JUCE Audio Recording & Playback
- [ ] Use JUCE::AudioFormatWriter to record inputBuffer (JELLIE) and reconstructedBuffer (PNBTR) to .wav (standalone/juce/recording.cpp)
- [ ] Use JUCE::AudioThumbnail to display recorded waveforms (standalone/juce/waveform.cpp)
- [ ] Use JUCE::AudioTransportSource for playback controls

## 6. Metrics & Analysis
- [ ] Implement SNR, THD, latency, reconstruction rate, gap fill, and quality calculations (src/metrics/metrics.cpp)
- [ ] Update metrics dashboard in real time (src/gui/metrics_display.cpp)

## 7. UI Wiring & Controls
- [ ] Wire up all oscilloscopes to real buffers (src/oscilloscope/oscilloscope.cpp)
- [ ] Wire up metrics bars to real calculations (src/gui/metrics_display.cpp)
- [ ] Wire up log/status window to all major events (src/gui/log.cpp)
- [ ] Implement Start/Stop/Export buttons (src/gui/controls.cpp)
- [ ] Implement sliders for packet loss, jitter, gain (src/gui/controls.cpp)

## 8. Export & Reporting
- [ ] Export .wav files and metrics report (timestamped, standalone/juce/recording.cpp)
- [ ] Ensure all exports use real data, not placeholders

## 9. Testing & Validation
- [ ] Test end-to-end: mic input → JELLIE → network → PNBTR → output
- [ ] Validate all metrics and recordings are correct
- [ ] Remove all placeholder/fake data code (search for: placeholder, fake, demo, test pattern)
- [ ] Run with debug logging enabled and verify all steps in the log

---

**Tips for a Smooth Build and Integration:**
- Every function and buffer must be implemented in the correct file/module as listed above.
- All CMakeLists.txt entries must match the actual file locations and names.
- All real-time data must use atomic/thread-safe containers.
- All UI updates must be dispatched to the main thread (use dispatch_async or JUCE MessageThread).
- All error and state changes must be logged to the log/status window.
- All placeholder/fake/demo/test pattern code must be removed before final build.
- After every build, run the app and verify:
    - Real mic input is visible in the input oscilloscope
    - Network simulation is active and adjustable
    - PNBTR output oscilloscope shows reconstructed waveform
    - Metrics dashboard updates in real time with plausible values
    - Recording and export functions produce valid .wav files and reports
    - All logs reflect real events and errors
- If any step fails, check the log window, console output, and CMake build output for errors.
- Use `git grep` or IDE search to ensure no placeholder/fake/demo code remains.
- Document any deviations from this schematic/roadmap in the README.