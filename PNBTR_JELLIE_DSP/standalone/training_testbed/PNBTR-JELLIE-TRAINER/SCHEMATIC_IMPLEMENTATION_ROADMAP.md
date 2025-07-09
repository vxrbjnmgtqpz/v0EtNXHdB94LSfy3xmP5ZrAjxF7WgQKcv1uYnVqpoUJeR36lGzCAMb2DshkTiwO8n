# PNBTR+JELLIE Training Testbed - Schematic Implementation Roadmap

## Target Layout (From User Specification)

```
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
```

## Current State Analysis

### What EXISTS:

- ✅ Basic MainComponent structure
- ✅ OscilloscopeComponent class (with MetalBridge integration)
- ✅ MetricsDashboard class (6 metrics: SNR, THD, Latency, ReconRate, GapFill, Quality)
- ✅ SessionManager for start/stop/export functionality
- ✅ MetalBridge for GPU buffer management
- ✅ JUCE build system with proper module linking

### What's WRONG with Current Implementation:

- ❌ Layout doesn't match schematic proportions
- ❌ No detailed text descriptions in each section
- ❌ Missing log/status window functionality
- ❌ Audio tracks are placeholder components, not real JUCE AudioThumbnails
- ❌ No visual separation between rows
- ❌ Missing technical annotations and flow indicators
- ❌ Controls row layout doesn't match specification

## Implementation Roadmap

### Phase 1: Fix Layout Structure (Foundation)

**Goal**: Make the basic 5-row layout match the schematic proportions exactly

#### Step 1.1: Analyze Current Layout Issues

- Current MainComponent uses arbitrary heights
- No proper visual row separators
- Components don't fill their allocated space properly
- Need to match the schematic's visual proportions

#### Step 1.2: Implement Exact Row Heights

- **Title Row**: 40px (matches schematic header)
- **Oscilloscope Row**: 200px (largest section, needs space for technical annotations)
- **Waveform Analysis Row**: 120px (medium height for dual oscilloscopes)
- **Audio Tracks Row**: 80px (compact for audio thumbnails)
- **Metrics Row**: 100px (space for 6 metrics with progress bars)
- **Controls Row**: 60px (compact for buttons and sliders)

#### Step 1.3: Add Visual Row Separators

- Draw horizontal lines between each row
- Use proper colors: `juce::Colour(0xff444444)` for separators
- Add subtle background color differences per row

### Phase 2: Row 1 - Four-Panel Oscilloscope Section

**Goal**: Implement the exact 4-panel layout with technical annotations

#### Step 2.1: Fix Oscilloscope Layout

- Divide row into exactly 4 equal sections
- Add 5px spacing between panels
- Ensure each oscilloscope fills its allocated space

#### Step 2.2: Add Technical Annotations

Each oscilloscope needs descriptive text overlay:

- **Input (Mic)**:
  - "(1) CoreAudio input callback (mic)"
  - "→ (2) JELLIE encode (48kHz→192kHz, 8ch)"
- **Network Sim**:
  - "(3) Simulate packet loss, jitter"
  - "→ (5) update network stats"
- **Log/Status**:
  - "(6) Log events, errors, metrics"
  - Real scrolling text log component
- **Output (Reconstructed)**:
  - "(4) PNBTR neural reconstruction (output buffer)"

#### Step 2.3: Implement Real Log/Status Window

- Create `LogStatusComponent` class
- Scrolling text display
- Real-time event logging
- Connect to SessionManager events

### Phase 3: Row 2 - Waveform Analysis Section

**Goal**: Implement dual oscilloscope comparison with proper labeling

#### Step 3.1: Create Dual Oscilloscope Layout

- Split row into 2 equal sections
- Left: "Original Waveform Oscilloscope (inputBuffer, real mic data)"
- Right: "Reconstructed Waveform Oscilloscope (reconstructedBuffer, after PNBTR)"

#### Step 3.2: Add Technical Annotations

- "updateOscilloscope(inputBuffer)" label on left
- "updateOscilloscope(reconstructedBuffer)" label on right
- "[src/oscilloscope/ or GUI class]" reference text

#### Step 3.3: Implement Real-Time Buffer Comparison

- Connect left oscilloscope to `audioInputBuffer`
- Connect right oscilloscope to `reconstructedBuffer`
- Synchronized time scales for comparison

### Phase 4: Row 3 - JUCE Audio Tracks Section

**Goal**: Implement real JUCE AudioThumbnail components

#### Step 4.1: Create Real AudioThumbnail Components

- Replace placeholder components with `juce::AudioThumbnail`
- Implement `juce::AudioThumbnailComponent` wrapper
- Add `juce::AudioFormatManager` and `juce::AudioThumbnailCache`

#### Step 4.2: Add Technical Labels

- Left: "JUCE::AudioThumbnail: JELLIE Track (recorded input, .wav)"
- Right: "JUCE::AudioThumbnail: PNBTR Track (reconstructed output, .wav)"
- Add transport controls: "JUCE::AudioTransportSource, JUCE::AudioFormatManager"

#### Step 4.3: Connect to Recording System

- Link to SessionManager's recording functionality
- Auto-update thumbnails when recording starts/stops
- Add playback controls

### Phase 5: Row 4 - Metrics Dashboard Enhancement

**Goal**: Ensure metrics display matches schematic exactly

#### Step 5.1: Verify Metrics Layout

- Confirm 6 metrics display horizontally: SNR, THD, Latency, ReconRate, GapFill, Quality
- Add progress bars/value displays as shown in schematic
- Ensure proper spacing and alignment

#### Step 5.2: Add Technical Annotations

- Add function name labels: "calculateSNR()", "calculateTHD()", etc.
- Add "updateMetricDisplay(metric, value, bar)" reference
- Add "[src/metrics/metrics.cpp, .h]" file reference

#### Step 5.3: Connect to Real Calculations

- Ensure MetricsDashboard connects to actual metric calculations
- Real-time updates during processing
- Proper value formatting and units

### Phase 6: Row 5 - Controls Section Enhancement

**Goal**: Match the exact control layout from schematic

#### Step 6.1: Fix Button Layout

- Three buttons: [Start] [Stop] [Export]
- Proper spacing and sizing
- Add technical annotations: "startAudio()", "stopAudio()", "exportWAV()"

#### Step 6.2: Fix Slider Layout

- Three sliders: Packet Loss, Jitter, Gain
- Horizontal layout with proper labels
- Add technical annotations: "setPacketLoss()", "setJitter()", "setGain()"

#### Step 6.3: Add File Reference

- Add "[src/gui/controls.cpp, .h]" reference text
- Connect to SessionManager parameter updates

### Phase 7: Visual Polish and Technical Annotations

**Goal**: Add all the technical details shown in the schematic

#### Step 7.1: Add File Path References

- Add the long file path reference in Row 1: "[PNBTR_JELLIE_DSP/standalone/vst3_plugin/src/PnbtrJellieGUIApp_Fixed.mm]"
- Add other file references as shown in schematic

#### Step 7.2: Add Flow Indicators

- Add "→" arrows showing data flow
- Add numbered process indicators: (1), (2), (3), etc.
- Match the exact technical flow shown in schematic

#### Step 7.3: Add Technical Descriptions

- Add all the technical descriptions exactly as shown
- Proper font sizing and positioning
- Use monospace font for code references

### Phase 8: Integration and Testing

**Goal**: Ensure everything works together as a cohesive system

#### Step 8.1: Test All Components

- Verify each row displays correctly
- Test real-time updates
- Verify MetalBridge integration

#### Step 8.2: Test Processing Flow

- Start/Stop functionality
- Real-time oscilloscope updates
- Metrics calculations
- Audio recording and playback

#### Step 8.3: Final Visual Verification

- Compare with original schematic
- Ensure exact proportions and layout
- Verify all technical annotations are present

## Technical Implementation Notes

### Key Classes to Modify:

1. **MainComponent**: Fix layout and add technical annotations
2. **OscilloscopeComponent**: Already exists, ensure proper integration
3. **MetricsDashboard**: Already exists, verify layout
4. **LogStatusComponent**: NEW - needs to be created
5. **AudioThumbnailComponent**: NEW - wrapper for JUCE AudioThumbnail

### Key Files to Create:

1. `LogStatusComponent.h/.cpp` - Real-time logging display
2. `AudioThumbnailComponent.h/.cpp` - JUCE audio track wrapper
3. Enhanced `MainComponent` with proper layout and annotations

### Dependencies Already Available:

- ✅ JUCE framework with all required modules
- ✅ MetalBridge for GPU integration
- ✅ SessionManager for control flow
- ✅ OscilloscopeComponent for waveform display
- ✅ MetricsDashboard for metrics display

### Critical Success Factors:

1. **Exact Layout Match**: Every row must match the schematic proportions
2. **Technical Annotations**: All the technical text must be present and accurate
3. **Real Functionality**: Not just visual - must actually work
4. **Integration**: All components must work together seamlessly

## Next Steps

1. **Review this roadmap** - Confirm the approach is correct
2. **Implement Phase 1** - Fix the basic layout structure first
3. **Proceed incrementally** - One phase at a time, testing each step
4. **Verify against schematic** - Constant comparison to ensure accuracy

This roadmap ensures we build exactly what you've specified, with real functionality, proper technical annotations, and the exact visual layout from your schematic.
