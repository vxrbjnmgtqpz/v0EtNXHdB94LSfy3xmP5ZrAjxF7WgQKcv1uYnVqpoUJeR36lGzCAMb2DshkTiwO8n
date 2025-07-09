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

Understood. I’ll focus first on producing a clear, precise implementation guide for what you’ve already built in the PNBTR_JELLIE_DSP repo — removing AI slop and ensuring detailed, non-handwavey context for building the app cleanly from the schematic and existing source.

Once that’s locked in, I’ll optionally follow up with a supplemental roadmap for the transport-phase GPU prioritization strategy, but only if it doesn’t compromise the clarity of the main build guide.

I’ll begin the research and get back to you shortly with a focused implementation breakdown.

# PNBTR+JELLIE Training Testbed Implementation Guide

This guide provides a detailed blueprint for implementing the **PNBTR+JELLIE Training Testbed** application exactly as specified. It is organized by each major UI **row** (section) according to the provided schematic layout and development roadmap. All module names, file paths, and function names are taken directly from the specification to ensure precise alignment with the planned architecture. No extraneous features or deviations are introduced beyond what the schematic and roadmap define. The focus is on exact component behavior, integration points, and layout details, with zero tolerance for ambiguity.

## Overall Layout and UI Structure (Foundation)

- **Fixed Row Heights:** Configure the main application window (`MainComponent`, defined in `PnbtrJellieGUIApp_Fixed.mm`) to contain a top title bar and five content rows stacked vertically. Assign each section an explicit height (in pixels) to match the schematic proportions:

  - _Title Bar:_ 40 px height (top header area).
  - _Oscilloscope Row:_ 200 px (Row 1 – the largest section for the four-panel oscilloscope and annotations).
  - _Waveform Analysis Row:_ 120 px (Row 2 – dual waveform comparison section).
  - _Audio Tracks Row:_ 80 px (Row 3 – recorded audio thumbnail displays).
  - _Metrics Dashboard Row:_ 100 px (Row 4 – six metrics with progress bars).
  - _Controls Row:_ 60 px (Row 5 – control buttons and sliders).

- **Layout Implementation:** In the `resized()` method of `MainComponent` (or equivalent UI resize handler), divide the vertical space according to the fixed heights above. Use absolute positioning or a layout manager (e.g. JUCE’s `FlexBox` or manual bounds setting) to position each row container. Each row should occupy the full window width (minus any margins) and its fixed height, with no overlap.
- **Row Separators:** After laying out the rows, draw a horizontal separator line between each adjacent row to visually delineate sections. This can be done by either painting lines in the `MainComponent`'s background or by inserting thin `Component` subclasses for dividers. Use the specified color `juce::Colour(0xff444444)` for these separator lines. Additionally, consider giving alternating rows a slight background color difference (e.g., alternating light/dark backgrounds or subtle shading) to match the schematic’s visual separation cues.
- **Title Bar Content:** The top 40px **Title Bar** should display the application title **“PNBTR+JELLIE Training Testbed”** prominently and centered. Implement this as a non-interactive `Label` or drawn text. Use a clear, bold font for visibility. The title bar has no other interactive UI elements; it serves as a header. Ensure its background contrasts with the main content (for example, a dark title bar with light text, or vice versa) so that it stands out as the header.

With the overall structure in place, we now detail each content row of the UI, following the row order from top (Row 1) to bottom (Row 5). Each section’s instructions cover layout specifics, component classes to use, what data or functionality they connect to, and the exact annotations and behaviors required.

## Row 1: Four-Panel Oscilloscope Section (Input, Network Sim, Log/Status, Output)

This top content row is a tall section (200px) divided into four equal-width panels side by side. The panels correspond to: **Input (Mic)**, **Network Sim**, **Log/Status**, and **Output (Reconstructed)**. Each panel will display real-time data or logs from different stages of the audio pipeline, with **exact technical annotations** overlayed as specified. The entire row is implemented within the main GUI component (MainComponent), likely in `PnbtrJellieGUIApp_Fixed.mm`.

- **Layout and Panels:** Split this row into **four columns of equal width**. Place a \~5px gap between columns for visual separation (you can account for these gaps in the positioning calculations). Each of the four sub-sections must fill its allotted space. Use child components for each panel:

  - **Input (Mic) Panel:** an oscilloscope display of the _microphone input waveform_.
  - **Network Sim Panel:** an oscilloscope display representing the _network-transmitted signal_ (after encoding and with simulated impairments).
  - **Log/Status Panel:** a text area for runtime logs and status messages.
  - **Output (Reconstructed) Panel:** an oscilloscope display of the _reconstructed output waveform_.

- **Oscilloscope Components:** Leverage the existing `OscilloscopeComponent` class (already available in the project) for the waveform panels. You will create **three instances** of `OscilloscopeComponent` – for Input, Network, and Output. Each oscilloscope should use the high-performance Metal rendering path (through the existing `MetalBridge` GPU integration) to draw waveforms efficiently. Ensure that each oscilloscope component is added to the MainComponent and its bounds set to cover one column of this row. The components must resize with the row and maintain their equal width distribution.

  - _MetalBridge Integration:_ Since `OscilloscopeComponent` is integrated with a Metal GPU context, verify that the Metal bridge is properly initialized and the component’s rendering context is active. This ensures real-time waveform drawing without CPU bottlenecks. No additional coding for Metal is needed beyond using the provided class; just ensure it’s included and functioning.
  - _Data Wiring:_ Connect each oscilloscope to the appropriate audio buffer source:

    - **Input Oscilloscope:** Display the live microphone input waveform. This should be fed from the **input audio buffer** that is filled by the CoreAudio input callback. In practice, when the audio input device provides new samples (via the audio IO callback in `PnbtrJellieGUIApp_Fixed.mm`), copy or push those samples into a shared `inputBuffer` (thread-safe), and call the input oscilloscope’s update method (e.g. `inputOscilloscope->updateWaveform(inputBuffer)`). This will visualize the raw mic audio in real time. The input callback (Step **1** in the processing flow) runs on the audio thread and captures mic data into `inputBuffer`, after which the **JELLIE encoder** step (Step **2**) processes that buffer to produce encoded data. (The JELLIE encoding happens in the background module, not directly visualized here except as it affects downstream data.)
    - **Network Sim Oscilloscope:** Display the waveform (or some representation) of the audio after network simulation. After encoding, the audio data (possibly an encoded multi-channel signal called `jellieEncoded`) passes through the **Network Simulation** stage (Step **3**) which introduces packet loss and jitter. The output of this stage (`networkProcessed`) can be visualized here. For example, if the network simulation yields a packetized audio stream or a reconstructed signal with dropouts, you might convert it to an audio waveform for display. Connect the network oscilloscope to the `networkProcessed` buffer or an equivalent signal that represents the transmitted audio post-simulation. Update this display whenever new network-processed data is available (e.g., after each network simulation cycle). This panel lets the developer see effects of packet loss/jitter on the signal.
    - **Output Oscilloscope:** Display the _reconstructed output waveform_. After network simulation, the **PNBTR neural reconstructor** processes the impaired data to generate a continuous output audio buffer (Step **4**). This final output buffer (`reconstructedBuffer`) contains the audio that will be played back. Connect the output oscilloscope to `reconstructedBuffer` and call its update method whenever new output audio is produced (e.g., after each PNBTR processing block). This shows the real-time waveform of the audio after neural reconstruction.

  - Ensure that the timebase (horizontal axis) and scaling of all oscilloscopes are appropriate (they should likely run at the same sample rate or time scale so that visually one can compare them if needed). They are in different panels, but roughly aligning their timeline can help correlate events.

- **Log/Status Panel:** The third panel in this row is a **scrolling log display** for status messages and events. Implement a new `LogStatusComponent` class (create files `src/gui/LogStatusComponent.h` and `.cpp`). This component should provide a multi-line text area that scrolls vertically as new log lines are added. Key implementation notes for the Log/Status panel:

  - Use a `juce::TextEditor` in read-only mode or a custom component drawing text lines to implement the scrolling text window. It must automatically scroll to show the latest entries at the bottom as new text is appended.
  - **Real-Time Event Logging:** Hook the `LogStatusComponent` up to receive logs from all major subsystems. For example, the existing `SessionManager` (the central controller for start/stop and processing) should invoke methods on `LogStatusComponent` (e.g., `logWindow->addEntry(String)`) whenever important events occur. Likewise, error handlers or status updates in the audio processing chain (JELLIE encoding, network sim, PNBTR, metrics calculations) should send messages to the log. This can be done via direct function calls, or by using a thread-safe queue that the LogStatusComponent checks periodically on the message thread.
  - **SessionManager Integration:** Ensure the SessionManager (if one exists in the codebase, as noted by the roadmap) is modified to broadcast events. For instance, when audio processing starts, when it stops, when a buffer underrun occurs, or when metrics update, call the log component to append a descriptive message. Use `juce::MessageManager::callAsync` or similar if these events originate from background threads, so that UI updates (the log component update) happen on the main thread.
  - **Visual Design:** The log window should be clearly separated (perhaps a border or different background since it’s text, not a waveform). It should display the label "Log/Status" at the top of its panel (as a section title or overlay text) for clarity.

- **Technical Annotations (Overlay Text):** Each of the four panels in this row requires specific overlay text as shown in the schematic. These annotations document the processing steps and data flow:

  - **Input (Mic) Panel Annotation:** Overlay two lines of text in the upper area of the input oscilloscope panel:

    1. “(1) CoreAudio input callback (mic)” – indicating that this panel shows data from the CoreAudio input stream (Step 1 of the pipeline).
    2. “→ (2) JELLIE encode (48kHz→192kHz, 8ch)” – with an arrow indicating that the mic input is fed into the JELLIE encoder step (Step 2), which upsamples 48 kHz audio to 192 kHz and distributes it over 8 channels (the JELLIE encoding process).

  - **Network Sim Panel Annotation:** Overlay similar text in the network panel:

    1. “(3) Simulate packet loss, jitter” – labeling the action in this stage (Step 3, network simulation applying packet loss and jitter to the encoded data).
    2. “→ (5) update network stats” – indicating that as a result of simulation, network statistics are updated (Step 5, feeding metrics like loss rate and jitter stats to the metrics system).

  - **Log/Status Panel Annotation:** Overlay the text “(6) Log events, errors, metrics” on the log window panel. This marks that the log is capturing Step 6 of the pipeline (logging of events, errors, and metrics updates). You may also include a second line or a sub-label like “Real-time Log Window” if needed (the roadmap explicitly notes to implement a real scrolling log component).
  - **Output (Reconstructed) Panel Annotation:** Overlay the text “(4) PNBTR neural reconstruction (output buffer)” on the output oscilloscope. This text identifies that panel as showing the output from the PNBTR neural network reconstructor (Step 4), i.e., the reconstructed audio buffer.
  - **Overlay Implementation:** These annotations can be implemented by drawing text in each panel’s `paint()` method (after drawing the waveform or background). Use a small, distinct font (e.g., a lighter color and smaller size) so as not to obstruct the main content. Ensure arrows “→” and numbering exactly match the specification (including parentheses around numbers). For any text that references code (like function names or file paths), use a monospaced font style to set it off (per the roadmap’s note on technical text styling). You might use `juce::Font::getDefaultMonospacedFontName()` for code annotations.

- **File Reference Annotation:** At the bottom of this Oscilloscope row (centered below the four panels, possibly spanning across them), display the reference path to the main GUI source file as given in the schematic: “\[PNBTR_JELLIE_DSP/standalone/vst3_plugin/src/PnbtrJellieGUIApp_Fixed.mm]”. This is a non-functional label purely for documentation, indicating which source file defines this portion of the UI. Render it in a subtle style (small font, possibly grey color) and monospace font. It can be drawn by the MainComponent after laying out the panels (since it spans the whole row) or placed as a non-interactive Label at a fixed position in this row’s space.

With Row 1 implemented, we will have a fully laid-out header section showing input, network simulation, log, and output waveforms side-by-side, each annotated with the corresponding step numbers and descriptions. The core audio input and PNBTR output are now visually trackable, the network effects can be seen, and the logging mechanism is in place for all events.

## Row 2: Waveform Analysis Row (Original vs. Reconstructed Waveforms)

Row 2 provides a side-by-side comparison of the original input audio versus the reconstructed output audio waveforms. This allows visual verification of the training system’s performance (e.g., how closely the reconstructed signal matches the original). The row is 120px tall and divided into two equal halves:

- **Dual Oscilloscope Layout:** Split this row into **two columns of equal width** with a small gap between them (similar to Row 1’s panel spacing). Reuse or instantiate two `OscilloscopeComponent` instances here – one for the left side (original waveform) and one for the right side (reconstructed waveform). Each should occupy the full height of the row and 50% of the width.
- **Left Panel (Original Waveform):** This oscilloscope displays the _original input audio buffer_ (microphone data) for the same segment of audio that is being processed. It should be fed by the same `inputBuffer` used in Row 1’s input panel, but here it serves for direct comparison purposes. In practice, this could be the _mic input_ signal aligned in time with the output. If the input buffer is streaming live, the left and right displays can be updated continuously. (Alternatively, if comparing a specific captured segment, you would display a recorded segment of input vs output. But given this is real-time, they will both show live streams.)
- **Right Panel (Reconstructed Waveform):** This oscilloscope displays the _PNBTR reconstructed audio buffer_, i.e., the output signal after network loss recovery and reconstruction. Use the same `reconstructedBuffer` that feeds Row 1’s output panel. The key here is that both left and right should be time-synchronized (e.g., if the app is running in real-time, they both scroll together showing corresponding segments of audio). This lets a viewer visually assess differences or delays between original and reconstructed signals.
- **Data Updates:** As the audio is processed, update these oscilloscopes each time new data is available. Likely, whenever a block of audio is processed by the PNBTR reconstructor (or at a steady refresh interval), call something like `leftOsc->updateWaveform(inputBuffer)` and `rightOsc->updateWaveform(reconstructedBuffer)` to refresh the displays. The update method might be the same as used in Row 1 (if the OscilloscopeComponent is generic and not tied to a specific buffer internally). Ensure thread safety: the input and output buffers are likely written by the audio thread, so use lock-free mechanisms or copy snapshots of the buffers for display to avoid data races.
- **Technical Annotations:** Add labels and annotations per the schematic:

  - Centered (above or within the left panel) the text: “Original Waveform Oscilloscope” and in parentheses below it “(inputBuffer, real mic data)”. This describes that the left oscilloscope is showing the original mic input data.
  - Similarly, above the right panel label it “Reconstructed Waveform Oscilloscope” with “(reconstructedBuffer, after PNBTR)” below, indicating the right side is the output of the PNBTR process.
  - Additionally, include small technical notes near the bottom of each panel:

    - On the left: “updateOscilloscope(inputBuffer)”
    - On the right: “updateOscilloscope(reconstructedBuffer)”.
      These annotations (in monospace font) correspond to the function calls that update the visuals. They serve as pseudo-code comments in the UI, showing how the data flow is connected to the UI update calls.

  - If space permits, also note the relevant source module for the oscilloscope (the schematic suggests a reference to `[src/oscilloscope/ or GUI class]` under each oscilloscope in this row). If the OscilloscopeComponent code resides in a specific file or module, you can annotate that (e.g., “OscilloscopeComponent (src/visual/Oscilloscope.cpp)” or similar) beneath each panel for clarity.

- **Behavior:** There is no distinct new processing happening in this row – it is purely visual. But ensure that both oscilloscopes start and stop in sync (for example, when the session is stopped via the control panel, these should cease updating). If the app supports pausing or if when no data flows both should idle. The **goal is to match the schematic’s intended layout and labeling exactly**, providing a clear visual comparison between input and output waveforms.

By completing Row 2, the UI now has a real-time waveform comparison. This is important for verifying the neural network’s performance and latency: the similarity (or differences) between the left and right waveforms can be observed, and any delay introduced by processing can be visually noted. This row sets the stage for the next, which deals with recorded tracks of those signals.

## Row 3: JUCE Audio Tracks Row (JELLIE & PNBTR Recordings)

Row 3 (80px height) presents two _audio track waveform displays_ side by side: one for the **JELLIE input track** (original audio as recorded) and one for the **PNBTR output track** (reconstructed audio as recorded). These are essentially mini wave displays of the recorded audio files, allowing the user to scroll through or visualize the entire captured session after (or during) recording. This uses JUCE’s audio visualization classes rather than the real-time oscilloscope, since these are for recorded files.

- **Layout:** Split this row into two equal halves (like Row 2) for the two tracks. Each half will contain a waveform thumbnail of a recorded audio file. Use a small gap between them. Label the entire row (perhaps via a small centered text at the top of this section) as "JUCE Audio Tracks: JELLIE & PNBTR Recordings" to match the schematic.
- **AudioThumbnailComponent:** Implement a new component class, e.g., `AudioThumbnailComponent`, as a wrapper around JUCE’s `AudioThumbnail` functionality. Create `AudioThumbnailComponent.h/cpp` in the project (`standalone/juce/` or `src/gui/`) as needed. Each instance of this class will encapsulate:

  - A `juce::AudioThumbnail` object (which holds the waveform data for a given audio source).
  - A `juce::AudioFormatManager` and `juce::AudioThumbnailCache` (usually one `AudioFormatManager` can be shared; it manages audio format readers, and the cache can be global or a decent size to hold thumbnails).
  - Possibly a `juce::AudioTransportSource` or similar if you plan to include playback controls (see below).
  - The component’s `paint()` should draw the waveform (using `AudioThumbnail::drawChannels`) to visualize the audio file’s waveform.

- **Left Track (JELLIE Input Recording):** This will display the waveform of the **recorded input audio** (which we can call the "JELLIE track"). The source of this is the audio that was captured from the mic and possibly encoded by JELLIE (though if JELLIE encoding is an internal representation, for recording we likely use the raw mic audio). According to the development plan, the system should record the microphone input buffer to a WAV file using JUCE’s `AudioFormatWriter` during the session. Implement this recording in `standalone/juce/recording.cpp` (as hinted) such that when the session is active (after Start is pressed), incoming input samples are written to a WAV file (let’s say JELLIE_input.wav). When recording is stopped (on Stop or Export), finalize the WAV file. The left AudioThumbnailComponent should then load this WAV file (JELLIE track) and display its waveform. The label under this waveform (as per the schematic) should read: "JUCE::AudioThumbnail: JELLIE Track (recorded input, .wav)". This text can be small and below or above the waveform display.
- **Right Track (PNBTR Output Recording):** Similarly, this displays the waveform of the **recorded reconstructed output audio** (the "PNBTR track"). The system should likewise record the `reconstructedBuffer` output to a separate WAV file (PNBTR_output.wav) during the session. Implement this parallel recording in `recording.cpp`. The right AudioThumbnailComponent then loads this output file and displays its waveform. Label it "JUCE::AudioThumbnail: PNBTR Track (reconstructed output, .wav)" below/above the waveform.
- **Thumbnail Loading & Updates:** Upon starting a new session (Start pressed), you might clear or create new audio files for recording. The AudioThumbnailComponents might remain blank or display an empty waveform until some data is recorded. When recording is ongoing, you can periodically update the thumbnail (AudioThumbnail has methods to add block of samples as they are recorded, if you prefer to show progress during recording). However, a simpler approach is to load the waveform once recording is finished (e.g., after Stop or on-demand when the user presses Export). The guide suggests auto-updating thumbnails when recording starts/stops. This means you should integrate the recording process with the thumbnails:

  - On **recording start**, you might initialize the AudioThumbnail with the file path and tell it to start collecting data (AudioThumbnail can be fed via `AudioThumbnail::addBlock()` each time a new chunk of samples is written).
  - On **recording stop**, finalize the file and then call `AudioThumbnail::setSource(new FileInputSource(file))` to reload the waveform from disk (or if you were adding on the fly, just finalize).
  - In either case, ensure the UI refreshes (call repaint on the thumbnail component) so the final waveform is drawn.

- **Playback Controls (Optional within this row):** The schematic text notes usage of `JUCE::AudioTransportSource` and `JUCE::AudioFormatManager`, implying that the app might allow playback of the recorded tracks. If desired, incorporate basic playback buttons or at least the ability to start/stop playback of these waveforms:

  - You might embed small play/stop buttons on each thumbnail or have a single playback control that toggles playback of one or both tracks. This goes beyond just visualization, but since the roadmap explicitly mentions AudioTransportSource, implementing playback is expected in this row.
  - Set up a `AudioTransportSource` for each track (or one that you swap sources in). Link it with an `AudioFormatReaderSource` reading from the recorded WAV file. Connect playback output either to the main audio output or allow the user to hear it (though if the main app is still running audio I/O, careful mixing or output selection is needed).
  - The UI for these controls could be minimal (even not explicitly drawn in the schematic), but functionally after recording, a developer could play back the input vs output file to audibly inspect the quality. Since the prompt focuses on matching the schematic, you may keep the controls subtle or hidden, but ensure the code structure supports using `AudioTransportSource` for these tracks.

- **Technical Annotation:** As with other rows, provide textual annotations in-place to clarify components:

  - Somewhere in the left half, list or overlay the text: “JUCE::AudioTransportSource, JUCE::AudioFormatManager” (these might be shared between tracks or one per track) to indicate the underlying JUCE classes enabling this feature. This text can be small and below the waveform or integrated into the label.
  - Also, include a reference to the source path for these features. The schematic gives “\[PNBTR_JELLIE_DSP/standalone/juce/]” as a reference for this row. Likely, all recording and waveform code is under `standalone/juce/` directory. You might append specific file names if clarity is needed (e.g., “recording.cpp” or “waveform.cpp”). The annotation could be placed at the bottom center of this row.

- **Integration:** Ensure that the recording system ties into the session control. For example, the Start button should initiate or arm recording (creating the files and preparing writers), Stop should finalize them, and Export might simply also finalize or copy them to an accessible location if needed. By the time Stop/Export is complete, the audio files should be ready for the thumbnail to display. Use the JUCE `TimeSliceThread` or background thread to generate thumbnails if the files are long, to avoid blocking the message thread. However, given typical use (short training sessions), this might not be an issue.

Once Row 3 is implemented, we have a means to **persist and review** the audio. The visual thumbnails of input vs output recordings allow an offline comparison over the entire capture duration (complementing Row 2’s live view). This also confirms that audio recording is working correctly. Next, we ensure that the metrics are calculated and shown.

## Row 4: Metrics Dashboard Row

Row 4 (100px height) contains the **Metrics Dashboard**, which displays six key performance metrics of the system: SNR, THD, Latency, Reconstruction Rate, Gap Fill, and Quality. Each metric is shown with a numeric value and/or a progress-bar style indicator, as depicted in the schematic. The `MetricsDashboard` class is already implemented to handle displaying metrics, but we need to ensure it is laid out correctly and fed real data from the processing pipeline.

- **Layout of Metrics:** Arrange six metrics **horizontally across this row**, each with equal space or a suitable proportion so that all six fit comfortably. Likely, each metric is presented as a small sub-panel containing a label (e.g., “SNR”) and either a horizontal bar, dial, or numeric readout. The schematic suggests a progress bar or value for each. You can implement each metric display as a custom component (if not already provided) or use `juce::ProgressBar` or `juce::Slider` in a read-only mode to visualize values. The MetricsDashboard might already aggregate these; if so, just ensure it uses the full width and divides into six segments.
- **Ensure Exact Metrics Order & Labels:** The metrics must appear in the exact order and naming as given: **SNR | THD | Latency | Recon Rate | Gap Fill | Quality**. Use short labels exactly as above (e.g., “Recon Rate” for reconstruction rate, “Gap Fill” for gap fill rate/metric). If the MetricsDashboard class wasn’t fully implemented, create subcomponents or draw text and bars for each.
- **Technical Annotations in UI:** Just below or within the metrics display, overlay the names of the functions that compute these metrics, as well as the update call:

  - In a smaller font underneath the metric labels, list the metric calculation function names corresponding to each column. The schematic shows: “(7) calculateSNR() calculateTHD() calculateLatency() calculateReconstructionRate() calculateGapFillQuality() calculateOverallQuality()” across two lines. You should place these exactly as formatted:

    - The number (7) corresponds to the entire metrics calculation stage in the pipeline (Step 7). You can place “(7)” at the beginning of the first line of function names to indicate these are part of step 7.
    - List `calculateSNR()`, `calculateTHD()`, `calculateLatency()`, `calculateReconstructionRate()`, `calculateGapFillQuality()`, and `calculateOverallQuality()` in order, separated by two spaces or a clear delimiter, to match the spacing in the schematic. If needed, break into two lines (as the schematic does) to fit.

  - Also include the text “updateMetricDisplay(metric, value, bar)” somewhere near the metrics, perhaps at the end or bottom of this section. This denotes the function that updates the UI for the metrics display. In practice, `MetricsDashboard` might have a method like `updateMetricDisplay(MetricType type, float newValue)` or similar; the schematic treats it generally. This annotation should be in monospace font to appear like code.
  - Finally, include a file reference annotation for the metrics source code: “\[src/metrics/metrics.cpp, .h]”. This can be placed at the bottom-right of the metrics row or centered below the function names. It indicates where the metric calculations are implemented.

- **Data Connections:** The metrics values must be calculated in real-time from the audio data:

  - Implement all the metric calculation functions in the metrics module (likely `src/metrics/metrics.cpp`) if not already done. For example:

    - `calculateSNR()` should compare the original vs reconstructed signal to compute signal-to-noise ratio.
    - `calculateTHD()` might measure total harmonic distortion between input and output.
    - `calculateLatency()` likely measures the end-to-end latency (perhaps by correlating input vs output or using timestamping).
    - `calculateReconstructionRate()` might be the percentage of packets correctly reconstructed or similar.
    - `calculateGapFillQuality()` possibly quantifies how well lost gaps were filled.
    - `calculateOverallQuality()` could be a composite metric or MOS-like quality measure.

  - Many of these will rely on comparing the `inputBuffer` and `reconstructedBuffer` (for example, SNR and quality) or using data from the network simulation (loss rate for gap fill, etc.). Ensure the `metrics.cpp` has access to the needed info. For instance, network stats (packet loss %, etc.) should be updated in a shared place (the roadmap mentions atomic counters in metrics for network stats).
  - The **MetricsDashboard UI** (perhaps implemented in `src/gui/MetricsDisplay.cpp` or integrated in MetricsDashboard class) should periodically update its displayed values. A straightforward approach is: after each audio processing block (or on a timer e.g. 10 times a second), recalc the metrics in `metrics.cpp` and then call `MetricsDashboard::updateMetricDisplay(...)` for each value. This could be orchestrated by the SessionManager or the main UI loop. Since performance metrics might not need frame-by-frame update, a timer on the message thread can pull new values and update the UI components.
  - All metric calculations must use real data. Remove any placeholder or fake metrics that may have been present. For example, if the current implementation was using dummy values, replace that with actual calculations.

- **Formatting:** Ensure each metric’s value is displayed with appropriate formatting and units (if applicable). For instance, latency might be in milliseconds, SNR in dB, THD in percentage, etc. You can display numeric values next to bars or as part of the bar (e.g., inside a progress bar).
- **Verification:** After implementing, verify that each metric responds when the system runs: e.g., if you introduce packet loss, SNR should drop, gap fill metric should reflect something, latency stays relatively constant, etc. The metrics row should update continuously without lag (since it's lightweight to update text/bars).

By completing Row 4, the application will present a fully functional metrics dashboard reflecting the live performance of the system. This, combined with logs and waveforms, gives a complete picture of system behavior.

## Row 5: Controls Row (Start/Stop/Export Buttons & Packet Loss/Jitter/Gain Sliders)

The bottom row (60px height) contains the **control interface** for the app’s operation. This includes three control buttons and three parameter sliders, laid out in a single horizontal band. The controls allow the user to start/stop the audio processing, export the results, and adjust network simulation parameters on the fly. It’s critical that their arrangement and labeling exactly match the spec.

- **Layout of Controls:** Arrange the controls in a logical and spaced manner within this 60px-high row. Typically, you would group the three buttons toward the left or center and the sliders to the right (or vice versa, whichever matches the intended UI schematic). Ensure there is adequate padding around each control and that the row is fully utilized without crowding. Use consistent sizes for buttons and sliders.

  - The schematic shows “\[Start] \[Stop] \[Export] \[Sliders: Packet Loss, Jitter, Gain]” all in one line. From this, we infer the three buttons are aligned, followed by some gap, then the three sliders grouped together. You can follow that pattern: left side for buttons, right side for sliders, with a bit of spacing in the middle.

- **Buttons Implementation:** Create three JUCE `TextButton` (or `TextButton` subclasses) for **Start**, **Stop**, and **Export**. Add them to the MainComponent (or a ControlsComponent if you encapsulate the row) and assign their onClick behaviors:

  - **Start Button:** onClick, triggers the start of audio processing. This should call a function like `SessionManager::startAudio()` or equivalent to begin capturing audio, enabling the processing pipeline, and recording. If SessionManager exists, wire the button to its start method. Otherwise, directly start the necessary subsystems: open audio IO streams, start network simulation threads, etc. Also ensure that pressing Start resets/clears previous data (like log window, metrics, etc., if appropriate).
  - **Stop Button:** onClick, stops the audio processing. Call `SessionManager::stopAudio()` or equivalent. This should halt the audio callback processing, stop network simulation, and finalize any recordings in progress. Also, after stop, perhaps the system could leave the last waveforms on screen and allow export or playback.
  - **Export Button:** onClick, initiates export of collected data. Typically this means writing the recorded audio buffers to disk (if not already done on stop) and possibly exporting a metrics report. The prompt specifically mentions an export to WAV, which we have implemented by recording to WAV in Row 3. So the Export button can either simply call the same finalization as Stop (if Stop didn’t already do it) or if Stop already wrote files, perhaps Export could open a file dialog or simply confirm that files are saved. Additionally, you might produce a metrics summary file or copy logs. If following the roadmap, implement a function like `exportWAV()` in SessionManager and call that.
  - Make sure these buttons are properly enabled/disabled in logic (e.g., while running, maybe Start is disabled and Stop enabled, etc., to avoid state confusion).

- **Sliders Implementation:** Create three sliders (JUCE `Slider` objects) for **Packet Loss**, **Jitter**, and **Gain** controls. These sliders allow runtime adjustment of the network simulation parameters and perhaps input gain:

  - **Packet Loss Slider:** controls the percentage of packet loss to simulate (e.g., 0% to some max like 5% or 10%). This value will affect the behavior of the network simulation module (`network_sim.cpp`). Ensure that moving this slider calls a function such as `SessionManager::setPacketLoss(float percent)` or directly sets an atomic variable that the network simulation uses each cycle. The function `setPacketLoss()` should be defined in the appropriate place (SessionManager or a global config) and the UI slider’s onValueChange should link to it. For instance, on slider change: `sessionManager.setPacketLoss(sliderValue)`.
  - **Jitter Slider:** controls the jitter simulation (perhaps in milliseconds of variance or a factor of reordering). Again, wire this to a `setJitter(float value)` function. The network simulation code should read this value to introduce timing jitter in packet delivery. Ensure thread-safe communication; likely SessionManager holds a config that network_sim checks.
  - **Gain Slider:** controls an audio gain parameter (possibly input gain to simulate different speaking volumes or output gain for listening). Wire this to `setGain(float value)`. Implementation could multiply the inputBuffer samples by this gain or adjust output volume. Place it in SessionManager or an audio pipeline component as appropriate.
  - All three sliders should be horizontal, with labels either as part of the slider (you can use `Slider::setTextValueSuffix` or just put static text labels next to them). Label them exactly “Packet Loss”, “Jitter”, “Gain”. Use ranges that make sense (e.g., 0–5% for loss, 0–50 ms for jitter, 0–1.0 or in dB for gain). The schematic likely expects generic positions, but ensure usability.

- **Technical Annotations:** Just like other rows, include overlay text near the controls to document their connected functions and file references:

  - Place the text “(8) startAudio() stopAudio() exportWAV()” near the buttons group. This indicates that these buttons are wired to those functions (Step 8 in the overall flow corresponds to control wiring). Use a monospace font and include the “(8)” prefix to denote this is step 8 of the process.
  - Near the sliders, place the text “setPacketLoss(), setJitter(), setGain()” to show the functions that are called when these sliders move. These can be right under the sliders or above them in a smaller font. Ensure the text matches exactly (including the comma separation as shown).
  - Also include the file reference “\[src/gui/controls.cpp, .h]” somewhere in this row (e.g., bottom-right). This indicates that the implementation of these controls is in the source files `controls.cpp`/`.h` in the GUI folder. In practice, you might implement the entire row in such a Controls class, so referencing that file path is appropriate.

- **SessionManager Integration:** As noted, all these controls should tie into the `SessionManager` or equivalent central logic (since the roadmap notes SessionManager exists for start/stop and parameter management). Ensure that:

  - The SessionManager has methods corresponding to each action (start, stop, export, setLoss, setJitter, setGain) and that they perform the necessary thread-safe operations (e.g., if adjusting packet loss, maybe setting an `std::atomic<float> packetLossRate` that the network thread reads).
  - Starting the session opens audio devices and starts any needed threads (for network simulation, metrics calculation if separate, etc.). Stopping closes them. Export perhaps just calls Stop then handles file output (or if files already written on the fly, maybe just collates logs/metrics).
  - The UI state updates accordingly (for instance, after Stop, you might want to automatically trigger the thumbnail components to load the new recordings, or update metrics one final time).

- **State Feedback:** It could be useful to log events to the LogStatusComponent when these actions occur. E.g., “Session started”, “Session stopped – audio files saved”, etc., to give feedback (the log panel is made for that and step 6 of pipeline logging would include these control events as well).

With Row 5 in place, the user/operator has full control over the testbed: they can initiate a session, terminate it, save the results, and tweak conditions in real time. All controls are labeled and annotated for clarity, matching the schematic exactly. The functional wiring of these controls closes the loop on the system: pressing Start begins the entire flow (CoreAudio input through PNBTR output) and all the above components (oscilloscopes, metrics, recordings) should respond accordingly, while Stop/Export cleanly shut it down and preserve data.

## **(Optional)** Secondary Roadmap: Advanced Runtime Refinements

_(The following are additional considerations and enhancements that go beyond the core schematic implementation. They focus on performance, reliability, and measurement improvements during runtime. These should be implemented only after the core functionality above is solid, so they do not confuse the primary development goals.)_

- **Transport-Phase GPU Protection:** Ensure that heavy GPU operations (such as Metal rendering for oscilloscopes) do not interfere with real-time audio processing. The audio input and output callbacks should remain real-time safe. To achieve this, decouple the OscilloscopeComponent updates from the audio thread. For example, use a lock-free FIFO or atomic flag to signal the availability of new audio data, and let the GUI thread (Message Thread) or a high-priority Timer handle pulling that data for rendering. This prevents any **GPU calls on the audio thread**. Additionally, consider double-buffering the waveform data: one buffer is written by the audio thread, while the GPU reads from another, swapping pointers safely (this avoids the audio thread waiting on GPU). By protecting the transport phase (moving data from audio thread to GPU/GUI), you ensure glitch-free audio even when the GPU is busy drawing waveforms.
- **Exclusive Metal Command Queue:** If the MetalBridge and JUCE's GPU context allow, use a dedicated Metal command queue for the oscilloscope rendering. This means the oscilloscope draws won't stall other GPU tasks. In practice, since JUCE’s graphics might be on CoreAnimation/Metal, having an exclusive `MTLCommandQueue` for the wave drawing layer can reduce contention. Implement this by configuring MetalBridge (or underlying Metal code) to submit commands on its own queue separate from the main UI composition queue. An exclusive queue ensures that if, for instance, a large waveform draw is happening, it does not block other render passes (or vice versa). This design will maintain smoother UI updates, especially if you extend the app with more GPU-based visuals.
- **Latency Tracking Instrumentation:** Introduce precise latency measurement throughout the pipeline. While we have a `calculateLatency()` metric, deeper instrumentation can help refine it:

  - Timestamp audio frames at capture (e.g., note the audio callback time for a particular sample index).
  - When that frame is reconstructed at output, compute the time difference. This can be done by embedding sequence numbers or markers in the data path (for example, tag the first sample of each input buffer with a steadily incrementing ID, carry it through JELLIE encoding and network simulation, and when PNBTR emits output, check the ID to match it to an input timestamp).
  - Use `juce::Time::getHighResolutionTicks` or performance counters to get sub-millisecond timing. Accumulate stats on latency (min, max, average) and present them in the log or as part of the metrics.
  - This instrumentation ensures the **Latency** metric in the dashboard is accurate and can also log outlier events (e.g., if a particular cycle was much slower, perhaps log it).

- **Threading and Real-Time Priority:** Review thread priorities to prioritize audio processing. The audio I/O and PNBTR processing should run in high-priority threads (often handled by CoreAudio/JUCE automatically for audio callbacks). The network simulation and logging can run in lower priority threads. If using additional threads (like for metrics computation if heavy, or for writing to disk), mark them as background priority to not interfere. Also, consider using a real-time safe queue for logging (to avoid locks on audio thread) and aggregate logs in batches to the UI.
- **GPU Frame Rate Throttling:** If the oscilloscopes or thumbnails are drawing too often, they could saturate GPU or CPU. Implement a throttle (e.g., redraw oscilloscope at most 30 or 60 FPS, using `Timer` callbacks) to balance smooth visuals with CPU/GPU usage. This can be tied to the Metal command queue – for instance, if a frame is still being rendered, skip posting a new one.
- **Error Handling and Recovery:** Enhance the system’s robustness by handling edge cases: if the audio device fails or disconnects, log it and possibly attempt to recover; if file writing fails, alert the user in the log; if the network simulation thread falls behind, drop packets rather than block, etc. These refinements ensure the app can run for long periods in a test without manual intervention.
- **Performance Monitoring:** Add internal counters or use JUCE’s `TimeSliceThread` to monitor CPU and GPU usage periodically. You could log if the CPU usage for processing is approaching real-time limits, or if render times are high. This is ancillary but helpful for a developer to tune the system (not necessarily exposed in UI, except maybe as verbose log info).
- **Exclusive Audio Device Mode:** If running on a system where exclusive access to the audio device lowers latency, consider enabling that (platform specific). This ensures minimal latency between input and output, making the latency metric as low as possible.
- **Continuous Integration Tests:** Although not a runtime feature, set up automated tests or scripts (if applicable) to run the whole pipeline with known signals and verify the metrics output (e.g., feed a test tone and check SNR is infinite if no loss, etc.). This goes beyond the immediate app implementation but is a refinement to ensure reliability of all parts.

_(The above advanced steps are optional and should be approached after the core implementation is verified against the schematic. They ensure that the application not only meets the spec visually and functionally, but also operates robustly and efficiently under the hood.)_

---

By following this implementation guide, a developer will construct the PNBTR+JELLIE training testbed app exactly as envisioned. Each UI row is built with precise dimensions and content, every component is connected to the correct data source or function, and all technical labels and behaviors match the provided schematic. After assembling each part and linking them through the SessionManager and underlying processing modules, the final system will allow real-time audio to flow from input to output (mic → JELLIE → network sim → PNBTR), while displaying waveforms, logging events, calculating metrics, and enabling control – all in a clear, developer-friendly interface. Always verify each phase against the schematic, and use the log and metrics to validate that real data (not placeholders) are driving the UI. The result is a highly transparent testbed application suitable for development and debugging of the PNBTR+JELLIE training pipeline, with exacting fidelity to the design specification.
