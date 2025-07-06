# TRANSPORT CONTROLLER HELP REQUEST

## CURRENT PROBLEM
The TOASTer application is not displaying the correct professional transport controller. Despite having a `ProfessionalTransportController` class, the app is showing a basic transport panel instead of the microsecond-precision professional transport interface.

## EXPECTED TRANSPORT INTERFACE
Based on the user's requirements, the transport should display:

### Layout & Visual Design
- Professional dark theme (matching JUCE dark UI)
- Clean, modern button design
- Proper spacing and typography
- Should fill the main application window area

### Transport Controls
- **Play/Stop/Pause/Record buttons** with proper state management
- Custom button graphics (not text-based)
- Visual feedback for button states (highlighted, pressed, etc.)

### Timing Display - CRITICAL REQUIREMENT
- **Session time with microsecond precision**: `00:00:00.000000` format
  - Hours:Minutes:Seconds.Microseconds (6-digit microsecond precision)
  - NOT the current format: `00:00:00.000.000` (which shows milliseconds.microseconds)
- **Bars/Beats display**: `001:1:000` format
  - Bar:Beat:Subdivision with proper zero-padding
- **BPM control**: Real-time slider with numeric display

### Transport State Management
- Proper play/stop/pause/record state transitions
- High-frequency timer updates (microsecond precision)
- Real-time clock synchronization
- Session time tracking that persists across pause/resume

## CURRENT CODE ISSUES

### 1. CMakeLists.txt Analysis
The build system includes:
- `Source/ProfessionalTransportController.cpp` ✓
- But also includes `Source/BasicMIDIPanel.cpp` which might be interfering
- Missing any audio timing modules for microsecond precision?

### 2. MainComponent Issues
- May be instantiating wrong transport class
- Layout might not be optimized for transport display
- Window sizing may be incorrect

### 3. ProfessionalTransportController Issues
- Time format might be wrong (showing milliseconds.microseconds instead of pure microseconds)
- Button rendering might not match expected professional appearance
- Timer frequency might not be sufficient for microsecond updates

## QUESTIONS THAT NEED ANSWERS

1. **Which exact transport class should be displayed?**
   - Is there a specific existing transport class that works correctly?
   - Should we modify ProfessionalTransportController or create a new one?

2. **What is the exact microsecond timing format required?**
   - Is it `HH:MM:SS.uuuuuu` (6-digit microseconds)?
   - Or `HH:MM:SS.mmm.uuu` (3-digit milliseconds + 3-digit microseconds)?
   - The user mentioned "00:00:00.000000" format specifically

3. **Are there existing working transport examples in the codebase?**
   - Should we examine `BasicTransportPanel.cpp` for reference?
   - Are there other transport controllers that work correctly?

4. **What JUCE modules are needed for microsecond timing?**
   - Do we need `juce_audio_basics` for high-resolution timing?
   - Should we use `juce_core` high-resolution timer classes?

## NEXT STEPS REQUESTED

Please help identify:
1. The correct class/file that should be used for the professional transport
2. The exact timing format specification
3. Any missing JUCE modules or dependencies
4. Whether the current ProfessionalTransportController can be fixed or needs replacement

## FILES TO EXAMINE
- `/Users/timothydowler/Projects/MIDIp2p/TOASTer/Source/ProfessionalTransportController.h`
- `/Users/timothydowler/Projects/MIDIp2p/TOASTer/Source/ProfessionalTransportController.cpp`
- `/Users/timothydowler/Projects/MIDIp2p/TOASTer/Source/BasicTransportPanel.h`
- `/Users/timothydowler/Projects/MIDIp2p/TOASTer/Source/MainComponent.cpp`
- Any other transport-related files in the codebase

## GOAL
Create a professional transport controller that displays microsecond-precision timing and matches the expected professional interface design, replacing whatever basic transport is currently being shown.
