#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include "jmid_gpu/gpu_jmid_framework.h"

//==============================================================================
/**
 * GPU-Native MIDI Manager for TOASTer
 * 
 * Replaces the legacy CPU-based MIDIManager with GPU-native MIDI processing.
 * All MIDI events are processed through the GPU JMID framework for ultra-low
 * latency and burst deduplication.
 */
class GPUMIDIManager : public juce::MidiInputCallback
{
public:
    GPUMIDIManager();
    ~GPUMIDIManager() override;

    // Initialize with GPU JMID framework
    bool initialize(jam::jmid_gpu::GPUJMIDFramework* jmidFramework);
    void shutdown();
    
    // MIDI I/O management
    bool openMIDIInput(int deviceIndex);
    bool openMIDIOutput(int deviceIndex);
    void closeMIDIInput();
    void closeMIDIOutput();
    
    // Get available MIDI devices
    juce::StringArray getMIDIInputDevices() const;
    juce::StringArray getMIDIOutputDevices() const;
    
    // GPU-native MIDI sending (through JMID framework)
    void sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2);
    void sendMIDINoteOn(int channel, int noteNumber, int velocity);
    void sendMIDINoteOff(int channel, int noteNumber, int velocity);
    void sendMIDIControlChange(int channel, int controller, int value);
    
    // MIDI callback (from JUCE input)
    void handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message) override;
    
    // GPU MIDI event callback (from network)
    void handleNetworkMIDIEvent(const jam::jmid_gpu::GPUMIDIEvent& event);
    
    // Status queries
    bool isMIDIInputOpen() const { return midiInput != nullptr; }
    bool isMIDIOutputOpen() const { return midiOutput != nullptr; }
    bool isGPUMIDIActive() const { return jmidFramework != nullptr; }
    
    // Performance metrics
    int getMIDIEventsPerSecond() const;
    double getAverageLatency() const;
    
    // Event filtering and routing
    void setMIDIChannelFilter(int channel, bool enabled);
    void setMIDITypeFilter(uint8_t messageType, bool enabled);
    
    // Callback registration for GUI updates
    std::function<void(const juce::MidiMessage&)> onMIDIInputReceived;
    std::function<void(const jam::jmid_gpu::GPUMIDIEvent&)> onNetworkMIDIReceived;
    std::function<void(const juce::String&)> onStatusUpdate;

private:
    // GPU-native MIDI framework
    jam::jmid_gpu::GPUJMIDFramework* jmidFramework = nullptr;
    
    // JUCE MIDI I/O (for local hardware)
    std::unique_ptr<juce::MidiInput> midiInput;
    std::unique_ptr<juce::MidiOutput> midiOutput;
    
    // Device management
    juce::AudioDeviceManager deviceManager;
    int currentInputDevice = -1;
    int currentOutputDevice = -1;
    
    // Event filtering
    std::array<bool, 16> channelFilter; // MIDI channels 0-15
    std::array<bool, 256> typeFilter;   // MIDI message types
    
    // Performance tracking
    mutable std::atomic<int> eventsPerSecond{0};
    mutable std::atomic<double> averageLatency{0.0};
    std::chrono::high_resolution_clock::time_point lastMetricsUpdate;
    int eventsSinceLastUpdate = 0;
    
    // GPU-native MIDI event conversion
    jam::jmid_gpu::GPUMIDIEvent juceMidiToGPUEvent(const juce::MidiMessage& message);
    juce::MidiMessage gpuEventToJuceMidi(const jam::jmid_gpu::GPUMIDIEvent& event);
    
    // Event filtering
    bool shouldProcessMIDIMessage(const juce::MidiMessage& message) const;
    bool shouldProcessGPUEvent(const jam::jmid_gpu::GPUMIDIEvent& event) const;
    
    // Performance metrics update
    void updatePerformanceMetrics();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUMIDIManager)
};
