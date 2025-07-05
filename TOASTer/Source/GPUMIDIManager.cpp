#include "GPUMIDIManager.h"
#include <chrono>

//==============================================================================
GPUMIDIManager::GPUMIDIManager()
{
    // Initialize all channel filters to enabled
    channelFilter.fill(true);
    
    // Initialize common MIDI message type filters
    typeFilter.fill(false);
    typeFilter[0x80] = true; // Note Off
    typeFilter[0x90] = true; // Note On
    typeFilter[0xA0] = true; // Aftertouch
    typeFilter[0xB0] = true; // Control Change
    typeFilter[0xC0] = true; // Program Change
    typeFilter[0xD0] = true; // Channel Pressure
    typeFilter[0xE0] = true; // Pitch Bend
    
    lastMetricsUpdate = std::chrono::high_resolution_clock::now();
}

GPUMIDIManager::~GPUMIDIManager()
{
    shutdown();
}

bool GPUMIDIManager::initialize(GPUJMIDFramework* framework)
{
    if (!framework) {
        if (onStatusUpdate) {
            onStatusUpdate("ERROR: GPUMIDIManager requires valid JMID framework");
        }
        return false;
    }
    
    jmidFramework = framework;
    
    // Initialize JUCE audio device manager for MIDI
    auto error = deviceManager.initialise(0, 0, nullptr, true);
    if (!error.isEmpty()) {
        if (onStatusUpdate) {
            onStatusUpdate("MIDI Device Manager Error: " + error);
        }
        return false;
    }
    
    if (onStatusUpdate) {
        onStatusUpdate("GPU MIDI Manager initialized successfully");
    }
    
    return true;
}

void GPUMIDIManager::shutdown()
{
    closeMIDIInput();
    closeMIDIOutput();
    
    jmidFramework = nullptr;
    
    deviceManager.closeAudioDevice();
    
    if (onStatusUpdate) {
        onStatusUpdate("GPU MIDI Manager shut down");
    }
}

//==============================================================================
// MIDI I/O management

bool GPUMIDIManager::openMIDIInput(int deviceIndex)
{
    closeMIDIInput();
    
    auto devices = juce::MidiInput::getAvailableDevices();
    if (deviceIndex < 0 || deviceIndex >= devices.size()) {
        if (onStatusUpdate) {
            onStatusUpdate("Invalid MIDI input device index: " + juce::String(deviceIndex));
        }
        return false;
    }
    
    auto device = devices[deviceIndex];
    midiInput = juce::MidiInput::openDevice(device.identifier, this);
    
    if (midiInput) {
        midiInput->start();
        currentInputDevice = deviceIndex;
        
        if (onStatusUpdate) {
            onStatusUpdate("MIDI Input opened: " + device.name);
        }
        return true;
    } else {
        if (onStatusUpdate) {
            onStatusUpdate("Failed to open MIDI input: " + device.name);
        }
        return false;
    }
}

bool GPUMIDIManager::openMIDIOutput(int deviceIndex)
{
    closeMIDIOutput();
    
    auto devices = juce::MidiOutput::getAvailableDevices();
    if (deviceIndex < 0 || deviceIndex >= devices.size()) {
        if (onStatusUpdate) {
            onStatusUpdate("Invalid MIDI output device index: " + juce::String(deviceIndex));
        }
        return false;
    }
    
    auto device = devices[deviceIndex];
    midiOutput = juce::MidiOutput::openDevice(device.identifier);
    
    if (midiOutput) {
        currentOutputDevice = deviceIndex;
        
        if (onStatusUpdate) {
            onStatusUpdate("MIDI Output opened: " + device.name);
        }
        return true;
    } else {
        if (onStatusUpdate) {
            onStatusUpdate("Failed to open MIDI output: " + device.name);
        }
        return false;
    }
}

void GPUMIDIManager::closeMIDIInput()
{
    if (midiInput) {
        midiInput->stop();
        midiInput.reset();
        currentInputDevice = -1;
        
        if (onStatusUpdate) {
            onStatusUpdate("MIDI Input closed");
        }
    }
}

void GPUMIDIManager::closeMIDIOutput()
{
    if (midiOutput) {
        midiOutput.reset();
        currentOutputDevice = -1;
        
        if (onStatusUpdate) {
            onStatusUpdate("MIDI Output closed");
        }
    }
}

//==============================================================================
// Device enumeration

juce::StringArray GPUMIDIManager::getMIDIInputDevices() const
{
    juce::StringArray deviceNames;
    auto devices = juce::MidiInput::getAvailableDevices();
    
    for (const auto& device : devices) {
        deviceNames.add(device.name);
    }
    
    return deviceNames;
}

juce::StringArray GPUMIDIManager::getMIDIOutputDevices() const
{
    juce::StringArray deviceNames;
    auto devices = juce::MidiOutput::getAvailableDevices();
    
    for (const auto& device : devices) {
        deviceNames.add(device.name);
    }
    
    return deviceNames;
}

//==============================================================================
// GPU-native MIDI sending

void GPUMIDIManager::sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2)
{
    // Send via GPU JMID framework for network transmission
    if (jmidFramework) {
        GPUJMIDEvent gpuEvent;
        gpuEvent.status = status;
        gpuEvent.data1 = data1;
        gpuEvent.data2 = data2;
        gpuEvent.timestamp = jmidFramework->getCurrentGPUTimestamp();
        gpuEvent.channel = status & 0x0F;
        
        // Send through GPU pipeline for network transmission
        jmidFramework->sendMIDIEvent(gpuEvent);
    }
    
    // Also send to local MIDI output if available
    if (midiOutput) {
        juce::MidiMessage message(status, data1, data2);
        midiOutput->sendMessageNow(message);
    }
    
    // Update performance metrics
    updatePerformanceMetrics();
}

void GPUMIDIManager::sendMIDINoteOn(int channel, int noteNumber, int velocity)
{
    uint8_t status = 0x90 | (channel & 0x0F);
    sendMIDIEvent(status, static_cast<uint8_t>(noteNumber), static_cast<uint8_t>(velocity));
}

void GPUMIDIManager::sendMIDINoteOff(int channel, int noteNumber, int velocity)
{
    uint8_t status = 0x80 | (channel & 0x0F);
    sendMIDIEvent(status, static_cast<uint8_t>(noteNumber), static_cast<uint8_t>(velocity));
}

void GPUMIDIManager::sendMIDIControlChange(int channel, int controller, int value)
{
    uint8_t status = 0xB0 | (channel & 0x0F);
    sendMIDIEvent(status, static_cast<uint8_t>(controller), static_cast<uint8_t>(value));
}

//==============================================================================
// MIDI callbacks

void GPUMIDIManager::handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message)
{
    if (!shouldProcessMIDIMessage(message)) return;
    
    // Convert to GPU event and send via JMID framework
    if (jmidFramework) {
        auto gpuEvent = juceMidiToGPUEvent(message);
        jmidFramework->sendMIDIEvent(gpuEvent);
    }
    
    // Notify GUI
    if (onMIDIInputReceived) {
        onMIDIInputReceived(message);
    }
    
    // Update performance metrics
    updatePerformanceMetrics();
}

void GPUMIDIManager::handleNetworkMIDIEvent(const GPUJMIDEvent& event)
{
    if (!shouldProcessGPUEvent(event)) return;
    
    // Send to local MIDI output if available
    if (midiOutput) {
        auto juceMidi = gpuEventToJuceMidi(event);
        midiOutput->sendMessageNow(juceMidi);
    }
    
    // Notify GUI
    if (onNetworkMIDIReceived) {
        onNetworkMIDIReceived(event);
    }
    
    // Update performance metrics
    updatePerformanceMetrics();
}

//==============================================================================
// Performance metrics

int GPUMIDIManager::getMIDIEventsPerSecond() const
{
    return eventsPerSecond.load();
}

double GPUMIDIManager::getAverageLatency() const
{
    return averageLatency.load();
}

//==============================================================================
// Event filtering

void GPUMIDIManager::setMIDIChannelFilter(int channel, bool enabled)
{
    if (channel >= 0 && channel < 16) {
        channelFilter[channel] = enabled;
    }
}

void GPUMIDIManager::setMIDITypeFilter(uint8_t messageType, bool enabled)
{
    typeFilter[messageType & 0xF0] = enabled;
}

//==============================================================================
// Private methods

GPUJMIDEvent GPUMIDIManager::juceMidiToGPUEvent(const juce::MidiMessage& message)
{
    GPUJMIDEvent event;
    
    if (message.isNoteOn()) {
        event.status = 0x90 | message.getChannel();
        event.data1 = static_cast<uint8_t>(message.getNoteNumber());
        event.data2 = static_cast<uint8_t>(message.getVelocity());
    } else if (message.isNoteOff()) {
        event.status = 0x80 | message.getChannel();
        event.data1 = static_cast<uint8_t>(message.getNoteNumber());
        event.data2 = static_cast<uint8_t>(message.getVelocity());
    } else if (message.isController()) {
        event.status = 0xB0 | message.getChannel();
        event.data1 = static_cast<uint8_t>(message.getControllerNumber());
        event.data2 = static_cast<uint8_t>(message.getControllerValue());
    } else {
        // Generic handling for other message types
        auto rawData = message.getRawData();
        event.status = rawData[0];
        event.data1 = message.getRawDataSize() > 1 ? rawData[1] : 0;
        event.data2 = message.getRawDataSize() > 2 ? rawData[2] : 0;
    }
    
    event.channel = message.getChannel();
    event.timestamp = jmidFramework ? jmidFramework->getCurrentGPUTimestamp() : 0;
    
    return event;
}

juce::MidiMessage GPUMIDIManager::gpuEventToJuceMidi(const GPUJMIDEvent& event)
{
    return juce::MidiMessage(event.status, event.data1, event.data2);
}

bool GPUMIDIManager::shouldProcessMIDIMessage(const juce::MidiMessage& message) const
{
    // Check channel filter
    int channel = message.getChannel() - 1; // JUCE uses 1-based channels
    if (channel >= 0 && channel < 16 && !channelFilter[channel]) {
        return false;
    }
    
    // Check message type filter
    uint8_t messageType = message.getRawData()[0] & 0xF0;
    if (!typeFilter[messageType]) {
        return false;
    }
    
    return true;
}

bool GPUMIDIManager::shouldProcessGPUEvent(const GPUJMIDEvent& event) const
{
    // Check channel filter
    if (event.channel >= 0 && event.channel < 16 && !channelFilter[event.channel]) {
        return false;
    }
    
    // Check message type filter
    uint8_t messageType = event.status & 0xF0;
    if (!typeFilter[messageType]) {
        return false;
    }
    
    return true;
}

void GPUMIDIManager::updatePerformanceMetrics()
{
    eventsSinceLastUpdate++;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastMetricsUpdate);
    
    if (elapsed.count() >= 1000) { // Update every second
        eventsPerSecond.store(eventsSinceLastUpdate);
        eventsSinceLastUpdate = 0;
        lastMetricsUpdate = now;
        
        // Calculate average latency if GPU framework is available
        if (jmidFramework) {
            averageLatency.store(jmidFramework->getAverageLatency());
        }
    }
}
