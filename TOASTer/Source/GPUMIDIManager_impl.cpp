#include "GPUMIDIManager.h"
#include <juce_audio_devices/juce_audio_devices.h>

//==============================================================================
GPUMIDIManager::GPUMIDIManager()
{
    // Initialize device manager
    deviceManager.initialiseWithDefaultDevices(0, 0); // Audio not needed, just MIDI
    
    // Initialize filters - enable all channels and message types by default
    channelFilter.fill(true);
    typeFilter.fill(true);
    
    lastMetricsUpdate = std::chrono::high_resolution_clock::now();
}

GPUMIDIManager::~GPUMIDIManager()
{
    shutdown();
}

bool GPUMIDIManager::initialize(jam::jmid_gpu::GPUJMIDFramework* framework)
{
    jmidFramework = framework;
    
    if (!jmidFramework) {
        if (onStatusUpdate) {
            onStatusUpdate("❌ GPU JMID Framework not available");
        }
        return false;
    }
    
    if (onStatusUpdate) {
        onStatusUpdate("✅ GPU MIDI Manager initialized with JMID framework");
    }
    
    return true;
}

void GPUMIDIManager::shutdown()
{
    closeMIDIInput();
    closeMIDIOutput();
    jmidFramework = nullptr;
}

//==============================================================================
// MIDI I/O management

bool GPUMIDIManager::openMIDIInput(int deviceIndex)
{
    closeMIDIInput(); // Close any existing input
    
    auto inputDevices = juce::MidiInput::getAvailableDevices();
    if (deviceIndex < 0 || deviceIndex >= inputDevices.size()) {
        return false;
    }
    
    auto device = inputDevices[deviceIndex];
    midiInput = juce::MidiInput::openDevice(device.identifier, this);
    
    if (midiInput) {
        midiInput->start();
        currentInputDevice = deviceIndex;
        
        if (onStatusUpdate) {
            onStatusUpdate("📥 GPU MIDI input opened: " + device.name);
        }
        
        return true;
    }
    
    return false;
}

bool GPUMIDIManager::openMIDIOutput(int deviceIndex)
{
    closeMIDIOutput(); // Close any existing output
    
    auto outputDevices = juce::MidiOutput::getAvailableDevices();
    if (deviceIndex < 0 || deviceIndex >= outputDevices.size()) {
        return false;
    }
    
    auto device = outputDevices[deviceIndex];
    midiOutput = juce::MidiOutput::openDevice(device.identifier);
    
    if (midiOutput) {
        currentOutputDevice = deviceIndex;
        
        if (onStatusUpdate) {
            onStatusUpdate("📤 GPU MIDI output opened: " + device.name);
        }
        
        return true;
    }
    
    return false;
}

void GPUMIDIManager::closeMIDIInput()
{
    if (midiInput) {
        midiInput->stop();
        midiInput.reset();
        currentInputDevice = -1;
        
        if (onStatusUpdate) {
            onStatusUpdate("📥 GPU MIDI input closed");
        }
    }
}

void GPUMIDIManager::closeMIDIOutput()
{
    if (midiOutput) {
        midiOutput.reset();
        currentOutputDevice = -1;
        
        if (onStatusUpdate) {
            onStatusUpdate("📤 GPU MIDI output closed");
        }
    }
}

//==============================================================================
// Get available MIDI devices

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
    // Create JUCE MIDI message for local output
    juce::MidiMessage localMessage(status, data1, data2);
    
    // Send to local MIDI output if available
    if (midiOutput && shouldProcessMIDIMessage(localMessage)) {
        midiOutput->sendMessageNow(localMessage);
    }
    
    // Convert to GPU JMID event for network transmission
    if (jmidFramework) {
        jam::jmid_gpu::GPUMIDIEvent gpuEvent;
        gpuEvent.status = status;
        gpuEvent.data1 = data1;
        gpuEvent.data2 = data2;
        gpuEvent.channel = status & 0x0F;
        gpuEvent.timestamp_frame = jam::gpu_native::GPUTimebase::is_initialized() ? jam::gpu_native::GPUTimebase::get_current_time_ns() : 0;
        
        if (shouldProcessGPUEvent(gpuEvent)) {
            // Schedule event through GPU event queue
            if (jmidFramework && jmidFramework->get_event_queue()) {
                jmidFramework->get_event_queue()->schedule_event(gpuEvent);
            }
            updatePerformanceMetrics();
        }
    }
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
// MIDI callback from JUCE input

void GPUMIDIManager::handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message)
{
    if (!shouldProcessMIDIMessage(message)) {
        return;
    }
    
    // Update performance metrics
    updatePerformanceMetrics();
    
    // Call local callback if set
    if (onMIDIInputReceived) {
        onMIDIInputReceived(message);
    }
    
    // Convert to GPU JMID event and send to network
    if (jmidFramework) {
        auto gpuEvent = juceMidiToGPUEvent(message);
        if (shouldProcessGPUEvent(gpuEvent)) {
            // Schedule event through GPU event queue
            if (jmidFramework->get_event_queue()) {
                jmidFramework->get_event_queue()->schedule_event(gpuEvent);
            }
        }
    }
}

//==============================================================================
// GPU MIDI event callback from network

void GPUMIDIManager::handleNetworkMIDIEvent(const jam::jmid_gpu::GPUMIDIEvent& event)
{
    if (!shouldProcessGPUEvent(event)) {
        return;
    }
    
    // Update performance metrics
    updatePerformanceMetrics();
    
    // Call network callback if set
    if (onNetworkMIDIReceived) {
        onNetworkMIDIReceived(event);
    }
    
    // Convert to JUCE MIDI and send to local output
    if (midiOutput) {
        auto juceMessage = gpuEventToJuceMidi(event);
        if (shouldProcessMIDIMessage(juceMessage)) {
            midiOutput->sendMessageNow(juceMessage);
        }
    }
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
    typeFilter[messageType] = enabled;
}

//==============================================================================
// Private helper methods

jam::jmid_gpu::GPUMIDIEvent GPUMIDIManager::juceMidiToGPUEvent(const juce::MidiMessage& message)
{
    jam::jmid_gpu::GPUMIDIEvent event;
    event.status = message.getRawData()[0];
    event.data1 = message.getRawDataSize() > 1 ? message.getRawData()[1] : 0;
    event.data2 = message.getRawDataSize() > 2 ? message.getRawData()[2] : 0;
    event.channel = message.getChannel() - 1; // JUCE uses 1-based, GPU uses 0-based
    event.timestamp_frame = jam::gpu_native::GPUTimebase::is_initialized() ? jam::gpu_native::GPUTimebase::get_current_time_ns() : 0;
    
    return event;
}

juce::MidiMessage GPUMIDIManager::gpuEventToJuceMidi(const jam::jmid_gpu::GPUMIDIEvent& event)
{
    return juce::MidiMessage(event.status, event.data1, event.data2);
}

bool GPUMIDIManager::shouldProcessMIDIMessage(const juce::MidiMessage& message) const
{
    // Check channel filter
    if (message.getChannel() > 0 && message.getChannel() <= 16) {
        if (!channelFilter[message.getChannel() - 1]) {
            return false;
        }
    }
    
    // Check message type filter
    uint8_t status = message.getRawData()[0] & 0xF0;
    if (!typeFilter[status]) {
        return false;
    }
    
    return true;
}

bool GPUMIDIManager::shouldProcessGPUEvent(const jam::jmid_gpu::GPUMIDIEvent& event) const
{
    // Check channel filter
    if (event.channel < 16 && !channelFilter[event.channel]) {
        return false;
    }
    
    // Check message type filter
    uint8_t status = event.status & 0xF0;
    if (!typeFilter[status]) {
        return false;
    }
    
    return true;
}

void GPUMIDIManager::updatePerformanceMetrics()
{
    eventsSinceLastUpdate++;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto timeSinceUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastMetricsUpdate).count();
    
    if (timeSinceUpdate >= 1000) { // Update every second
        eventsPerSecond.store(eventsSinceLastUpdate);
        eventsSinceLastUpdate = 0;
        lastMetricsUpdate = now;
        
        // Calculate average latency using performance stats
        if (jmidFramework) {
            auto stats = jmidFramework->get_performance_stats();
            averageLatency.store(stats.average_dispatch_latency_us);
        }
    }
}
