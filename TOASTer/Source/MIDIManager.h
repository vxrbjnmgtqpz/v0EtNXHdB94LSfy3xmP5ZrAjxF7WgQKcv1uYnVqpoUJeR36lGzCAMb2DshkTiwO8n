#pragma once

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_events/juce_events.h>
#include <memory>
#include <vector>
#include <functional>
#include "JSONMIDIMessage.h"
#include "LockFreeQueue.h"

//==============================================================================
/**
 * MIDI I/O Manager for TOASTer application
 * Handles real-time MIDI input/output and integrates with JSONMIDI framework
 */
class MIDIManager : public juce::MidiInputCallback,
                    public juce::ChangeBroadcaster
{
public:
    MIDIManager();
    ~MIDIManager() override;

    // Device Management
    void refreshDeviceList();
    juce::StringArray getInputDeviceNames() const;
    juce::StringArray getOutputDeviceNames() const;
    
    bool openInputDevice(int deviceIndex);
    bool openOutputDevice(int deviceIndex);
    void closeInputDevice();
    void closeOutputDevice();
    
    bool isInputDeviceOpen() const;
    bool isOutputDeviceOpen() const;
    juce::String getCurrentInputDeviceName() const;
    juce::String getCurrentOutputDeviceName() const;

    // MIDI I/O
    void sendMIDIMessage(const juce::MidiMessage& message);
    void sendJSONMIDIMessage(std::shared_ptr<JSONMIDI::MIDIMessage> jsonMidiMessage);
    
    // Real-time message queue access
    using MessageCallback = std::function<void(std::shared_ptr<JSONMIDI::MIDIMessage>)>;
    void setMessageCallback(MessageCallback callback);
    
    // Statistics
    struct Statistics {
        uint64_t messagesReceived = 0;
        uint64_t messagesSent = 0;
        uint64_t bytesReceived = 0;
        uint64_t bytesSent = 0;
        double averageLatencyMs = 0.0;
        uint32_t droppedMessages = 0;
    };
    
    Statistics getStatistics() const;
    void resetStatistics();

    // MidiInputCallback interface
    void handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message) override;

private:
    // Device lists
    juce::Array<juce::MidiDeviceInfo> inputDevices;
    juce::Array<juce::MidiDeviceInfo> outputDevices;
    
    // Device change monitoring
    juce::MidiDeviceListConnection deviceListConnection;
    
    // Active devices
    std::unique_ptr<juce::MidiInput> currentInputDevice;
    std::unique_ptr<juce::MidiOutput> currentOutputDevice;
    
    // Message processing
    MessageCallback messageCallback;
    
    // Lock-free queues for real-time safety
    JSONMIDI::LockFreeQueue<std::shared_ptr<JSONMIDI::MIDIMessage>, 1024> incomingMessageQueue;
    
    // For outgoing MIDI, use a simple struct to hold the raw data
    struct MIDIData {
        std::array<uint8_t, 16> data;
        size_t size;
        
        MIDIData() : size(0) {}
        MIDIData(const juce::MidiMessage& msg) : size(std::min(msg.getRawDataSize(), 16)) {
            std::memcpy(data.data(), msg.getRawData(), size);
        }
        
        juce::MidiMessage toJuceMessage() const {
            return juce::MidiMessage(data.data(), static_cast<int>(size));
        }
    };
    
    JSONMIDI::LockFreeQueue<MIDIData, 1024> outgoingMessageQueue;
    
    // Statistics tracking
    mutable std::mutex statisticsMutex;
    Statistics statistics;
    
    // Helper methods
    std::shared_ptr<JSONMIDI::MIDIMessage> convertJuceMidiToJSONMIDI(const juce::MidiMessage& juceMidi);
    juce::MidiMessage convertJSONMIDIToJuce(std::shared_ptr<JSONMIDI::MIDIMessage> jsonMidi);
    
    // Message processing timer
    class MessageProcessor : public juce::Timer
    {
    public:
        MessageProcessor(MIDIManager& owner) : owner_(owner) {}
        void timerCallback() override;
    private:
        MIDIManager& owner_;
    } messageProcessor;
    
    friend class MessageProcessor;
    void processMessages();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MIDIManager)
};
