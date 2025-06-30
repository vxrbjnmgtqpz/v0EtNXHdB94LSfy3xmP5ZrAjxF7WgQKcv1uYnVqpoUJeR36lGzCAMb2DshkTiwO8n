#include "MIDIManager.h"
#include "JSONMIDIParser.h"
#include <chrono>

//==============================================================================
MIDIManager::MIDIManager() : messageProcessor(*this)
{
    // Initial device list population
    refreshDeviceList();
    
    // Set up automatic device change monitoring using JUCE's system
    deviceListConnection = juce::MidiDeviceListConnection::make([this]() {
        // This callback will only be triggered when devices are actually plugged/unplugged
        refreshDeviceList();
    });
    
    // Start message processing timer at 60 FPS (16.67ms intervals)
    messageProcessor.startTimer(16);
}

MIDIManager::~MIDIManager()
{
    messageProcessor.stopTimer();
    closeInputDevice();
    closeOutputDevice();
}

//==============================================================================
void MIDIManager::refreshDeviceList()
{
    auto newInputDevices = juce::MidiInput::getAvailableDevices();
    auto newOutputDevices = juce::MidiOutput::getAvailableDevices();
    
    // Only send change message if the device lists actually changed
    bool devicesChanged = false;
    
    if (newInputDevices.size() != inputDevices.size() || 
        newOutputDevices.size() != outputDevices.size())
    {
        devicesChanged = true;
    }
    else
    {
        // Check if device names/identifiers changed
        for (int i = 0; i < newInputDevices.size(); ++i)
        {
            if (i >= inputDevices.size() || 
                newInputDevices[i].identifier != inputDevices[i].identifier ||
                newInputDevices[i].name != inputDevices[i].name)
            {
                devicesChanged = true;
                break;
            }
        }
        
        if (!devicesChanged)
        {
            for (int i = 0; i < newOutputDevices.size(); ++i)
            {
                if (i >= outputDevices.size() || 
                    newOutputDevices[i].identifier != outputDevices[i].identifier ||
                    newOutputDevices[i].name != outputDevices[i].name)
                {
                    devicesChanged = true;
                    break;
                }
            }
        }
    }
    
    // Update the cached device lists
    inputDevices = std::move(newInputDevices);
    outputDevices = std::move(newOutputDevices);
    
    // Only notify listeners if something actually changed
    if (devicesChanged)
    {
        sendChangeMessage();
    }
}

juce::StringArray MIDIManager::getInputDeviceNames() const
{
    juce::StringArray names;
    for (const auto& device : inputDevices)
        names.add(device.name);
    return names;
}

juce::StringArray MIDIManager::getOutputDeviceNames() const
{
    juce::StringArray names;
    for (const auto& device : outputDevices)
        names.add(device.name);
    return names;
}

//==============================================================================
bool MIDIManager::openInputDevice(int deviceIndex)
{
    closeInputDevice();
    
    if (juce::isPositiveAndBelow(deviceIndex, inputDevices.size()))
    {
        auto device = juce::MidiInput::openDevice(inputDevices[deviceIndex].identifier, this);
        if (device != nullptr)
        {
            currentInputDevice = std::move(device);
            currentInputDevice->start();
            sendChangeMessage();
            return true;
        }
    }
    return false;
}

bool MIDIManager::openOutputDevice(int deviceIndex)
{
    closeOutputDevice();
    
    if (juce::isPositiveAndBelow(deviceIndex, outputDevices.size()))
    {
        auto device = juce::MidiOutput::openDevice(outputDevices[deviceIndex].identifier);
        if (device != nullptr)
        {
            currentOutputDevice = std::move(device);
            sendChangeMessage();
            return true;
        }
    }
    return false;
}

void MIDIManager::closeInputDevice()
{
    if (currentInputDevice != nullptr)
    {
        currentInputDevice->stop();
        currentInputDevice.reset();
        sendChangeMessage();
    }
}

void MIDIManager::closeOutputDevice()
{
    if (currentOutputDevice != nullptr)
    {
        currentOutputDevice.reset();
        sendChangeMessage();
    }
}

//==============================================================================
bool MIDIManager::isInputDeviceOpen() const
{
    return currentInputDevice != nullptr;
}

bool MIDIManager::isOutputDeviceOpen() const
{
    return currentOutputDevice != nullptr;
}

juce::String MIDIManager::getCurrentInputDeviceName() const
{
    return currentInputDevice ? currentInputDevice->getName() : "None";
}

juce::String MIDIManager::getCurrentOutputDeviceName() const
{
    return currentOutputDevice ? currentOutputDevice->getName() : "None";
}

//==============================================================================
void MIDIManager::sendMIDIMessage(const juce::MidiMessage& message)
{
    if (currentOutputDevice != nullptr)
    {
        // For real-time safety, queue the message as MIDIData
        MIDIData midiData(message);
        if (outgoingMessageQueue.tryPush(midiData))
        {
            std::lock_guard<std::mutex> lock(statisticsMutex);
            statistics.messagesSent++;
            statistics.bytesSent += static_cast<uint64_t>(message.getRawDataSize());
        }
        else
        {
            std::lock_guard<std::mutex> lock(statisticsMutex);
            statistics.droppedMessages++;
        }
    }
}

void MIDIManager::sendJSONMIDIMessage(std::shared_ptr<JSONMIDI::MIDIMessage> jsonMidiMessage)
{
    if (jsonMidiMessage != nullptr)
    {
        auto juceMidiMessage = convertJSONMIDIToJuce(jsonMidiMessage);
        sendMIDIMessage(juceMidiMessage);
    }
}

//==============================================================================
void MIDIManager::setMessageCallback(MessageCallback callback)
{
    messageCallback = std::move(callback);
}

//==============================================================================
MIDIManager::Statistics MIDIManager::getStatistics() const
{
    std::lock_guard<std::mutex> lock(statisticsMutex);
    return statistics;
}

void MIDIManager::resetStatistics()
{
    std::lock_guard<std::mutex> lock(statisticsMutex);
    statistics = Statistics{};
}

//==============================================================================
void MIDIManager::handleIncomingMidiMessage(juce::MidiInput* /*source*/, const juce::MidiMessage& message)
{
    // Convert JUCE MIDI message to JSONMIDI message
    auto jsonMidiMessage = convertJuceMidiToJSONMIDI(message);
    
    if (jsonMidiMessage != nullptr)
    {
        // Queue for processing on the message thread
        if (incomingMessageQueue.tryPush(jsonMidiMessage))
        {
            std::lock_guard<std::mutex> lock(statisticsMutex);
            statistics.messagesReceived++;
            statistics.bytesReceived += static_cast<uint64_t>(message.getRawDataSize());
        }
        else
        {
            std::lock_guard<std::mutex> lock(statisticsMutex);
            statistics.droppedMessages++;
        }
    }
}

//==============================================================================
std::shared_ptr<JSONMIDI::MIDIMessage> MIDIManager::convertJuceMidiToJSONMIDI(const juce::MidiMessage& juceMidi)
{
    auto timestamp = std::chrono::high_resolution_clock::now();
    
    try {
        if (juceMidi.isNoteOn())
        {
            return std::make_shared<JSONMIDI::NoteOnMessage>(
                juceMidi.getChannel(),
                juceMidi.getNoteNumber(),
                juceMidi.getVelocity(),
                timestamp
            );
        }
        else if (juceMidi.isNoteOff())
        {
            return std::make_shared<JSONMIDI::NoteOffMessage>(
                juceMidi.getChannel(),
                juceMidi.getNoteNumber(),
                juceMidi.getVelocity(),
                timestamp
            );
        }
        else if (juceMidi.isController())
        {
            return std::make_shared<JSONMIDI::ControlChangeMessage>(
                juceMidi.getChannel(),
                juceMidi.getControllerNumber(),
                juceMidi.getControllerValue(),
                timestamp
            );
        }
        // Add more message types as needed
    }
    catch (const std::exception&)
    {
        // Log error in real implementation
    }
    
    return nullptr;
}

juce::MidiMessage MIDIManager::convertJSONMIDIToJuce(std::shared_ptr<JSONMIDI::MIDIMessage> jsonMidi)
{
    if (jsonMidi == nullptr)
        return juce::MidiMessage();
    
    auto midiBytes = jsonMidi->toMIDIBytes();
    if (midiBytes.empty())
        return juce::MidiMessage();
    
    return juce::MidiMessage(midiBytes.data(), static_cast<int>(midiBytes.size()));
}

//==============================================================================
void MIDIManager::processMessages()
{
    // Process incoming messages
    std::shared_ptr<JSONMIDI::MIDIMessage> incomingMessage;
    while (incomingMessageQueue.tryPop(incomingMessage))
    {
        if (messageCallback && incomingMessage)
        {
            messageCallback(incomingMessage);
        }
    }
    
    // Process outgoing messages
    MIDIData outgoingData;
    while (outgoingMessageQueue.tryPop(outgoingData))
    {
        if (currentOutputDevice != nullptr)
        {
            auto juceMessage = outgoingData.toJuceMessage();
            currentOutputDevice->sendMessageNow(juceMessage);
        }
    }
}

//==============================================================================
void MIDIManager::MessageProcessor::timerCallback()
{
    owner_.processMessages();
}
