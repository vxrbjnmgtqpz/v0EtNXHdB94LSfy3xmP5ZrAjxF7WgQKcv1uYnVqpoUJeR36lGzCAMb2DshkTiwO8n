#include "MIDITestingPanel.h"
#include "MIDIManager.h"
#include "MessageFactory.h"
#include <nlohmann/json.hpp>

//==============================================================================
MIDITestingPanel::MIDITestingPanel()
{
    // Initialize BassoonParser (stubbed for now)
    try {
        parser = std::make_unique<JMID::BassoonParser>();
    } catch (...) {
        // BassoonParser might not be fully implemented yet
        parser = nullptr;
    }

    // Title
    titleLabel.setText("MIDI Testing & Device Selection", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    // Device selection
    inputDeviceLabel.setText("MIDI Input:", juce::dontSendNotification);
    addAndMakeVisible(inputDeviceLabel);
    
    inputDeviceSelector.addItem("No Input Device", 1);
    inputDeviceSelector.setSelectedId(1);
    inputDeviceSelector.onChange = [this] { inputDeviceChanged(); };
    addAndMakeVisible(inputDeviceSelector);
    
    outputDeviceLabel.setText("MIDI Output:", juce::dontSendNotification);
    addAndMakeVisible(outputDeviceLabel);
    
    outputDeviceSelector.addItem("No Output Device", 1);
    outputDeviceSelector.setSelectedId(1);
    outputDeviceSelector.onChange = [this] { outputDeviceChanged(); };
    addAndMakeVisible(outputDeviceSelector);
    
    refreshDevicesButton.setButtonText("Refresh Devices");
    refreshDevicesButton.onClick = [this] { refreshDeviceLists(); };
    addAndMakeVisible(refreshDevicesButton);

    // Test controls
    channelLabel.setText("Channel:", juce::dontSendNotification);
    addAndMakeVisible(channelLabel);
    
    midiChannelSelector.addItem("1", 1);
    for (int i = 2; i <= 16; ++i)
        midiChannelSelector.addItem(juce::String(i), i);
    midiChannelSelector.setSelectedId(1);
    addAndMakeVisible(midiChannelSelector);

    noteLabel.setText("Note:", juce::dontSendNotification);
    addAndMakeVisible(noteLabel);
    
    noteSlider.setRange(0, 127, 1);
    noteSlider.setValue(60); // Middle C
    noteSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    noteSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50);
    addAndMakeVisible(noteSlider);

    velocityLabel.setText("Velocity:", juce::dontSendNotification);
    addAndMakeVisible(velocityLabel);
    
    velocitySlider.setRange(0, 127, 1);
    velocitySlider.setValue(100);
    velocitySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    velocitySlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50);
    addAndMakeVisible(velocitySlider);

    sendTestNoteButton.setButtonText("Send Test Note");
    sendTestNoteButton.onClick = [this] { sendTestNoteClicked(); };
    addAndMakeVisible(sendTestNoteButton);

    clearLogButton.setButtonText("Clear Log");
    clearLogButton.onClick = [this] { clearLogClicked(); };
    addAndMakeVisible(clearLogButton);

    logDisplay.setMultiLine(true);
    logDisplay.setReturnKeyStartsNewLine(true);
    logDisplay.setReadOnly(true);
    logDisplay.setScrollbarsShown(true);
    logDisplay.setCaretVisible(false);
    logDisplay.setPopupMenuEnabled(false);
    logDisplay.setColour(juce::TextEditor::backgroundColourId, juce::Colours::black);
    logDisplay.setColour(juce::TextEditor::textColourId, juce::Colours::green);
    logDisplay.setText("MIDI Log (ready for real I/O):\n");
    addAndMakeVisible(logDisplay);
}

MIDITestingPanel::~MIDITestingPanel()
{
    if (midiManager != nullptr)
        midiManager->removeChangeListener(this);
}

//==============================================================================
void MIDITestingPanel::setMIDIManager(MIDIManager* manager)
{
    if (midiManager != nullptr)
        midiManager->removeChangeListener(this);
    
    midiManager = manager;
    
    if (midiManager != nullptr)
    {
        midiManager->addChangeListener(this);
        
        // Set up message callback to receive incoming MIDI
        midiManager->setMessageCallback(
            [this](std::shared_ptr<JMID::MIDIMessage> message) {
                onMIDIMessageReceived(message);
            });
        
        refreshDeviceLists();
    }
}

void MIDITestingPanel::changeListenerCallback(juce::ChangeBroadcaster* /*source*/)
{
    // MIDI device list has changed, refresh our combo boxes
    refreshDeviceLists();
}

//==============================================================================
void MIDITestingPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void MIDITestingPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    // Device selection row
    auto deviceRow = bounds.removeFromTop(25);
    inputDeviceLabel.setBounds(deviceRow.removeFromLeft(80));
    inputDeviceSelector.setBounds(deviceRow.removeFromLeft(150));
    deviceRow.removeFromLeft(10);
    outputDeviceLabel.setBounds(deviceRow.removeFromLeft(85));
    outputDeviceSelector.setBounds(deviceRow.removeFromLeft(150));
    deviceRow.removeFromLeft(10);
    refreshDevicesButton.setBounds(deviceRow.removeFromLeft(120));
    
    bounds.removeFromTop(10);
    
    // Control row
    auto controlRow = bounds.removeFromTop(25);
    channelLabel.setBounds(controlRow.removeFromLeft(60));
    midiChannelSelector.setBounds(controlRow.removeFromLeft(60));
    controlRow.removeFromLeft(10);
    noteLabel.setBounds(controlRow.removeFromLeft(40));
    noteSlider.setBounds(controlRow.removeFromLeft(120));
    controlRow.removeFromLeft(10);
    velocityLabel.setBounds(controlRow.removeFromLeft(60));
    velocitySlider.setBounds(controlRow.removeFromLeft(120));
    
    bounds.removeFromTop(5);
    
    // Button row
    auto buttonRow = bounds.removeFromTop(25);
    sendTestNoteButton.setBounds(buttonRow.removeFromLeft(120));
    buttonRow.removeFromLeft(5);
    clearLogButton.setBounds(buttonRow.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    logDisplay.setBounds(bounds);
}

//==============================================================================
void MIDITestingPanel::refreshDeviceLists()
{
    if (midiManager == nullptr)
        return;
    
    // Refresh the device lists
    midiManager->refreshDeviceList();
    
    // Update input device selector
    inputDeviceSelector.clear();
    inputDeviceSelector.addItem("No Input Device", 1);
    auto inputNames = midiManager->getInputDeviceNames();
    for (int i = 0; i < inputNames.size(); ++i)
        inputDeviceSelector.addItem(inputNames[i], i + 2);
    
    // Update output device selector
    outputDeviceSelector.clear();
    outputDeviceSelector.addItem("No Output Device", 1);
    auto outputNames = midiManager->getOutputDeviceNames();
    for (int i = 0; i < outputNames.size(); ++i)
        outputDeviceSelector.addItem(outputNames[i], i + 2);
    
    logMessage("🔄 Device lists refreshed");
}

void MIDITestingPanel::inputDeviceChanged()
{
    if (midiManager == nullptr)
        return;
    
    int selectedId = inputDeviceSelector.getSelectedId();
    if (selectedId == 1)
    {
        midiManager->closeInputDevice();
        logMessage("📥 Input device closed");
    }
    else
    {
        int deviceIndex = selectedId - 2;
        if (midiManager->openInputDevice(deviceIndex))
        {
            logMessage("📥 Opened input: " + midiManager->getCurrentInputDeviceName());
        }
        else
        {
            logMessage("❌ Failed to open input device");
        }
    }
}

void MIDITestingPanel::outputDeviceChanged()
{
    if (midiManager == nullptr)
        return;
    
    int selectedId = outputDeviceSelector.getSelectedId();
    if (selectedId == 1)
    {
        midiManager->closeOutputDevice();
        logMessage("📤 Output device closed");
    }
    else
    {
        int deviceIndex = selectedId - 2;
        if (midiManager->openOutputDevice(deviceIndex))
        {
            logMessage("📤 Opened output: " + midiManager->getCurrentOutputDeviceName());
        }
        else
        {
            logMessage("❌ Failed to open output device");
        }
    }
}

//==============================================================================
void MIDITestingPanel::sendTestNoteClicked()
{
    if (midiManager == nullptr)
    {
        logMessage("❌ MIDI Manager not available");
        return;
    }
    
    auto note = static_cast<int>(noteSlider.getValue());
    auto velocity = static_cast<int>(velocitySlider.getValue());
    auto channel = midiChannelSelector.getSelectedId();
    
    try {
        // Create JMID NoteOn message
        auto timestamp = std::chrono::high_resolution_clock::now();
        auto noteOnMessage = std::make_shared<JMID::NoteOnMessage>(
            static_cast<uint8_t>(channel),
            static_cast<uint8_t>(note),
            static_cast<uint32_t>(velocity),
            timestamp
        );
        
        // Send via MIDI manager
        midiManager->sendJMIDMessage(noteOnMessage);
        
        // Log the message
        std::string json = noteOnMessage->toJSON();
        logMessage("🎵 Sent Note On: Ch=" + juce::String(channel) + 
                  ", Note=" + juce::String(note) + 
                  ", Vel=" + juce::String(velocity));
        logMessage("   JSON: " + juce::String(json.substr(0, 100)) + "...");
        
        // Schedule note off after 500ms
        juce::Timer::callAfterDelay(500, [this, channel, note, timestamp]() {
            auto noteOffMessage = std::make_shared<JMID::NoteOffMessage>(
                static_cast<uint8_t>(channel),
                static_cast<uint8_t>(note),
                0, // Release velocity
                timestamp + std::chrono::milliseconds(500)
            );
            
            midiManager->sendJMIDMessage(noteOffMessage);
            logMessage("🎵 Sent Note Off: Ch=" + juce::String(channel) + ", Note=" + juce::String(note));
        });
        
    } catch (const std::exception& e) {
        logMessage("❌ Exception: " + juce::String(e.what()));
    }
}

void MIDITestingPanel::onMIDIMessageReceived(std::shared_ptr<JMID::MIDIMessage> message)
{
    if (message == nullptr)
        return;
    
    // This will be called from the message processing thread
    juce::MessageManager::callAsync([this, message]() {
        std::string json = message->toJSON();
        
        switch (message->getType()) {
            case JMID::MessageType::NOTE_ON:
            {
                auto noteOn = std::dynamic_pointer_cast<JMID::NoteOnMessage>(message);
                if (noteOn) {
                    logMessage("🎵 Received Note On: Ch=" + juce::String(noteOn->getChannel()) + 
                              ", Note=" + juce::String(noteOn->getNote()) + 
                              ", Vel=" + juce::String(noteOn->getVelocity()));
                }
                break;
            }
            case JMID::MessageType::NOTE_OFF:
            {
                auto noteOff = std::dynamic_pointer_cast<JMID::NoteOffMessage>(message);
                if (noteOff) {
                    logMessage("🎵 Received Note Off: Ch=" + juce::String(noteOff->getChannel()) + 
                              ", Note=" + juce::String(noteOff->getNote()));
                }
                break;
            }
            case JMID::MessageType::CONTROL_CHANGE:
            {
                auto cc = std::dynamic_pointer_cast<JMID::ControlChangeMessage>(message);
                if (cc) {
                    logMessage("🎛️ Received CC: Ch=" + juce::String(cc->getChannel()) + 
                              ", CC=" + juce::String(cc->getController()) + 
                              ", Val=" + juce::String(cc->getValue()));
                }
                break;
            }
            default:
                logMessage("📨 Received MIDI: " + juce::String(json.substr(0, 80)) + "...");
                break;
        }
    });
}

void MIDITestingPanel::logMessage(const juce::String& message)
{
    auto timestamp = juce::Time::getCurrentTime().toString(true, true, true, true);
    auto logText = logDisplay.getText();
    logText += "[" + timestamp + "] " + message + "\n";
    logDisplay.setText(logText);
    logDisplay.moveCaretToEnd();
}

void MIDITestingPanel::clearLogClicked()
{
    logDisplay.setText("MIDI Log (ready for real I/O):\n");
}
