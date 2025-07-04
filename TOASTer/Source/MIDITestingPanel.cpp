#include "MIDITestingPanel.h"
#include "MIDIManager.h"
#include "JMIDParser.h"
#include <nlohmann/json.hpp>

//==============================================================================
// Helper function for emoji-compatible font setup
juce::Font MIDITestingPanel::getEmojiCompatibleFont(float size)
{
    // On macOS, prefer system fonts that support emoji
    #if JUCE_MAC
        return juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
    #elif JUCE_WINDOWS
        return juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
    #else
        return juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
    #endif
}

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
    titleLabel.setFont(juce::Font(juce::FontOptions(16.0f).withStyle("bold")));
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
    
    noteSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    noteSlider.setRange(0, 127, 1);
    noteSlider.setValue(60); // Middle C
    noteSlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 60, 20);
    addAndMakeVisible(noteSlider);
    
    velocityLabel.setText("Velocity:", juce::dontSendNotification);
    addAndMakeVisible(velocityLabel);
    
    velocitySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    velocitySlider.setRange(1, 127, 1);
    velocitySlider.setValue(100);
    velocitySlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 60, 20);
    addAndMakeVisible(velocitySlider);
    
    sendTestNoteButton.setButtonText("Send Note");
    sendTestNoteButton.onClick = [this] { sendTestNoteClicked(); };
    addAndMakeVisible(sendTestNoteButton);

    // MIDI log
    logLabel.setText("MIDI Log:", juce::dontSendNotification);
    addAndMakeVisible(logLabel);
    
    logDisplay.setMultiLine(true);
    logDisplay.setReadOnly(true);
    logDisplay.setScrollbarsShown(true);
    logDisplay.setCaretVisible(false);
    logDisplay.setFont(getEmojiCompatibleFont(10.0f)); // Use emoji-compatible font
    addAndMakeVisible(logDisplay);
    
    clearLogButton.setButtonText("Clear Log");
    clearLogButton.onClick = [this] { logDisplay.clear(); };
    addAndMakeVisible(clearLogButton);

    startTimer(50); // Update display at 20 FPS
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
        
        // Only refresh once on initial setup
        refreshDeviceLists();
    }
}

void MIDITestingPanel::changeListenerCallback(juce::ChangeBroadcaster* /*source*/)
{
    // MIDI device list has changed, refresh our combo boxes
    // This should only happen when devices are actually plugged/unplugged
    refreshDeviceLists();
}

void MIDITestingPanel::timerCallback()
{
    // This could be used for periodic updates if needed
    // For now, it's just to maintain the timer functionality
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
    
    // Only call refreshDeviceList when we actually need to update the UI
    // This should be driven by real device change events, not polling
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
    
    logMessage("🔄 Device lists refreshed (event-driven)");
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
