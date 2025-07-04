#include "MIDITestingPanel.h"
#include <nlohmann/json.hpp>
#include <chrono>

MIDITestingPanel::MIDITestingPanel()
    : sendTestNoteButton("Send Test Note"), clearLogButton("Clear Log")
{
    // Initialize JMID Framework components
    parser = std::make_unique<JMID::BassoonParser>();
    
    // Set up title
    titleLabel.setText("MIDI Testing", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);
    
    // Set up MIDI channel selector
    midiChannelSelector.addItem("Channel 1", 1);
    midiChannelSelector.addItem("Channel 2", 2);
    midiChannelSelector.addItem("Channel 3", 3);
    midiChannelSelector.addItem("Channel 4", 4);
    midiChannelSelector.setSelectedId(1);
    addAndMakeVisible(midiChannelSelector);
    
    // Set up note slider
    noteSlider.setRange(21, 108, 1); // Piano range
    noteSlider.setValue(60); // Middle C
    noteSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    noteSlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 50, 20);
    addAndMakeVisible(noteSlider);
    
    // Set up velocity slider
    velocitySlider.setRange(1, 127, 1);
    velocitySlider.setValue(100);
    velocitySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    velocitySlider.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 50, 20);
    addAndMakeVisible(velocitySlider);
    
    // Set up buttons
    sendTestNoteButton.onClick = [this] { sendTestNoteClicked(); };
    addAndMakeVisible(sendTestNoteButton);
    
    clearLogButton.onClick = [this] { clearLogClicked(); };
    addAndMakeVisible(clearLogButton);
    
    // Set up log display
    logDisplay.setMultiLine(true);
    logDisplay.setReadOnly(true);
    logDisplay.setColour(juce::TextEditor::backgroundColourId, juce::Colours::black);
    logDisplay.setColour(juce::TextEditor::textColourId, juce::Colours::green);
    logDisplay.setText("MIDI Log:\n");
    addAndMakeVisible(logDisplay);
}

MIDITestingPanel::~MIDITestingPanel()
{
}

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
    
    auto row = bounds.removeFromTop(25);
    midiChannelSelector.setBounds(row.removeFromLeft(100));
    
    bounds.removeFromTop(5);
    
    row = bounds.removeFromTop(25);
    auto labelWidth = 60;
    auto sliderWidth = 120;
    
    // Note slider
    row.removeFromLeft(labelWidth);
    noteSlider.setBounds(row.removeFromLeft(sliderWidth));
    row.removeFromLeft(10);
    
    // Velocity slider  
    row.removeFromLeft(labelWidth);
    velocitySlider.setBounds(row.removeFromLeft(sliderWidth));
    
    bounds.removeFromTop(5);
    
    row = bounds.removeFromTop(25);
    sendTestNoteButton.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(5);
    clearLogButton.setBounds(row.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    logDisplay.setBounds(bounds);
}

void MIDITestingPanel::sendTestNoteClicked()
{
    auto note = static_cast<int>(noteSlider.getValue());
    auto velocity = static_cast<int>(velocitySlider.getValue());
    auto channel = midiChannelSelector.getSelectedId();
    
    try {
        // Create JSON data for note on message
        nlohmann::json noteOnData;
        noteOnData["type"] = "note_on";
        noteOnData["channel"] = channel;
        noteOnData["note"] = note;
        noteOnData["velocity"] = velocity;
        noteOnData["timestamp"] = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        std::string json = noteOnData.dump();
        
        // Create JMID message using our framework
        auto noteOnMessage = JMID::MessageFactory::createFromJSON(json);
        
        if (noteOnMessage) {
            // Test parsing
            auto parsedMessage = parser->parseMessage(json);
            
            if (parsedMessage) {
                logMessage("✅ JMID Note On created and parsed successfully");
                logMessage("Channel: " + juce::String(channel) + ", Note: " + juce::String(note) + ", Velocity: " + juce::String(velocity));
                logMessage("JSON: " + juce::String(json.substr(0, 80)) + "...");
                
                // Convert to MIDI bytes for verification
                auto midiBytes = noteOnMessage->toMIDIBytes();
                juce::String midiHex = "MIDI bytes: ";
                for (auto byte : midiBytes) {
                    midiHex += juce::String::toHexString(byte) + " ";
                }
                logMessage(midiHex);
            } else {
                logMessage("❌ Failed to parse JMID message");
            }
        } else {
            logMessage("❌ Failed to create JMID message");
        }
        
    } catch (const std::exception& e) {
        logMessage("❌ Exception: " + juce::String(e.what()));
    }
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
    logDisplay.setText("MIDI Log:\n");
}