#include "PluginProcessor.h"
#include "PluginEditor.h"

PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    setSize (400, 150);

    addAndMakeVisible(inputBox);
    inputBox.setMultiLine(false);
    inputBox.setTextToShowWhenEmpty("Type a message...", juce::Colours::grey);

    addAndMakeVisible(sendButton);
    sendButton.setButtonText("Send");

    sendButton.onClick = [this]() {
        auto msg = inputBox.getText();
        juce::DynamicObject::Ptr json = new juce::DynamicObject();
        json->setProperty("type", "message");
        json->setProperty("text", msg);
        json->setProperty("timestamp", juce::Time::getCurrentTime().toString(true, true));

        juce::var payload(json.get());
        juce::File outFile = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                             .getChildFile("plugin_message.json");
        outFile.replaceWithText(juce::JSON::toString(payload));
    };
}

void PluginEditor::resized()
{
    inputBox.setBounds(10, 10, getWidth() - 20, 40);
    sendButton.setBounds(10, 60, getWidth() - 20, 30);
}
