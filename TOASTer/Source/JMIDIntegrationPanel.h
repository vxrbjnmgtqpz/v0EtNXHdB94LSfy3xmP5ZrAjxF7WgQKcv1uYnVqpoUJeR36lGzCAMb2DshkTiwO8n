#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include "JMIDMessage.h"
#include "JMIDParser.h"
#include <queue>
#include <mutex>

//==============================================================================
/**
 * Panel for testing and demonstrating the JMID Framework integration
 * Features real-time message processing, performance monitoring, and validation testing
 */
class JMIDIntegrationPanel : public juce::Component, public juce::Timer
{
public:
    JMIDIntegrationPanel();
    ~JMIDIntegrationPanel() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    // UI Components
    juce::Label titleLabel;
    juce::Label statusLabel;
    juce::Label performanceLabel;
    
    // Message Creation Controls
    juce::ComboBox messageTypeSelector;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;
    juce::Slider channelSlider;
    juce::TextButton sendMessageButton;
    juce::TextButton sendBurstButton;
    
    // Parser Testing Controls
    juce::ComboBox parserSelector;
    juce::TextButton validateSchemaButton;
    juce::TextButton benchmarkButton;
    
    // Display Areas
    juce::TextEditor jsonDisplay;
    juce::TextEditor logDisplay;
    juce::TextEditor performanceDisplay;
    
    // Framework Components
    // std::unique_ptr<JMID::BassoonParser> bassoonParser; // TODO: Enable when fully implemented
    // std::unique_ptr<JMID::PerformanceProfiler> profiler; // TODO: Enable when framework implements this
    JMID::LockFreeQueue<std::shared_ptr<JMID::MIDIMessage>, 1024> messageQueue;
    
    // Performance Tracking
    std::atomic<uint64_t> messagesProcessed{0};
    std::atomic<uint64_t> totalProcessingTime{0};
    double averageProcessingTime = 0.0;
    
    // Event Handlers
    void sendMessageClicked();
    void sendBurstClicked();
    void validateSchemaClicked();
    void benchmarkClicked();
    void updatePerformanceDisplay();
    void processMessageQueue();
    
    // Helper Methods
    void logMessage(const juce::String& message);
    void updateJSONDisplay(const std::string& json);
    std::shared_ptr<JMID::MIDIMessage> createSelectedMessage();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JMIDIntegrationPanel)
};
