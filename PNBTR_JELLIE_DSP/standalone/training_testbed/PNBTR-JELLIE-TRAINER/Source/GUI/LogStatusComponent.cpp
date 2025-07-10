#include "LogStatusComponent.h"

LogStatusComponent::LogStatusComponent()
{
    logBox.setMultiLine(true);
    logBox.setReadOnly(true);
    logBox.setScrollbarsShown(true);
    logBox.setCaretVisible(false);
    logBox.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain)));
    logBox.setColour(juce::TextEditor::backgroundColourId, juce::Colours::black);
    logBox.setColour(juce::TextEditor::textColourId, juce::Colours::lightgreen);
    addAndMakeVisible(logBox);

    startTimerHz(5); // 🎮 VIDEO GAME ENGINE: Much slower to prevent string race conditions
    
    // Add initial log entry
    addEntry("🚀 PNBTR+JELLIE Training Testbed initialized");
}

LogStatusComponent::~LogStatusComponent() = default;

void LogStatusComponent::resized()
{
    logBox.setBounds(getLocalBounds().reduced(4));
}

void LogStatusComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds());
}

void LogStatusComponent::addEntry(const juce::String& message)
{
    const std::lock_guard<std::mutex> lock(queueMutex);
    
    // Add timestamp
    auto time = juce::Time::getCurrentTime();
    juce::String timestampedMessage = time.toString(true, true, true, true) + " - " + message;
    
    pendingQueue.push_back(timestampedMessage);
    if (pendingQueue.size() > 200)
        pendingQueue.pop_front();
}

void LogStatusComponent::timerCallback()
{
    std::deque<juce::String> localQueue;
    {
        const std::lock_guard<std::mutex> lock(queueMutex);
        std::swap(localQueue, pendingQueue);
    }

    for (const auto& msg : localQueue)
    {
        logBox.moveCaretToEnd();
        logBox.insertTextAtCaret(msg + "\n");
    }
    
    // Auto-scroll to bottom
    if (!localQueue.empty())
    {
        logBox.moveCaretToEnd();
        logBox.scrollEditorToPositionCaret(0, logBox.getTextHeight());
    }
}
