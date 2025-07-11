#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <deque>
#include <mutex>

class LogStatusComponent : public juce::Component, private juce::Timer
{
public:
    LogStatusComponent();
    ~LogStatusComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    void addEntry(const juce::String& message);

private:
    void timerCallback() override;

    juce::TextEditor logBox;
    std::deque<juce::String> pendingQueue;
    std::mutex queueMutex;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LogStatusComponent)
};
