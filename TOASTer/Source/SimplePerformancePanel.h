#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

class SimplePerformancePanel : public juce::Component, private juce::Timer
{
public:
    SimplePerformancePanel()
    {
        addAndMakeVisible(cpuLabel);
        addAndMakeVisible(memoryLabel);
        addAndMakeVisible(latencyLabel);
        addAndMakeVisible(throughputLabel);
        
        cpuLabel.setText("CPU: 0%", juce::dontSendNotification);
        memoryLabel.setText("Memory: 0 MB", juce::dontSendNotification);
        latencyLabel.setText("Latency: 0 μs", juce::dontSendNotification);
        throughputLabel.setText("Throughput: 0 msgs/s", juce::dontSendNotification);
        
        cpuLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
        memoryLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        throughputLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
        
        startTimer(1000); // Update every second
    }

    ~SimplePerformancePanel() override
    {
        stopTimer();
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff1a1a1a));
        g.setColour(juce::Colour(0xff3a3a3a));
        g.drawRect(getLocalBounds(), 1);
        
        g.setColour(juce::Colours::white);
        g.setFont(16.0f);
        g.drawText("Performance Monitor", getLocalBounds().removeFromTop(30), 
                   juce::Justification::centred, true);
    }

    void resized() override
    {
        auto bounds = getLocalBounds();
        bounds.removeFromTop(30); // Title area
        bounds.reduce(10, 5);
        
        auto labelHeight = bounds.getHeight() / 4;
        
        cpuLabel.setBounds(bounds.removeFromTop(labelHeight));
        memoryLabel.setBounds(bounds.removeFromTop(labelHeight));
        latencyLabel.setBounds(bounds.removeFromTop(labelHeight));
        throughputLabel.setBounds(bounds);
    }

private:
    void timerCallback() override
    {
        // Simulate performance metrics
        static int counter = 0;
        counter++;
        
        auto cpuUsage = (counter % 100);
        auto memoryUsage = 50 + (counter % 200);
        auto latency = 100 + (counter % 50);
        auto throughput = 1000 + (counter % 500);
        
        cpuLabel.setText("CPU: " + juce::String(cpuUsage) + "%", juce::dontSendNotification);
        memoryLabel.setText("Memory: " + juce::String(memoryUsage) + " MB", juce::dontSendNotification);
        latencyLabel.setText("Latency: " + juce::String(latency) + " μs", juce::dontSendNotification);
        throughputLabel.setText("Throughput: " + juce::String(throughput) + " msgs/s", juce::dontSendNotification);
    }

    juce::Label cpuLabel;
    juce::Label memoryLabel;
    juce::Label latencyLabel;
    juce::Label throughputLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimplePerformancePanel)
};
