#include "PerformanceMonitorPanel.h"

PerformanceMonitorPanel::PerformanceMonitorPanel()
{
    // Set up title
    titleLabel.setText("Performance Monitor", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);
    
    // Set up metric labels
    latencyLabel.setText("Latency: 0.0ms", juce::dontSendNotification);
    addAndMakeVisible(latencyLabel);
    
    cpuUsageLabel.setText("CPU Usage: 0%", juce::dontSendNotification);
    addAndMakeVisible(cpuUsageLabel);
    
    memoryUsageLabel.setText("Memory: 0MB", juce::dontSendNotification);
    addAndMakeVisible(memoryUsageLabel);
    
    networkStatsLabel.setText("Network: 0 Kb/s", juce::dontSendNotification);
    addAndMakeVisible(networkStatsLabel);
    
    midiThroughputLabel.setText("MIDI Throughput: 0 msg/s", juce::dontSendNotification);
    addAndMakeVisible(midiThroughputLabel);
    
    // Start timer to update metrics
    startTimer(1000); // Update every second
}

PerformanceMonitorPanel::~PerformanceMonitorPanel()
{
}

void PerformanceMonitorPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void PerformanceMonitorPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(5);
    
    latencyLabel.setBounds(bounds.removeFromTop(20));
    cpuUsageLabel.setBounds(bounds.removeFromTop(20));
    memoryUsageLabel.setBounds(bounds.removeFromTop(20));
    networkStatsLabel.setBounds(bounds.removeFromTop(20));
    midiThroughputLabel.setBounds(bounds.removeFromTop(20));
}

void PerformanceMonitorPanel::updateMetrics()
{
    // Placeholder implementation - would get actual metrics in real implementation
    static int counter = 0;
    counter++;
    
    latencyLabel.setText("Latency: " + juce::String(counter % 10) + ".0ms", juce::dontSendNotification);
    cpuUsageLabel.setText("CPU Usage: " + juce::String(counter % 100) + "%", juce::dontSendNotification);
    memoryUsageLabel.setText("Memory: " + juce::String(50 + (counter % 50)) + "MB", juce::dontSendNotification);
    networkStatsLabel.setText("Network: " + juce::String(counter % 1000) + " Kb/s", juce::dontSendNotification);
    midiThroughputLabel.setText("MIDI Throughput: " + juce::String(counter % 500) + " msg/s", juce::dontSendNotification);
}