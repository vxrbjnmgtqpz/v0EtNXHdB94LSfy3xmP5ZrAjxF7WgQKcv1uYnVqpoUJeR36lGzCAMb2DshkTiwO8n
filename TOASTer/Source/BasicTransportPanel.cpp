#include "BasicTransportPanel.h"
#include <chrono>

BasicTransportPanel::BasicTransportPanel()
{
    addAndMakeVisible(playButton);
    addAndMakeVisible(stopButton);
    addAndMakeVisible(resetButton);
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(tempoLabel);
    addAndMakeVisible(positionLabel);
    addAndMakeVisible(tempoSlider);
    
    playButton.setButtonText("Play");
    playButton.onClick = [this] { startTransport(); };
    
    stopButton.setButtonText("Stop");
    stopButton.onClick = [this] { stopTransport(); };
    stopButton.setEnabled(false);
    
    resetButton.setButtonText("Reset");
    resetButton.onClick = [this] { resetTransport(); };
    
    statusLabel.setText("Transport: Stopped", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    
    tempoLabel.setText("Tempo: 120 BPM", juce::dontSendNotification);
    tempoLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    
    positionLabel.setText("Position: 00:00.000", juce::dontSendNotification);
    positionLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    
    tempoSlider.setRange(60.0, 200.0, 1.0);
    tempoSlider.setValue(120.0);
    tempoSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    tempoSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    tempoSlider.onValueChange = [this] { 
        currentTempo = tempoSlider.getValue();
        updateDisplay();
    };
    
    startTimer(50); // 20 FPS updates
}

BasicTransportPanel::~BasicTransportPanel()
{
    stopTimer();
}

void BasicTransportPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a1a));
    
    g.setColour(juce::Colour(0xff3a3a3a));
    g.drawRect(getLocalBounds(), 1);
    
    g.setColour(juce::Colours::white);
    g.setFont(16.0f);
    g.drawText("Transport Control", 
               getLocalBounds().removeFromTop(25).reduced(5), 
               juce::Justification::centredLeft, true);
}

void BasicTransportPanel::resized()
{
    auto bounds = getLocalBounds();
    bounds.removeFromTop(25); // Title area
    bounds.reduce(10, 5);
    
    auto topRow = bounds.removeFromTop(30);
    playButton.setBounds(topRow.removeFromLeft(60));
    topRow.removeFromLeft(5);
    stopButton.setBounds(topRow.removeFromLeft(60));
    topRow.removeFromLeft(5);
    resetButton.setBounds(topRow.removeFromLeft(60));
    
    bounds.removeFromTop(10);
    
    auto statusRow = bounds.removeFromTop(25);
    statusLabel.setBounds(statusRow);
    
    bounds.removeFromTop(5);
    auto positionRow = bounds.removeFromTop(25);
    positionLabel.setBounds(positionRow);
    
    bounds.removeFromTop(10);
    auto tempoRow = bounds.removeFromTop(25);
    tempoLabel.setBounds(tempoRow.removeFromLeft(120));
    tempoRow.removeFromLeft(10);
    tempoSlider.setBounds(tempoRow);
}

void BasicTransportPanel::timerCallback()
{
    if (isPlaying)
    {
        updateDisplay();
    }
}

void BasicTransportPanel::startTransport()
{
    if (!isPlaying)
    {
        isPlaying = true;
        startTime = std::chrono::steady_clock::now();
        
        playButton.setEnabled(false);
        stopButton.setEnabled(true);
        
        statusLabel.setText("Transport: Playing", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    }
}

void BasicTransportPanel::stopTransport()
{
    if (isPlaying)
    {
        isPlaying = false;
        
        playButton.setEnabled(true);
        stopButton.setEnabled(false);
        
        statusLabel.setText("Transport: Stopped", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
}

void BasicTransportPanel::resetTransport()
{
    bool wasPlaying = isPlaying;
    
    if (isPlaying)
        stopTransport();
    
    currentSample = 0;
    startTime = std::chrono::steady_clock::now();
    
    updateDisplay();
    
    if (wasPlaying)
        startTransport();
}

void BasicTransportPanel::updateDisplay()
{
    // Update tempo display
    tempoLabel.setText("Tempo: " + juce::String(currentTempo.load(), 0) + " BPM", 
                       juce::dontSendNotification);
    
    // Calculate current position
    if (isPlaying)
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime);
        currentSample = static_cast<int64_t>((elapsed.count() / 1000000.0) * sampleRate);
    }
    
    // Convert samples to minutes:seconds.milliseconds
    double seconds = currentSample.load() / sampleRate;
    int minutes = static_cast<int>(seconds / 60.0);
    seconds = fmod(seconds, 60.0);
    
    juce::String positionText = juce::String::formatted("%02d:%06.3f", minutes, seconds);
    positionLabel.setText("Position: " + positionText, juce::dontSendNotification);
}
