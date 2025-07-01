#include "TransportController.h"

// Helper function for emoji-compatible font setup
static juce::Font getEmojiCompatibleFont(float size = 12.0f, bool bold = false)
{
    // On macOS, prefer system fonts that support emoji
    #if JUCE_MAC
        auto font = juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
        if (bold) font = font.boldened();
        return font;
    #elif JUCE_WINDOWS
        auto font = juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
        if (bold) font = font.boldened();
        return font;
    #else
        auto font = juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
        if (bold) font = font.boldened();
        return font;
    #endif
}

//==============================================================================
void TransportButton::paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown)
{
    auto bounds = getLocalBounds().toFloat().reduced(1.0f);
    
    // Background color based on button type and state
    juce::Colour bgColor = juce::Colours::darkgrey;
    juce::Colour symbolColor = juce::Colours::white;
    
    if (buttonType == Play)
    {
        if (getToggleState()) // playing/paused
        {
            bgColor = juce::Colours::orange;
            symbolColor = juce::Colours::black;
        }
        else
        {
            bgColor = juce::Colours::darkgreen;
        }
    }
    else if (buttonType == Stop)
    {
        bgColor = juce::Colours::darkred;
    }
    else if (buttonType == Record)
    {
        if (getToggleState()) // recording
        {
            bgColor = juce::Colours::hotpink;
            symbolColor = juce::Colours::white;
        }
        else
        {
            bgColor = juce::Colours::purple.darker(0.3f);
        }
    }
    
    // Adjust for interaction states
    if (shouldDrawButtonAsDown)
        bgColor = bgColor.darker(0.2f);
    else if (shouldDrawButtonAsHighlighted)
        bgColor = bgColor.brighter(0.1f);
    
    // Draw rounded rectangle background
    g.setColour(bgColor);
    g.fillRoundedRectangle(bounds, 4.0f);
    
    // Draw border
    g.setColour(juce::Colours::black);
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
    
    // Draw the symbol using geometric shapes
    g.setColour(symbolColor);
    auto symbolBounds = bounds.reduced(8.0f);
    auto centerX = symbolBounds.getCentreX();
    auto centerY = symbolBounds.getCentreY();
    
    if (buttonType == Play)
    {
        if (getToggleState()) // Pause - draw two vertical rectangles
        {
            auto pauseWidth = symbolBounds.getWidth() * 0.25f;
            auto pauseHeight = symbolBounds.getHeight() * 0.7f;
            auto spacing = symbolBounds.getWidth() * 0.15f;
            
            auto leftRect = juce::Rectangle<float>(
                centerX - spacing - pauseWidth, 
                centerY - pauseHeight * 0.5f, 
                pauseWidth, 
                pauseHeight
            );
            auto rightRect = juce::Rectangle<float>(
                centerX + spacing, 
                centerY - pauseHeight * 0.5f, 
                pauseWidth, 
                pauseHeight
            );
            
            g.fillRect(leftRect);
            g.fillRect(rightRect);
        }
        else // Play - draw right-pointing triangle
        {
            auto triangleSize = symbolBounds.getHeight() * 0.6f;
            juce::Path triangle;
            triangle.addTriangle(
                centerX - triangleSize * 0.3f, centerY - triangleSize * 0.5f,  // top left
                centerX - triangleSize * 0.3f, centerY + triangleSize * 0.5f,  // bottom left  
                centerX + triangleSize * 0.4f, centerY                         // right point
            );
            g.fillPath(triangle);
        }
    }
    else if (buttonType == Stop)
    {
        // Stop - draw square
        auto squareSize = symbolBounds.getHeight() * 0.6f;
        auto square = juce::Rectangle<float>(
            centerX - squareSize * 0.5f, 
            centerY - squareSize * 0.5f, 
            squareSize, 
            squareSize
        );
        g.fillRect(square);
    }
    else if (buttonType == Record)
    {
        // Record - draw circle
        auto circleSize = symbolBounds.getHeight() * 0.6f;
        auto circle = juce::Rectangle<float>(
            centerX - circleSize * 0.5f, 
            centerY - circleSize * 0.5f, 
            circleSize, 
            circleSize
        );
        g.fillEllipse(circle);
    }
}

//==============================================================================
TransportController::TransportController()
    : playButton("Play", TransportButton::Play)
    , stopButton("Stop", TransportButton::Stop)
    , recordButton("Record", TransportButton::Record)
{
    // Set up play button
    playButton.onClick = [this] { playButtonClicked(); };
    addAndMakeVisible(playButton);
    
    // Set up stop button
    stopButton.onClick = [this] { stopButtonClicked(); };
    addAndMakeVisible(stopButton);
    
    // Set up record button
    recordButton.onClick = [this] { recordButtonClicked(); };
    addAndMakeVisible(recordButton);
    
    // Set up labels
    sessionTimeLabel.setText("SESSION TIME: 00:00:00", juce::dontSendNotification);
    sessionTimeLabel.setJustificationType(juce::Justification::centred);
    sessionTimeLabel.setFont(getEmojiCompatibleFont(14.0f, true));
    sessionTimeLabel.setColour(juce::Label::textColourId, juce::Colours::lightyellow);
    addAndMakeVisible(sessionTimeLabel);
    
    barsBeatsLabel.setText("BARS: 1.1.1", juce::dontSendNotification);
    barsBeatsLabel.setJustificationType(juce::Justification::centred);
    barsBeatsLabel.setFont(getEmojiCompatibleFont(14.0f, true));
    barsBeatsLabel.setColour(juce::Label::textColourId, juce::Colours::lightyellow);
    addAndMakeVisible(barsBeatsLabel);
}

TransportController::~TransportController()
{
}

void TransportController::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::grey);
    g.drawRect(getLocalBounds(), 2);
}

void TransportController::resized()
{
    auto bounds = getLocalBounds().reduced(5);
    auto buttonSize = 35; // Make buttons slightly larger and square
    auto buttonHeight = buttonSize;
    
    // Position buttons on the left
    playButton.setBounds(bounds.removeFromLeft(buttonSize).removeFromTop(buttonHeight));
    bounds.removeFromLeft(8); // spacing
    stopButton.setBounds(bounds.removeFromLeft(buttonSize).removeFromTop(buttonHeight));
    bounds.removeFromLeft(8); // spacing
    recordButton.setBounds(bounds.removeFromLeft(buttonSize).removeFromTop(buttonHeight));
    
    bounds.removeFromLeft(25); // larger spacing
    
    // Session time in the middle
    auto labelWidth = 160;
    sessionTimeLabel.setBounds(bounds.removeFromLeft(labelWidth).removeFromTop(buttonHeight));
    
    bounds.removeFromLeft(25); // spacing
    
    // Bars/beats on the right
    barsBeatsLabel.setBounds(bounds.removeFromLeft(labelWidth).removeFromTop(buttonHeight));
}

void TransportController::playButtonClicked()
{
    isPlaying = !isPlaying;
    playButton.setToggleState(isPlaying, juce::dontSendNotification);
    updateDisplay();
}

void TransportController::stopButtonClicked()
{
    isPlaying = false;
    isRecording = false;
    playButton.setToggleState(false, juce::dontSendNotification);
    recordButton.setToggleState(false, juce::dontSendNotification);
    updateDisplay();
}

void TransportController::recordButtonClicked()
{
    isRecording = !isRecording;
    recordButton.setToggleState(isRecording, juce::dontSendNotification);
    updateDisplay();
}

void TransportController::updateDisplay()
{
    // This would update the time and bars/beats display
    // For now just placeholder
}