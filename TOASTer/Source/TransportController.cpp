#include "TransportController.h"
#include "JAMNetworkPanel.h"  // Updated for JAM Framework v2

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
    sessionTimeLabel.setText("SESSION TIME: 00:00:00.000.000", juce::dontSendNotification);
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
    auto labelWidth = 220; // Increased width for microsecond precision
    sessionTimeLabel.setBounds(bounds.removeFromLeft(labelWidth).removeFromTop(buttonHeight));
    
    bounds.removeFromLeft(25); // spacing
    
    // Bars/beats on the right
    barsBeatsLabel.setBounds(bounds.removeFromLeft(labelWidth).removeFromTop(buttonHeight));
}

void TransportController::playButtonClicked()
{
    isPlaying = !isPlaying;
    playButton.setToggleState(isPlaying, juce::dontSendNotification);
    
    if (isPlaying) {
        startTransport();
    } else {
        stopTimer(); // Pause the transport
    }
    
    // Send transport sync to network if connected
    syncTransportStateToNetwork();
    
    updateDisplay();
}

void TransportController::stopButtonClicked()
{
    isPlaying = false;
    isRecording = false;
    playButton.setToggleState(false, juce::dontSendNotification);
    recordButton.setToggleState(false, juce::dontSendNotification);
    stopTransport();
    
    // Send transport sync to network if connected
    syncTransportStateToNetwork();
    
    updateDisplay();
}

void TransportController::recordButtonClicked()
{
    isRecording = !isRecording;
    recordButton.setToggleState(isRecording, juce::dontSendNotification);
    updateDisplay();
}

void TransportController::startTransport()
{
    transportStartTime = std::chrono::high_resolution_clock::now();
    startTimer(16); // Update display at ~60 FPS for smooth microsecond display
}

void TransportController::stopTransport()
{
    stopTimer();
    currentPosition = std::chrono::microseconds{0}; // Reset to beginning
}

void TransportController::timerCallback()
{
    if (isPlaying) {
        auto now = std::chrono::high_resolution_clock::now();
        currentPosition = std::chrono::duration_cast<std::chrono::microseconds>(
            now - transportStartTime);
        updateDisplay();
    }
}

void TransportController::updateDisplay()
{
    // Update session time display with microsecond precision
    auto totalMicroseconds = currentPosition.count();
    auto hours = totalMicroseconds / 3600000000LL;
    auto minutes = (totalMicroseconds % 3600000000LL) / 60000000LL;
    auto seconds = (totalMicroseconds % 60000000LL) / 1000000LL;
    auto milliseconds = (totalMicroseconds % 1000000LL) / 1000LL;
    auto microseconds = totalMicroseconds % 1000LL;
    
    auto timeString = juce::String::formatted("SESSION TIME: %02d:%02d:%02d.%03d.%03d", 
                                            (int)hours, (int)minutes, (int)seconds, 
                                            (int)milliseconds, (int)microseconds);
    sessionTimeLabel.setText(timeString, juce::dontSendNotification);
    
    // Update bars/beats display (4/4 time) - convert microseconds to seconds for calculation
    double totalSeconds = totalMicroseconds / 1000000.0;
    double totalBeats = totalSeconds * (bpm / 60.0);
    int bars = (int)(totalBeats / beatsPerBar) + 1;  // 1-based
    int beats = (int)(totalBeats) % beatsPerBar + 1; // 1-based
    int ticks = (int)((totalBeats - (int)totalBeats) * 96); // 96 ticks per beat
    
    auto barsBeatsString = juce::String::formatted("BARS: %d.%d.%03d", bars, beats, ticks);
    barsBeatsLabel.setText(barsBeatsString, juce::dontSendNotification);
}

void TransportController::syncTransportStateToNetwork()
{
    if (!jamNetworkPanel || !jamNetworkPanel->isConnected()) return;
    
    // Send transport command using JAM Framework v2 transport protocol
    std::string command = isPlaying ? "PLAY" : "STOP";
    uint64_t currentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Send via JAM Framework v2 new transport system
    jamNetworkPanel->sendTransportCommand(command, currentTime, 
                                        static_cast<double>(currentPosition.count()), bpm);
    
    juce::Logger::writeToLog("🎛️ Transport sync sent via JAM Framework v2: " + juce::String(command) +
                           " (pos: " + juce::String(currentPosition.count()) + 
                           ", bpm: " + juce::String(bpm) + ")");
}

void TransportController::handleNetworkTransportCommand(const std::string& command, uint64_t timestamp,
                                                       double position, double bpm)
{
    if (isMaster) return; // Master doesn't accept transport commands
    
    if (command == "PLAY" && !isPlaying) {
        isPlaying = true;
        playButton.setToggleState(true, juce::dontSendNotification);
        
        // Sync to network timestamp and position
        auto networkTime = std::chrono::microseconds{timestamp};
        auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch());
        transportStartTime = std::chrono::high_resolution_clock::now() - (now - networkTime);
        
        // Set position and BPM from network
        currentPosition = std::chrono::milliseconds(static_cast<int64_t>(position));
        this->bpm = bpm;
        
        startTimer(16);
        juce::Logger::writeToLog("🎛️ Received PLAY command - pos: " + juce::String(position) + 
                                " bpm: " + juce::String(bpm));
    }
    else if (command == "STOP" && isPlaying) {
        isPlaying = false;
        playButton.setToggleState(false, juce::dontSendNotification);
        stopTimer();
        juce::Logger::writeToLog("🎛️ Received STOP command");
    }
    else if (command == "POSITION") {
        currentPosition = std::chrono::milliseconds(static_cast<int64_t>(position));
        juce::Logger::writeToLog("🎛️ Received POSITION command: " + juce::String(position));
    }
    else if (command == "BPM") {
        this->bpm = bpm;
        juce::Logger::writeToLog("🎛️ Received BPM command: " + juce::String(bpm));
    }
    
    updateDisplay();
}