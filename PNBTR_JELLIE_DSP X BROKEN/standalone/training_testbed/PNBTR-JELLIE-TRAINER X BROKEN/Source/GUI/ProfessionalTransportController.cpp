#include "ProfessionalTransportController.h"
#include "../DSP/PNBTRTrainer.h"
#include <sstream>
#include <iomanip>

ProfessionalTransportController::ProfessionalTransportController()
{
    // Packet Loss slider
    packetLossLabel = std::make_unique<juce::Label>("PacketLoss", "Packet Loss [%]");
    packetLossLabel->setFont(juce::Font(11.0f));
    packetLossLabel->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    packetLossSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::NoTextBox);
    packetLossSlider->setRange(0.0, 100.0, 0.1);
    packetLossSlider->setValue(2.0);
    addAndMakeVisible(packetLossLabel.get());
    addAndMakeVisible(packetLossSlider.get());
    packetLossSlider->onValueChange = [this] { packetLossSliderChanged(); };

    // Jitter slider
    jitterLabel = std::make_unique<juce::Label>("Jitter", "Jitter [ms]");
    jitterLabel->setFont(juce::Font(11.0f));
    jitterLabel->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    jitterSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::NoTextBox);
    jitterSlider->setRange(0.0, 50.0, 0.1);
    jitterSlider->setValue(1.0);
    addAndMakeVisible(jitterLabel.get());
    addAndMakeVisible(jitterSlider.get());
    jitterSlider->onValueChange = [this] { jitterSliderChanged(); };
    // Create transport buttons - USING STANDARD TEXTBUTTON FOR RELIABLE CLICKING
    playButton = std::make_unique<juce::TextButton>("Play");
    stopButton = std::make_unique<juce::TextButton>("Stop");
    pauseButton = std::make_unique<juce::TextButton>("Pause");
    recordButton = std::make_unique<juce::TextButton>("Record");
    
    // Style the buttons to match the design
    playButton->setColour(juce::TextButton::buttonColourId, juce::Colours::green.withAlpha(0.7f));
    stopButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.7f));
    pauseButton->setColour(juce::TextButton::buttonColourId, juce::Colours::orange.withAlpha(0.7f));
    recordButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.7f));
    
    // Create labels
    sessionTimeLabel = std::make_unique<juce::Label>("SessionTime", "SESSION TIME: 00:00:00.000.000");
    barsBeatsLabel = std::make_unique<juce::Label>("BarsBeats", "BARS: 1.1.1");
    bpmLabel = std::make_unique<juce::Label>("BPM", "BPM: 120");
    
    // Create BPM slider
    bpmSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::NoTextBox);
    bpmSlider->setRange(60.0, 200.0, 0.1);
    bpmSlider->setValue(120.0);
    
    // Style the labels to match your design
    sessionTimeLabel->setFont(juce::Font(12.0f, juce::Font::bold));
    sessionTimeLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    sessionTimeLabel->setJustificationType(juce::Justification::centred);
    
    barsBeatsLabel->setFont(juce::Font(12.0f, juce::Font::bold));
    barsBeatsLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    barsBeatsLabel->setJustificationType(juce::Justification::centred);
    
    bpmLabel->setFont(juce::Font(11.0f));
    bpmLabel->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    
    // Add components
    addAndMakeVisible(playButton.get());
    addAndMakeVisible(stopButton.get());
    addAndMakeVisible(pauseButton.get());
    addAndMakeVisible(recordButton.get());
    addAndMakeVisible(sessionTimeLabel.get());
    addAndMakeVisible(barsBeatsLabel.get());
    addAndMakeVisible(bpmLabel.get());
    addAndMakeVisible(bpmSlider.get());
    
    // Set up button callbacks
    playButton->onClick = [this] { playButtonClicked(); };
    stopButton->onClick = [this] { stopButtonClicked(); };
    pauseButton->onClick = [this] { pauseButtonClicked(); };
    recordButton->onClick = [this] { recordButtonClicked(); };
    bpmSlider->onValueChange = [this] { bpmSliderChanged(); };
    
    // �� VIDEO GAME ENGINE: Normal timer frequency for responsive UI updates
    startTimer(100); // 10 Hz - proper responsive UI updates
}

ProfessionalTransportController::~ProfessionalTransportController()
{
    stopTimer();
}

void ProfessionalTransportController::paint(juce::Graphics& g)
{
    // Dark background like in your screenshot
    g.fillAll(juce::Colour(0xff1a1a1a));
    
    // Draw border
    g.setColour(juce::Colour(0xff3a3a3a));
    g.drawRect(getLocalBounds(), 1);
}

void ProfessionalTransportController::resized()
{
    auto bounds = getLocalBounds().reduced(5);
    
    // Single horizontal row: All components side by side
    int buttonWidth = 35;
    int buttonSpacing = 5;
    int componentSpacing = 15;
    
    // Transport buttons (left side)
    playButton->setBounds(bounds.removeFromLeft(buttonWidth));
    bounds.removeFromLeft(buttonSpacing);
    stopButton->setBounds(bounds.removeFromLeft(buttonWidth));
    bounds.removeFromLeft(buttonSpacing);
    pauseButton->setBounds(bounds.removeFromLeft(buttonWidth));
    bounds.removeFromLeft(buttonSpacing);
    recordButton->setBounds(bounds.removeFromLeft(buttonWidth));
    
    bounds.removeFromLeft(componentSpacing);
    
    // Session time (middle-left)
    sessionTimeLabel->setBounds(bounds.removeFromLeft(200));
    
    bounds.removeFromLeft(componentSpacing);
    
    // Bars display (middle-right)
    barsBeatsLabel->setBounds(bounds.removeFromLeft(100));
    
    bounds.removeFromLeft(componentSpacing);
    
    // BPM controls (right side)
    bpmLabel->setBounds(bounds.removeFromLeft(60));
    bounds.removeFromLeft(5);
    bpmSlider->setBounds(bounds.removeFromLeft(100));

    bounds.removeFromLeft(10);
    packetLossLabel->setBounds(bounds.removeFromLeft(80));
    packetLossSlider->setBounds(bounds.removeFromLeft(80));
    bounds.removeFromLeft(10);
    jitterLabel->setBounds(bounds.removeFromLeft(60));

    jitterSlider->setBounds(bounds.removeFromLeft(80));
}

void ProfessionalTransportController::packetLossSliderChanged()
{
    if (pnbtrTrainer)
        pnbtrTrainer->setPacketLossPercentage(static_cast<float>(packetLossSlider->getValue()));
}

void ProfessionalTransportController::jitterSliderChanged()
{
    if (pnbtrTrainer)
        pnbtrTrainer->setJitterAmount(static_cast<float>(jitterSlider->getValue()));
}

void ProfessionalTransportController::timerCallback()
{
    if (currentState.load() == TransportState::Playing || 
        currentState.load() == TransportState::Recording)
    {
        updateSessionTimeDisplay();
        updateBarsBeatsDisplay();
    }
}

void ProfessionalTransportController::play()
{
    if (currentState.load() == TransportState::Paused)
    {
        // Resume from pause
        auto pauseDuration = std::chrono::steady_clock::now() - pausedTime;
        playStartTime += pauseDuration;
    }
    else
    {
        // Start fresh
        playStartTime = std::chrono::steady_clock::now();
        currentSample = 0;
    }
    
    currentState = TransportState::Playing;
}

void ProfessionalTransportController::stop()
{
    currentState = TransportState::Stopped;
    currentSample = 0;
    
    // Reset displays
    sessionTimeLabel->setText("SESSION TIME: 00:00:00.000.000", juce::dontSendNotification);
    barsBeatsLabel->setText("BARS: 1.1.1", juce::dontSendNotification);
}

void ProfessionalTransportController::pause()
{
    if (currentState.load() == TransportState::Playing || 
        currentState.load() == TransportState::Recording)
    {
        pausedTime = std::chrono::steady_clock::now();
        currentState = TransportState::Paused;
    }
}

void ProfessionalTransportController::record()
{
    if (currentState.load() != TransportState::Recording)
    {
        if (currentState.load() != TransportState::Playing)
        {
            playStartTime = std::chrono::steady_clock::now();
            currentSample = 0;
        }
        currentState = TransportState::Recording;
    }
}

void ProfessionalTransportController::updateSessionTimeDisplay()
{
    if (!sessionTimeLabel) return; // Safety check
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - playStartTime);
    
    // Calculate current sample position
    currentSample = static_cast<uint64_t>((elapsed.count() / 1000000.0) * sampleRate);
    
    // Convert to hours:minutes:seconds.microseconds (6-digit precision)
    uint64_t totalMicroseconds = elapsed.count();
    uint64_t hours = totalMicroseconds / (1000000ULL * 60 * 60);
    uint64_t minutes = (totalMicroseconds / (1000000ULL * 60)) % 60;
    uint64_t seconds = (totalMicroseconds / 1000000ULL) % 60;
    uint64_t microseconds = totalMicroseconds % 1000000; // Full 6-digit microseconds
    
    std::ostringstream oss;
    oss << "SESSION TIME: " 
        << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "."
        << std::setfill('0') << std::setw(6) << microseconds; // 6-digit microsecond precision
    
    sessionTimeLabel->setText(oss.str(), juce::dontSendNotification);
}

void ProfessionalTransportController::updateBarsBeatsDisplay()
{
    if (!barsBeatsLabel) return; // Safety check
    
    double currentTimeSeconds = getCurrentTimeInSeconds();
    double bpm = currentBPM.load();
    
    // Calculate bars, beats, and subdivisions
    double beatsPerSecond = bpm / 60.0;
    double totalBeats = currentTimeSeconds * beatsPerSecond;
    
    int currentBar = static_cast<int>(totalBeats / beatsPerBar.load()) + 1;
    int currentBeat = static_cast<int>(fmod(totalBeats, beatsPerBar.load())) + 1;
    double fractionalBeat = fmod(totalBeats, 1.0);
    int currentSubdivision = static_cast<int>(fractionalBeat * subdivision.load()) + 1;
    
    std::ostringstream oss;
    oss << "BARS: " << currentBar << "." << currentBeat << "." << currentSubdivision;
    
    barsBeatsLabel->setText(oss.str(), juce::dontSendNotification);
}

// Button callbacks
void ProfessionalTransportController::playButtonClicked()
{
    if (onPlay) onPlay();
    play();
}
void ProfessionalTransportController::stopButtonClicked() {
    if (onStop) onStop();
    stop();
}
void ProfessionalTransportController::pauseButtonClicked() { pause(); }
void ProfessionalTransportController::recordButtonClicked() {
    if (onRecord) onRecord();
    record();
}

void ProfessionalTransportController::bpmSliderChanged()
{
    currentBPM = bpmSlider->getValue();
    bpmLabel->setText("BPM: " + juce::String(currentBPM.load(), 1), juce::dontSendNotification);
    // Wire BPM slider to DSP
    if (pnbtrTrainer) {
        pnbtrTrainer->setPacketLossPercentage(static_cast<float>(bpmSlider->getValue())); // Example: wire to packet loss for demo
        // Replace with: pnbtrTrainer->setBPM(...) if you add such a method
    }
}

// Getters
bool ProfessionalTransportController::isPlaying() const 
{ 
    return currentState.load() == TransportState::Playing; 
}

bool ProfessionalTransportController::isRecording() const 
{ 
    return currentState.load() == TransportState::Recording; 
}

bool ProfessionalTransportController::isPaused() const 
{ 
    return currentState.load() == TransportState::Paused; 
}

double ProfessionalTransportController::getCurrentTimeInSeconds() const
{
    if (currentState.load() == TransportState::Stopped)
        return 0.0;
        
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - playStartTime);
    return elapsed.count() / 1000000.0;
}

uint64_t ProfessionalTransportController::getCurrentSample() const
{
    return currentSample.load();
}

void ProfessionalTransportController::setBPM(double bpm)
{
    currentBPM = bpm;
    bpmSlider->setValue(bpm, juce::dontSendNotification);
    bpmLabel->setText("BPM: " + juce::String(bpm, 1), juce::dontSendNotification);
}
