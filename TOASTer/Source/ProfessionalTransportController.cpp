#include "ProfessionalTransportController.h"
#include <sstream>
#include <iomanip>

ProfessionalTransportController::ProfessionalTransportController()
{
    // Create transport buttons
    playButton = std::make_unique<TransportButton>("Play", TransportButton::Play);
    stopButton = std::make_unique<TransportButton>("Stop", TransportButton::Stop);
    pauseButton = std::make_unique<TransportButton>("Pause", TransportButton::Pause);
    recordButton = std::make_unique<TransportButton>("Record", TransportButton::Record);
    
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
    
    // Start high-frequency timer for microsecond updates
    startTimer(16); // ~60 FPS for smooth updates
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
    
    // Top row: Transport buttons
    auto buttonRow = bounds.removeFromTop(40);
    int buttonWidth = 35;
    int buttonSpacing = 5;
    
    playButton->setBounds(buttonRow.removeFromLeft(buttonWidth));
    buttonRow.removeFromLeft(buttonSpacing);
    stopButton->setBounds(buttonRow.removeFromLeft(buttonWidth));
    buttonRow.removeFromLeft(buttonSpacing);
    pauseButton->setBounds(buttonRow.removeFromLeft(buttonWidth));
    buttonRow.removeFromLeft(buttonSpacing);
    recordButton->setBounds(buttonRow.removeFromLeft(buttonWidth));
    
    bounds.removeFromTop(10);
    
    // Second row: Session time and bars
    auto timeRow = bounds.removeFromTop(25);
    sessionTimeLabel->setBounds(timeRow.removeFromLeft(200));
    timeRow.removeFromLeft(20);
    barsBeatsLabel->setBounds(timeRow.removeFromLeft(100));
    
    bounds.removeFromTop(10);
    
    // Third row: BPM controls
    auto bpmRow = bounds.removeFromTop(25);
    bpmLabel->setBounds(bpmRow.removeFromLeft(60));
    bpmSlider->setBounds(bpmRow.removeFromLeft(150));
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
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - playStartTime);
    
    // Calculate current sample position
    currentSample = static_cast<uint64_t>((elapsed.count() / 1000000.0) * sampleRate);
    
    // Convert to hours:minutes:seconds.milliseconds.microseconds
    uint64_t totalMicroseconds = elapsed.count();
    uint64_t hours = totalMicroseconds / (1000000ULL * 60 * 60);
    uint64_t minutes = (totalMicroseconds / (1000000ULL * 60)) % 60;
    uint64_t seconds = (totalMicroseconds / 1000000ULL) % 60;
    uint64_t milliseconds = (totalMicroseconds / 1000ULL) % 1000;
    uint64_t microseconds = totalMicroseconds % 1000;
    
    std::ostringstream oss;
    oss << "SESSION TIME: " 
        << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << "."
        << std::setfill('0') << std::setw(3) << milliseconds << "."
        << std::setfill('0') << std::setw(3) << microseconds;
    
    sessionTimeLabel->setText(oss.str(), juce::dontSendNotification);
}

void ProfessionalTransportController::updateBarsBeatsDisplay()
{
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
void ProfessionalTransportController::playButtonClicked() { play(); }
void ProfessionalTransportController::stopButtonClicked() { stop(); }
void ProfessionalTransportController::pauseButtonClicked() { pause(); }
void ProfessionalTransportController::recordButtonClicked() { record(); }

void ProfessionalTransportController::bpmSliderChanged()
{
    currentBPM = bpmSlider->getValue();
    bpmLabel->setText("BPM: " + juce::String(currentBPM.load(), 1), juce::dontSendNotification);
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
