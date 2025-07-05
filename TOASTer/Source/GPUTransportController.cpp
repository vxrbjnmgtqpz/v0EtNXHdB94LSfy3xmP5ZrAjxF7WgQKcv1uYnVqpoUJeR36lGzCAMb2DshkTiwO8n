#include "GPUTransportController.h"
#include "JAMNetworkPanel.h"

//==============================================================================
GPUTransportController::GPUTransportController()
{
    // Initialize GPU timebase if not already done (static)
    if (!jam::gpu_native::GPUTimebase::is_initialized()) {
        jam::gpu_native::GPUTimebase::initialize();
    }
    
    // Initialize shared timeline manager (static)
    if (!jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUSharedTimelineManager::initialize();
    }
    
    // Create GUI components
    playButton = std::make_unique<juce::TextButton>("▶️ Play");
    playButton->onClick = [this]() { playButtonClicked(); };
    playButton->setColour(juce::TextButton::buttonColourId, juce::Colours::green.withAlpha(0.3f));
    addAndMakeVisible(playButton.get());
    
    stopButton = std::make_unique<juce::TextButton>("⏹️ Stop");
    stopButton->onClick = [this]() { stopButtonClicked(); };
    stopButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.3f));
    addAndMakeVisible(stopButton.get());
    
    pauseButton = std::make_unique<juce::TextButton>("⏸️ Pause");
    pauseButton->onClick = [this]() { pauseButtonClicked(); };
    pauseButton->setColour(juce::TextButton::buttonColourId, juce::Colours::orange.withAlpha(0.3f));
    addAndMakeVisible(pauseButton.get());
    
    recordButton = std::make_unique<juce::TextButton>("🔴 Record");
    recordButton->onClick = [this]() { recordButtonClicked(); };
    recordButton->setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.5f));
    addAndMakeVisible(recordButton.get());
    
    // Position display
    positionLabel = std::make_unique<juce::Label>("Position", "00:00.000");
    positionLabel->setFont(juce::Font(16.0f, juce::Font::bold));
    positionLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(positionLabel.get());
    
    // BPM controls
    bpmLabel = std::make_unique<juce::Label>("BPM", "BPM:");
    addAndMakeVisible(bpmLabel.get());
    
    bpmSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    bpmSlider->setRange(60.0, 200.0, 0.1);
    bpmSlider->setValue(120.0);
    bpmSlider->onValueChange = [this]() { bpmSliderChanged(); };
    addAndMakeVisible(bpmSlider.get());
    
    // Start timer for position updates (synchronized with GPU timebase)
    startTimer(50); // 20 FPS for smooth position updates
}

GPUTransportController::~GPUTransportController()
{
    stopTimer();
}

void GPUTransportController::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::darkgrey.darker());
    
    // Draw GPU status indicator
    auto statusArea = getLocalBounds().removeFromRight(100).reduced(10);
    
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        g.setColour(juce::Colours::green);
        g.fillEllipse(statusArea.removeFromTop(20).toFloat());
        g.setColour(juce::Colours::white);
        g.setFont(12.0f);
        g.drawText("GPU Active", statusArea, juce::Justification::centred, true);
    } else {
        g.setColour(juce::Colours::red);
        g.fillEllipse(statusArea.removeFromTop(20).toFloat());
        g.setColour(juce::Colours::white);
        g.setFont(12.0f);
        g.drawText("CPU Fallback", statusArea, juce::Justification::centred, true);
    }
}

void GPUTransportController::resized()
{
    auto bounds = getLocalBounds().reduced(5);
    
    // Transport buttons
    auto buttonArea = bounds.removeFromLeft(300);
    auto buttonWidth = buttonArea.getWidth() / 4;
    
    playButton->setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(2));
    stopButton->setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(2));
    pauseButton->setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(2));
    recordButton->setBounds(buttonArea.removeFromLeft(buttonWidth).reduced(2));
    
    // Position display
    positionLabel->setBounds(bounds.removeFromLeft(120).reduced(5));
    
    // BPM controls
    auto bpmArea = bounds.removeFromLeft(200);
    bpmLabel->setBounds(bpmArea.removeFromLeft(40));
    bpmSlider->setBounds(bpmArea);
}

void GPUTransportController::timerCallback()
{
    updatePositionDisplay();
}

//==============================================================================
// GPU-native transport controls

void GPUTransportController::play()
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) {
        juce::Logger::writeToLog("WARNING: GPU timebase not available for play command");
        return;
    }
    
    if (currentState == GPUTransportState::Paused) {
        // Resume from paused position
        playStartFrame = pausedFrame;
    } else {
        // Start from current GPU frame
        playStartFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
    
    currentState = GPUTransportState::Playing;
    jam::gpu_native::GPUTimebase::set_transport_state(jam::gpu_native::GPUTransportState::PLAYING);
    
    // Schedule transport event on GPU timeline (static API)
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUTimelineEvent event;
        event.timestamp_ns = playStartFrame;
        event.type = jam::gpu_native::EventType::TRANSPORT_CHANGE;
        jam::gpu_native::GPUSharedTimelineManager::scheduleEvent(event);
    }
    
    // Send to network peers
    sendTransportCommand("play");
    
    juce::Logger::writeToLog("GPU Transport: Play at frame " + juce::String((int64_t)playStartFrame));
}

void GPUTransportController::stop()
{
    auto stopFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    currentState = GPUTransportState::Stopped;
    playStartFrame = 0;
    pausedFrame = 0;
    
    jam::gpu_native::GPUTimebase::set_transport_state(jam::gpu_native::GPUTransportState::STOPPED);
    
    // Schedule stop event on GPU timeline
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUTimelineEvent event;
        event.timestamp_ns = stopFrame;
        event.type = jam::gpu_native::EventType::TRANSPORT_CHANGE;
        jam::gpu_native::GPUSharedTimelineManager::scheduleEvent(event);
    }
    
    // Send to network peers
    sendTransportCommand("stop");
    
    juce::Logger::writeToLog("GPU Transport: Stop at frame " + juce::String((int64_t)stopFrame));
}

void GPUTransportController::pause()
{
    pausedFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    currentState = GPUTransportState::Paused;
    
    jam::gpu_native::GPUTimebase::set_transport_state(jam::gpu_native::GPUTransportState::PAUSED);
    
    // Schedule pause event on GPU timeline
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUTimelineEvent event;
        event.timestamp_ns = pausedFrame;
        event.type = jam::gpu_native::EventType::TRANSPORT_CHANGE;
        jam::gpu_native::GPUSharedTimelineManager::scheduleEvent(event);
    }
    
    // Send to network peers
    sendTransportCommand("pause");
    
    juce::Logger::writeToLog("GPU Transport: Pause at frame " + juce::String((int64_t)pausedFrame));
}

void GPUTransportController::record()
{
    auto recordFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    currentState = GPUTransportState::Recording;
    playStartFrame = recordFrame;
    
    jam::gpu_native::GPUTimebase::set_transport_state(jam::gpu_native::GPUTransportState::RECORDING);
    
    // Schedule record event on GPU timeline
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUTimelineEvent event;
        event.timestamp_ns = recordFrame;
        event.type = jam::gpu_native::EventType::TRANSPORT_CHANGE;
        jam::gpu_native::GPUSharedTimelineManager::scheduleEvent(event);
    }
    
    // Send to network peers
    sendTransportCommand("record");
    
    juce::Logger::writeToLog("GPU Transport: Record at frame " + juce::String((int64_t)recordFrame));
}

void GPUTransportController::seek(uint64_t gpuFrame)
{
    playStartFrame = gpuFrame;
    if (currentState == GPUTransportState::Paused) {
        pausedFrame = gpuFrame;
    }
    
    jam::gpu_native::GPUTimebase::set_transport_position_ns(gpuFrame);
    
    // Schedule seek event on GPU timeline
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUTimelineEvent event;
        event.timestamp_ns = gpuFrame;
        event.type = jam::gpu_native::EventType::TRANSPORT_CHANGE;
        jam::gpu_native::GPUSharedTimelineManager::scheduleEvent(event);
    }
    
    // Send to network peers
    sendTransportCommand(("seek:" + juce::String((int64_t)gpuFrame)).toStdString());
    
    juce::Logger::writeToLog("GPU Transport: Seek to frame " + juce::String((int64_t)gpuFrame));
}

//==============================================================================
// GPU timeline queries

bool GPUTransportController::isPlaying() const
{
    return currentState == GPUTransportState::Playing || currentState == GPUTransportState::Recording;
}

bool GPUTransportController::isRecording() const
{
    return currentState == GPUTransportState::Recording;
}

bool GPUTransportController::isPaused() const
{
    return currentState == GPUTransportState::Paused;
}

uint64_t GPUTransportController::getCurrentGPUFrame() const
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) return 0;
    
    if (currentState == GPUTransportState::Paused) {
        return pausedFrame;
    }
    
    return jam::gpu_native::GPUTimebase::get_current_time_ns();
}

double GPUTransportController::getCurrentTimeInSeconds() const
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) return 0.0;
    
    auto currentFrame = getCurrentGPUFrame();
    
    // Convert nanoseconds to seconds
    return static_cast<double>(currentFrame) / 1000000000.0;
}

void GPUTransportController::setBPM(double bpm)
{
    currentBPM = juce::jlimit(60.0, 200.0, bpm);
    bpmSlider->setValue(currentBPM, juce::dontSendNotification);
    
    // Update GPU timebase tempo
    jam::gpu_native::GPUTimebase::set_bpm(static_cast<uint32_t>(currentBPM));
    
    // Send BPM change to network peers
    sendTransportCommand(("bpm:" + juce::String(currentBPM, 1)).toStdString());
}

//==============================================================================
// Network sync

void GPUTransportController::handleRemoteTransportCommand(const std::string& command, uint64_t gpuTimestamp)
{
    if (command == "play") {
        play();
    } else if (command == "stop") {
        stop();
    } else if (command == "pause") {
        pause();
    } else if (command == "record") {
        record();
    } else if (command.substr(0, 4) == "seek") {
        auto colonPos = command.find(':');
        if (colonPos != std::string::npos) {
            auto frameStr = command.substr(colonPos + 1);
            uint64_t frame = std::stoull(frameStr);
            seek(frame);
        }
    } else if (command.substr(0, 3) == "bpm") {
        auto colonPos = command.find(':');
        if (colonPos != std::string::npos) {
            auto bpmStr = command.substr(colonPos + 1);
            double bpm = std::stod(bpmStr);
            setBPM(bpm);
        }
    }
}

void GPUTransportController::sendTransportCommand(const std::string& command)
{
    if (networkPanel) {
        // Send via JAM Framework v2 network panel
        auto timestamp = jam::gpu_native::GPUTimebase::is_initialized() ? jam::gpu_native::GPUTimebase::get_current_time_ns() : 0;
        networkPanel->sendTransportCommand(command, timestamp);
    }
}

//==============================================================================
// Private methods

void GPUTransportController::playButtonClicked()
{
    play();
}

void GPUTransportController::stopButtonClicked()
{
    stop();
}

void GPUTransportController::pauseButtonClicked()
{
    pause();
}

void GPUTransportController::recordButtonClicked()
{
    record();
}

void GPUTransportController::bpmSliderChanged()
{
    setBPM(bpmSlider->getValue());
}

void GPUTransportController::updatePositionDisplay()
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) {
        positionLabel->setText("--:--:---", juce::dontSendNotification);
        return;
    }
    
    double timeInSeconds = getCurrentTimeInSeconds();
    int minutes = static_cast<int>(timeInSeconds) / 60;
    int seconds = static_cast<int>(timeInSeconds) % 60;
    int milliseconds = static_cast<int>((timeInSeconds - static_cast<int>(timeInSeconds)) * 1000);
    
    juce::String timeString = juce::String::formatted("%02d:%02d.%03d", minutes, seconds, milliseconds);
    positionLabel->setText(timeString, juce::dontSendNotification);
}
