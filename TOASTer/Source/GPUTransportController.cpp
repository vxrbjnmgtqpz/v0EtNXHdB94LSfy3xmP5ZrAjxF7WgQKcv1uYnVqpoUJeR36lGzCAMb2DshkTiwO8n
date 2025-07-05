#include "GPUTransportController.h"
#include "JAMNetworkPanel.h"

//==============================================================================
GPUTransportController::GPUTransportController()
{
    // Initialize GPU transport manager first
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transportManager.isInitialized()) {
        bool initSuccess = transportManager.initialize();
        if (!initSuccess) {
            juce::Logger::writeToLog("⚠️ Failed to initialize GPU Transport Manager - falling back to CPU timing");
        } else {
            juce::Logger::writeToLog("✅ GPU Transport Manager initialized successfully");
        }
    } else {
        juce::Logger::writeToLog("✅ GPU Transport Manager already initialized");
    }
    
    // Set up state change callback for real-time GPU state updates
    transportManager.setStateChangeCallback([this](::GPUTransportState oldState, ::GPUTransportState newState) {
        // Convert to local enum and update UI
        currentState = static_cast<GPUTransportState>(newState);
        
        // Trigger UI update on message thread
        juce::MessageManager::callAsync([this]() {
            updatePositionDisplay();
            repaint();
        });
    });
    
    // Initialize default time signature and subdivision for GPU bars/beats calculation
    transportManager.setTimeSignature(4, 4);    // 4/4 time signature
    transportManager.setSubdivision(1000);      // 1000 subdivisions per beat for precise display
    
    // Initialize GPU timebase if not already done (static)
    if (!jam::gpu_native::GPUTimebase::is_initialized()) {
        jam::gpu_native::GPUTimebase::initialize();
    }
    
    // Initialize shared timeline manager (static)
    if (!jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        jam::gpu_native::GPUSharedTimelineManager::initialize();
    }
    
    // Create GUI components with custom canvas rendering
    playButton = std::make_unique<GPUTransportButton>("Play", GPUTransportButton::Play);
    playButton->onClick = [this]() { playButtonClicked(); };
    addAndMakeVisible(playButton.get());
    
    stopButton = std::make_unique<GPUTransportButton>("Stop", GPUTransportButton::Stop);
    stopButton->onClick = [this]() { stopButtonClicked(); };
    addAndMakeVisible(stopButton.get());
    
    pauseButton = std::make_unique<GPUTransportButton>("Pause", GPUTransportButton::Pause);
    pauseButton->onClick = [this]() { pauseButtonClicked(); };
    addAndMakeVisible(pauseButton.get());
    
    recordButton = std::make_unique<GPUTransportButton>("Record", GPUTransportButton::Record);
    recordButton->onClick = [this]() { recordButtonClicked(); };
    addAndMakeVisible(recordButton.get());
    
    // Position display
    positionLabel = std::make_unique<juce::Label>("Position", "00:00.000.000");
    positionLabel->setFont(juce::Font(juce::FontOptions(16.0f).withStyle("bold")));
    positionLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(positionLabel.get());
    
    // Bars/Beats display
    barsBeatsLabel = std::make_unique<juce::Label>("BarsBeats", "001.01.000");
    barsBeatsLabel->setFont(juce::Font(juce::FontOptions(16.0f).withStyle("bold")));
    barsBeatsLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(barsBeatsLabel.get());
    
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
    
    // Position displays - put them side by side instead of stacked
    auto displayArea = bounds.removeFromLeft(400);  // Make it wider for both displays
    positionLabel->setBounds(displayArea.removeFromLeft(200).reduced(2));  // Time display on left
    barsBeatsLabel->setBounds(displayArea.reduced(2));  // Bars/beats on right
    
    // BPM controls
    auto bpmArea = bounds.removeFromLeft(200);
    bpmLabel->setBounds(bpmArea.removeFromLeft(40));
    bpmSlider->setBounds(bpmArea);
}

void GPUTransportController::timerCallback()
{
    // Update GPU Transport Manager to process timeline events
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        transportManager.update();
    }
    
    updatePositionDisplay();
    updateBarsBeatsDisplay();  // Also update bars/beats
}

//==============================================================================
// GPU-native transport controls

void GPUTransportController::play()
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    bool isInit = transportManager.isInitialized();
    juce::Logger::writeToLog("GPU Transport Manager initialized status: " + juce::String(isInit ? "YES" : "NO"));
    
    if (!isInit) {
        juce::Logger::writeToLog("WARNING: GPU Transport Manager not available for play command");
        return;
    }
    
    // Use GPU Transport Manager for actual transport control
    uint64_t startFrame = 0;
    if (currentState == GPUTransportState::Paused) {
        startFrame = pausedFrame;
    }
    
    transportManager.play(startFrame);
    
    // Update local state (will be synced via callback)
    if (currentState == GPUTransportState::Paused) {
        playStartFrame = pausedFrame;
    } else {
        playStartFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
    
    // Send to network peers
    sendTransportCommand("play");
    
    juce::Logger::writeToLog("GPU Transport: Play command sent to GPU");
}

void GPUTransportController::stop()
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transportManager.isInitialized()) {
        juce::Logger::writeToLog("WARNING: GPU Transport Manager not available for stop command");
        return;
    }
    
    // Use GPU Transport Manager for actual transport control
    transportManager.stop();
    
    // Update local state (will be synced via callback)
    playStartFrame = 0;
    pausedFrame = 0;
    
    // Send to network peers
    sendTransportCommand("stop");
    
    juce::Logger::writeToLog("GPU Transport: Stop command sent to GPU");
}

void GPUTransportController::pause()
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transportManager.isInitialized()) {
        juce::Logger::writeToLog("WARNING: GPU Transport Manager not available for pause command");
        return;
    }
    
    // Use GPU Transport Manager for actual transport control
    transportManager.pause();
    
    // Update local state (will be synced via callback)
    pausedFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    
    // Send to network peers
    sendTransportCommand("pause");
    
    juce::Logger::writeToLog("GPU Transport: Pause command sent to GPU");
}

void GPUTransportController::record()
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transportManager.isInitialized()) {
        juce::Logger::writeToLog("WARNING: GPU Transport Manager not available for record command");
        return;
    }
    
    // Use GPU Transport Manager for actual transport control
    uint64_t recordFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    transportManager.record(recordFrame);
    
    // Update local state (will be synced via callback)
    playStartFrame = recordFrame;
    
    // Send to network peers
    sendTransportCommand("record");
    
    juce::Logger::writeToLog("GPU Transport: Record command sent to GPU");
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
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        return transportManager.isPlaying();
    }
    return currentState == GPUTransportState::Playing || currentState == GPUTransportState::Recording;
}

bool GPUTransportController::isRecording() const
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        return transportManager.isRecording();
    }
    return currentState == GPUTransportState::Recording;
}

bool GPUTransportController::isPaused() const
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        return transportManager.isPaused();
    }
    return currentState == GPUTransportState::Paused;
}

uint64_t GPUTransportController::getCurrentGPUFrame() const
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        return transportManager.getCurrentFrame();
    }
    
    if (!jam::gpu_native::GPUTimebase::is_initialized()) return 0;
    
    if (currentState == GPUTransportState::Paused) {
        return pausedFrame;
    }
    
    return jam::gpu_native::GPUTimebase::get_current_time_ns();
}

double GPUTransportController::getCurrentTimeInSeconds() const
{
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        return transportManager.getPositionSeconds();
    }
    
    if (!jam::gpu_native::GPUTimebase::is_initialized()) return 0.0;
    
    auto currentFrame = getCurrentGPUFrame();
    
    // Convert nanoseconds to seconds
    return static_cast<double>(currentFrame) / 1000000000.0;
}

void GPUTransportController::setBPM(double bpm)
{
    currentBPM = juce::jlimit(60.0, 200.0, bpm);
    bpmSlider->setValue(currentBPM, juce::dontSendNotification);
    
    // Update GPU Transport Manager BPM
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (transportManager.isInitialized()) {
        transportManager.setBPM(static_cast<float>(currentBPM));
    } else {
        // Fallback to GPU timebase
        jam::gpu_native::GPUTimebase::set_bpm(static_cast<uint32_t>(currentBPM));
    }
    
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
        double position = getCurrentTimeInSeconds();
        networkPanel->sendTransportCommand(command, timestamp, position, currentBPM);
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
        positionLabel->setText("--:--:---.---", juce::dontSendNotification);
        return;
    }
    
    double timeInSeconds = getCurrentTimeInSeconds();
    int minutes = static_cast<int>(timeInSeconds) / 60;
    int seconds = static_cast<int>(timeInSeconds) % 60;
    double fractionalSeconds = timeInSeconds - static_cast<int>(timeInSeconds);
    int milliseconds = static_cast<int>(fractionalSeconds * 1000) % 1000;
    int microseconds = static_cast<int>(fractionalSeconds * 1000000) % 1000;
    
    juce::String timeString = juce::String::formatted("%02d:%02d.%03d.%03d", 
                                                      minutes, seconds, milliseconds, microseconds);
    positionLabel->setText(timeString, juce::dontSendNotification);
}

void GPUTransportController::updateBarsBeatsDisplay()
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) {
        barsBeatsLabel->setText("---.---.---", juce::dontSendNotification);
        return;
    }
    
    // Get current bars/beats info directly from GPU transport manager
    auto& transportManager = jam::gpu_transport::GPUTransportManager::getInstance();
    if (!transportManager.isInitialized()) {
        barsBeatsLabel->setText("---.---.---", juce::dontSendNotification);
        return;
    }
    
    // Get GPU-native bars/beats calculation 
    GPUBarsBeatsBuffer barsBeatsInfo = transportManager.getBarsBeatsInfo();
    
    // Format as "BAR.BEAT.TICKS" like a DAW (e.g., "001.01.000")
    juce::String barsBeatsString = juce::String::formatted("%03d.%02d.%03d", 
                                                           barsBeatsInfo.bars, 
                                                           barsBeatsInfo.beats, 
                                                           barsBeatsInfo.subdivisions);
    barsBeatsLabel->setText(barsBeatsString, juce::dontSendNotification);
}
