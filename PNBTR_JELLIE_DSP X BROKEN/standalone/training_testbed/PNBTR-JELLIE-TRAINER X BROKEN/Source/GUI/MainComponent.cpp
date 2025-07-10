/*
  ==============================================================================

    MainComponent.cpp
    Created: Main GUI component for PNBTR+JELLIE Training Testbed

    Implements the exact schematic layout with fixed row heights:
    - Title Bar: 40px
    - Row 1 (Oscilloscopes): 200px  
    - Row 2 (Waveform Analysis): 120px
    - Row 3 (Audio Tracks): 80px
    - Row 4 (Metrics Dashboard): 100px
    - Row 5 (Controls): 60px

  ==============================================================================
*/


#include "MainComponent.h"
// JUCE module includes (must be first)
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "MainComponent.h"
#include "../Audio/AudioEngine.h"

using namespace juce;

//==============================================================================


MainComponent::MainComponent()
    : lastTimerCall(std::chrono::steady_clock::now())
{
    setSize(1200, 800);
    
    // Initialize AudioDeviceManager
    deviceManager.initialiseWithDefaultDevices(2, 2);
    deviceManager.addAudioCallback(this);
    
    // Initialize progressive loading
    loadingLabel = std::make_unique<juce::Label>("Loading", "Initializing...");
    loadingLabel->setFont(juce::Font(16.0f, juce::Font::bold));
    loadingLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    loadingLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(loadingLabel.get());
    
    startTimer(100); // Start progressive loading
}

void MainComponent::handleTransportPlay()
{
    // Initialize and start YOUR AudioEngine system
    if (!audioEngine) {
        audioEngine = std::make_unique<AudioEngine>();
        audioEngine->setTrainer(pnbtrTrainer.get());
        audioEngine->initialize(48000.0, 512);
    }
    
    // Start YOUR game engine audio architecture
    audioEngine->startProcessing();
    
    if (pnbtrTrainer) {
        pnbtrTrainer->startTraining();
    }
}

void MainComponent::handleTransportStop()
{
    if (pnbtrTrainer) {
        pnbtrTrainer->stopTraining();
    }
    
    if (audioEngine) {
        audioEngine->stopProcessing();
    }
}

void MainComponent::handleTransportRecord()
{
    // Start recording (set recordingActive)
    if (pnbtrTrainer)
        pnbtrTrainer->recordingActive.store(true);
    handleTransportPlay();
}
// Audio device management
void MainComponent::updateDeviceLists()
{
    // Make device enumeration async to prevent UI blocking
    juce::Timer::callAfterDelay(10, [this]() {
        juce::Logger::writeToLog("[DEVICE] Updating device lists...");
        
        inputDeviceBox->clear();
        outputDeviceBox->clear();
        auto setup = deviceManager.getAudioDeviceSetup();
        
        // Device enumeration can be slow - this is the main blocking operation
        const auto& deviceTypes = deviceManager.getAvailableDeviceTypes();
        if (deviceTypes.isEmpty()) {
            juce::Logger::writeToLog("[DEVICE] No device types available");
            return;
        }
        
        juce::StringArray inputNames = deviceTypes[0]->getDeviceNames(true);
        juce::StringArray outputNames = deviceTypes[0]->getDeviceNames(false);
        
        for (int i = 0; i < inputNames.size(); ++i)
            inputDeviceBox->addItem(inputNames[i], i + 1);
        for (int i = 0; i < outputNames.size(); ++i)
            outputDeviceBox->addItem(outputNames[i], i + 1);
        
        inputDeviceBox->setSelectedId(inputNames.indexOf(setup.inputDeviceName) + 1, juce::dontSendNotification);
        outputDeviceBox->setSelectedId(outputNames.indexOf(setup.outputDeviceName) + 1, juce::dontSendNotification);
        
        juce::Logger::writeToLog("[DEVICE] Device lists updated - Input: " + juce::String(inputNames.size()) + 
                                ", Output: " + juce::String(outputNames.size()));
    });
}

void MainComponent::inputDeviceChanged()
{
    // Defer device change to avoid blocking UI thread
    juce::Timer::callAfterDelay(50, [this]() {
        auto setup = deviceManager.getAudioDeviceSetup();
        setup.inputDeviceName = inputDeviceBox->getText();
        juce::Logger::writeToLog("[DEVICE] Changing input to: " + setup.inputDeviceName);
        
        // Use async device setup (false = non-blocking)
        juce::String error = deviceManager.setAudioDeviceSetup(setup, false);
        if (error.isNotEmpty()) {
            juce::Logger::writeToLog("[DEVICE] Input change error: " + error);
        }
    });
}

void MainComponent::outputDeviceChanged()
{
    // Defer device change to avoid blocking UI thread  
    juce::Timer::callAfterDelay(50, [this]() {
        auto setup = deviceManager.getAudioDeviceSetup();
        setup.outputDeviceName = outputDeviceBox->getText();
        juce::Logger::writeToLog("[DEVICE] Changing output to: " + setup.outputDeviceName);
        
        // Use async device setup (false = non-blocking)
        juce::String error = deviceManager.setAudioDeviceSetup(setup, false);
        if (error.isNotEmpty()) {
            juce::Logger::writeToLog("[DEVICE] Output change error: " + error);
        }
    });
}

void MainComponent::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    juce::Logger::writeToLog("[AUDIO ENGINE] Device starting: " + device->getName() + 
                            " (" + juce::String(device->getCurrentSampleRate()) + "Hz, " +
                            juce::String(device->getCurrentBufferSizeSamples()) + " samples)");
    
    if (pnbtrTrainer) {
        pnbtrTrainer->prepareToPlay(device->getCurrentSampleRate(), device->getCurrentBufferSizeSamples());
        juce::Logger::writeToLog("[AUDIO ENGINE] PNBTRTrainer prepared for YOUR AudioEngine");
    }
    
    // Initialize YOUR AudioEngine with proper parameters
    if (!audioEngine) {
        audioEngine = std::make_unique<AudioEngine>();
        audioEngine->setTrainer(pnbtrTrainer.get());
        audioEngine->initialize(device->getCurrentSampleRate(), device->getCurrentBufferSizeSamples());
        juce::Logger::writeToLog("[AUDIO ENGINE] YOUR AudioEngine initialized with device parameters");
    }
}

void MainComponent::audioDeviceStopped()
{
    if (audioEngine) {
        audioEngine->stopProcessing();
        juce::Logger::writeToLog("[AUDIO ENGINE] YOUR AudioEngine stopped");
    }
    
    if (pnbtrTrainer)
        pnbtrTrainer->releaseResources();
}

// Minimal JUCE callback that delegates to YOUR AudioEngine system
void MainComponent::audioDeviceIOCallback(const float** inputChannelData, int numInputChannels,
                                          float** outputChannelData, int numOutputChannels, int numSamples)
{
    // Clear output first for safety
    for (int ch = 0; ch < numOutputChannels; ++ch)
        std::memset(outputChannelData[ch], 0, sizeof(float) * numSamples);

    // Delegate to YOUR AudioEngine instead of basic JUCE processing
    if (audioEngine && audioEngine->isProcessing()) {
        // YOUR AudioEngine handles all the sophisticated processing
        // This is just a minimal bridge to the real system
        
        // Let YOUR AudioEngine do the real work
        audioEngine->processAudioCallback(
            inputChannelData ? inputChannelData[0] : nullptr, 
            outputChannelData[0], 
            numOutputChannels, 
            numSamples
        );
        
    } else {
        // AudioEngine not running - output silence
    }
}

MainComponent::~MainComponent()
{
    // 🔥 CRITICAL: Stop timer and cleanup AudioDeviceManager to prevent crashes
    stopTimer();
    printf("[DESTRUCTOR] Cleaning up AudioDeviceManager...\n");
    fflush(stdout);
    deviceManager.removeAudioCallback(this);
    deviceManager.closeAudioDevice();
    printf("[DESTRUCTOR] AudioDeviceManager cleanup completed\n");
    fflush(stdout);
}

//==============================================================================
void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::darkgrey);

    // Optional: draw horizontal lines between rows
    const int rowHeights[] = {40, 48, 200, 240, 160, 100, 60};
    int y = rowHeights[0];
    for (int i = 1; i < 7; ++i)
    {
        y += rowHeights[i - 1];
        g.drawLine(0.0f, (float)y, (float)getWidth(), (float)y, 1.0f);
    }
}

void MainComponent::resized()
{
    auto area = getLocalBounds();
    
    // Show loading screen until fully loaded
    if (!isFullyLoaded) {
        if (loadingLabel) {
            loadingLabel->setBounds(area);
        }
        return;
    }
    
    // Full UI layout (only when loaded) - REMOVED redundant title row
    const int rowHeights[] = {48, 200, 240, 160, 100, 60};

    if (transportBar) transportBar->setBounds(area.removeFromTop(rowHeights[0]));
    
    // Place device dropdowns above oscilloscope row
    auto deviceBar = area.removeFromTop(32);
    if (inputDeviceBox) inputDeviceBox->setBounds(deviceBar.removeFromLeft(getWidth() / 2).reduced(8, 4));
    if (outputDeviceBox) outputDeviceBox->setBounds(deviceBar.reduced(8, 4));
    
    if (oscilloscopeRow) oscilloscopeRow->setBounds(area.removeFromTop(rowHeights[1]));
    if (waveformAnalysisRow) waveformAnalysisRow->setBounds(area.removeFromTop(rowHeights[2]));
    if (audioTracksRow) audioTracksRow->setBounds(area.removeFromTop(rowHeights[3]));
    if (metricsDashboard) metricsDashboard->setBounds(area.removeFromTop(rowHeights[4]));
    if (controlsRow) controlsRow->setBounds(area.removeFromTop(rowHeights[5]));
}

void MainComponent::timerCallback()
{
    if (initializationStep >= maxSteps) {
        stopTimer();
        return;
    }

    auto currentTime = std::chrono::steady_clock::now();
    auto deltaMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTimerCall).count();
    lastTimerCall = currentTime;

    // Progressive loading steps
    switch (initializationStep) {
        case 0: {
            auto step0Start = std::chrono::steady_clock::now();
            
            juce::Logger::writeToLog("[INIT] Step 0 - Basic initialization");
            
            auto step0End = std::chrono::steady_clock::now();
            auto step0Time = std::chrono::duration_cast<std::chrono::milliseconds>(step0End - step0Start).count();
            juce::Logger::writeToLog("[INIT] Step 0 completed in " + juce::String(step0Time) + "ms");
            break;
        }
        
        case 1: {
            auto step1Start = std::chrono::steady_clock::now();
            
            juce::Logger::writeToLog("[INIT] Step 1 - Creating transport bar");
            
            transportBar = std::make_unique<ProfessionalTransportController>();
            addAndMakeVisible(transportBar.get());
            
            transportBar->onPlay = [this]() {
                handleTransportPlay();
            };
            transportBar->onStop = [this]() {
                handleTransportStop(); 
            };
            transportBar->onRecord = [this]() {
                handleTransportRecord();
            };
            transportBar->onLogMessage = [this](const juce::String& message) {
                // Silent logging
            };
            
            auto step1End = std::chrono::steady_clock::now();
            auto step1Time = std::chrono::duration_cast<std::chrono::milliseconds>(step1End - step1Start).count();
            juce::Logger::writeToLog("[INIT] Step 1 completed in " + juce::String(step1Time) + "ms");
            break;
        }
        
        case 2: {
            auto step2Start = std::chrono::steady_clock::now();
            
            juce::Logger::writeToLog("[INIT] Step 2 - Creating device dropdowns");
            
            inputDeviceBox = std::make_unique<juce::ComboBox>("Input Device");
            outputDeviceBox = std::make_unique<juce::ComboBox>("Output Device");
            
            addAndMakeVisible(inputDeviceBox.get());
            addAndMakeVisible(outputDeviceBox.get());
            
            inputDeviceBox->onChange = [this] { inputDeviceChanged(); };
            outputDeviceBox->onChange = [this] { outputDeviceChanged(); };
            
            updateDeviceLists();
            
            auto step2End = std::chrono::steady_clock::now();
            auto step2Time = std::chrono::duration_cast<std::chrono::milliseconds>(step2End - step2Start).count();
            juce::Logger::writeToLog("[INIT] Step 2 completed in " + juce::String(step2Time) + "ms");
            break;
        }
        
        case 3: {
            auto step3Start = std::chrono::steady_clock::now();
            
            juce::Logger::writeToLog("[INIT] Step 3 - Creating DSP engine");
            
            // Create DSP engine asynchronously
            juce::MessageManager::callAsync([this]() {
                pnbtrTrainer = std::make_unique<PNBTRTrainer>();
                juce::Logger::writeToLog("[INIT] PNBTRTrainer created successfully");
            });
            
            auto step3End = std::chrono::steady_clock::now();
            auto step3Time = std::chrono::duration_cast<std::chrono::milliseconds>(step3End - step3Start).count();
            juce::Logger::writeToLog("[INIT] Step 3 completed in " + juce::String(step3Time) + "ms");
            break;
        }
        
        // Additional steps 4-9 would be here...
        
        case 10: {
            auto step10Start = std::chrono::steady_clock::now();
            
            juce::Logger::writeToLog("[INIT] Step 10 - Final initialization");
            
            isFullyLoaded = true;
            
            if (loadingLabel) {
                removeChildComponent(loadingLabel.get());
                loadingLabel.reset();
            }
            
            if (transportBar) {
                transportBar->onLogMessage = [this](const juce::String& message) {
                    // Silent logging
                };
            }
            
            resized();
            
            auto step10End = std::chrono::steady_clock::now();
            auto step10Time = std::chrono::duration_cast<std::chrono::milliseconds>(step10End - step10Start).count();
            juce::Logger::writeToLog("[INIT] Final step completed in " + juce::String(step10Time) + "ms - App ready");
            break;
        }
    }
    
    initializationStep++;
}
