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

using namespace juce;

//==============================================================================


MainComponent::MainComponent()
{
    printf("[CONSTRUCTOR] MainComponent constructor starting...\n");
    fflush(stdout);
    
    // GAME ENGINE APPROACH: Minimal synchronous setup
    setSize(1280, 720);
    printf("[CONSTRUCTOR] Size set to 1280x720\n");
    
    // Create loading screen first (immediate, lightweight)
    loadingLabel = std::make_unique<juce::Label>("Loading", "Initializing PNBTR+JELLIE Training System...");
    loadingLabel->setJustificationType(juce::Justification::centred);
    loadingLabel->setFont(juce::Font(24.0f, juce::Font::bold));
    loadingLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(loadingLabel.get());
    printf("[CONSTRUCTOR] Loading label created and added\n");
    
    // Start background initialization timer (video game style)
    initializationStep = 0;
    printf("[CONSTRUCTOR] About to call startTimer(16)...\n");
    fflush(stdout);
    startTimer(16); // 60 FPS update rate like a game engine
    printf("[CONSTRUCTOR] startTimer(16) completed successfully\n");
    fflush(stdout);
    
    juce::Logger::writeToLog("[INIT] MainComponent constructor completed - starting background loading...");
    printf("[CONSTRUCTOR] MainComponent constructor completed!\n");
    fflush(stdout);
}

void MainComponent::handleTransportPlay()
{
    // Start audio device if not running
    auto* device = deviceManager.getCurrentAudioDevice();
    if (!device || !device->isOpen()) {
        juce::Logger::writeToLog("[TRANSPORT] Restarting audio device");
        deviceManager.restartLastAudioDevice();
    }
    if (pnbtrTrainer)
        pnbtrTrainer->startTraining();
    juce::Logger::writeToLog("[TRANSPORT] Play pressed, training started");
}

void MainComponent::handleTransportStop()
{
    // Stop audio device
    juce::Logger::writeToLog("[TRANSPORT] Stop pressed, closing audio device");
    deviceManager.closeAudioDevice();
    if (pnbtrTrainer)
        pnbtrTrainer->stopTraining();
}

void MainComponent::handleTransportRecord()
{
    // Start recording (set recordingActive)
    if (pnbtrTrainer)
        pnbtrTrainer->recordingActive.store(true);
    handleTransportPlay();
}

// ADDED: Record arm state methods for connecting UI to audio pipeline
bool MainComponent::isJellieTrackRecordArmed() const
{
    return jellieTrack ? jellieTrack->isRecordArmed() : false;
}

bool MainComponent::isPNBTRTrackRecordArmed() const
{
    return pnbtrTrack ? pnbtrTrack->isRecordArmed() : false;
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
    juce::Logger::writeToLog("[AUDIO] Device starting: " + device->getName() + 
                            " (" + juce::String(device->getCurrentSampleRate()) + "Hz, " +
                            juce::String(device->getCurrentBufferSizeSamples()) + " samples)");
    
    if (pnbtrTrainer) {
        pnbtrTrainer->prepareToPlay(device->getCurrentSampleRate(), device->getCurrentBufferSizeSamples());
        
        // Auto-start training when audio device starts (no manual button required)
        pnbtrTrainer->startTraining();
        juce::Logger::writeToLog("[AUDIO] Training auto-started with audio device");
    }
}

void MainComponent::audioDeviceStopped()
{
    if (pnbtrTrainer)
        pnbtrTrainer->releaseResources();
}

void MainComponent::audioDeviceIOCallback(const float** inputChannelData, int numInputChannels,
                                          float** outputChannelData, int numOutputChannels, int numSamples)
{
    // CRITICAL FIX: Verify callback is being called and show input levels
    static int callbackCount = 0;
    static int audioDebugCount = 0;
    
    // FORCE CONSOLE OUTPUT - ALWAYS log first few callbacks to confirm execution
    if (callbackCount < 5) {
        printf("[AUDIO CALLBACK #%d] CHANNELS IN:%d OUT:%d SAMPLES:%d\n", 
               callbackCount + 1, numInputChannels, numOutputChannels, numSamples);
        fflush(stdout);  // Force immediate output
    }
    
    if (++callbackCount % 100 == 0) {
        juce::Logger::writeToLog("[AUDIO CALLBACK] Running (" + juce::String(callbackCount) + ")");
    }
    
    // ENHANCED DEBUG: Check if we're actually receiving microphone input and show levels
    if (++audioDebugCount % 50 == 0) { // Every ~1 second
        float maxInputLevel = 0.0f;
        float rmsLevel = 0.0f;
        int samplesWithSignal = 0;
        
        if (inputChannelData && numInputChannels > 0 && inputChannelData[0]) {
            for (int i = 0; i < numSamples; ++i) {
                float sample = std::abs(inputChannelData[0][i]);
                maxInputLevel = std::max(maxInputLevel, sample);
                rmsLevel += sample * sample;
                if (sample > 0.001f) samplesWithSignal++; // Count samples above noise floor
            }
            rmsLevel = std::sqrt(rmsLevel / numSamples);
        }
        
        printf("[🎤 INPUT LEVEL] Max: %.4f, RMS: %.4f, Active samples: %d/%d\n", 
               maxInputLevel, rmsLevel, samplesWithSignal, numSamples);
        
        juce::Logger::writeToLog("[INPUT DEBUG] Channels: " + juce::String(numInputChannels) + 
                                ", Max Level: " + juce::String(maxInputLevel, 4) + 
                                ", RMS Level: " + juce::String(rmsLevel, 4) +
                                ", Active Samples: " + juce::String(samplesWithSignal) + "/" + juce::String(numSamples) +
                                ", Data Ptr: " + juce::String(inputChannelData ? "valid" : "null"));
    }
    
    // Defensive: zero output in case of crash or uninitialized buffer
    for (int ch = 0; ch < numOutputChannels; ++ch)
        std::memset(outputChannelData[ch], 0, sizeof(float) * numSamples);

    // CORRECTED: Create proper JUCE buffer and copy input data (no const cast issues)
    juce::AudioBuffer<float> buffer(std::max(numInputChannels, 2), numSamples);
    buffer.clear(); // Start with silence
    
    // Copy input channels to buffer (safe copy, not cast)
    if (inputChannelData && numInputChannels > 0) {
        for (int ch = 0; ch < std::min(numInputChannels, buffer.getNumChannels()); ++ch) {
            if (inputChannelData[ch]) {
                buffer.copyFrom(ch, 0, inputChannelData[ch], numSamples);
            }
        }
    }
    
    // Ensure we have at least stereo for processing
    if (buffer.getNumChannels() == 1) {
        buffer.setSize(2, numSamples, true, true, true); // Make stereo, copy existing data
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples); // Duplicate mono to stereo
    }
    
    if (pnbtrTrainer)
    {
        // Try/catch to prevent any crash from killing the audio thread
        try {
            juce::MidiBuffer midi;
            
            // CRITICAL FIX: Pass record arm states to the audio processor
            bool jellieRecordArmed = isJellieTrackRecordArmed();
            bool pnbtrRecordArmed = isPNBTRTrackRecordArmed();
            
            // Set record arm states in the trainer so GPU pipeline can respect them
            pnbtrTrainer->setJellieRecordArmed(jellieRecordArmed);
            pnbtrTrainer->setPNBTRRecordArmed(pnbtrRecordArmed);
            
            pnbtrTrainer->processBlock(buffer, midi);
        } catch (const std::exception& e) {
            juce::Logger::writeToLog("[AUDIO CALLBACK] Exception: " + juce::String(e.what()));
        } catch (...) {
            juce::Logger::writeToLog("[AUDIO CALLBACK] Unknown exception");
        }
    }
    
    // Copy processed buffer to output (ensure we don't exceed output channel count)
    for (int ch = 0; ch < numOutputChannels; ++ch)
    {
        if (ch < buffer.getNumChannels()) {
            std::memcpy(outputChannelData[ch], buffer.getReadPointer(ch), sizeof(float) * numSamples);
        } else {
            // Zero out extra output channels
            std::memset(outputChannelData[ch], 0, sizeof(float) * numSamples);
        }
    }
}

MainComponent::~MainComponent() = default;

//==============================================================================
void MainComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::darkgrey);

    // FIXED: Consistent row heights matching resized() method
    const int rowHeights[] = {48, 32, 200, 160, 160, 100, 60}; // Added device bar height
    int y = 0;
    for (int i = 0; i < 6; ++i)
    {
        y += rowHeights[i];
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
    
    // FIXED: Consistent row heights and complete layout
    const int rowHeights[] = {48, 32, 200, 160, 160, 100, 60};

    // Row 1: Transport Bar (48px)
    if (transportBar) transportBar->setBounds(area.removeFromTop(rowHeights[0]));
    
    // Row 2: Device dropdowns (32px)
    auto deviceBar = area.removeFromTop(rowHeights[1]);
    if (inputDeviceBox) inputDeviceBox->setBounds(deviceBar.removeFromLeft(getWidth() / 2).reduced(8, 4));
    if (outputDeviceBox) outputDeviceBox->setBounds(deviceBar.reduced(8, 4));
    
    // Row 3: Oscilloscopes (200px)
    if (oscilloscopeRow) oscilloscopeRow->setBounds(area.removeFromTop(rowHeights[2]));
    
    // Row 4: JELLIE Track (160px)
    if (jellieTrack) jellieTrack->setBounds(area.removeFromTop(rowHeights[3]));
    
    // Row 5: PNBTR Track (160px)
    if (pnbtrTrack) pnbtrTrack->setBounds(area.removeFromTop(rowHeights[4]));
    
    // Row 6: Metrics Dashboard (100px)
    if (metricsDashboard) metricsDashboard->setBounds(area.removeFromTop(rowHeights[5]));
    
    // Row 7: Controls Row (60px) - FIXED: This was missing!
    if (controlsRow) controlsRow->setBounds(area.removeFromTop(rowHeights[6]));
}

void MainComponent::timerCallback()
{
    printf("[TIMER] Timer callback called - Step %d\n", initializationStep);
    fflush(stdout);
    
    // VIDEO GAME ENGINE LOADING: One component per frame to prevent blocking
    switch (initializationStep) {
        case 0:
            printf("[INIT] Step 0: Creating transport bar...\n");
            juce::Logger::writeToLog("[INIT] Step 0: Creating transport bar...");
            transportBar = std::make_unique<ProfessionalTransportController>();
            addAndMakeVisible(transportBar.get());
            loadingLabel->setText("Loading Transport Controls...", juce::dontSendNotification);
            printf("[INIT] Step 0 completed successfully\n");
            break;
            
        case 1:
            juce::Logger::writeToLog("[INIT] Step 1: Creating audio device dropdowns...");
            inputDeviceBox = std::make_unique<juce::ComboBox>("InputDevice");
            outputDeviceBox = std::make_unique<juce::ComboBox>("OutputDevice");
            addAndMakeVisible(inputDeviceBox.get());
            addAndMakeVisible(outputDeviceBox.get());
            inputDeviceBox->onChange = [this] { inputDeviceChanged(); };
            outputDeviceBox->onChange = [this] { outputDeviceChanged(); };
            loadingLabel->setText("Loading Audio Device Controls...", juce::dontSendNotification);
            break;
            
        case 2:
            juce::Logger::writeToLog("[INIT] Step 2: Creating audio device dropdowns...");
            inputDeviceBox = std::make_unique<juce::ComboBox>("InputDevice");
            outputDeviceBox = std::make_unique<juce::ComboBox>("OutputDevice");
            addAndMakeVisible(inputDeviceBox.get());
            addAndMakeVisible(outputDeviceBox.get());
            inputDeviceBox->onChange = [this] { inputDeviceChanged(); };
            outputDeviceBox->onChange = [this] { outputDeviceChanged(); };
            loadingLabel->setText("Loading Audio Device Controls...", juce::dontSendNotification);
            break;
            
        case 3:
            juce::Logger::writeToLog("[INIT] Step 3: Creating DSP engine (background)...");
            // This is the heavy one - DSP engine with Metal initialization
            pnbtrTrainer = std::make_unique<PNBTRTrainer>();
            loadingLabel->setText("Loading DSP Engine & GPU Resources...", juce::dontSendNotification);
            break;
            
        case 4:
            juce::Logger::writeToLog("[INIT] Step 4: Creating oscilloscopes...");
            oscilloscopeRow = std::make_unique<OscilloscopeRow>();
            addAndMakeVisible(oscilloscopeRow.get());
            loadingLabel->setText("Loading Real-Time Visualizations...", juce::dontSendNotification);
            break;
            
        case 5:
            juce::Logger::writeToLog("[INIT] Step 5: Creating waveform analysis...");
            waveformAnalysisRow = std::make_unique<WaveformAnalysisRow>();
            addAndMakeVisible(waveformAnalysisRow.get());
            loadingLabel->setText("Loading Waveform Analysis...", juce::dontSendNotification);
            break;
            
        case 6:
            juce::Logger::writeToLog("[INIT] Step 6: Creating spectral audio tracks...");
            jellieTrack = std::make_unique<SpectralAudioTrack>(SpectralAudioTrack::TrackType::JELLIE_INPUT, "JELLIE");
            pnbtrTrack = std::make_unique<SpectralAudioTrack>(SpectralAudioTrack::TrackType::PNBTR_OUTPUT, "PNBTR");
            addAndMakeVisible(jellieTrack.get());
            addAndMakeVisible(pnbtrTrack.get());
            loadingLabel->setText("Loading Spectral Audio Tracks...", juce::dontSendNotification);
            break;
            
        case 7:
            juce::Logger::writeToLog("[INIT] Step 7: Creating metrics dashboard...");
            metricsDashboard = std::make_unique<MetricsDashboard>();
            addAndMakeVisible(metricsDashboard.get());
            loadingLabel->setText("Loading Metrics Dashboard...", juce::dontSendNotification);
            break;
            
        case 8:
            juce::Logger::writeToLog("[INIT] Step 8: Creating controls row...");
            controlsRow = std::make_unique<ControlsRow>();
            addAndMakeVisible(controlsRow.get());
            loadingLabel->setText("Loading Control Panel...", juce::dontSendNotification);
            break;
            
        case 9:
            juce::Logger::writeToLog("[INIT] Step 9: Wiring components together...");
            // Wire transport bar to audio device and DSP
            transportBar->onPlay = [this] { handleTransportPlay(); };
            transportBar->onStop = [this] { handleTransportStop(); };
            transportBar->onRecord = [this] { handleTransportRecord(); };
            
            // Connect controls and visualizations to DSP pipeline
            transportBar->setTrainer(pnbtrTrainer.get());
            controlsRow->setTrainer(pnbtrTrainer.get());
            oscilloscopeRow->setTrainer(pnbtrTrainer.get());
            waveformAnalysisRow->setTrainer(pnbtrTrainer.get());
            metricsDashboard->setTrainer(pnbtrTrainer.get());
            jellieTrack->setTrainer(pnbtrTrainer.get());
            pnbtrTrack->setTrainer(pnbtrTrainer.get());
            
            // Connect TOAST network oscilloscope to metrics dashboard
            metricsDashboard->setTOASTNetworkOscilloscope(&oscilloscopeRow->getNetworkOsc());
            loadingLabel->setText("Connecting Components...", juce::dontSendNotification);
            break;
            
        case 10: {
            printf("[INIT] Step 10: Initializing audio system...\n");
            juce::Logger::writeToLog("[INIT] Step 10: Initializing audio system...");
            
            // CRITICAL FIX FOR macOS: Request microphone permissions explicitly
            #if JUCE_MAC
            printf("[PERMISSIONS] Requesting microphone permissions on macOS...\n");
            juce::Logger::writeToLog("[PERMISSIONS] Requesting microphone permissions on macOS...");
            
            // Request microphone permission (this will show the permission dialog if needed)
            juce::RuntimePermissions::request(juce::RuntimePermissions::recordAudio, 
                [this](bool granted) {
                    if (granted) {
                        printf("[PERMISSIONS] ✅ Microphone permission granted!\n");
                        juce::Logger::writeToLog("[PERMISSIONS] ✅ Microphone permission granted!");
                    } else {
                        printf("[PERMISSIONS] ❌ Microphone permission denied!\n");
                        juce::Logger::writeToLog("[PERMISSIONS] ❌ Microphone permission denied!");
                    }
                });
            
            // Wait a moment for permission dialog to be handled
            juce::Timer::callAfterDelay(100, [this]() {
                bool hasPermission = juce::RuntimePermissions::isGranted(juce::RuntimePermissions::recordAudio);
                printf("[PERMISSIONS] Microphone permission status: %s\n", hasPermission ? "GRANTED" : "DENIED");
                juce::Logger::writeToLog("[PERMISSIONS] Microphone permission status: " + 
                                       juce::String(hasPermission ? "GRANTED" : "DENIED"));
            });
            #endif
            
            // CRITICAL FIX: Enable input channels explicitly and request microphone permissions
            printf("[DEVICE] Requesting microphone permissions and initializing with input channels...\n");
            juce::String error = deviceManager.initialise(2, 2, nullptr, true);
            
            if (error.isNotEmpty()) {
                printf("[DEVICE] ERROR: %s\n", error.toUTF8().getAddress());
                juce::Logger::writeToLog("[DEVICE] Initialization error: " + error);
                // Try fallback with different settings
                printf("[DEVICE] Trying fallback initialise(1, 2, nullptr, false)...\n");
                error = deviceManager.initialise(1, 2, nullptr, false);
                if (error.isNotEmpty()) {
                    printf("[DEVICE] FALLBACK ALSO FAILED: %s\n", error.toUTF8().getAddress());
                    juce::Logger::writeToLog("[DEVICE] Fallback initialization also failed: " + error);
                }
            } else {
                printf("[DEVICE] Audio device manager initialized successfully\n");
                juce::Logger::writeToLog("[DEVICE] Audio device manager initialized successfully");
                
                // CRITICAL FIX: Add callback BEFORE configuring device
                printf("[CALLBACK] Adding audio callback to device manager...\n");
                juce::Logger::writeToLog("[CALLBACK] Adding audio callback to device manager...");
                deviceManager.addAudioCallback(this);
                printf("[CALLBACK] Audio callback registered successfully\n");
                juce::Logger::writeToLog("[CALLBACK] Audio callback registered successfully");
                
                // CRITICAL FIX: Explicitly enable input channels and configure device
                auto setup = deviceManager.getAudioDeviceSetup();
                printf("[DEVICE] Current setup - Input: '%s', Output: '%s'\n", 
                       setup.inputDeviceName.toUTF8().getAddress(),
                       setup.outputDeviceName.toUTF8().getAddress());
                
                // Ensure we have input devices selected
                if (setup.inputDeviceName.isEmpty()) {
                    auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
                    auto inputDevices = deviceType->getDeviceNames(true);
                    if (!inputDevices.isEmpty()) {
                        setup.inputDeviceName = inputDevices[0];
                        printf("[DEVICE] Selected default input device: '%s'\n", 
                               setup.inputDeviceName.toUTF8().getAddress());
                    }
                }
                
                // CRITICAL: Set up input channels explicitly
                setup.useDefaultInputChannels = true;
                setup.inputChannels.setRange(0, 2, true);  // Enable first 2 input channels
                setup.useDefaultOutputChannels = true;
                setup.outputChannels.setRange(0, 2, true); // Enable first 2 output channels
                setup.sampleRate = 48000.0;
                setup.bufferSize = 512;
                
                printf("[DEVICE] Applying audio setup with input channels enabled...\n");
                error = deviceManager.setAudioDeviceSetup(setup, true); // true = strict requirements
                
                if (error.isNotEmpty()) {
                    printf("[DEVICE] Setup error: %s\n", error.toUTF8().getAddress());
                    juce::Logger::writeToLog("[DEVICE] Audio setup error: " + error);
                } else {
                    printf("[DEVICE] Audio device setup completed successfully\n");
                    juce::Logger::writeToLog("[DEVICE] Audio device setup completed successfully");
                    
                    // Verify the device is actually running and has input channels
                    auto* currentDevice = deviceManager.getCurrentAudioDevice();
                    if (currentDevice && currentDevice->isOpen()) {
                        auto activeInputs = currentDevice->getActiveInputChannels();
                        auto activeOutputs = currentDevice->getActiveOutputChannels();
                        printf("[DEVICE] ✅ Device running: '%s'\n", currentDevice->getName().toUTF8().getAddress());
                        printf("[DEVICE] ✅ Active input channels: %s\n", activeInputs.toString(2).toUTF8().getAddress());
                        printf("[DEVICE] ✅ Active output channels: %s\n", activeOutputs.toString(2).toUTF8().getAddress());
                        printf("[DEVICE] ✅ Sample rate: %.0f Hz\n", currentDevice->getCurrentSampleRate());
                        printf("[DEVICE] ✅ Buffer size: %d samples\n", currentDevice->getCurrentBufferSizeSamples());
                        
                        juce::Logger::writeToLog("[DEVICE] ✅ Audio device fully operational with input channels!");
                        juce::Logger::writeToLog("[DEVICE] Input channels: " + activeInputs.toString(2));
                        juce::Logger::writeToLog("[DEVICE] Output channels: " + activeOutputs.toString(2));
                    } else {
                        printf("[DEVICE] ❌ Device not running after setup\n");
                        juce::Logger::writeToLog("[DEVICE] ❌ Device not running after setup");
                    }
                }
                
                updateDeviceLists();
                
                // CRITICAL FIX: Verify input channels are configured  
                juce::Logger::writeToLog("[DEVICE] Input channels requested: 2");
                juce::Logger::writeToLog("[DEVICE] Output channels requested: 2");
            }
            loadingLabel->setText("Initializing Audio System...", juce::dontSendNotification);
            break;
        }
            
        case 11: {
            juce::Logger::writeToLog("[INIT] Step 11: Finalizing layout...");
            // Hide loading screen and show final UI
            loadingLabel->setVisible(false);
            isFullyLoaded = true;
            resized(); // Trigger layout
            
            // DIAGNOSTIC: Check audio device status
            auto setup = deviceManager.getAudioDeviceSetup();
            juce::Logger::writeToLog("[DIAGNOSTIC] Input device: " + setup.inputDeviceName);
            juce::Logger::writeToLog("[DIAGNOSTIC] Output device: " + setup.outputDeviceName);
            juce::Logger::writeToLog("[DIAGNOSTIC] Sample rate: " + juce::String(setup.sampleRate));
            juce::Logger::writeToLog("[DIAGNOSTIC] Buffer size: " + juce::String(setup.bufferSize));
            
            // CRITICAL OSCILLOSCOPE FIX: Ensure input device is actually selected
            if (setup.inputDeviceName.isEmpty())
            {
                auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
                auto inputs = deviceType->getDeviceNames(true); // true = input devices
                if (! inputs.isEmpty())
                {
                    setup.inputDeviceName = inputs[0]; // select first available input
                    juce::Logger::writeToLog("[OSCILLOSCOPE FIX] Selected input device: " + setup.inputDeviceName);
                    deviceManager.setAudioDeviceSetup(setup, true);
                }
            }
            
            // Force audio device start for oscilloscope data - FIX: Actually start a device
            auto* currentDevice = deviceManager.getCurrentAudioDevice();
            if (!currentDevice || !currentDevice->isOpen()) {
                juce::Logger::writeToLog("[DIAGNOSTIC] No audio device running - starting default device...");
                
                // Get default audio device setup and force it to start
                auto setup = deviceManager.getAudioDeviceSetup();
                setup.bufferSize = 512;
                setup.sampleRate = 48000.0;
                
                // Ensure we have input and output devices selected
                if (setup.inputDeviceName.isEmpty()) {
                    auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
                    auto inputDevices = deviceType->getDeviceNames(true);
                    if (!inputDevices.isEmpty()) {
                        setup.inputDeviceName = inputDevices[0];
                        juce::Logger::writeToLog("[DIAGNOSTIC] Selected input: " + setup.inputDeviceName);
                    }
                }
                
                if (setup.outputDeviceName.isEmpty()) {
                    auto* deviceType = deviceManager.getAvailableDeviceTypes()[0];
                    auto outputDevices = deviceType->getDeviceNames(false);
                    if (!outputDevices.isEmpty()) {
                        setup.outputDeviceName = outputDevices[0];
                        juce::Logger::writeToLog("[DIAGNOSTIC] Selected output: " + setup.outputDeviceName);
                    }
                }
                
                // CRITICAL FIX: Force explicit device startup sequence
                juce::String error = deviceManager.setAudioDeviceSetup(setup, true);
                if (error.isNotEmpty()) {
                    juce::Logger::writeToLog("[DIAGNOSTIC] Failed to start audio device: " + error);
                } else {
                    juce::Logger::writeToLog("[DIAGNOSTIC] Audio device setup completed");
                    
                    // FORCE: Explicitly restart the device to ensure it starts
                    deviceManager.restartLastAudioDevice();
                    
                    // VERIFY: Check if device is actually running now
                    auto* nowRunning = deviceManager.getCurrentAudioDevice();
                    if (nowRunning && nowRunning->isOpen()) {
                        juce::Logger::writeToLog("[DIAGNOSTIC] ✅ AUDIO DEVICE IS NOW RUNNING!");
                        juce::Logger::writeToLog("[DIAGNOSTIC] Device: " + nowRunning->getName());
                        juce::Logger::writeToLog("[DIAGNOSTIC] Sample Rate: " + juce::String(nowRunning->getCurrentSampleRate()));
                        juce::Logger::writeToLog("[DIAGNOSTIC] Buffer Size: " + juce::String(nowRunning->getCurrentBufferSizeSamples()));
                        juce::Logger::writeToLog("[DIAGNOSTIC] Input Channels: " + nowRunning->getActiveInputChannels().toString(2));
                    } else {
                        juce::Logger::writeToLog("[DIAGNOSTIC] ❌ DEVICE STILL NOT RUNNING - DEEPER ISSUE");
                    }
                }
            } else {
                juce::Logger::writeToLog("[DIAGNOSTIC] Audio device already running: " + currentDevice->getName());
            }
            
            // Auto-start training for immediate oscilloscope response
            if (pnbtrTrainer) {
                pnbtrTrainer->startTraining();
                juce::Logger::writeToLog("[DIAGNOSTIC] Auto-started training for oscilloscope data");
            }
            
            stopTimer(); // Stop the loading timer
            juce::Logger::writeToLog("[INIT] ✅ FULLY LOADED - App ready for real-time operation!");
            break;
        }
            
        default:
            stopTimer();
            break;
    }
    
    initializationStep++;
    
    // Only repaint during loading phase (stops when initialization complete)
    if (initializationStep <= 11) {
        repaint(); // Trigger UI update only during loading
    }
}
