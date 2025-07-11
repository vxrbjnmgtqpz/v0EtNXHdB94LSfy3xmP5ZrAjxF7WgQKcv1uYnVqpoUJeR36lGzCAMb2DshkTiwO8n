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
    : isFullyLoaded(false),
      coreAudioBridge(nullptr),
      initializationStep(0)
{
    // Initialize Core Audio → Metal bridge immediately
    initializeCoreAudioBridge();
    
    // Create loading label
    loadingLabel = std::make_unique<juce::Label>("Loading", "Loading PNBTR+JELLIE Training System...");
    loadingLabel->setFont(juce::Font(24.0f, juce::Font::bold));
    loadingLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    loadingLabel->setJustificationType(juce::Justification::centred);
    addAndMakeVisible(loadingLabel.get());
    
    // Start initialization timer
    startTimer(200); // 200ms intervals for smooth loading animation
}

void MainComponent::handleTransportPlay()
{
    // Start Core Audio capture when transport plays
    if (coreAudioBridge) {
        startCoreAudioCapture();
        juce::Logger::writeToLog("[TRANSPORT] Play pressed, Core Audio capture started");
    }
}

void MainComponent::handleTransportStop()
{
    // Stop Core Audio capture when transport stops
    if (coreAudioBridge) {
        stopCoreAudioCapture();
        juce::Logger::writeToLog("[TRANSPORT] Stop pressed, Core Audio capture stopped");
    }
}

void MainComponent::handleTransportRecord()
{
    // Start recording - record arm states control actual recording
    handleTransportPlay();
    juce::Logger::writeToLog("[TRANSPORT] Record pressed, capture started (record arm states control actual recording)");
}

// Core Audio → Metal bridge management
void MainComponent::initializeCoreAudioBridge()
{
    juce::Logger::writeToLog("[CoreAudio→Metal] Initializing Core Audio → Metal bridge...");
    coreAudioBridge = createCoreAudioGPUBridge();
    if (coreAudioBridge) {
        juce::Logger::writeToLog("[CoreAudio→Metal] Bridge initialized successfully");
    } else {
        juce::Logger::writeToLog("[CoreAudio→Metal] Failed to initialize bridge");
    }
}

void MainComponent::shutdownCoreAudioBridge()
{
    if (coreAudioBridge) {
        destroyCoreAudioGPUBridge();
        coreAudioBridge = nullptr;
        juce::Logger::writeToLog("[CoreAudio→Metal] Bridge shutdown completed");
    }
}

// ADDED: Record arm state methods for connecting UI to Core Audio → Metal pipeline
bool MainComponent::isJellieTrackRecordArmed() const
{
    return jellieTrack ? jellieTrack->isRecordArmed() : false;
}

bool MainComponent::isPNBTRTrackRecordArmed() const
{
    return pnbtrTrack ? pnbtrTrack->isRecordArmed() : false;
}

void MainComponent::updateRecordArmStates()
{
    if (coreAudioBridge) {
        bool jellieArmed = isJellieTrackRecordArmed();
        bool pnbtrArmed = isPNBTRTrackRecordArmed();
        
        // 🎯 DIAGNOSTIC: Always log record arm state checks (every 10 calls)
        static int updateCount = 0;
        bool shouldLog = (++updateCount % 10 == 0) || jellieArmed || pnbtrArmed;
        
        if (shouldLog) {
            juce::Logger::writeToLog("[🔍 RECORD ARM CHECK #" + juce::String(updateCount) + "] JELLIE: " + 
                                   juce::String(jellieArmed ? "ARMED" : "DISARMED") + 
                                   ", PNBTR: " + juce::String(pnbtrArmed ? "ARMED" : "DISARMED"));
        }
        
        // Update Core Audio bridge with record arm states
        setCoreAudioRecordArmStates(jellieArmed, pnbtrArmed);
        
        // Extra logging when states change
        static bool lastJellieArmed = false;
        static bool lastPnbtrArmed = false;
        
        if (jellieArmed != lastJellieArmed || pnbtrArmed != lastPnbtrArmed) {
            juce::Logger::writeToLog("[🚨 RECORD ARM CHANGED] JELLIE: " + juce::String(jellieArmed ? "ARMED" : "DISARMED") + 
                                   ", PNBTR: " + juce::String(pnbtrArmed ? "ARMED" : "DISARMED"));
            lastJellieArmed = jellieArmed;
            lastPnbtrArmed = pnbtrArmed;
            
            // 🎯 CRITICAL FIX: Auto-start capture when any track becomes armed
            if (jellieArmed || pnbtrArmed) {
                juce::Logger::writeToLog("[🚀 AUTO-START] Track armed - starting Core Audio capture automatically");
                startCoreAudioCapture();
            } else {
                juce::Logger::writeToLog("[🛑 AUTO-STOP] No tracks armed - stopping Core Audio capture");
                stopCoreAudioCapture();
            }
        }
    }
}

// Core Audio device management
void MainComponent::updateDeviceLists()
{
    if (!coreAudioBridge) {
        juce::Logger::writeToLog("[DEVICE] Core Audio bridge not initialized");
        return;
    }
    
    juce::Logger::writeToLog("[DEVICE] Updating Core Audio device lists...");
    
    // Clear existing items
    inputDeviceBox->clear();
    outputDeviceBox->clear();
        
    // Get input devices
    int inputCount = getCoreAudioInputDeviceCount();
    for (int i = 0; i < inputCount; ++i) {
        const char* deviceName = getCoreAudioInputDeviceName(i);
        if (deviceName && strlen(deviceName) > 0) {
            inputDeviceBox->addItem(juce::String(deviceName), i + 1);
        }
    }
    
    // Get output devices
    int outputCount = getCoreAudioOutputDeviceCount();
    for (int i = 0; i < outputCount; ++i) {
        const char* deviceName = getCoreAudioOutputDeviceName(i);
        if (deviceName && strlen(deviceName) > 0) {
            outputDeviceBox->addItem(juce::String(deviceName), i + 1);
        }
    }
    
    // Set default selections
    if (inputCount > 0) {
        inputDeviceBox->setSelectedId(1, juce::dontSendNotification);
    }
    if (outputCount > 0) {
        outputDeviceBox->setSelectedId(1, juce::dontSendNotification);
            }
    
    juce::Logger::writeToLog("[DEVICE] Core Audio device lists updated - Input: " + 
                            juce::String(inputCount) + ", Output: " + juce::String(outputCount));
    }
    
void MainComponent::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged == inputDeviceBox.get()) {
        int selectedIndex = inputDeviceBox->getSelectedId() - 1;
        if (selectedIndex >= 0) {
            setCoreAudioInputDevice(selectedIndex);
            juce::Logger::writeToLog("[DEVICE] Input device changed to index: " + juce::String(selectedIndex));
    }
    } else if (comboBoxThatHasChanged == outputDeviceBox.get()) {
        int selectedIndex = outputDeviceBox->getSelectedId() - 1;
        if (selectedIndex >= 0) {
            setCoreAudioOutputDevice(selectedIndex);
            juce::Logger::writeToLog("[DEVICE] Output device changed to index: " + juce::String(selectedIndex));
        }
    }
}

// Core Audio → Metal Pipeline
// Audio processing now happens directly in CoreAudioGPUBridge → MetalBridge
// No JUCE audio callback needed since we bypass JUCE audio processing

MainComponent::~MainComponent()
{
    // Shutdown Core Audio bridge before destruction
    shutdownCoreAudioBridge();
}

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
    
    // DEBUGGING: Add debugging buttons at the bottom
    if (useDefaultInputButton && enableSineTestButton && checkMetalBridgeButton && forceCallbackButton) {
        auto debugArea = area.removeFromTop(40); // 40px for debug buttons
        int buttonWidth = debugArea.getWidth() / 4;
        
        useDefaultInputButton->setBounds(debugArea.removeFromLeft(buttonWidth).reduced(2));
        enableSineTestButton->setBounds(debugArea.removeFromLeft(buttonWidth).reduced(2));
        checkMetalBridgeButton->setBounds(debugArea.removeFromLeft(buttonWidth).reduced(2));
        forceCallbackButton->setBounds(debugArea.reduced(2));
    }
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
            inputDeviceBox->addListener(this);
            outputDeviceBox->addListener(this);
            addAndMakeVisible(inputDeviceBox.get());
            addAndMakeVisible(outputDeviceBox.get());
            loadingLabel->setText("Loading Audio Device Controls...", juce::dontSendNotification);
            break;
            
        case 2:
            juce::Logger::writeToLog("[INIT] Step 2: Populating Core Audio device lists...");
            // Populate device lists from Core Audio bridge
            updateDeviceLists();
            loadingLabel->setText("Loading Audio Device Lists...", juce::dontSendNotification);
            break;
            
        case 3:
            juce::Logger::writeToLog("[INIT] Step 3: Creating oscilloscopes...");
            oscilloscopeRow = std::make_unique<OscilloscopeRow>();
            addAndMakeVisible(oscilloscopeRow.get());
            loadingLabel->setText("Loading Real-Time Visualizations...", juce::dontSendNotification);
            break;
            
        case 4:
            juce::Logger::writeToLog("[INIT] Step 4: Creating waveform analysis...");
            waveformAnalysisRow = std::make_unique<WaveformAnalysisRow>();
            addAndMakeVisible(waveformAnalysisRow.get());
            loadingLabel->setText("Loading Waveform Analysis...", juce::dontSendNotification);
            break;
            
        case 5:
            juce::Logger::writeToLog("[INIT] Step 5: Creating spectral audio tracks...");
            jellieTrack = std::make_unique<SpectralAudioTrack>(SpectralAudioTrack::TrackType::JELLIE_INPUT, "JELLIE");
            pnbtrTrack = std::make_unique<SpectralAudioTrack>(SpectralAudioTrack::TrackType::PNBTR_OUTPUT, "PNBTR");
            addAndMakeVisible(jellieTrack.get());
            addAndMakeVisible(pnbtrTrack.get());
            loadingLabel->setText("Loading Spectral Audio Tracks...", juce::dontSendNotification);
            break;
            
        case 6:
            juce::Logger::writeToLog("[INIT] Step 6: Creating metrics dashboard...");
            metricsDashboard = std::make_unique<MetricsDashboard>();
            addAndMakeVisible(metricsDashboard.get());
            loadingLabel->setText("Loading Metrics Dashboard...", juce::dontSendNotification);
            break;
            
        case 7:
            juce::Logger::writeToLog("[INIT] Step 7: Creating controls row...");
            controlsRow = std::make_unique<ControlsRow>();
            addAndMakeVisible(controlsRow.get());
            
            // DEBUGGING: Add debugging buttons for testing audio pipeline
            setupDebuggingButtons();
            
            loadingLabel->setText("Loading Control Panel...", juce::dontSendNotification);
            break;
            
        case 8:
            juce::Logger::writeToLog("[INIT] Step 8: Wiring components together...");
            // Wire transport bar to Core Audio bridge
            transportBar->onPlay = [this] { handleTransportPlay(); };
            transportBar->onStop = [this] { handleTransportStop(); };
            transportBar->onRecord = [this] { handleTransportRecord(); };
            
            // Connect TOAST network oscilloscope to metrics dashboard
            metricsDashboard->setTOASTNetworkOscilloscope(&oscilloscopeRow->getNetworkOsc());
            loadingLabel->setText("Connecting Components...", juce::dontSendNotification);
            break;
            
        case 9: {
            juce::Logger::writeToLog("[INIT] Step 9: Initializing Core Audio → Metal pipeline...");
            
            // Request microphone permissions for Core Audio
            #if JUCE_MAC
            juce::RuntimePermissions::request(juce::RuntimePermissions::recordAudio, 
                [this](bool granted) {
                    if (granted) {
                        juce::Logger::writeToLog("[PERMISSIONS] ✅ Microphone permission granted!");
                    } else {
                        juce::Logger::writeToLog("[PERMISSIONS] ❌ Microphone permission denied!");
                    }
                });
            #endif
            
            // Core Audio bridge is already initialized in constructor
            // Just verify it's ready
            if (coreAudioBridge) {
                juce::Logger::writeToLog("[CoreAudio→Metal] ✅ Core Audio bridge ready for audio processing");
            } else {
                juce::Logger::writeToLog("[CoreAudio→Metal] ❌ Core Audio bridge not initialized");
            }
            
            loadingLabel->setText("Initializing Core Audio → Metal Pipeline...", juce::dontSendNotification);
            break;
        }
            
        case 10: {
            juce::Logger::writeToLog("[INIT] Step 10: Finalizing Core Audio → Metal pipeline...");
            // Hide loading screen and show final UI
            loadingLabel->setVisible(false);
            isFullyLoaded = true;
            resized(); // Trigger layout
            
            // 🧪 TEST: Auto-arm JELLIE track to verify pipeline functionality
            if (jellieTrack) {
                jellieTrack->setRecordArmed(true);
                juce::Logger::writeToLog("[🧪 TEST] Auto-armed JELLIE track for pipeline testing");
                    }
            
            // Start monitoring record arm states for Core Audio bridge
            startTimer(100); // Start 10Hz timer for record arm state monitoring
            
            juce::Logger::writeToLog("[CoreAudio→Metal] ✅ FULLY LOADED - Core Audio → Metal pipeline ready!");
            juce::Logger::writeToLog("[RECORD ARM] System ready to respond to record arm button states");
            break;
        }
            
        default:
            // 🔧 FIXED: Don't stop timer - let it continue for record arm monitoring
            // The timer now continues running at 100ms intervals to monitor record arm states
            break;
    }
    
    initializationStep++;
    
    // Only repaint during loading phase (stops when initialization complete)
    if (initializationStep <= 10) {
        repaint(); // Trigger UI update only during loading
    } else if (isFullyLoaded) {
        // After initialization, monitor record arm states for Core Audio bridge
        updateRecordArmStates();
    }
}

// DEBUGGING: Setup debugging buttons for testing audio pipeline
void MainComponent::setupDebuggingButtons()
{
    // Use Default Input Device button
    useDefaultInputButton = std::make_unique<juce::TextButton>("Use Default Input");
    useDefaultInputButton->setButtonText("Use Default Input");
    useDefaultInputButton->setColour(juce::TextButton::buttonColourId, juce::Colours::darkblue);
    useDefaultInputButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    useDefaultInputButton->onClick = [this] {
        useDefaultInputDevice();
        juce::Logger::writeToLog("[DEBUG] Use Default Input Device button pressed");
    };
    addAndMakeVisible(useDefaultInputButton.get());
    
    // Enable Sine Test button
    enableSineTestButton = std::make_unique<juce::TextButton>("Enable Sine Test");
    enableSineTestButton->setButtonText("Enable Sine Test");
    enableSineTestButton->setColour(juce::TextButton::buttonColourId, juce::Colours::darkgreen);
    enableSineTestButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    enableSineTestButton->setClickingTogglesState(true);
    enableSineTestButton->onClick = [this] {
        bool enabled = enableSineTestButton->getToggleState();
        enableCoreAudioSineTest(enabled);
        enableSineTestButton->setButtonText(enabled ? "Disable Sine Test" : "Enable Sine Test");
        juce::Logger::writeToLog("[DEBUG] Sine Test " + juce::String(enabled ? "ENABLED" : "DISABLED"));
    };
    addAndMakeVisible(enableSineTestButton.get());
    
    // Check MetalBridge Status button
    checkMetalBridgeButton = std::make_unique<juce::TextButton>("Check MetalBridge");
    checkMetalBridgeButton->setButtonText("Check MetalBridge");
    checkMetalBridgeButton->setColour(juce::TextButton::buttonColourId, juce::Colours::purple);
    checkMetalBridgeButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    checkMetalBridgeButton->onClick = [this] {
        checkMetalBridgeStatus();
        juce::Logger::writeToLog("[DEBUG] MetalBridge Status check requested");
    };
    addAndMakeVisible(checkMetalBridgeButton.get());
    
    // Force Callback button
    forceCallbackButton = std::make_unique<juce::TextButton>("Force Callback");
    forceCallbackButton->setButtonText("Force Callback");
    forceCallbackButton->setColour(juce::TextButton::buttonColourId, juce::Colours::darkorange);
    forceCallbackButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    forceCallbackButton->onClick = [this] {
        forceCoreAudioCallback();
        juce::Logger::writeToLog("[DEBUG] Force Callback button pressed");
    };
    addAndMakeVisible(forceCallbackButton.get());
}
