#include "MainComponent.h"
#include "GPUTransportController.h"  // GPU-native transport controller
#include "JAMNetworkPanel.h"         // Using JAM Framework v2 panel
#include "WiFiNetworkDiscovery.h"    // WiFi peer discovery
#include "MIDITestingPanel.h"
#include "PerformanceMonitorPanel.h"
#include "ClockSyncPanel.h"
#include "JMIDIntegrationPanel.h"
#include "GPUMIDIManager.h"          // GPU-native MIDI manager

//==============================================================================
MainComponent::MainComponent()
{
    // Initialize GPU-native infrastructure first
    if (!jam::gpu_native::GPUTimebase::initialize()) {
        juce::AlertWindow::showMessageBoxAsync(
            juce::MessageBoxIconType::WarningIcon,
            "GPU Initialization Warning",
            "GPU timebase initialization failed. Falling back to CPU timing.\n"
            "For best performance, ensure Metal/Vulkan drivers are installed."
        );
    }
    
    // Initialize shared timeline manager (static)
    if (!jam::gpu_native::GPUSharedTimelineManager::initialize()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "GPU Error", "Failed to initialize GPU shared timeline!");
        return;
    }
    
    // Initialize GPU-native multimedia frameworks
    jmidFramework = std::make_unique<jam::jmid_gpu::GPUJMIDFramework>();
    if (!jmidFramework->initialize()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "GPU Error", "Failed to initialize JMID framework!");
        return;
    }
    
    // Note: JDAT and JVID frameworks need configuration objects, create with defaults
    jam::jdat::GPUJDATFramework::GPUAudioConfig audioConfig;
    jdatFramework = std::make_unique<jam::jdat::GPUJDATFramework>(audioConfig);
    if (!jdatFramework->initialize()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "GPU Error", "Failed to initialize JDAT framework!");
        return;
    }
    
    jam::jvid::GPUJVIDFramework::GPUVideoConfig videoConfig;
    jvidFramework = std::make_unique<jam::jvid::GPUJVIDFramework>(videoConfig);
    if (!jvidFramework->initialize()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "GPU Error", "Failed to initialize JVID framework!");
        return;
    }
    
    // Initialize GPU-native MIDI I/O system first
    midiManager = std::make_unique<GPUMIDIManager>();
    midiManager->initialize(jmidFramework.get());
    
    // Create and add child components with GPU-native backends
    transportController = std::make_unique<GPUTransportController>();
    addAndMakeVisible(transportController.get());
    
    jamNetworkPanel = std::make_unique<JAMNetworkPanel>();  // Using JAM Framework v2 panel
    addAndMakeVisible(jamNetworkPanel.get());
    
    // Add WiFi network discovery panel
    wifiDiscovery = std::make_unique<WiFiNetworkDiscovery>();
    addAndMakeVisible(wifiDiscovery.get());
    
    midiPanel = std::make_unique<MIDITestingPanel>();
    midiPanel->setGPUMIDIManager(midiManager.get()); // Connect GPU MIDI manager
    addAndMakeVisible(midiPanel.get());
    
    performancePanel = std::make_unique<PerformanceMonitorPanel>();
    addAndMakeVisible(performancePanel.get());
    
    clockSyncPanel = std::make_unique<ClockSyncPanel>();
    addAndMakeVisible(clockSyncPanel.get());
    
    jmidPanel = std::make_unique<JMIDIntegrationPanel>();
    addAndMakeVisible(jmidPanel.get());
    
    // Connect GPU TransportController to Network Panel for automatic sync (bidirectional)
    transportController->setNetworkPanel(jamNetworkPanel.get());
    jamNetworkPanel->setTransportController(transportController.get());
    
    // Connect JAMNetworkPanel to notify MainComponent of network changes
    jamNetworkPanel->setNetworkStatusCallback([this](bool connected, int peers, const std::string& address, int port) {
        updateNetworkState(connected, peers, address, port);
    });
    
    // Connect ClockSyncPanel to get network status updates
    // Note: Will need to add callback in JAMNetworkPanel to notify ClockSyncPanel of connection changes
    
    // Start timer synchronized with GPU timebase
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        // Update at ~60 FPS synchronized with GPU timeline
        startTimer(16); // ~60 FPS for UI updates
    } else {
        // Fallback to slower updates if GPU is not available
        startTimer(250); // Update 4 times per second
    }
    
    // Set the main window size (increased for new panel)
    setSize(1200, 800);
}

MainComponent::~MainComponent()
{
    stopTimer();
}

void MainComponent::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::darkgrey);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Transport bar at the top (fixed height)
    transportController.get()->setBounds(bounds.removeFromTop(50));
    
    // Divide remaining space into panels (now 2x3 grid to accommodate WiFi discovery)
    auto remainingBounds = bounds.reduced(10);
    auto panelHeight = remainingBounds.getHeight() / 2;
    auto panelWidth = remainingBounds.getWidth() / 3;
    
    // Top row - three panels (Network-related panels together)
    auto topRow = remainingBounds.removeFromTop(panelHeight);
    jamNetworkPanel.get()->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));  // JAM Framework v2 panel
    wifiDiscovery.get()->setBounds(topRow.removeFromLeft(panelWidth).reduced(5));   // WiFi discovery panel
    midiPanel.get()->setBounds(topRow.reduced(5));
    
    // Bottom row - three panels  
    auto bottomRow = remainingBounds;
    jmidPanel.get()->setBounds(bottomRow.removeFromLeft(panelWidth).reduced(5)); // JMID panel
    performancePanel.get()->setBounds(bottomRow.removeFromLeft(panelWidth).reduced(5));
    clockSyncPanel.get()->setBounds(bottomRow.reduced(5));
}

void MainComponent::timerCallback()
{
    // Update GPU timeline timestamp (not CPU clock)
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
        
        // Check for GPU performance metrics
        updateGPUPerformance();
    }
    
    // Push current GPU-synchronized state to all panels
    performancePanel.get()->setConnectionState(gpuAppState.isNetworkConnected, gpuAppState.activeConnections);
    performancePanel.get()->setNetworkLatency(gpuAppState.networkLatency);
    performancePanel.get()->setClockAccuracy(gpuAppState.clockAccuracy);
    performancePanel->setMessageProcessingRate(gpuAppState.messageProcessingRate);
    performancePanel->setMIDIThroughput(gpuAppState.midiThroughput);
    
    // Update ClockSyncPanel with current network status
    clockSyncPanel->setNetworkConnected(gpuAppState.isNetworkConnected, gpuAppState.activeConnections);
}

void MainComponent::updateGPUPerformance()
{
    if (!jam::gpu_native::GPUTimebase::is_initialized()) return;
    
    // Update GPU performance metrics
    auto currentFrame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    auto deltaFrames = currentFrame - lastGPUFrame;
    lastGPUFrame = currentFrame;
    
    // Calculate processing rates based on GPU timeline
    if (jmidFramework) {
        // Note: Implementation pending - need to add metrics to GPUJMIDFramework
        // gpuAppState.midiThroughput = jmidFramework->getProcessedEventsPerSecond();
    }
    
    // Get processing rate from shared timeline manager (static)
    if (jam::gpu_native::GPUSharedTimelineManager::isInitialized()) {
        // Note: Implementation pending - need to add metrics to GPUSharedTimelineManager
        // gpuAppState.messageProcessingRate = jam::gpu_native::GPUSharedTimelineManager::getEventsProcessedPerSecond();
    }
}

void MainComponent::updateNetworkState(bool connected, int connections, const std::string& ip, int port)
{
    gpuAppState.isNetworkConnected = connected;
    gpuAppState.activeConnections = connections;
    gpuAppState.connectedIP = ip;
    gpuAppState.connectedPort = port;
    
    // Update GPU timestamp
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
    
    // Update JAM Network Panel performance metrics
    if (jamNetworkPanel) {
        // Note: getCurrentLatency method needs to be implemented in JAMNetworkPanel
        // gpuAppState.networkLatency = jamNetworkPanel->getCurrentLatency();
    }
}

void MainComponent::updateNetworkLatency(double latencyMs)
{
    gpuAppState.networkLatency = latencyMs;
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
}

void MainComponent::updateClockSync(bool enabled, double accuracy, double offset, uint64_t rtt)
{
    gpuAppState.isClockSyncEnabled = enabled;
    gpuAppState.clockAccuracy = accuracy;
    gpuAppState.clockOffset = offset;
    gpuAppState.roundTripTime = rtt;
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
}

void MainComponent::updatePerformanceMetrics(int msgRate, int midiRate)
{
    gpuAppState.messageProcessingRate = msgRate;
    gpuAppState.midiThroughput = midiRate;
    if (jam::gpu_native::GPUTimebase::is_initialized()) {
        gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
    }
}

void MainComponent::sendMIDIEventViaJAM(uint8_t status, uint8_t data1, uint8_t data2)
{
    if (jamNetworkPanel && jamNetworkPanel->isConnected()) {
        jamNetworkPanel->sendMIDIEvent(status, data1, data2);
        
        // Update performance metrics
        gpuAppState.midiThroughput++;
        if (jam::gpu_native::GPUTimebase::is_initialized()) {
            gpuAppState.lastGPUTimestamp = jam::gpu_native::GPUTimebase::get_current_time_ns();
        }
    }
}

bool MainComponent::isJAMFrameworkConnected() const
{
    return jamNetworkPanel && jamNetworkPanel->isConnected();
}