#include "JAMNetworkPanel.h"
#include "GPUTransportController.h"

// Helper function for emoji-compatible font setup
juce::Font JAMNetworkPanel::getEmojiCompatibleFont(float size) {
    #if JUCE_MAC
        return juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
    #elif JUCE_WINDOWS
        return juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
    #else
        return juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
    #endif
}

JAMNetworkPanel::JAMNetworkPanel()
    : connectButton("🚀 Connect JAM v2"), 
      disconnectButton("⏹️ Disconnect"),
      pnbtrAudioToggle("🎵 PNBTR Audio"),
      pnbtrVideoToggle("🎬 PNBTR Video"),
      burstTransmissionToggle("📡 Burst Transmission"),
      gpuAccelerationToggle("💻 GPU Acceleration") {
    
    // Initialize JAM Framework v2
    jamFramework = std::make_unique<JAMFrameworkIntegration>();
    
    // Set up callbacks
    jamFramework->setStatusCallback([this](const std::string& status, bool connected) {
        onJAMStatusChanged(status, connected);
    });
    
    jamFramework->setPerformanceCallback([this](double latency_us, double throughput_mbps, int active_peers) {
        onJAMPerformanceUpdate(latency_us, throughput_mbps, active_peers);
    });
    
    jamFramework->setMIDICallback([this](uint8_t status, uint8_t data1, uint8_t data2, uint32_t timestamp) {
        onJAMMIDIReceived(status, data1, data2, timestamp);
    });
    
    jamFramework->setTransportCallback([this](const std::string& command, uint64_t timestamp, double position, double bpm) {
        onJAMTransportReceived(command, timestamp, position, bpm);
    });
    
    // Title label
    titleLabel.setText("🌟 JAM Framework v2 Network Panel", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(16.0f));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);
    
    // Status label
    statusLabel.setText("⏸️ Disconnected - Ready to initialize JAM Framework v2", juce::dontSendNotification);
    statusLabel.setFont(getEmojiCompatibleFont(12.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    addAndMakeVisible(statusLabel);
    
    // Session configuration
    sessionLabel.setText("Session:", juce::dontSendNotification);
    sessionLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(sessionLabel);
    
    sessionNameEditor.setText("TOASTer_Session");
    sessionNameEditor.setFont(getEmojiCompatibleFont(10.0f));
    sessionNameEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(sessionNameEditor);
    
    multicastLabel.setText("Multicast:", juce::dontSendNotification);
    multicastLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(multicastLabel);
    
    multicastAddressEditor.setText("239.255.77.77");
    multicastAddressEditor.setFont(getEmojiCompatibleFont(10.0f));
    multicastAddressEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(multicastAddressEditor);
    
    portLabel.setText("Port:", juce::dontSendNotification);
    portLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(portLabel);
    
    portEditor.setText("7777");
    portEditor.setFont(getEmojiCompatibleFont(10.0f));
    portEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(portEditor);
    
    // Connection buttons
    connectButton.onClick = [this] { connectButtonClicked(); };
    connectButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green.withAlpha(0.7f));
    addAndMakeVisible(connectButton);
    
    disconnectButton.onClick = [this] { disconnectButtonClicked(); };
    disconnectButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red.withAlpha(0.7f));
    disconnectButton.setEnabled(false);
    addAndMakeVisible(disconnectButton);
    
    // Feature toggles - NOW AUTOMATIC (always enabled for optimal performance)
    pnbtrAudioToggle.setToggleState(true, juce::dontSendNotification);
    pnbtrAudioToggle.setEnabled(false); // Always on - not user configurable
    pnbtrAudioToggle.setAlpha(0.6f); // Visual indicator that it's automatic
    pnbtrAudioToggle.onStateChange = [this] {
        // Always enabled for seamless audio continuity - no user control
        if (jamFramework) {
            jamFramework->setPNBTRAudioPrediction(true);
        }
    };
    addAndMakeVisible(pnbtrAudioToggle);
    
    pnbtrVideoToggle.setToggleState(true, juce::dontSendNotification);
    pnbtrVideoToggle.setEnabled(false); // Always on - not user configurable
    pnbtrVideoToggle.setAlpha(0.6f); // Visual indicator that it's automatic
    pnbtrVideoToggle.onStateChange = [this] {
        // Always enabled for seamless video continuity - no user control
        if (jamFramework) {
            jamFramework->setPNBTRVideoPrediction(true);
        }
    };
    addAndMakeVisible(pnbtrVideoToggle);
    
    burstTransmissionToggle.setToggleState(true, juce::dontSendNotification);
    burstTransmissionToggle.setEnabled(false); // Always on for redundancy
    burstTransmissionToggle.setAlpha(0.6f); // Visual indicator that it's automatic
    burstTransmissionToggle.onStateChange = [this] {
        // Always enabled for packet loss protection - no user control
        if (jamFramework) {
            jamFramework->setBurstConfig(3, 500, true); // Always use burst
        }
    };
    addAndMakeVisible(burstTransmissionToggle);
    
    gpuAccelerationToggle.setToggleState(true, juce::dontSendNotification);
    gpuAccelerationToggle.setEnabled(false); // Always on for performance
    gpuAccelerationToggle.setAlpha(0.6f); // Visual indicator that it's automatic
    gpuAccelerationToggle.onStateChange = [this] {
        // GPU is already initialized in GPU-native architecture - no action needed
    };
    addAndMakeVisible(gpuAccelerationToggle);
    
    // Performance metrics display
    latencyLabel.setText("Latency: -- μs", juce::dontSendNotification);
    latencyLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(latencyLabel);
    
    throughputLabel.setText("Throughput: -- Mbps", juce::dontSendNotification);
    throughputLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(throughputLabel);
    
    peersLabel.setText("Peers: 0", juce::dontSendNotification);
    peersLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(peersLabel);
    
    predictionLabel.setText("Prediction: --", juce::dontSendNotification);
    predictionLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(predictionLabel);
    
    // PNBTR status
    pnbtrStatusLabel.setText("🧠 PNBTR Predictive Neural Buffered Transient Recovery ready", juce::dontSendNotification);
    pnbtrStatusLabel.setFont(getEmojiCompatibleFont(10.0f));
    pnbtrStatusLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(pnbtrStatusLabel);
    
    // Bonjour discovery for fallback
    bonjourDiscovery = std::make_unique<BonjourDiscovery>();
    bonjourDiscovery->addListener(this);
    addAndMakeVisible(*bonjourDiscovery);
    
    // Start 250ms update timer
    startTimer(250);
}

JAMNetworkPanel::~JAMNetworkPanel() {
    if (bonjourDiscovery) {
        bonjourDiscovery->removeListener(this);
    }
    
    if (jamFramework && networkConnected) {
        jamFramework->stopNetwork();
    }
}

void JAMNetworkPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::cyan);
    g.drawRect(getLocalBounds(), 2);
    
    // Draw JAM Framework v2 branding
    g.setColour(juce::Colours::cyan.withAlpha(0.3f));
    g.fillRect(getLocalBounds().reduced(5));
}

void JAMNetworkPanel::resized() {
    auto bounds = getLocalBounds().reduced(10);
    
    // Title
    titleLabel.setBounds(bounds.removeFromTop(25));
    statusLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);
    
    // Configuration row 1: Session name
    auto row = bounds.removeFromTop(25);
    sessionLabel.setBounds(row.removeFromLeft(60));
    sessionNameEditor.setBounds(row.removeFromLeft(150));
    
    bounds.removeFromTop(3);
    
    // Configuration row 2: Multicast address and port
    row = bounds.removeFromTop(25);
    multicastLabel.setBounds(row.removeFromLeft(60));
    multicastAddressEditor.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(10);
    portLabel.setBounds(row.removeFromLeft(40));
    portEditor.setBounds(row.removeFromLeft(60));
    
    bounds.removeFromTop(5);
    
    // Connection buttons
    row = bounds.removeFromTop(30);
    connectButton.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(5);
    disconnectButton.setBounds(row.removeFromLeft(100));
    
    bounds.removeFromTop(5);
    
    // Feature toggles row 1
    row = bounds.removeFromTop(25);
    pnbtrAudioToggle.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(10);
    pnbtrVideoToggle.setBounds(row.removeFromLeft(120));
    
    bounds.removeFromTop(3);
    
    // Feature toggles row 2
    row = bounds.removeFromTop(25);
    burstTransmissionToggle.setBounds(row.removeFromLeft(140));
    row.removeFromLeft(10);
    gpuAccelerationToggle.setBounds(row.removeFromLeft(140));
    
    bounds.removeFromTop(5);
    
    // Performance metrics
    row = bounds.removeFromTop(20);
    latencyLabel.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(5);
    throughputLabel.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(5);
    peersLabel.setBounds(row.removeFromLeft(60));
    row.removeFromLeft(5);
    predictionLabel.setBounds(row);
    
    bounds.removeFromTop(3);
    
    // PNBTR status
    pnbtrStatusLabel.setBounds(bounds.removeFromTop(15));
    
    bounds.removeFromTop(5);
    
    // Bonjour discovery (remaining space)
    if (bounds.getHeight() > 50) {
        bonjourDiscovery->setBounds(bounds.removeFromTop(60));
    }
}

bool JAMNetworkPanel::isConnected() const {
    return networkConnected;
}

int JAMNetworkPanel::getActivePeers() const {
    return activePeers;
}

double JAMNetworkPanel::getCurrentLatency() const {
    return currentLatency;
}

double JAMNetworkPanel::getCurrentThroughput() const {
    return currentThroughput;
}

void JAMNetworkPanel::sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2) {
    if (jamFramework && networkConnected) {
        bool useBurst = burstTransmissionToggle.getToggleState();
        jamFramework->sendMIDIEvent(status, data1, data2, useBurst);
    }
}

void JAMNetworkPanel::sendMIDIData(const uint8_t* data, size_t size) {
    if (jamFramework && networkConnected && data && size > 0) {
        bool useBurst = burstTransmissionToggle.getToggleState();
        jamFramework->sendMIDIData(data, size, useBurst);
    }
}

void JAMNetworkPanel::setSessionName(const juce::String& sessionName) {
    currentSessionName = sessionName;
    sessionNameEditor.setText(sessionName);
}

void JAMNetworkPanel::setMulticastAddress(const juce::String& address) {
    currentMulticastAddr = address;
    multicastAddressEditor.setText(address);
}

void JAMNetworkPanel::setUDPPort(int port) {
    currentUDPPort = port;
    portEditor.setText(juce::String(port));
}

void JAMNetworkPanel::enablePNBTRPrediction(bool audio, bool video) {
    pnbtrAudioToggle.setToggleState(audio, juce::sendNotification);
    pnbtrVideoToggle.setToggleState(video, juce::sendNotification);
}

void JAMNetworkPanel::enableGPUAcceleration(bool enable) {
    gpuAccelerationToggle.setToggleState(enable, juce::sendNotification);
}

void JAMNetworkPanel::connectButtonClicked() {
    if (!jamFramework) {
        statusLabel.setText("❌ JAM Framework not initialized", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        return;
    }
    
    // Update configuration from UI
    currentSessionName = sessionNameEditor.getText();
    currentMulticastAddr = multicastAddressEditor.getText();
    currentUDPPort = portEditor.getText().getIntValue();
    
    if (currentSessionName.isEmpty()) {
        currentSessionName = "TOASTer_Session";
        sessionNameEditor.setText(currentSessionName);
    }
    
    if (currentMulticastAddr.isEmpty()) {
        currentMulticastAddr = "239.255.77.77";
        multicastAddressEditor.setText(currentMulticastAddr);
    }
    
    if (currentUDPPort <= 0 || currentUDPPort > 65535) {
        currentUDPPort = 7777;
        portEditor.setText(juce::String(currentUDPPort));
    }
    
    statusLabel.setText("🔄 Initializing JAM Framework v2...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    // Initialize JAM Framework
    bool initSuccess = jamFramework->initialize(
        currentMulticastAddr.toStdString(),
        currentUDPPort,
        currentSessionName.toStdString()
    );
    
    if (!initSuccess) {
        statusLabel.setText("❌ Failed to initialize JAM Framework v2", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        return;
    }
    
    // Start network
    statusLabel.setText("🔄 Starting UDP multicast network...", juce::dontSendNotification);
    
    bool networkSuccess = jamFramework->startNetwork();
    
    if (networkSuccess) {
        networkConnected = true;
        connectButton.setEnabled(false);
        disconnectButton.setEnabled(true);
        
        statusLabel.setText("✅ Connected via JAM Framework v2 UDP multicast", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        
        // GPU is already initialized in GPU-native architecture
        // All GPU acceleration is automatic and always enabled
        
        // AUTO-ENABLE all optimal settings (no user choice needed)
        jamFramework->setPNBTRAudioPrediction(true);    // Always on
        jamFramework->setPNBTRVideoPrediction(true);    // Always on
        jamFramework->setBurstConfig(3, 500, true);     // Always use burst redundancy
        
        juce::Logger::writeToLog("🚀 AUTO-CONFIGURED: PNBTR, GPU acceleration, and burst transmission enabled");
        
    } else {
        statusLabel.setText("❌ Failed to start UDP multicast network", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
}

void JAMNetworkPanel::disconnectButtonClicked() {
    if (jamFramework && networkConnected) {
        jamFramework->stopNetwork();
        networkConnected = false;
        activePeers = 0;
        currentLatency = 0.0;
        currentThroughput = 0.0;
        
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
        
        statusLabel.setText("⏹️ Disconnected from JAM Framework v2", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        
        updatePerformanceDisplay();
    }
}

void JAMNetworkPanel::sessionConfigurationChanged() {
    // Session configuration changed - will take effect on next connect
    if (networkConnected) {
        statusLabel.setText("⚠️ Disconnect and reconnect to apply changes", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
}

void JAMNetworkPanel::onJAMStatusChanged(const std::string& status, bool connected) {
    networkConnected = connected;
    
    statusLabel.setText(juce::String(status), juce::dontSendNotification);
    
    if (connected) {
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);
        connectButton.setEnabled(false);
        disconnectButton.setEnabled(true);
    } else {
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
    }
}

void JAMNetworkPanel::onJAMPerformanceUpdate(double latency_us, double throughput_mbps, int active_peers) {
    currentLatency = latency_us;
    currentThroughput = throughput_mbps;
    activePeers = active_peers;
    
    if (jamFramework) {
        predictionAccuracy = jamFramework->getPredictionConfidence();
    }
    
    updatePerformanceDisplay();
}

void JAMNetworkPanel::onJAMMIDIReceived(uint8_t status, uint8_t data1, uint8_t data2, uint32_t timestamp) {
    // Handle incoming MIDI - this would typically be forwarded to the MIDI panel
    juce::Logger::writeToLog("JAM MIDI Received: " + 
                           juce::String::toHexString(status) + " " +
                           juce::String::toHexString(data1) + " " + 
                           juce::String::toHexString(data2));
}

void JAMNetworkPanel::deviceDiscovered(const BonjourDiscovery::DiscoveredDevice& device) {
    // Bonjour discovered a device - could auto-suggest connection
    juce::Logger::writeToLog("Bonjour discovered device: " + juce::String(device.name) + " at " + juce::String(device.hostname));
}

void JAMNetworkPanel::deviceLost(const std::string& name) {
    juce::Logger::writeToLog("Bonjour lost device: " + juce::String(name));
}

void JAMNetworkPanel::deviceConnected(const BonjourDiscovery::DiscoveredDevice& device) {
    juce::Logger::writeToLog("Bonjour connected to device: " + juce::String(device.name));
}

void JAMNetworkPanel::timerCallback() {
    updateUI();
}

void JAMNetworkPanel::updateUI() {
    // Update any dynamic UI elements that need periodic refresh
    if (jamFramework && networkConnected) {
        // Get latest performance metrics
        auto metrics = jamFramework->getPerformanceMetrics();
        
        currentLatency = metrics.latency_us;
        currentThroughput = metrics.throughput_mbps;
        activePeers = metrics.active_peers;
        predictionAccuracy = metrics.prediction_accuracy;
        
        updatePerformanceDisplay();
    }
}

void JAMNetworkPanel::updatePerformanceDisplay() {
    latencyLabel.setText("Latency: " + juce::String(currentLatency, 1) + " μs", juce::dontSendNotification);
    throughputLabel.setText("Throughput: " + juce::String(currentThroughput, 2) + " Mbps", juce::dontSendNotification);
    peersLabel.setText("Peers: " + juce::String(activePeers), juce::dontSendNotification);
    
    if (predictionAccuracy > 0.0) {
        predictionLabel.setText("Prediction: " + juce::String(predictionAccuracy * 100.0, 1) + "%", juce::dontSendNotification);
    } else {
        predictionLabel.setText("Prediction: --", juce::dontSendNotification);
    }
    
    // Color code latency
    if (currentLatency < 100.0) {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::green);
    } else if (currentLatency < 1000.0) {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    } else {
        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
}

void JAMNetworkPanel::sendTransportCommand(const std::string& command, uint64_t timestamp, 
                                          double position, double bpm) {
    if (jamFramework && networkConnected) {
        jamFramework->sendTransportCommand(command, timestamp, position, bpm);
    }
}

void JAMNetworkPanel::onJAMTransportReceived(const std::string& command, uint64_t timestamp, 
                                             double position, double bpm) {
    // Forward transport command to the GPUTransportController for proper bidirectional sync
    juce::Logger::writeToLog("🎛️ JAM Transport Received: " + juce::String(command) + 
                           " pos=" + juce::String(position) + " bpm=" + juce::String(bpm));
    
    if (transportController) {
        transportController->handleRemoteTransportCommand(command, timestamp);
    }
}
