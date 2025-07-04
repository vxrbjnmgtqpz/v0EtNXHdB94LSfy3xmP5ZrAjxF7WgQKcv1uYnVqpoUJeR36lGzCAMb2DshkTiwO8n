#include "JAMNetworkPanel.h"

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
      gpuInitButton("⚡ Init GPU"),
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
    
    // Set up title
    titleLabel.setText("🌐 JAM Framework v2 Network (UDP Multicast)", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(16.0f));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    addAndMakeVisible(titleLabel);
    
    // Status label
    statusLabel.setText("💡 Ready to connect - UDP multicast with GPU acceleration", juce::dontSendNotification);
    statusLabel.setFont(getEmojiCompatibleFont(12.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(statusLabel);
    
    // Configuration section
    sessionLabel.setText("Session:", juce::dontSendNotification);
    sessionLabel.setFont(getEmojiCompatibleFont(12.0f));
    addAndMakeVisible(sessionLabel);
    
    sessionNameEditor.setText(currentSessionName);
    sessionNameEditor.setTextToShowWhenEmpty("Session Name", juce::Colours::grey);
    sessionNameEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(sessionNameEditor);
    
    multicastLabel.setText("Multicast:", juce::dontSendNotification);
    multicastLabel.setFont(getEmojiCompatibleFont(12.0f));
    addAndMakeVisible(multicastLabel);
    
    multicastAddressEditor.setText(currentMulticastAddr);
    multicastAddressEditor.setTextToShowWhenEmpty("239.255.77.77", juce::Colours::grey);
    multicastAddressEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(multicastAddressEditor);
    
    portLabel.setText("Port:", juce::dontSendNotification);
    portLabel.setFont(getEmojiCompatibleFont(12.0f));
    addAndMakeVisible(portLabel);
    
    portEditor.setText(juce::String(currentUDPPort));
    portEditor.setTextToShowWhenEmpty("7777", juce::Colours::grey);
    portEditor.onTextChange = [this] { sessionConfigurationChanged(); };
    addAndMakeVisible(portEditor);
    
    // Connection buttons
    connectButton.onClick = [this] { connectButtonClicked(); };
    addAndMakeVisible(connectButton);
    
    disconnectButton.onClick = [this] { disconnectButtonClicked(); };
    disconnectButton.setEnabled(false);
    addAndMakeVisible(disconnectButton);
    
    gpuInitButton.onClick = [this] { gpuInitButtonClicked(); };
    addAndMakeVisible(gpuInitButton);
    
    // Feature toggles
    pnbtrAudioToggle.setToggleState(true, juce::dontSendNotification);
    pnbtrAudioToggle.onStateChange = [this] {
        jamFramework->setPNBTRAudioPrediction(pnbtrAudioToggle.getToggleState());
    };
    addAndMakeVisible(pnbtrAudioToggle);
    
    pnbtrVideoToggle.setToggleState(true, juce::dontSendNotification);
    pnbtrVideoToggle.onStateChange = [this] {
        jamFramework->setPNBTRVideoPrediction(pnbtrVideoToggle.getToggleState());
    };
    addAndMakeVisible(pnbtrVideoToggle);
    
    burstTransmissionToggle.setToggleState(true, juce::dontSendNotification);
    burstTransmissionToggle.onStateChange = [this] {
        if (burstTransmissionToggle.getToggleState()) {
            jamFramework->setBurstConfig(3, 500, true); // 3 packets, 500μs jitter, redundancy on\n        } else {\n            jamFramework->setBurstConfig(1, 0, false); // Single packet mode\n        }\n    };\n    addAndMakeVisible(burstTransmissionToggle);\n    \n    gpuAccelerationToggle.setToggleState(false, juce::dontSendNotification);\n    gpuAccelerationToggle.onStateChange = [this] {\n        if (gpuAccelerationToggle.getToggleState() && !gpuInitialized) {\n            gpuInitButtonClicked();\n        }\n    };\n    addAndMakeVisible(gpuAccelerationToggle);\n    \n    // Performance metrics display\n    latencyLabel.setText(\"Latency: -- μs\", juce::dontSendNotification);\n    latencyLabel.setFont(getEmojiCompatibleFont(10.0f));\n    addAndMakeVisible(latencyLabel);\n    \n    throughputLabel.setText(\"Throughput: -- Mbps\", juce::dontSendNotification);\n    throughputLabel.setFont(getEmojiCompatibleFont(10.0f));\n    addAndMakeVisible(throughputLabel);\n    \n    peersLabel.setText(\"Peers: 0\", juce::dontSendNotification);\n    peersLabel.setFont(getEmojiCompatibleFont(10.0f));\n    addAndMakeVisible(peersLabel);\n    \n    predictionLabel.setText(\"Prediction: --\", juce::dontSendNotification);\n    predictionLabel.setFont(getEmojiCompatibleFont(10.0f));\n    addAndMakeVisible(predictionLabel);\n    \n    // PNBTR status\n    pnbtrStatusLabel.setText(\"🧠 PNBTR Predictive Neural Buffered Transient Recovery ready\", juce::dontSendNotification);\n    pnbtrStatusLabel.setFont(getEmojiCompatibleFont(10.0f));\n    pnbtrStatusLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);\n    addAndMakeVisible(pnbtrStatusLabel);\n    \n    // Bonjour discovery for fallback\n    bonjourDiscovery = std::make_unique<BonjourDiscovery>();\n    bonjourDiscovery->addListener(this);\n    addAndMakeVisible(*bonjourDiscovery);\n    \n    // Start 250ms update timer\n    startTimer(250);\n}\n\nJAMNetworkPanel::~JAMNetworkPanel() {\n    if (bonjourDiscovery) {\n        bonjourDiscovery->removeListener(this);\n    }\n    \n    if (jamFramework && networkConnected) {\n        jamFramework->stopNetwork();\n    }\n}\n\nvoid JAMNetworkPanel::paint(juce::Graphics& g) {\n    g.fillAll(juce::Colours::black);\n    g.setColour(juce::Colours::cyan);\n    g.drawRect(getLocalBounds(), 2);\n    \n    // Draw JAM Framework v2 branding\n    g.setColour(juce::Colours::cyan.withAlpha(0.3f));\n    g.fillRect(getLocalBounds().reduced(5));\n}\n\nvoid JAMNetworkPanel::resized() {\n    auto bounds = getLocalBounds().reduced(10);\n    \n    // Title\n    titleLabel.setBounds(bounds.removeFromTop(25));\n    statusLabel.setBounds(bounds.removeFromTop(20));\n    bounds.removeFromTop(5);\n    \n    // Configuration row 1: Session name\n    auto row = bounds.removeFromTop(25);\n    sessionLabel.setBounds(row.removeFromLeft(60));\n    sessionNameEditor.setBounds(row.removeFromLeft(150));\n    \n    bounds.removeFromTop(3);\n    \n    // Configuration row 2: Multicast address and port\n    row = bounds.removeFromTop(25);\n    multicastLabel.setBounds(row.removeFromLeft(60));\n    multicastAddressEditor.setBounds(row.removeFromLeft(100));\n    row.removeFromLeft(10);\n    portLabel.setBounds(row.removeFromLeft(40));\n    portEditor.setBounds(row.removeFromLeft(60));\n    \n    bounds.removeFromTop(5);\n    \n    // Connection buttons\n    row = bounds.removeFromTop(30);\n    connectButton.setBounds(row.removeFromLeft(100));\n    row.removeFromLeft(5);\n    disconnectButton.setBounds(row.removeFromLeft(100));\n    row.removeFromLeft(5);\n    gpuInitButton.setBounds(row.removeFromLeft(80));\n    \n    bounds.removeFromTop(5);\n    \n    // Feature toggles row 1\n    row = bounds.removeFromTop(25);\n    pnbtrAudioToggle.setBounds(row.removeFromLeft(120));\n    row.removeFromLeft(10);\n    pnbtrVideoToggle.setBounds(row.removeFromLeft(120));\n    \n    bounds.removeFromTop(3);\n    \n    // Feature toggles row 2\n    row = bounds.removeFromTop(25);\n    burstTransmissionToggle.setBounds(row.removeFromLeft(140));\n    row.removeFromLeft(10);\n    gpuAccelerationToggle.setBounds(row.removeFromLeft(140));\n    \n    bounds.removeFromTop(5);\n    \n    // Performance metrics\n    row = bounds.removeFromTop(20);\n    latencyLabel.setBounds(row.removeFromLeft(100));\n    row.removeFromLeft(5);\n    throughputLabel.setBounds(row.removeFromLeft(120));\n    row.removeFromLeft(5);\n    peersLabel.setBounds(row.removeFromLeft(60));\n    row.removeFromLeft(5);\n    predictionLabel.setBounds(row);\n    \n    bounds.removeFromTop(3);\n    \n    // PNBTR status\n    pnbtrStatusLabel.setBounds(bounds.removeFromTop(15));\n    \n    bounds.removeFromTop(5);\n    \n    // Bonjour discovery (remaining space)\n    if (bounds.getHeight() > 50) {\n        bonjourDiscovery->setBounds(bounds.removeFromTop(60));\n    }\n}\n\nbool JAMNetworkPanel::isConnected() const {\n    return networkConnected;\n}\n\nint JAMNetworkPanel::getActivePeers() const {\n    return activePeers;\n}\n\ndouble JAMNetworkPanel::getCurrentLatency() const {\n    return currentLatency;\n}\n\ndouble JAMNetworkPanel::getCurrentThroughput() const {\n    return currentThroughput;\n}\n\nvoid JAMNetworkPanel::sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2) {\n    if (jamFramework && networkConnected) {\n        bool useBurst = burstTransmissionToggle.getToggleState();\n        jamFramework->sendMIDIEvent(status, data1, data2, useBurst);\n    }\n}\n\nvoid JAMNetworkPanel::sendMIDIData(const uint8_t* data, size_t size) {\n    if (jamFramework && networkConnected && data && size > 0) {\n        bool useBurst = burstTransmissionToggle.getToggleState();\n        jamFramework->sendMIDIData(data, size, useBurst);\n    }\n}\n\nvoid JAMNetworkPanel::setSessionName(const juce::String& sessionName) {\n    currentSessionName = sessionName;\n    sessionNameEditor.setText(sessionName);\n}\n\nvoid JAMNetworkPanel::setMulticastAddress(const juce::String& address) {\n    currentMulticastAddr = address;\n    multicastAddressEditor.setText(address);\n}\n\nvoid JAMNetworkPanel::setUDPPort(int port) {\n    currentUDPPort = port;\n    portEditor.setText(juce::String(port));\n}\n\nvoid JAMNetworkPanel::enablePNBTRPrediction(bool audio, bool video) {\n    pnbtrAudioToggle.setToggleState(audio, juce::sendNotification);\n    pnbtrVideoToggle.setToggleState(video, juce::sendNotification);\n}\n\nvoid JAMNetworkPanel::enableGPUAcceleration(bool enable) {\n    gpuAccelerationToggle.setToggleState(enable, juce::sendNotification);\n}\n\nvoid JAMNetworkPanel::connectButtonClicked() {\n    if (!jamFramework) {\n        statusLabel.setText(\"❌ JAM Framework not initialized\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n        return;\n    }\n    \n    // Update configuration from UI\n    currentSessionName = sessionNameEditor.getText();\n    currentMulticastAddr = multicastAddressEditor.getText();\n    currentUDPPort = portEditor.getText().getIntValue();\n    \n    if (currentSessionName.isEmpty()) {\n        currentSessionName = \"TOASTer_Session\";\n        sessionNameEditor.setText(currentSessionName);\n    }\n    \n    if (currentMulticastAddr.isEmpty()) {\n        currentMulticastAddr = \"239.255.77.77\";\n        multicastAddressEditor.setText(currentMulticastAddr);\n    }\n    \n    if (currentUDPPort <= 0 || currentUDPPort > 65535) {\n        currentUDPPort = 7777;\n        portEditor.setText(juce::String(currentUDPPort));\n    }\n    \n    statusLabel.setText(\"🔄 Initializing JAM Framework v2...\", juce::dontSendNotification);\n    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);\n    \n    // Initialize JAM Framework\n    bool initSuccess = jamFramework->initialize(\n        currentMulticastAddr.toStdString(),\n        currentUDPPort,\n        currentSessionName.toStdString()\n    );\n    \n    if (!initSuccess) {\n        statusLabel.setText(\"❌ Failed to initialize JAM Framework v2\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n        return;\n    }\n    \n    // Start network\n    statusLabel.setText(\"🔄 Starting UDP multicast network...\", juce::dontSendNotification);\n    \n    bool networkSuccess = jamFramework->startNetwork();\n    \n    if (networkSuccess) {\n        networkConnected = true;\n        connectButton.setEnabled(false);\n        disconnectButton.setEnabled(true);\n        \n        statusLabel.setText(\"✅ Connected via JAM Framework v2 UDP multicast\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);\n        \n        // Update PNBTR settings\n        jamFramework->setPNBTRAudioPrediction(pnbtrAudioToggle.getToggleState());\n        jamFramework->setPNBTRVideoPrediction(pnbtrVideoToggle.getToggleState());\n        \n        // Update burst settings\n        if (burstTransmissionToggle.getToggleState()) {\n            jamFramework->setBurstConfig(3, 500, true);\n        } else {\n            jamFramework->setBurstConfig(1, 0, false);\n        }\n        \n    } else {\n        statusLabel.setText(\"❌ Failed to start UDP multicast network\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n    }\n}\n\nvoid JAMNetworkPanel::disconnectButtonClicked() {\n    if (jamFramework && networkConnected) {\n        jamFramework->stopNetwork();\n        networkConnected = false;\n        activePeers = 0;\n        currentLatency = 0.0;\n        currentThroughput = 0.0;\n        \n        connectButton.setEnabled(true);\n        disconnectButton.setEnabled(false);\n        \n        statusLabel.setText(\"⏹️ Disconnected from JAM Framework v2\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);\n        \n        updatePerformanceDisplay();\n    }\n}\n\nvoid JAMNetworkPanel::gpuInitButtonClicked() {\n    if (!jamFramework) {\n        statusLabel.setText(\"❌ JAM Framework not initialized\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n        return;\n    }\n    \n    bool gpuSuccess = jamFramework->initializeGPU();\n    gpuInitialized = gpuSuccess;\n    \n    if (gpuSuccess) {\n        gpuInitButton.setEnabled(false);\n        gpuAccelerationToggle.setToggleState(true, juce::dontSendNotification);\n        gpuAccelerationToggle.setEnabled(false);\n        \n        pnbtrStatusLabel.setText(\"⚡ PNBTR GPU acceleration enabled (Metal backend)\", juce::dontSendNotification);\n        pnbtrStatusLabel.setColour(juce::Label::textColourId, juce::Colours::cyan);\n        \n    } else {\n        pnbtrStatusLabel.setText(\"❌ GPU acceleration unavailable\", juce::dontSendNotification);\n        pnbtrStatusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n    }\n}\n\nvoid JAMNetworkPanel::sessionConfigurationChanged() {\n    // Session configuration changed - will take effect on next connect\n    if (networkConnected) {\n        statusLabel.setText(\"⚠️ Disconnect and reconnect to apply changes\", juce::dontSendNotification);\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);\n    }\n}\n\nvoid JAMNetworkPanel::onJAMStatusChanged(const std::string& status, bool connected) {\n    networkConnected = connected;\n    \n    statusLabel.setText(juce::String(status), juce::dontSendNotification);\n    \n    if (connected) {\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::green);\n        connectButton.setEnabled(false);\n        disconnectButton.setEnabled(true);\n    } else {\n        statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n        connectButton.setEnabled(true);\n        disconnectButton.setEnabled(false);\n    }\n}\n\nvoid JAMNetworkPanel::onJAMPerformanceUpdate(double latency_us, double throughput_mbps, int active_peers) {\n    currentLatency = latency_us;\n    currentThroughput = throughput_mbps;\n    activePeers = active_peers;\n    \n    if (jamFramework) {\n        predictionAccuracy = jamFramework->getPredictionConfidence();\n    }\n    \n    updatePerformanceDisplay();\n}\n\nvoid JAMNetworkPanel::onJAMMIDIReceived(uint8_t status, uint8_t data1, uint8_t data2, uint32_t timestamp) {\n    // Handle incoming MIDI - this would typically be forwarded to the MIDI panel\n    juce::Logger::writeToLog(\"JAM MIDI Received: \" + \n                           juce::String::toHexString(status) + \" \" +\n                           juce::String::toHexString(data1) + \" \" + \n                           juce::String::toHexString(data2));\n}\n\nvoid JAMNetworkPanel::deviceFound(const std::string& name, const std::string& ip, int port) {\n    // Bonjour discovered a device - could auto-suggest connection\n    juce::Logger::writeToLog(\"Bonjour found device: \" + juce::String(name) + \" at \" + juce::String(ip));\n}\n\nvoid JAMNetworkPanel::deviceLost(const std::string& name) {\n    juce::Logger::writeToLog(\"Bonjour lost device: \" + juce::String(name));\n}\n\nvoid JAMNetworkPanel::timerCallback() {\n    updateUI();\n}\n\nvoid JAMNetworkPanel::updateUI() {\n    // Update any dynamic UI elements that need periodic refresh\n    if (jamFramework && networkConnected) {\n        // Get latest performance metrics\n        auto metrics = jamFramework->getPerformanceMetrics();\n        \n        currentLatency = metrics.latency_us;\n        currentThroughput = metrics.throughput_mbps;\n        activePeers = metrics.active_peers;\n        predictionAccuracy = metrics.prediction_accuracy;\n        \n        updatePerformanceDisplay();\n    }\n}\n\nvoid JAMNetworkPanel::updatePerformanceDisplay() {\n    latencyLabel.setText(\"Latency: \" + juce::String(currentLatency, 1) + \" μs\", juce::dontSendNotification);\n    throughputLabel.setText(\"Throughput: \" + juce::String(currentThroughput, 2) + \" Mbps\", juce::dontSendNotification);\n    peersLabel.setText(\"Peers: \" + juce::String(activePeers), juce::dontSendNotification);\n    \n    if (predictionAccuracy > 0.0) {\n        predictionLabel.setText(\"Prediction: \" + juce::String(predictionAccuracy * 100.0, 1) + \"%\", juce::dontSendNotification);\n    } else {\n        predictionLabel.setText(\"Prediction: --\", juce::dontSendNotification);\n    }\n    \n    // Color code latency\n    if (currentLatency < 100.0) {\n        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::green);\n    } else if (currentLatency < 1000.0) {\n        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);\n    } else {\n        latencyLabel.setColour(juce::Label::textColourId, juce::Colours::red);\n    }\n}
