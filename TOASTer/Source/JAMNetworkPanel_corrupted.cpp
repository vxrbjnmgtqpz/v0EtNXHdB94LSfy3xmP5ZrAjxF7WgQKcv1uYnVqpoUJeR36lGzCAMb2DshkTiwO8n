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
            jamFramework->setBurstConfig(3, 500, true); // 3 packets, 500μs jitter, redundancy on
        } else {
            jamFramework->setBurstConfig(1, 0, false); // Single packet mode
        }
    };
    addAndMakeVisible(burstTransmissionToggle);
    
    gpuAccelerationToggle.setToggleState(false, juce::dontSendNotification);
    gpuAccelerationToggle.onStateChange = [this] {
        if (gpuAccelerationToggle.getToggleState() && !gpuInitialized) {
            gpuInitButtonClicked();
        }
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
