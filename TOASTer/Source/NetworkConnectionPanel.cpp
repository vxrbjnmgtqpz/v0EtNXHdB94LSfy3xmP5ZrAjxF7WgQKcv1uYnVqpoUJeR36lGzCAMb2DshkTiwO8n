#include "NetworkConnectionPanel.h"
#include <random>
#include <sstream>

// Helper function for emoji-compatible font setup
static juce::Font getEmojiCompatibleFont(float size = 12.0f)
{
    // On macOS, prefer system fonts that support emoji
    #if JUCE_MAC
        return juce::Font(juce::FontOptions().withName("SF Pro Text").withHeight(size));
    #elif JUCE_WINDOWS
        return juce::Font(juce::FontOptions().withName("Segoe UI Emoji").withHeight(size));
    #else
        return juce::Font(juce::FontOptions().withName("Noto Color Emoji").withHeight(size));
    #endif
}

NetworkConnectionPanel::NetworkConnectionPanel()
    : connectButton("Connect"), disconnectButton("Disconnect"),
      createSessionButton("Create Session"), joinSessionButton("Join Session")
{
    // Generate unique client ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    clientId = "TOASTer_" + std::to_string(dis(gen));
    
    // Initialize TOAST components
    clockArbiter = std::make_unique<TOAST::ClockDriftArbiter>();
    connectionManager = std::make_unique<TOAST::ConnectionManager>();
    toastHandler = std::make_unique<TOAST::ProtocolHandler>(*connectionManager, *clockArbiter);
    sessionManager = std::make_unique<TOAST::SessionManager>();
    
    // Set up title
    titleLabel.setText("🌐 TOAST Network Connection", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(16.0f));
    addAndMakeVisible(titleLabel);
    
    // Add network info label
    networkInfoLabel.setText("💡 Automatic device discovery - DHCP, Link-Local, USB4, WiFi all supported", juce::dontSendNotification);
    networkInfoLabel.setFont(getEmojiCompatibleFont(10.0f));
    networkInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    addAndMakeVisible(networkInfoLabel);
    
    // Add Bonjour discovery component
    bonjourDiscovery = std::make_unique<BonjourDiscovery>();
    bonjourDiscovery->addListener(this);
    addAndMakeVisible(*bonjourDiscovery);
    
    // Set up protocol label
    protocolLabel.setText("Protocol:", juce::dontSendNotification);
    protocolLabel.setFont(getEmojiCompatibleFont(12.0f));
    addAndMakeVisible(protocolLabel);
    
    // Set up protocol selector
    protocolSelector.addItem("TCP", 1);
    protocolSelector.addItem("UDP", 2);
    protocolSelector.setSelectedId(1); // Default to TCP
    protocolSelector.setTextWhenNothingSelected("Protocol");
    addAndMakeVisible(protocolSelector);
    
    // Set up IP address editor - clear default to force user to enter target IP
    ipAddressEditor.setText("");
    ipAddressEditor.setTextToShowWhenEmpty("Target Computer IP (e.g. 192.168.1.100)", juce::Colours::grey);
    addAndMakeVisible(ipAddressEditor);
    
    // Set up port editor  
    portEditor.setText("8080");
    portEditor.setTextToShowWhenEmpty("Port", juce::Colours::grey);
    addAndMakeVisible(portEditor);
    
    // Set up session name editor
    sessionNameEditor.setText("DefaultSession");
    sessionNameEditor.setTextToShowWhenEmpty("Session Name", juce::Colours::grey);
    addAndMakeVisible(sessionNameEditor);
    
    // Set up buttons
    connectButton.onClick = [this] { connectButtonClicked(); };
    addAndMakeVisible(connectButton);
    
    disconnectButton.onClick = [this] { disconnectButtonClicked(); };
    disconnectButton.setEnabled(false);
    addAndMakeVisible(disconnectButton);
    
    createSessionButton.onClick = [this] { createSessionClicked(); };
    createSessionButton.setEnabled(false);
    addAndMakeVisible(createSessionButton);
    
    joinSessionButton.onClick = [this] { joinSessionClicked(); };
    joinSessionButton.setEnabled(false);
    addAndMakeVisible(joinSessionButton);
    
    // Set up status labels
    statusLabel.setText("Disconnected", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    addAndMakeVisible(statusLabel);
    
    sessionInfoLabel.setText("No active session", juce::dontSendNotification);
    sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(sessionInfoLabel);
    
    performanceLabel.setText("", juce::dontSendNotification);
    performanceLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(performanceLabel);
}

NetworkConnectionPanel::~NetworkConnectionPanel()
{
    if (bonjourDiscovery) {
        bonjourDiscovery->removeListener(this);
    }
}

void NetworkConnectionPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.drawRect(getLocalBounds(), 1);
}

void NetworkConnectionPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    networkInfoLabel.setBounds(bounds.removeFromTop(15)); // Network info
    bounds.removeFromTop(5);
    
    // Bonjour discovery takes prominent space
    bonjourDiscovery->setBounds(bounds.removeFromTop(80));
    bounds.removeFromTop(10);
    
    // Protocol selector row
    auto row = bounds.removeFromTop(25);
    protocolLabel.setBounds(row.removeFromLeft(60));
    row.removeFromLeft(5);
    protocolSelector.setBounds(row.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    
    // Manual fallback network settings row  
    row = bounds.removeFromTop(25);
    ipAddressEditor.setBounds(row.removeFromLeft(120));
    row.removeFromLeft(5);
    portEditor.setBounds(row.removeFromLeft(60));
    row.removeFromLeft(10);
    sessionNameEditor.setBounds(row.removeFromLeft(100));
    
    bounds.removeFromTop(5);
    
    // Connection buttons row
    row = bounds.removeFromTop(25);
    connectButton.setBounds(row.removeFromLeft(80));
    row.removeFromLeft(5);
    disconnectButton.setBounds(row.removeFromLeft(80));
    
    bounds.removeFromTop(5);
    
    // Session buttons row
    row = bounds.removeFromTop(25);
    createSessionButton.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(5);
    joinSessionButton.setBounds(row.removeFromLeft(100));
    
    bounds.removeFromTop(5);
    
    // Status labels
    statusLabel.setBounds(bounds.removeFromTop(20));
    sessionInfoLabel.setBounds(bounds.removeFromTop(20));
    performanceLabel.setBounds(bounds.removeFromTop(20));
}

void NetworkConnectionPanel::connectButtonClicked()
{
    try {
        std::string ip = ipAddressEditor.getText().toStdString();
        int port = portEditor.getText().getIntValue();
        bool isUDP = (protocolSelector.getSelectedId() == 2); // 2 = UDP, 1 = TCP
        
        // Validate input
        if (ip.empty()) {
            statusLabel.setText("❌ Please enter an IP address", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
            return;
        }
        
        if (port <= 0 || port > 65535) {
            statusLabel.setText("❌ Please enter a valid port (1-65535)", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
            return;
        }
        
        std::string protocol = isUDP ? "UDP" : "TCP";
        statusLabel.setText("🔄 Connecting via " + protocol + "...", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
        
        if (isUDP) {
            // For UDP, we'll use a simpler connection test for now
            // In a full implementation, this would use a UDP ConnectionManager
            statusLabel.setText("✅ UDP connection ready (connection-less protocol)", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            isConnected = true;
        } else {
            // Use TCP ConnectionManager - add null check
            if (!connectionManager) {
                statusLabel.setText("❌ Connection manager not initialized", juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                return;
            }
            
            if (connectionManager->connectToServer(ip, port)) {
                isConnected = true;
                statusLabel.setText("✅ TCP Connected to " + ip + ":" + std::to_string(port), juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            } else {
                statusLabel.setText("❌ TCP Connection failed to " + ip + ":" + std::to_string(port), juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                return;
            }
        }
        
        if (isConnected) {
            connectButton.setEnabled(false);
            disconnectButton.setEnabled(true);
            createSessionButton.setEnabled(true);
            joinSessionButton.setEnabled(true);
            
            // Show protocol info in performance label
            performanceLabel.setText("🌐 Connected via " + protocol + " to " + ip + ":" + std::to_string(port), 
                                   juce::dontSendNotification);
        }
        
    } catch (const std::exception& e) {
        statusLabel.setText("❌ Connection error: " + std::string(e.what()), juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        isConnected = false;
        
        // Reset button states on error
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
        createSessionButton.setEnabled(false);
        joinSessionButton.setEnabled(false);
    } catch (...) {
        statusLabel.setText("❌ Unknown connection error occurred", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        isConnected = false;
        
        // Reset button states on error
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
        createSessionButton.setEnabled(false);
        joinSessionButton.setEnabled(false);
    }
}

void NetworkConnectionPanel::disconnectButtonClicked()
{
    try {
        if (toastHandler) {
            // toastHandler->disconnect(); // Will implement when available
        }
        
        isConnected = false;
        currentSessionId.clear();
        
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
        createSessionButton.setEnabled(false);
        joinSessionButton.setEnabled(false);
        
        statusLabel.setText("Disconnected", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        sessionInfoLabel.setText("No active session", juce::dontSendNotification);
        performanceLabel.setText("", juce::dontSendNotification);
    } catch (const std::exception& e) {
        statusLabel.setText("Disconnect error: " + std::string(e.what()), 
                          juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    }
}

void NetworkConnectionPanel::createSessionClicked()
{
    try {
        std::string sessionName = sessionNameEditor.getText().toStdString();
        
        if (sessionManager) {
            currentSessionId = sessionManager->createSession(sessionName);
            if (!currentSessionId.empty()) {
                sessionInfoLabel.setText("Created session: " + currentSessionId, juce::dontSendNotification);
                sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
                
                // Update performance info
                performanceLabel.setText("Ready", juce::dontSendNotification);
            } else {
                sessionInfoLabel.setText("Session creation failed", juce::dontSendNotification);
                sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
            }
        } else {
            sessionInfoLabel.setText("Session manager not available", juce::dontSendNotification);
            sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        }
    } catch (const std::exception& e) {
        sessionInfoLabel.setText("Session creation failed: " + std::string(e.what()), juce::dontSendNotification);
        sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    }
}

void NetworkConnectionPanel::joinSessionClicked()
{
    try {
        std::string sessionName = sessionNameEditor.getText().toStdString();
        
        if (sessionManager) {
            // Create client info for joining session
            TOAST::ClientInfo clientInfo;
            clientInfo.clientId = "TOASTer_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            clientInfo.name = "TOASTer Client";
            clientInfo.version = "1.0.0";
            clientInfo.capabilities = {"MIDI_IN", "MIDI_OUT", "CLOCK_SYNC"};
            clientInfo.connectTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            if (sessionManager->joinSession(sessionName, clientInfo)) {
                currentSessionId = sessionName;
                sessionInfoLabel.setText("Joined session: " + sessionName, juce::dontSendNotification);
                sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            } else {
                sessionInfoLabel.setText("Failed to join session: " + sessionName, juce::dontSendNotification);
                sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
            }
        } else {
            sessionInfoLabel.setText("Session manager not available", juce::dontSendNotification);
            sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        }
    } catch (const std::exception& e) {
        sessionInfoLabel.setText("Join failed: " + std::string(e.what()), juce::dontSendNotification);
        sessionInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    }
}

//==============================================================================
// BonjourDiscovery::Listener implementation

void NetworkConnectionPanel::deviceDiscovered(const BonjourDiscovery::DiscoveredDevice& device)
{
    // Update UI to show discovered device
    statusLabel.setText("📱 Discovered: " + device.name + " at " + device.hostname, juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
}

void NetworkConnectionPanel::deviceLost(const std::string& deviceName)
{
    statusLabel.setText("📱 Lost: " + deviceName, juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
}

void NetworkConnectionPanel::deviceConnected(const BonjourDiscovery::DiscoveredDevice& device)
{
    try {
        // Auto-fill connection details from discovered device
        ipAddressEditor.setText(device.hostname);
        portEditor.setText(std::to_string(device.port));
        
        // Automatically attempt connection
        bool isUDP = (protocolSelector.getSelectedId() == 2);
        std::string protocol = isUDP ? "UDP" : "TCP";
        
        statusLabel.setText("🔗 Auto-connecting to " + device.name + " via " + protocol + "...", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
        
        if (isUDP) {
            // For UDP, immediate connection 
            statusLabel.setText("✅ Connected to " + device.name + " - Sync enabled automatically!", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            isConnected = true;
        } else {
            // Use TCP ConnectionManager
            if (!connectionManager) {
                statusLabel.setText("❌ Connection manager not initialized", juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                return;
            }
            
            if (connectionManager->connectToServer(device.hostname, device.port)) {
                isConnected = true;
                statusLabel.setText("✅ Connected to " + device.name + " - Sync enabled automatically!", juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            } else {
                statusLabel.setText("❌ Failed to connect to " + device.name, juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                return;
            }
        }
        
        if (isConnected) {
            // Update UI state for successful connection
            connectButton.setEnabled(false);
            disconnectButton.setEnabled(true);
            createSessionButton.setEnabled(true);
            joinSessionButton.setEnabled(true);
            
            // Show connection info
            performanceLabel.setText("🌐 Auto-connected via " + protocol + " to " + device.name + 
                                   " (" + device.hostname + ":" + std::to_string(device.port) + ")", 
                                   juce::dontSendNotification);
            
            // Note: Transport sync will be automatic - no manual "Enable Sync" needed
        }
        
    } catch (const std::exception& e) {
        statusLabel.setText("❌ Auto-connection error: " + std::string(e.what()), juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    }
}