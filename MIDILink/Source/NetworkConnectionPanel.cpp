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
    clientId = "MIDILink_" + std::to_string(dis(gen));
    
    // Initialize TOAST components
    clockArbiter = std::make_unique<TOAST::ClockDriftArbiter>();
    connectionManager = std::make_unique<TOAST::ConnectionManager>();
    toastHandler = std::make_unique<TOAST::ProtocolHandler>(*connectionManager, *clockArbiter);
    sessionManager = std::make_unique<TOAST::SessionManager>();
    
    // Set up title
    titleLabel.setText("🌐 TOAST Network Connection", juce::dontSendNotification);
    titleLabel.setFont(getEmojiCompatibleFont(16.0f));
    addAndMakeVisible(titleLabel);
    
    // Set up IP address editor
    ipAddressEditor.setText("127.0.0.1");
    ipAddressEditor.setTextToShowWhenEmpty("IP Address", juce::Colours::grey);
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
    statusLabel.setText("🔴 Disconnected", juce::dontSendNotification);
    statusLabel.setFont(getEmojiCompatibleFont(12.0f));
    addAndMakeVisible(statusLabel);
    
    sessionInfoLabel.setText("No active session", juce::dontSendNotification);
    sessionInfoLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(sessionInfoLabel);
    
    performanceLabel.setText("", juce::dontSendNotification);
    performanceLabel.setFont(getEmojiCompatibleFont(10.0f));
    addAndMakeVisible(performanceLabel);
}

NetworkConnectionPanel::~NetworkConnectionPanel()
{
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
    bounds.removeFromTop(5);
    
    // Network settings row
    auto row = bounds.removeFromTop(25);
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
        
        // Initialize TOAST connection (simplified for now)
        if (toastHandler) {
            isConnected = true;
            connectButton.setEnabled(false);
            disconnectButton.setEnabled(true);
            createSessionButton.setEnabled(true);
            joinSessionButton.setEnabled(true);
            
            statusLabel.setText("🟡 Connected to " + ip + ":" + std::to_string(port), 
                              juce::dontSendNotification);
        } else {
            statusLabel.setText("🔴 Connection failed", juce::dontSendNotification);
        }
    } catch (const std::exception& e) {
        statusLabel.setText("🔴 Error: " + std::string(e.what()), juce::dontSendNotification);
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
        
        statusLabel.setText("🔴 Disconnected", juce::dontSendNotification);
        sessionInfoLabel.setText("No active session", juce::dontSendNotification);
        performanceLabel.setText("", juce::dontSendNotification);
    } catch (const std::exception& e) {
        statusLabel.setText("🔴 Disconnect error: " + std::string(e.what()), 
                          juce::dontSendNotification);
    }
}

void NetworkConnectionPanel::createSessionClicked()
{
    if (!isConnected) return;
    
    try {
        std::string sessionName = sessionNameEditor.getText().toStdString();
        currentSessionId = sessionManager->createSession(sessionName);
        
        sessionInfoLabel.setText("📡 Created session: " + currentSessionId, 
                                juce::dontSendNotification);
        
        // Join our own session
        TOAST::ClientInfo clientInfo;
        clientInfo.clientId = clientId;
        clientInfo.capabilities = {"MIDI_IN", "MIDI_OUT", "CLOCK_SYNC"};
        
        if (sessionManager->joinSession(currentSessionId, clientInfo)) {
            sessionInfoLabel.setText("✅ Hosting session: " + sessionName, 
                                    juce::dontSendNotification);
        }
    } catch (const std::exception& e) {
        sessionInfoLabel.setText("❌ Session creation failed: " + std::string(e.what()), 
                                juce::dontSendNotification);
    }
}

void NetworkConnectionPanel::joinSessionClicked()
{
    if (!isConnected) return;
    
    try {
        std::string sessionName = sessionNameEditor.getText().toStdString();
        // For demo purposes, assume session ID matches session name
        currentSessionId = sessionName;
        
        TOAST::ClientInfo clientInfo;
        clientInfo.clientId = clientId;
        clientInfo.capabilities = {"MIDI_IN", "MIDI_OUT", "CLOCK_SYNC"};
        
        if (sessionManager->joinSession(currentSessionId, clientInfo)) {
            sessionInfoLabel.setText("✅ Joined session: " + sessionName, 
                                    juce::dontSendNotification);
        } else {
            sessionInfoLabel.setText("❌ Failed to join session: " + sessionName, 
                                    juce::dontSendNotification);
        }
    } catch (const std::exception& e) {
        sessionInfoLabel.setText("❌ Join failed: " + std::string(e.what()), 
                                juce::dontSendNotification);
    }
}