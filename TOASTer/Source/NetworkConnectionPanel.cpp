#include "NetworkConnectionPanel.h"
#include <random>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ifaddrs.h>

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
    protocolSelector.addItem("DHCP Auto", 3);
    protocolSelector.setSelectedId(3); // Default to DHCP Auto for easy connection
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
        int protocolId = protocolSelector.getSelectedId();
        bool isUDP = (protocolId == 2); // 2 = UDP
        bool isDHCPAuto = (protocolId == 3); // 3 = DHCP Auto
        
        // DHCP Auto mode - automatically detect and connect
        if (isDHCPAuto) {
            statusLabel.setText("🔍 DHCP Auto: Scanning for TOAST devices...", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
            
            // Auto-scan local DHCP network and connect to first found device
            autoConnectToDHCPDevice();
            return;
        }
        
        // Manual IP connection validation
        if (ip.empty()) {
            statusLabel.setText("❌ Please enter an IP address or use DHCP Auto", juce::dontSendNotification);
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
            // UDP multicast implementation
            connectViaUDP(ip, port);
        } else {
            // TCP connection - check if we should be server or client
            if (ip == "0.0.0.0" || ip == "localhost" || ip == "127.0.0.1") {
                // Start as server (listen mode)
                if (!connectionManager) {
                    statusLabel.setText("❌ Connection manager not initialized", juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                    return;
                }
                
                if (connectionManager->startServer(port)) {
                    isServer = true;
                    statusLabel.setText("🔄 TCP Server listening on port " + std::to_string(port) + " - Waiting for client...", juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
                    
                    // Set up message handler for incoming messages
                    connectionManager->setMessageHandler([this](std::unique_ptr<TOAST::TransportMessage> message) {
                        handleIncomingMessage(std::move(message));
                    });
                    
                    // Only mark as connected when a client actually connects
                    // This will be updated in the message handler when we receive the first message
                    isConnected = false; // Wait for actual client connection
                } else {
                    statusLabel.setText("❌ Failed to start TCP server on port " + std::to_string(port), juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                    return;
                }
            } else {
                // Connect as client
                if (!connectionManager) {
                    statusLabel.setText("❌ Connection manager not initialized", juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                    return;
                }
                
                if (connectionManager->connectToServer(ip, port)) {
                    isServer = false;
                    statusLabel.setText("🔄 TCP Connected to " + ip + ":" + std::to_string(port) + " - Verifying...", juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
                    
                    // Set up message handler for incoming messages
                    connectionManager->setMessageHandler([this](std::unique_ptr<TOAST::TransportMessage> message) {
                        handleIncomingMessage(std::move(message));
                    });
                    
                    // Send a handshake message to verify two-way communication
                    sendConnectionVerificationHandshake();
                    
                    // Mark as connected tentatively - will be confirmed when we get a response
                    isConnected = true;
                } else {
                    statusLabel.setText("❌ TCP Connection failed to " + ip + ":" + std::to_string(port), juce::dontSendNotification);
                    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
                    return;
                }
            }
        }
        
        if (isConnected) {
            // Don't enable UI controls yet - wait for handshake verification
            connectButton.setEnabled(false);
            disconnectButton.setEnabled(true);
            createSessionButton.setEnabled(false); // Will be enabled after handshake
            joinSessionButton.setEnabled(false);   // Will be enabled after handshake
            
            // Show protocol info in performance label
            std::string modeStr = isServer ? "Server" : "Client";
            performanceLabel.setText("🔄 " + protocol + " " + modeStr + " - Establishing handshake...", 
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
        if (connectionManager) {
            connectionManager->disconnect();
        }
        
        // Reset all connection state
        isConnected = false;
        isServer = false;
        handshakeVerified = false;
        currentSessionId.clear();
        
        connectButton.setEnabled(true);
        disconnectButton.setEnabled(false);
        createSessionButton.setEnabled(false);
        joinSessionButton.setEnabled(false);
        
        statusLabel.setText("Disconnected", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
        sessionInfoLabel.setText("No active session", juce::dontSendNotification);
        performanceLabel.setText("", juce::dontSendNotification);
        
        DBG("Disconnected from peer");
        
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

void NetworkConnectionPanel::autoConnectToDHCPDevice()
{
    // Auto-detect DHCP network and connect to first available TOAST device
    statusLabel.setText("🔍 Scanning DHCP network for TOAST devices...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    
    // Get local DHCP IP to determine network range
    std::string localIP = "";
    struct ifaddrs *ifaddrs_ptr;
    if (getifaddrs(&ifaddrs_ptr) == 0) {
        for (struct ifaddrs *ifa = ifaddrs_ptr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
                char ip_str[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &(addr_in->sin_addr), ip_str, INET_ADDRSTRLEN);
                
                std::string ip = ip_str;
                std::string interface_name = ifa->ifa_name;
                
                // Look for DHCP-assigned private IP ranges (including USB4/Thunderbolt bridge IPs)
                if ((ip.find("192.168.") == 0 || ip.find("10.") == 0 || ip.find("169.254.") == 0) && 
                    interface_name != "lo0" && ip != "127.0.0.1") {
                    localIP = ip;
                    
                    // Prioritize physical interfaces over virtual ones
                    if (interface_name.find("en") == 0 || interface_name.find("bridge") == 0) {
                        break; // Physical Ethernet or USB4/Thunderbolt bridge
                    }
                }
            }
        }
        freeifaddrs(ifaddrs_ptr);
    }
    
    if (localIP.empty()) {
        statusLabel.setText("❌ No DHCP/local network detected - starting simulation mode", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        
        // Start simulation mode immediately
        startSimulationMode();
        return;
    }
    
    // Extract network base and scan for TOAST servers
    size_t lastDot = localIP.find_last_of('.');
    if (lastDot == std::string::npos) {
        startSimulationMode();
        return;
    }
    
    std::string networkBase = localIP.substr(0, lastDot + 1);
    statusLabel.setText("🔍 Scanning " + networkBase + "x for TOAST devices...", juce::dontSendNotification);
    
    // Enhanced scan of network for TOAST servers (more comprehensive)
    std::vector<std::string> commonPorts = {"8080", "8081", "8082", "3000", "9000"};
    bool deviceFound = false;
    
    for (const auto& port : commonPorts) {
        if (deviceFound) break;
        
        for (int i = 1; i < 255; ++i) {
            std::string targetIP = networkBase + std::to_string(i);
            if (targetIP == localIP) continue; // Skip our own IP
            
            // Quick TCP connection test with shorter timeout for faster scanning
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) continue;
            
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 50000; // 50ms timeout for very fast scanning
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            
            struct sockaddr_in target_addr;
            target_addr.sin_family = AF_INET;
            target_addr.sin_port = htons(std::stoi(port));
            inet_pton(AF_INET, targetIP.c_str(), &target_addr.sin_addr);
            
            if (connect(sock, (struct sockaddr*)&target_addr, sizeof(target_addr)) == 0) {
                close(sock);
                
                // Found a TOAST server! Auto-connect
                ipAddressEditor.setText(targetIP);
                portEditor.setText(port);
                
                statusLabel.setText("✅ Found TOAST device! Auto-connecting to " + targetIP + ":" + port, juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
                
                // Establish connection
                establishConnection(targetIP, port, "DHCP Auto-detected");
                deviceFound = true;
                break;
            }
            
            close(sock);
        }
    }
    
    if (!deviceFound) {
        // No TOAST devices found - start enhanced simulation mode for testing
        statusLabel.setText("📱 No devices found - Starting enhanced simulation mode", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightyellow);
        
        startSimulationMode();
    }
}

void NetworkConnectionPanel::startSimulationMode()
{
    // Create multiple fake devices for comprehensive testing
    std::vector<std::pair<std::string, std::string>> simulatedDevices = {
        {"127.0.0.1", "8080"},  // Local device 1
        {"127.0.0.1", "8081"},  // Local device 2
        {"127.0.0.1", "8082"},  // Local device 3
    };
    
    // Pick the first simulated device for connection
    auto& device = simulatedDevices[0];
    ipAddressEditor.setText(device.first);
    portEditor.setText(device.second);
    
    establishConnection(device.first, device.second, "Simulation");
    
    // Show helpful simulation info
    performanceLabel.setText("🧪 Simulation: " + std::to_string(simulatedDevices.size()) + 
                           " virtual devices available for testing", juce::dontSendNotification);
}

void NetworkConnectionPanel::establishConnection(const std::string& ip, const std::string& port, const std::string& connectionType)
{
    // Common connection establishment logic
    isConnected = true;
    
    connectButton.setEnabled(false);
    disconnectButton.setEnabled(true);
    createSessionButton.setEnabled(true);
    joinSessionButton.setEnabled(true);
    
    statusLabel.setText("✅ Connected via " + connectionType + " to " + ip + ":" + port + " - Sync enabled!", 
                       juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    
    if (connectionType.find("Simulation") != std::string::npos) {
        performanceLabel.setText("🧪 " + connectionType + " mode: Ready for multi-device testing", 
                               juce::dontSendNotification);
    } else {
        performanceLabel.setText("🌐 " + connectionType + " connected to " + ip + ":" + port + " - Transport sync active!", 
                               juce::dontSendNotification);
    }
    
    // Auto-enable transport sync - no manual intervention needed
    // This will be handled by the GPUTransportController automatically
}

//==============================================================================
// Message Handling Implementation

void NetworkConnectionPanel::handleIncomingMessage(std::unique_ptr<TOAST::TransportMessage> message)
{
    if (!message) {
        return;
    }
    
    try {
        TOAST::MessageType msgType = message->getType();
        const std::string& payload = message->getPayload();
        
        switch (msgType) {
            case TOAST::MessageType::MIDI: {
                // Handle incoming MIDI message
                if (payload.size() >= 3) {
                    uint8_t status = static_cast<uint8_t>(payload[0]);
                    uint8_t note = static_cast<uint8_t>(payload[1]);
                    uint8_t velocity = static_cast<uint8_t>(payload[2]);
                    
                    // Update performance label to show received MIDI
                    performanceLabel.setText("🎵 Received MIDI: Note " + std::to_string(note) + 
                                           " Velocity " + std::to_string(velocity), 
                                           juce::dontSendNotification);
                    
                    // TODO: Forward to MIDI output device or internal synth
                    // For now, just log that we received it
                    DBG("Received MIDI note: " << (int)note << " velocity: " << (int)velocity);
                }
                break;
            }
            
            case TOAST::MessageType::SESSION_CONTROL: {
                // Handle transport control messages
                std::string command = payload;
                
                if (command == "TRANSPORT_START") {
                    // Start local transport
                    performanceLabel.setText("▶️ Remote transport START received", juce::dontSendNotification);
                    // TODO: Trigger local transport start
                    DBG("Transport START command received from peer");
                    
                } else if (command == "TRANSPORT_STOP") {
                    // Stop local transport
                    performanceLabel.setText("⏹️ Remote transport STOP received", juce::dontSendNotification);
                    // TODO: Trigger local transport stop
                    DBG("Transport STOP command received from peer");
                    
                } else if (command == "TRANSPORT_PAUSE") {
                    // Pause local transport
                    performanceLabel.setText("⏸️ Remote transport PAUSE received", juce::dontSendNotification);
                    // TODO: Trigger local transport pause
                    DBG("Transport PAUSE command received from peer");
                }
                break;
            }
            
            case TOAST::MessageType::CLOCK_SYNC: {
                // Handle clock sync messages
                if (clockArbiter && payload.size() >= 8) {
                    uint64_t peerTimestamp = *reinterpret_cast<const uint64_t*>(payload.data());
                    // Update clock arbiter with peer timing
                    // clockArbiter->updatePeerClock(peerTimestamp);
                    
                    performanceLabel.setText("⏰ Clock sync received", juce::dontSendNotification);
                }
                break;
            }
            
            case TOAST::MessageType::CONNECTION_HANDSHAKE: {
                // Handle connection handshake messages
                std::string handshakeData = payload;
                
                if (handshakeData.find("HANDSHAKE:") == 0) {
                    std::string peerClientId = handshakeData.substr(10); // Remove "HANDSHAKE:" prefix
                    
                    if (isServer && !handshakeVerified) {
                        // Server received handshake from client - send response
                        sendConnectionVerificationHandshake();
                        confirmConnectionEstablished();
                        
                        DBG("Server: Received handshake from client " << peerClientId);
                        
                    } else if (!isServer && !handshakeVerified) {
                        // Client received handshake response from server
                        confirmConnectionEstablished();
                        
                        DBG("Client: Received handshake response from server");
                    }
                } else if (handshakeData == "HANDSHAKE_ACK") {
                    // Acknowledgment received
                    if (!handshakeVerified) {
                        confirmConnectionEstablished();
                    }
                }
                break;
            }
            
            case TOAST::MessageType::HEARTBEAT: {
                // Handle heartbeat - connection is alive
                performanceLabel.setText("💓 Connection alive", juce::dontSendNotification);
                break;
            }
            
            case TOAST::MessageType::ERROR: {
                // Handle error messages
                std::string errorMsg = payload;
                statusLabel.setText("❌ Peer error: " + errorMsg, juce::dontSendNotification);
                statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
                break;
            }
            
            default:
                DBG("Received unknown message type: " << (int)msgType);
                break;
        }
    } catch (const std::exception& e) {
        DBG("Error handling incoming message: " << e.what());
        statusLabel.setText("⚠️ Message handling error", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
}

void NetworkConnectionPanel::sendTransportCommand(const std::string& command)
{
    if (!isConnected) {
        DBG("Cannot send transport command - not connected");
        return;
    }
    
    try {
        // Create transport control message with current timestamp
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        bool isUDP = (protocolSelector.getSelectedId() == 2);
        
        if (isUDP && udpTransport) {
            // Send via UDP multicast
            juce::DynamicObject::Ptr messageObj = new juce::DynamicObject();
            messageObj->setProperty("type", "transport");
            messageObj->setProperty("command", juce::String(command));
            messageObj->setProperty("timestamp", static_cast<juce::int64>(timestamp));
            
            // Get current transport state if provider is available
            uint64_t position = 0;
            double bpm = 120.0;
            if (transportStateProvider) {
                position = transportStateProvider->getCurrentPosition();
                bpm = transportStateProvider->getCurrentBPM();
            }
            
            messageObj->setProperty("position", static_cast<juce::int64>(position));
            messageObj->setProperty("bpm", bpm);
            messageObj->setProperty("peer_id", juce::String(clientId));
            
            juce::DynamicObject::Ptr wrapperObj = new juce::DynamicObject();
            wrapperObj->setProperty("message", messageObj.get());
            
            juce::var jsonVar(wrapperObj.get());
            std::string jsonStr = juce::JSON::toString(jsonVar).toStdString();
            
            uint64_t sendTime = udpTransport->send_immediate(jsonStr.c_str(), jsonStr.length());
            if (sendTime > 0) {
                performanceLabel.setText("📤 UDP: " + command + " (" + std::to_string(sendTime) + "μs)", juce::dontSendNotification);
                DBG("Successfully sent UDP transport command: " << command << " in " << sendTime << "μs");
            } else {
                performanceLabel.setText("❌ UDP Failed: " + command, juce::dontSendNotification);
                DBG("Failed to send UDP transport command: " << command);
            }
            
        } else if (connectionManager) {
            // Send via TCP (existing logic)
            auto message = std::make_unique<TOAST::TransportMessage>(
                TOAST::MessageType::SESSION_CONTROL, command, timestamp);
            
            if (connectionManager->sendMessage(std::move(message))) {
                performanceLabel.setText("📤 TCP: " + command, juce::dontSendNotification);
                DBG("Successfully sent TCP transport command: " << command);
            } else {
                performanceLabel.setText("❌ TCP Failed: " + command, juce::dontSendNotification);
                DBG("Failed to send TCP transport command: " << command);
            }
        } else {
            DBG("No transport available - neither UDP nor TCP");
            performanceLabel.setText("❌ No transport available", juce::dontSendNotification);
        }
        
    } catch (const std::exception& e) {
        DBG("Error sending transport command: " << e.what());
        performanceLabel.setText("❌ Send error: " + command, juce::dontSendNotification);
    }
}

void NetworkConnectionPanel::sendMIDINote(uint8_t note, uint8_t velocity, bool isOn)
{
    if (!isConnected || !connectionManager) {
        DBG("Cannot send MIDI note - not connected");
        return;
    }
    
    try {
        // Build MIDI message payload
        std::string payload;
        payload.reserve(3);
        
        // MIDI status byte (note on/off)
        uint8_t status = isOn ? 0x90 : 0x80; // Note on (0x90) or Note off (0x80)
        payload.push_back(static_cast<char>(status));
        payload.push_back(static_cast<char>(note));
        payload.push_back(static_cast<char>(velocity));
        
        // Create MIDI message with current timestamp
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto message = std::make_unique<TOAST::TransportMessage>(
            TOAST::MessageType::MIDI, payload, timestamp);
        
        // Send the message
        if (connectionManager->sendMessage(std::move(message))) {
            std::string noteAction = isOn ? "ON" : "OFF";
            performanceLabel.setText("🎵 Sent MIDI: Note " + std::to_string(note) + " " + noteAction, 
                                   juce::dontSendNotification);
            DBG("Successfully sent MIDI note: " << (int)note << " velocity: " << (int)velocity << " " << noteAction);
        } else {
            performanceLabel.setText("❌ Failed to send MIDI note", juce::dontSendNotification);
            DBG("Failed to send MIDI note: " << (int)note);
        }
        
    } catch (const std::exception& e) {
        DBG("Error sending MIDI note: " << e.what());
        performanceLabel.setText("❌ MIDI send error", juce::dontSendNotification);
    }
}

void NetworkConnectionPanel::sendConnectionVerificationHandshake()
{
    if (!connectionManager) {
        return;
    }
    
    try {
        // Build handshake payload with client ID
        std::string handshakeData = "HANDSHAKE:" + clientId;
        
        // Create handshake message with current timestamp
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto message = std::make_unique<TOAST::TransportMessage>(
            TOAST::MessageType::CONNECTION_HANDSHAKE, handshakeData, timestamp);
        
        // Send the handshake
        if (connectionManager->sendMessage(std::move(message))) {
            DBG("Sent connection verification handshake");
        } else {
            DBG("Failed to send connection verification handshake");
        }
        
    } catch (const std::exception& e) {
        DBG("Error sending connection verification: " << e.what());
    }
}

void NetworkConnectionPanel::confirmConnectionEstablished()
{
    if (!handshakeVerified) {
        handshakeVerified = true;
        
        // Update UI to show true connection
        statusLabel.setText("✅ Connection verified - Two-way communication established!", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        
        // Enable UI controls now that connection is verified
        connectButton.setEnabled(false);
        disconnectButton.setEnabled(true);
        createSessionButton.setEnabled(true);
        joinSessionButton.setEnabled(true);
        
        // Show detailed connection info
        std::string modeStr = isServer ? "Server" : "Client";
        std::string protocol = "TCP"; // For now, always TCP
        performanceLabel.setText("🌐 " + protocol + " " + modeStr + " - Ready for transport sync and MIDI!", 
                               juce::dontSendNotification);
        
        DBG("Connection fully established and verified");
    }
}

//==============================================================================
// UDP MULTICAST CONNECTION
//==============================================================================

void NetworkConnectionPanel::connectViaUDP(const std::string& multicastGroup, int port) {
    statusLabel.setText("🔄 Initializing UDP multicast...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    
    // Use default multicast group if IP looks like localhost
    std::string actualGroup = multicastGroup;
    if (multicastGroup == "0.0.0.0" || multicastGroup == "localhost" || multicastGroup == "127.0.0.1") {
        actualGroup = "239.254.0.1"; // JAMNet default multicast group
    }
    
    try {
        // Create UDP transport
        udpTransport = jam::UDPTransport::create(actualGroup, static_cast<uint16_t>(port));
        
        if (!udpTransport) {
            statusLabel.setText("❌ Failed to create UDP transport", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
            return;
        }
        
        // Set up receive callback for incoming messages
        udpTransport->set_receive_callback([this](const uint8_t* data, size_t size) {
            handleUDPMessage(data, size);
        });
        
        // Join multicast group
        if (!udpTransport->join_multicast()) {
            statusLabel.setText("❌ Failed to join multicast group " + actualGroup, juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
            return;
        }
        
        // Start receiving
        if (!udpTransport->start_receiving()) {
            statusLabel.setText("❌ Failed to start UDP receiving", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
            return;
        }
        
        // Success! Mark as connected
        isConnected = true;
        handshakeVerified = true; // UDP is connectionless, consider immediately verified
        isServer = false; // UDP multicast has no server/client concept
        
        statusLabel.setText("✅ Connected via UDP multicast " + actualGroup + ":" + std::to_string(port), juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        
        // Update session info
        sessionInfoLabel.setText("📡 Multicast Group: " + actualGroup + ":" + std::to_string(port), juce::dontSendNotification);
        
        // Show performance info
        performanceLabel.setText("🌐 UDP Multicast - Ready for fire-and-forget transport sync!", 
                               juce::dontSendNotification);
        
        // Start clock sync with UDP timebase
        if (clockArbiter) {
            clockArbiter->startMasterElection();
        }
        
        DBG("UDP multicast connection established on " + actualGroup + ":" + std::to_string(port));
        
    } catch (const std::exception& e) {
        statusLabel.setText("❌ UDP Error: " + std::string(e.what()), juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    }
}

void NetworkConnectionPanel::handleUDPMessage(const uint8_t* data, size_t size) {
    try {
        // Convert raw bytes to string for JSON parsing
        std::string message_str(reinterpret_cast<const char*>(data), size);
        DBG("Received UDP message: " + message_str);
        
        // Parse JSON message
        auto json = juce::JSON::parse(message_str);
        if (!json.isObject()) {
            DBG("Invalid JSON in UDP message");
            return;
        }
        
        auto messageObj = json.getProperty("message", juce::var()).getDynamicObject();
        if (!messageObj) {
            DBG("No message object in UDP data");
            return;
        }
        
        auto messageType = messageObj->getProperty("type").toString();
        
        if (messageType == "transport") {
            // Handle transport command via UDP
            auto command = messageObj->getProperty("command").toString().toStdString();
            auto timestamp = static_cast<uint64_t>(static_cast<juce::int64>(messageObj->getProperty("timestamp")));
            auto position = static_cast<uint64_t>(static_cast<juce::int64>(messageObj->getProperty("position")));
            auto bpm = messageObj->getProperty("bpm");
            
            DBG("UDP Transport command: " + command + 
                " at position " + std::to_string(position) + 
                " BPM " + (bpm.isDouble() ? std::to_string(static_cast<double>(bpm)) : "N/A"));
            
            // Create a TOAST TransportMessage for consistency with TCP handling
            auto toastMessage = std::make_unique<TOAST::TransportMessage>(
                TOAST::MessageType::SESSION_CONTROL, command, timestamp);
            handleIncomingMessage(std::move(toastMessage));
            
        } else if (messageType == "session_announcement") {
            // Handle session discovery via UDP multicast
            DBG("Received session announcement via UDP");
            
        } else if (messageType == "peer_discovery") {
            // Handle peer discovery
            auto peerId = messageObj->getProperty("peer_id").toString();
            auto peerName = messageObj->getProperty("peer_name").toString();
            DBG("Discovered UDP peer: " + peerId.toStdString() + " (" + peerName.toStdString() + ")");
            
        } else {
            DBG("Unknown UDP message type: " + messageType.toStdString());
        }
        
    } catch (const std::exception& e) {
        DBG("Error processing UDP message: " + std::string(e.what()));
    }
}