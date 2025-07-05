#include "JAMFrameworkIntegration.h"

// Include JAM Framework v2 headers
#include "jam_toast.h"
#include "jam_core.h" 
#include "compute_pipeline.h"
#include "gpu_manager.h"
#include "network_state_detector.h"

#include <juce_core/juce_core.h>

JAMFrameworkIntegration::JAMFrameworkIntegration() {
    // Initialize PNBTR manager
    pnbtrManager = std::make_unique<PNBTRManager>();
    
    // Initialize network state detector for real connectivity testing
    networkStateDetector = std::make_unique<jam::NetworkStateDetector>();
    
    // Enable automatic features for seamless operation
    auto_connection_enabled_ = true;    // Always auto-connect
    minimum_peers_ = 1;                 // Connect as soon as one peer found
    pnbtrAudioEnabled = true;          // Always use PNBTR audio prediction
    pnbtrVideoEnabled = true;          // Always use PNBTR video prediction
    
    // Initialize with 100ms update timer for performance monitoring
    startTimer(100);
}

JAMFrameworkIntegration::~JAMFrameworkIntegration() {
    stopNetwork();
    
    // Shutdown PNBTR manager
    if (pnbtrManager) {
        pnbtrManager->shutdown();
    }
}

bool JAMFrameworkIntegration::initialize(const std::string& multicast_addr, 
                                       int port, 
                                       const std::string& session_name) {
    sessionName = session_name;
    multicastAddress = multicast_addr;
    udpPort = port;
    
    try {
        // Initialize TOAST v2 protocol
        toastProtocol = std::make_unique<jam::TOASTv2Protocol>();
        
        // Set up frame callback
        toastProtocol->set_midi_callback([this](const jam::TOASTFrame& frame) {
            handleIncomingFrame(frame);
        });
        
        // Set up discovery callback for peer detection with auto-connection
        toastProtocol->set_discovery_callback([this](const jam::TOASTFrame& frame) {
            std::string peer_id = "peer_" + std::to_string(frame.header.session_id);
            
            {
                std::lock_guard<std::mutex> lock(discovered_peers_mutex_);
                auto it = std::find(discovered_peers_.begin(), discovered_peers_.end(), peer_id);
                if (it == discovered_peers_.end()) {
                    discovered_peers_.push_back(peer_id);
                    juce::Logger::writeToLog("🔍 Discovered new peer: " + juce::String(peer_id));
                }
            }
            
            activePeers++;
            
            // Auto-connection logic: automatically "connect" to discovered peers
            if (auto_connection_enabled_ && activePeers >= minimum_peers_) {
                if (!networkActive) {
                    juce::Logger::writeToLog("🚀 Auto-connecting - minimum peers reached!");
                    // Network is already active, just update status
                }
                notifyStatusChange("Auto-connected! Active peers: " + std::to_string(activePeers), true);
            } else {
                notifyStatusChange("Peer discovered! Active peers: " + std::to_string(activePeers), true);
            }
        });
        
        // Set up heartbeat callback with peer maintenance
        toastProtocol->set_heartbeat_callback([this](const jam::TOASTFrame& frame) {
            juce::Logger::writeToLog("💓 Received heartbeat from peer");
            
            // Maintain peer list and auto-connection
            std::string peer_id = "peer_" + std::to_string(frame.header.session_id);
            {
                std::lock_guard<std::mutex> lock(discovered_peers_mutex_);
                auto it = std::find(discovered_peers_.begin(), discovered_peers_.end(), peer_id);
                if (it == discovered_peers_.end()) {
                    discovered_peers_.push_back(peer_id);
                    juce::Logger::writeToLog("💓 Heartbeat from new peer: " + juce::String(peer_id));
                    
                    // Auto-connect to heartbeat peers too
                    if (auto_connection_enabled_) {
                        activePeers++;
                        notifyStatusChange("Auto-connected via heartbeat! Active peers: " + std::to_string(activePeers), true);
                    }
                }
            }
        });
        
        // Set up error callback
        toastProtocol->set_error_callback([this](const std::string& error) {
            juce::Logger::writeToLog("JAM Framework Error: " + juce::String(error));
            notifyStatusChange("Error: " + error, false);
        });
        
        // Initialize the protocol with session name converted to ID
        uint32_t session_id = std::hash<std::string>{}(session_name);
        bool success = toastProtocol->initialize(multicast_addr, port, session_id);
        
        if (success) {
            notifyStatusChange("JAM Framework v2 initialized", false);
            return true;
        } else {
            notifyStatusChange("Failed to initialize JAM Framework v2", false);
            return false;
        }
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("JAM Framework initialization failed: " + juce::String(e.what()));
        notifyStatusChange("Initialization failed: " + std::string(e.what()), false);
        return false;
    }
}



bool JAMFrameworkIntegration::startNetwork() {
    if (!toastProtocol) {
        notifyStatusChange("TOAST protocol not initialized", false);
        return false;
    }
    
    // CRITICAL FIX: Test real network state before claiming connected
    juce::Logger::writeToLog("🔍 Testing real network connectivity...");
    notifyStatusChange("Testing network connectivity...", false);
    
    // Step 1: Check network permission (macOS specific issue)
    if (!networkStateDetector->hasNetworkPermission()) {
        std::string error = "❌ Network permission denied - please allow network access";
        juce::Logger::writeToLog(error);
        notifyStatusChange(error, false);
        return false;
    }
    
    // Step 2: Check if network interface is actually ready (not just DHCP pending)
    if (!networkStateDetector->isNetworkInterfaceReady()) {
        std::string error = "❌ Network interface not ready - check connection";
        juce::Logger::writeToLog(error);
        notifyStatusChange(error, false);
        return false;
    }
    
    // Step 3: Test actual UDP connectivity with our settings
    if (!networkStateDetector->testUDPConnectivity(multicastAddress, udpPort)) {
        std::string error = "❌ UDP connectivity test failed on " + multicastAddress + ":" + std::to_string(udpPort);
        juce::Logger::writeToLog(error);
        notifyStatusChange(error, false);
        return false;
    }
    
    // Step 4: Test multicast capability (critical for discovery)
    if (!networkStateDetector->testMulticastCapability(multicastAddress, udpPort, 2000)) {
        std::string error = "❌ Multicast test failed - check firewall/network settings";
        juce::Logger::writeToLog(error);
        notifyStatusChange(error, false);
        return false;
    }
    
    juce::Logger::writeToLog("✅ All network connectivity tests passed");
    
    try {
        // Now start the actual TOAST protocol with validated network
        bool success = toastProtocol->start_processing();
        
        if (success) {
            networkActive = true;
            juce::Logger::writeToLog("JAM Framework v2 network started on " + 
                                   juce::String(multicastAddress) + ":" + 
                                   juce::String(udpPort));
            
            // Send discovery message to announce our presence
            toastProtocol->send_discovery();
            juce::Logger::writeToLog("🔍 Sent discovery message to announce presence");
            
            // Start periodic heartbeat (we'll do this with the timer)
            notifyStatusChange("Connected via UDP multicast - Looking for peers...", true);
        } else {
            juce::Logger::writeToLog("Failed to start JAM Framework v2 network");
            notifyStatusChange("Failed to connect - socket initialization failed", false);
        }
        
        return success;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Network start failed: " + juce::String(e.what()));
        notifyStatusChange("Network error: " + std::string(e.what()), false);
        networkActive = false;
        return false;
    }
}

void JAMFrameworkIntegration::stopNetwork() {
    if (toastProtocol && networkActive) {
        toastProtocol->stop_processing();
        networkActive = false;
        activePeers = 0;
        notifyStatusChange("Disconnected", false);
        juce::Logger::writeToLog("JAM Framework v2 network stopped");
    }
}

void JAMFrameworkIntegration::sendMIDIEvent(uint8_t status, uint8_t data1, uint8_t data2, bool use_burst) {
    if (!toastProtocol || !networkActive) {
        return;
    }
    
    try {
        // Create MIDI event payload
        std::vector<uint8_t> midiData = {status, data1, data2};
        
        // Create TOAST frame
        jam::TOASTFrame frame;
        frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::MIDI);
        frame.header.timestamp_us = static_cast<uint32_t>(
            juce::Time::getHighResolutionTicks() / 1000); // Convert to microseconds
        frame.payload = midiData;
        
        // Configure burst transmission if requested
        if (use_burst) {
            jam::BurstConfig burstConfig;
            burstConfig.burst_size = 3;  // Send 3 packets per MIDI event
            burstConfig.jitter_window_us = 500;  // 500μs jitter window
            burstConfig.enable_redundancy = true;
        }
        
        // Send frame with burst setting
        toastProtocol->send_frame(frame, use_burst);
        
        // Update throughput metrics
        currentMetrics.throughput_mbps += (midiData.size() * 8.0) / 1000000.0;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("MIDI send failed: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::sendMIDIData(const uint8_t* data, size_t size, bool use_burst) {
    if (!toastProtocol || !networkActive || !data || size == 0) {
        return;
    }
    
    try {
        // Create TOAST frame
        jam::TOASTFrame frame;
        frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::MIDI);
        frame.header.timestamp_us = static_cast<uint32_t>(
            juce::Time::getHighResolutionTicks() / 1000);
        frame.payload.assign(data, data + size);
        
        if (use_burst) {
            jam::BurstConfig burstConfig;
            burstConfig.burst_size = 3;
            burstConfig.jitter_window_us = 500;
            burstConfig.enable_redundancy = true;
        }
        
        // Send frame with burst setting
        toastProtocol->send_frame(frame, use_burst);
        
        // Update metrics
        currentMetrics.throughput_mbps += (size * 8.0) / 1000000.0;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("MIDI data send failed: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::sendAudioData(const float* samples, int numSamples, int sampleRate, bool enablePrediction) {
    if (!toastProtocol || !networkActive || !samples || numSamples <= 0) {
        return;
    }
    
    try {
        // Convert float samples to bytes
        std::vector<uint8_t> audioData(numSamples * sizeof(float));
        std::memcpy(audioData.data(), samples, audioData.size());
        
        // Apply PNBTR audio prediction if enabled
        if (enablePrediction && pnbtrAudioEnabled && pnbtrManager) {
            try {
                // Use last few samples as context for prediction
                std::vector<float> context(samples, samples + std::min(numSamples, 128));
                
                // Predict missing samples (simulating packet loss prediction)
                auto prediction = pnbtrManager->predictAudio(context, numSamples / 4, sampleRate);
                
                if (prediction.success) {
                    juce::Logger::writeToLog(juce::String::formatted(
                        "PNBTR: Audio prediction ready (%.1f%% confidence)", 
                        prediction.confidence * 100.0f));
                    
                    // Store prediction for potential use in packet loss recovery
                    predictionConfidence = prediction.confidence;
                } else {
                    juce::Logger::writeToLog("PNBTR: Audio prediction failed");
                }
            } catch (const std::exception& e) {
                juce::Logger::writeToLog("PNBTR: Audio prediction exception: " + juce::String(e.what()));
            }
        }
        
        // Create TOAST frame
        jam::TOASTFrame frame;
        frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::AUDIO);
        frame.header.timestamp_us = static_cast<uint32_t>(
            juce::Time::getHighResolutionTicks() / 1000);
        frame.payload = audioData;
        
        // Send frame
        toastProtocol->send_frame(frame);
        
        // Update metrics
        currentMetrics.throughput_mbps += (audioData.size() * 8.0) / 1000000.0;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Audio send failed: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::sendVideoFrame(const uint8_t* frameData, int width, int height, 
                                           const std::string& format, bool enablePrediction) {
    if (!toastProtocol || !networkActive || !frameData || width <= 0 || height <= 0) {
        return;
    }
    
    try {
        // Calculate frame size based on format
        int bytesPerPixel = 3; // Default to RGB24
        if (format == "RGBA32") bytesPerPixel = 4;
        else if (format == "RGB24") bytesPerPixel = 3;
        else if (format == "YUV420") bytesPerPixel = 2; // Approximate
        
        size_t frameSize = width * height * bytesPerPixel;
        
        // Create payload with frame data
        std::vector<uint8_t> videoData(frameData, frameData + frameSize);
        
        // Apply PNBTR-JVID video prediction if enabled
        if (enablePrediction && pnbtrVideoEnabled && pnbtrManager) {
            try {
                // Maintain frame history for prediction context (simple implementation)
                static std::vector<std::vector<uint8_t>> frameHistory;
                frameHistory.push_back(videoData);
                
                // Keep only last 5 frames for prediction context
                if (frameHistory.size() > 5) {
                    frameHistory.erase(frameHistory.begin());
                }
                
                // Only predict if we have enough history
                if (frameHistory.size() >= 3) {
                    auto prediction = pnbtrManager->predictVideoFrame(frameHistory, frameSize);
                    
                    if (prediction.success) {
                        juce::Logger::writeToLog(juce::String::formatted(
                            "PNBTR-JVID: Video prediction ready (%.1f%% confidence)", 
                            prediction.confidence * 100.0f));
                        
                        predictionConfidence = std::max(predictionConfidence, (double)prediction.confidence);
                    } else {
                        juce::Logger::writeToLog("PNBTR-JVID: Video prediction failed");
                    }
                }
            } catch (const std::exception& e) {
                juce::Logger::writeToLog("PNBTR-JVID: Video prediction exception: " + juce::String(e.what()));
            }
        }
        
        // Create TOAST frame
        jam::TOASTFrame frame;
        frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::VIDEO);
        frame.header.timestamp_us = static_cast<uint32_t>(
            juce::Time::getHighResolutionTicks() / 1000);
        frame.payload = videoData;
        
        // Send frame (video frames are typically large, consider compression)
        toastProtocol->send_frame(frame);
        
        // Update metrics
        currentMetrics.throughput_mbps += (videoData.size() * 8.0) / 1000000.0;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Video send failed: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::setBurstConfig(int burstSize, int jitterWindow_us, bool enableRedundancy) {
    if (toastProtocol) {
        jam::BurstConfig config;
        config.burst_size = static_cast<uint8_t>(juce::jlimit(1, 255, burstSize));
        config.jitter_window_us = static_cast<uint16_t>(juce::jlimit(0, 65535, jitterWindow_us));
        config.enable_redundancy = enableRedundancy;
        
        toastProtocol->set_burst_config(config);
        
        juce::Logger::writeToLog("Burst config updated: size=" + juce::String(burstSize) + 
                               ", jitter=" + juce::String(jitterWindow_us) + "μs");
    }
}

void JAMFrameworkIntegration::handleIncomingFrame(const jam::TOASTFrame& frame) {
    try {
        switch (static_cast<jam::TOASTFrameType>(frame.header.frame_type)) {
            case jam::TOASTFrameType::MIDI:
                if (midiCallback && frame.payload.size() >= 3) {
                    uint8_t status = frame.payload[0];
                    uint8_t data1 = frame.payload[1];
                    uint8_t data2 = frame.payload[2];
                    midiCallback(status, data1, data2, frame.header.timestamp_us);
                }
                break;
                
            case jam::TOASTFrameType::AUDIO:
                if (audioCallback && frame.payload.size() >= sizeof(float)) {
                    const float* samples = reinterpret_cast<const float*>(frame.payload.data());
                    int numSamples = static_cast<int>(frame.payload.size() / sizeof(float));
                    audioCallback(samples, numSamples, frame.header.timestamp_us);
                }
                break;
                
            case jam::TOASTFrameType::VIDEO:
                if (videoCallback && frame.payload.size() > 0) {
                    // For now, assume basic frame format - this would need header info for dimensions
                    videoCallback(frame.payload.data(), 640, 480, frame.header.timestamp_us);
                }
                break;
                
            case jam::TOASTFrameType::SYNC:
                // Handle transport sync commands with full bidirectional support
                if (frame.payload.size() > 0 && transportCallback) {
                    std::string syncMessage(frame.payload.begin(), frame.payload.end());
                    juce::Logger::writeToLog("🎵 Received transport sync: " + juce::String(syncMessage));
                    
                    // Parse JSON transport command and extract all parameters
                    try {
                        std::string command;
                        uint64_t timestamp = frame.header.timestamp_us;
                        double position = 0.0;
                        double bpm = 120.0;
                        
                        // Extract command
                        if (syncMessage.find("\"PLAY\"") != std::string::npos) {
                            command = "PLAY";
                        } else if (syncMessage.find("\"STOP\"") != std::string::npos) {
                            command = "STOP";
                        }
                        
                        // Extract position (simple parsing)
                        auto posPos = syncMessage.find("\"position\":");
                        if (posPos != std::string::npos) {
                            auto start = syncMessage.find(':', posPos) + 1;
                            auto end = syncMessage.find_first_of(",}", start);
                            if (end != std::string::npos) {
                                position = std::stod(syncMessage.substr(start, end - start));
                            }
                        }
                        
                        // Extract BPM
                        auto bpmPos = syncMessage.find("\"bpm\":");
                        if (bpmPos != std::string::npos) {
                            auto start = syncMessage.find(':', bpmPos) + 1;
                            auto end = syncMessage.find_first_of(",}", start);
                            if (end != std::string::npos) {
                                bpm = std::stod(syncMessage.substr(start, end - start));
                            }
                        }
                        
                        // Call transport callback with full parameters
                        if (!command.empty()) {
                            transportCallback(command, timestamp, position, bpm);
                            juce::Logger::writeToLog("🎵 Transport command processed: " + juce::String(command) + 
                                                   " pos=" + juce::String(position) + " bpm=" + juce::String(bpm));
                        }
                        
                    } catch (const std::exception& e) {
                        juce::Logger::writeToLog("Transport sync parse error: " + juce::String(e.what()));
                    }
                }
                break;
                
            case jam::TOASTFrameType::TRANSPORT:
                // Handle dedicated transport commands with full bidirectional support
                if (frame.payload.size() > 0) {
                    std::string transportMessage(frame.payload.begin(), frame.payload.end());
                    juce::Logger::writeToLog("🎛️ Received transport frame: " + juce::String(transportMessage));
                    
                    // Parse JSON transport command
                    try {
                        std::string command;
                        uint64_t timestamp = frame.header.timestamp_us * 1000; // Convert back to microseconds
                        double position = 0.0;
                        double bpm = 120.0;
                        
                        // Extract command
                        if (transportMessage.find("\"PLAY\"") != std::string::npos) {
                            command = "PLAY";
                        } else if (transportMessage.find("\"STOP\"") != std::string::npos) {
                            command = "STOP";
                        } else if (transportMessage.find("\"POSITION\"") != std::string::npos) {
                            command = "POSITION";
                        } else if (transportMessage.find("\"BPM\"") != std::string::npos) {
                            command = "BPM";
                        }
                        
                        // Extract position (simple parsing)
                        auto posPos = transportMessage.find("\"position\":");
                        if (posPos != std::string::npos) {
                            auto start = transportMessage.find(':', posPos) + 1;
                            auto end = transportMessage.find_first_of(",}", start);
                            if (end != std::string::npos) {
                                position = std::stod(transportMessage.substr(start, end - start));
                            }
                        }
                        
                        // Extract BPM
                        auto bpmPos = transportMessage.find("\"bpm\":");
                        if (bpmPos != std::string::npos) {
                            auto start = transportMessage.find(':', bpmPos) + 1;
                            auto end = transportMessage.find_first_of(",}", start);
                            if (end != std::string::npos) {
                                bpm = std::stod(transportMessage.substr(start, end - start));
                            }
                        }
                        
                        // Handle transport command via callback
                        if (!command.empty()) {
                            handleTransportCommand(command, timestamp, position, bpm);
                        }
                        
                    } catch (const std::exception& e) {
                        juce::Logger::writeToLog("Transport frame parse error: " + juce::String(e.what()));
                    }
                }
                break;
                
            case jam::TOASTFrameType::DISCOVERY:
                // Handle peer discovery frames
                juce::Logger::writeToLog("🔍 Received discovery frame");
                break;
                
            case jam::TOASTFrameType::HEARTBEAT:
                // Handle heartbeat frames
                juce::Logger::writeToLog("💓 Received heartbeat frame");
                break;
                
            case jam::TOASTFrameType::BURST_HEADER:
                // Handle burst header frames
                juce::Logger::writeToLog("📦 Received burst header frame");
                break;
                
            default:
                juce::Logger::writeToLog("Unknown frame type received: " + 
                                       juce::String(static_cast<int>(frame.header.frame_type)));
                break;
        }
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Frame handling error: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::updatePerformanceMetrics() {
    if (toastProtocol && networkActive) {
        // Get metrics from TOAST protocol - placeholder values for now
        currentMetrics.active_peers = activePeers; // toastProtocol->getActivePeerCount();
        currentMetrics.latency_us = 50.0; // toastProtocol->getAverageLatency();
        currentMetrics.packet_loss_rate = 0.01; // toastProtocol->getPacketLossRate();
        
        // Update prediction accuracy if GPU backend is available
        if (gpuPipeline && jam::gpu_native::GPUTimebase::is_initialized()) {
            // TODO: Get prediction accuracy from PNBTR system
            currentMetrics.prediction_accuracy = predictionConfidence;
        }
        
        // Notify performance callback if registered
        if (performanceCallback) {
            performanceCallback(currentMetrics.latency_us, currentMetrics.throughput_mbps, currentMetrics.active_peers);
        }
        
        // Reset throughput counter (will be accumulated over next period)
        currentMetrics.throughput_mbps = 0.0;
    }
}

void JAMFrameworkIntegration::notifyStatusChange(const std::string& status, bool connected) {
    if (statusCallback) {
        statusCallback(status, connected);
    }
    
    // Also log to JUCE logger
    juce::Logger::writeToLog("JAM Framework Status: " + juce::String(status) + 
                           " (Connected: " + (connected ? "Yes" : "No") + ")");
}

void JAMFrameworkIntegration::timerCallback() {
    // Update performance metrics every 100ms
    updatePerformanceMetrics();
    
    // GPU is already initialized in GPU-native architecture - no auto-init needed
    
    // Send heartbeat every second if connected
    static int heartbeatCounter = 0;
    heartbeatCounter++;
    
    if (toastProtocol && networkActive && heartbeatCounter >= 10) { // Every 1 second (10 * 100ms)
        heartbeatCounter = 0;
        
        try {
            toastProtocol->send_heartbeat();
            juce::Logger::writeToLog("💓 Sent heartbeat to peers");
        } catch (const std::exception& e) {
            // Heartbeat failure indicates network issues
            juce::Logger::writeToLog("Heartbeat failed: " + juce::String(e.what()));
        }
    }
    
    // Auto-enable features if network becomes active
    if (networkActive) {
        // Ensure PNBTR prediction is always enabled
        if (pnbtrManager && !pnbtrAudioEnabled) {
            pnbtrAudioEnabled = true;
            pnbtrManager->setAudioPredictionEnabled(true);
        }
        
        if (pnbtrManager && !pnbtrVideoEnabled) {
            pnbtrVideoEnabled = true; 
            pnbtrManager->setVideoPredictionEnabled(true);
        }
    }
}

void JAMFrameworkIntegration::sendTransportCommand(const std::string& command, uint64_t timestamp, 
                                                   double position, double bpm) {
    if (!toastProtocol || !networkActive) {
        return;
    }
    
    try {
        // Create JSON transport command
        std::string transportMessage = 
            "{\"type\":\"transport\","
            "\"command\":\"" + command + "\","
            "\"timestamp\":" + std::to_string(timestamp) + ","
            "\"position\":" + std::to_string(position) + ","
            "\"bpm\":" + std::to_string(bpm) + "}";
        
        // Create TOAST frame with TRANSPORT type
        jam::TOASTFrame frame;
        frame.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::TRANSPORT);
        frame.header.timestamp_us = static_cast<uint32_t>(timestamp / 1000); // Convert to milliseconds 
        frame.payload.assign(transportMessage.begin(), transportMessage.end());
        
        // Send frame with burst transmission for reliability
        toastProtocol->send_frame(frame, true);
        
        juce::Logger::writeToLog("📡 Sent transport command: " + juce::String(command) + 
                               " (pos: " + juce::String(position) + ", bpm: " + juce::String(bpm) + ")");
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Transport command send failed: " + juce::String(e.what()));
    }
}

void JAMFrameworkIntegration::handleTransportCommand(const std::string& command, uint64_t timestamp, 
                                                    double position, double bpm) {
    if (transportCallback) {
        transportCallback(command, timestamp, position, bpm);
        juce::Logger::writeToLog("🎛️ Received transport command: " + juce::String(command) + 
                               " (pos: " + juce::String(position) + ", bpm: " + juce::String(bpm) + ")");
    }
}

std::vector<std::string> JAMFrameworkIntegration::getDiscoveredPeers() const {
    std::lock_guard<std::mutex> lock(discovered_peers_mutex_);
    return discovered_peers_;
}
