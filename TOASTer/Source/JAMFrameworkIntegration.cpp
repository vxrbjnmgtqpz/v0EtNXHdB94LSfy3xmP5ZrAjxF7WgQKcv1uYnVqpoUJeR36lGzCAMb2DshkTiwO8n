#include "JAMFrameworkIntegration.h"

// Include JAM Framework v2 headers
#include "jam_toast.h"
#include "jam_core.h" 
#include "compute_pipeline.h"
#include "gpu_manager.h"

#include <juce_core/juce_core.h>

JAMFrameworkIntegration::JAMFrameworkIntegration() {
    // Initialize PNBTR manager
    pnbtrManager = std::make_unique<PNBTRManager>();
    
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

bool JAMFrameworkIntegration::initializeGPU() {
    try {
        // Initialize GPU Manager first
        gpuManager = std::make_unique<jam::GPUManager>();
        if (!gpuManager->initialize()) {
            juce::Logger::writeToLog("JAM Framework GPU Manager initialization failed");
            return false;
        }
        
        // Initialize Metal GPU backend on macOS
        gpuPipeline = std::make_unique<jam::ComputePipeline>();
        
        bool success = gpuPipeline->initialize();
        gpuInitialized = success;
        
        if (success) {
            // Initialize PNBTR prediction system with GPU backend
            if (pnbtrManager && !pnbtrManager->initialize(gpuManager.get())) {
                juce::Logger::writeToLog("PNBTR initialization failed - continuing without prediction");
            } else {
                juce::Logger::writeToLog("PNBTR GPU-accelerated prediction system initialized");
            }
            
            juce::Logger::writeToLog("JAM Framework GPU backend initialized (Metal)");
            notifyStatusChange("GPU acceleration enabled", networkActive);
        } else {
            juce::Logger::writeToLog("JAM Framework GPU backend initialization failed");
            notifyStatusChange("GPU acceleration unavailable", networkActive);
        }
        
        return success;
        
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("GPU backend initialization failed: " + juce::String(e.what()));
        gpuInitialized = false;
        return false;
    }
}

bool JAMFrameworkIntegration::startNetwork() {
    if (!toastProtocol) {
        notifyStatusChange("TOAST protocol not initialized", false);
        return false;
    }
    
    try {
        bool success = toastProtocol->start_processing();
        networkActive = success;
        
        if (success) {
            juce::Logger::writeToLog("JAM Framework v2 network started on " + 
                                   juce::String(multicastAddress) + ":" + 
                                   juce::String(udpPort));
            notifyStatusChange("Connected via UDP multicast", true);
        } else {
            juce::Logger::writeToLog("Failed to start JAM Framework v2 network");
            notifyStatusChange("Failed to connect", false);
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
                
            case jam::TOASTFrameType::DISCOVERY:
                // Handle peer discovery - placeholder value for now
                activePeers = 1; // toastProtocol->getActivePeerCount();
                break;
                
            case jam::TOASTFrameType::HEARTBEAT:
                // Update peer count and latency - placeholder values for now
                currentMetrics.latency_us = 50.0; // toastProtocol->getAverageLatency();
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
        if (gpuPipeline && gpuInitialized) {
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
    
    // Send heartbeat if connected
    if (toastProtocol && networkActive) {
        jam::TOASTFrame heartbeat;
        heartbeat.header.frame_type = static_cast<uint8_t>(jam::TOASTFrameType::HEARTBEAT);
        heartbeat.header.timestamp_us = static_cast<uint32_t>(
            juce::Time::getHighResolutionTicks() / 1000);
        
        try {
            toastProtocol->send_frame(heartbeat);
        } catch (const std::exception& e) {
            // Heartbeat failure indicates network issues
            juce::Logger::writeToLog("Heartbeat failed: " + juce::String(e.what()));
        }
    }
}
