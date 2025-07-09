#pragma once

#include <cstddef>

/**
 * Pure C++ interface to MetalBridge
 * Avoids Objective-C Metal framework includes in C++ GUI compilation
 * Implementation is in MetalBridge.mm (Objective-C++)
 */

struct AudioMetrics {
    float snr_db = 0.0f;
    float thd_percent = 0.0f;
    float latency_ms = 0.0f;
    float reconstruction_rate_percent = 0.0f;
    float gap_fill_quality = 0.0f;
    float overall_quality = 0.0f;
    bool is_active = false;
};

class MetalBridgeInterface {
public:
    virtual ~MetalBridgeInterface() = default;
    
    // Audio buffer access (returns read-only pointers)
    virtual const float* getAudioInputBuffer(size_t& bufferSize) = 0;
    virtual const float* getJellieBuffer(size_t& bufferSize) = 0; 
    virtual const float* getNetworkBuffer(size_t& bufferSize) = 0;
    virtual const float* getReconstructedBuffer(size_t& bufferSize) = 0;
    
    // Metrics access
    virtual AudioMetrics getLatestMetrics() = 0;
    
    // Session control
    virtual bool isSessionActive() = 0;
    virtual void startSession() = 0;
    virtual void stopSession() = 0;
    
    // Singleton access (implemented in MetalBridge.mm)
    static MetalBridgeInterface& getInstance();
};

// Buffer type enumeration for GUI components
enum class MetalBufferType {
    AudioInput,
    JellieEncoded,
    NetworkProcessed,
    Reconstructed
}; 