#pragma once

#include <juce_core/juce_core.h>
#include <vector>
#include <memory>

// Forward declarations for JAM Framework v2
namespace jam {
    class GPUManager;
    class ComputePipeline;
}

/**
 * PNBTRManager - GPU-accelerated Predictive Neural Buffered Transient Recovery
 * 
 * Provides real-time audio and video prediction during network packet loss using
 * specialized Metal and GLSL compute shaders for sub-50μs processing latency.
 */
class PNBTRManager
{
public:
    struct AudioPredictionResult {
        std::vector<float> predictedSamples;
        float confidence;          // 0.0 - 1.0
        bool success;
    };
    
    struct VideoPredictionResult {
        std::vector<uint8_t> predictedFrame;
        float confidence;          // 0.0 - 1.0
        bool success;
    };
    
    PNBTRManager();
    ~PNBTRManager();
    
    // === Initialization ===
    bool initialize(jam::GPUManager* gpuManager);
    void shutdown();
    
    // === Audio Prediction (PNBTR) ===
    /**
     * Predict missing audio samples using context from available packets
     * @param context Recent audio samples for prediction context
     * @param missingSampleCount Number of samples to predict
     * @param sampleRate Audio sample rate (44100, 48000, etc.)
     * @return Predicted audio samples with confidence score
     */
    AudioPredictionResult predictAudio(const std::vector<float>& context, 
                                     int missingSampleCount, 
                                     double sampleRate);
    
    // === Video Prediction (PNBTR-JVID) ===
    /**
     * Predict missing video frame using temporal context
     * @param frameHistory Previous frames for motion analysis
     * @param frameSize Width * Height * Channels
     * @return Predicted frame data with confidence score
     */
    VideoPredictionResult predictVideoFrame(const std::vector<std::vector<uint8_t>>& frameHistory,
                                          int frameSize);
    
    // === Configuration ===
    void setAudioPredictionEnabled(bool enabled) { audioEnabled = enabled; }
    void setVideoPredictionEnabled(bool enabled) { videoEnabled = enabled; }
    bool isAudioPredictionEnabled() const { return audioEnabled; }
    bool isVideoPredictionEnabled() const { return videoEnabled; }
    
    // === Statistics ===
    struct Statistics {
        uint64_t audioPredictions = 0;
        uint64_t videoPredictions = 0;
        float averageAudioConfidence = 0.0f;
        float averageVideoConfidence = 0.0f;
        float averageProcessingTime = 0.0f;  // μs
    };
    
    Statistics getStatistics() const { return stats; }
    void resetStatistics() { stats = {}; }
    
private:
    // GPU Management (placeholder for future implementation)
    jam::GPUManager* gpu = nullptr;
    
    // Configuration
    bool audioEnabled = true;
    bool videoEnabled = true;
    bool initialized = false;
    
    // Statistics
    mutable Statistics stats;
    
    // === Shader Loading ===
    bool loadAudioShaders();
    bool loadVideoShaders();
    std::vector<uint8_t> loadShaderFile(const std::string& filename);
    
    // === Prediction Algorithms ===
    AudioPredictionResult runAudioPrediction(const std::vector<float>& context, int samples, double sampleRate);
    VideoPredictionResult runVideoPrediction(const std::vector<std::vector<uint8_t>>& frames, int frameSize);
    
    // === Confidence Assessment ===
    float calculateAudioConfidence(const std::vector<float>& prediction, const std::vector<float>& context);
    float calculateVideoConfidence(const std::vector<uint8_t>& prediction, const std::vector<std::vector<uint8_t>>& history);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PNBTRManager)
};
