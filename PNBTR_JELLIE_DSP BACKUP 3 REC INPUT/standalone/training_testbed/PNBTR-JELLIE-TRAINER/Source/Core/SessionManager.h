#pragma once

#include <string>
#include <memory>
#include <vector>

// Forward declarations
struct SessionConfig;
struct AudioMetrics;

/**
 * SessionManager handles JSON-based configuration and export system
 * Following the roadmap for complete session control
 */
class SessionManager {
public:
    SessionManager();
    ~SessionManager();
    
    //==============================================================================
    // Session Configuration
    struct Config {
        // Audio settings
        double sampleRate = 48000.0;
        int blockSize = 512;
        int numChannels = 2;
        
        // Network simulation
        float packetLossPercent = 2.0f;
        float jitterMs = 1.0f;
        
        // Processing settings
        bool enableJellie = true;
        bool enablePnbtr = true;
        bool enableMetrics = true;
        
        // GPU settings
        bool useGpuProcessing = true;
        int metalBufferSize = 1024;
        
        // Visualization
        int waveformWidth = 800;
        int waveformHeight = 200;
        bool showInputWaveform = true;
        bool showOutputWaveform = true;
        
        // Export settings
        std::string outputDirectory = "./exports";
        bool exportWav = true;
        bool exportPng = true;
        bool exportCsv = true;
        bool exportJson = true;
    };
    
    //==============================================================================
    // Session Management
    bool loadSession(const std::string& jsonPath);
    bool saveSession(const std::string& jsonPath) const;
    bool createDefaultSession();
    
    // Configuration access
    const Config& getConfig() const { return config; }
    void updateConfig(const Config& newConfig);
    
    //==============================================================================
    // Runtime Control
    void startSession();
    void stopSession();
    void pauseSession();
    void resumeSession();
    bool isSessionActive() const { return sessionActive; }
    
    //==============================================================================
    // Metrics Collection
    struct SessionMetrics {
        std::vector<float> snrHistory;
        std::vector<float> latencyHistory;
        std::vector<float> gapQualityHistory;
        std::vector<float> processingTimeHistory;
        
        double sessionDuration = 0.0;
        int totalSamplesProcessed = 0;
        int totalPacketsLost = 0;
        
        // Real-time values
        float currentSnr = 0.0f;
        float currentLatency = 0.0f;
        float currentGapQuality = 0.0f;
        float currentProcessingTime = 0.0f;
    };
    
    void recordMetrics(const AudioMetrics& metrics);
    const SessionMetrics& getSessionMetrics() const { return sessionMetrics; }
    void clearMetrics();
    
    //==============================================================================
    // Export System
    struct ExportOptions {
        bool includeWaveforms;
        bool includeMetrics;
        bool includeConfig;
        std::string sessionName;
        std::string timestamp;
        
        ExportOptions() 
            : includeWaveforms(true)
            , includeMetrics(true)
            , includeConfig(true)
            , sessionName("pnbtr_jellie_session")
            , timestamp("")
        {}
    };
    
    bool exportSession(const ExportOptions& options = ExportOptions{});
    bool exportWaveforms(const std::string& basePath);
    bool exportMetrics(const std::string& csvPath);
    bool exportConfig(const std::string& jsonPath);
    bool exportWaveformImages(const std::string& basePath);
    
    //==============================================================================
    // Audio Buffer Management
    void setInputBuffer(const float* buffer, size_t numSamples);
    void setOutputBuffer(const float* buffer, size_t numSamples);
    const std::vector<float>& getInputBuffer() const { return inputBuffer; }
    const std::vector<float>& getOutputBuffer() const { return outputBuffer; }
    
private:
    //==============================================================================
    // Internal state
    Config config;
    SessionMetrics sessionMetrics;
    bool sessionActive;
    bool sessionPaused;
    
    // Audio buffers for export
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    
    // Session timing
    double sessionStartTime;
    double totalPauseTime;
    
    //==============================================================================
    // Internal methods
    std::string generateTimestamp() const;
    std::string generateSessionPath(const std::string& extension) const;
    bool createExportDirectory(const std::string& path) const;
    
    // JSON serialization
    std::string configToJson() const;
    bool configFromJson(const std::string& json);
    std::string metricsToJson() const;
    std::string metricsToCSV() const;
}; 