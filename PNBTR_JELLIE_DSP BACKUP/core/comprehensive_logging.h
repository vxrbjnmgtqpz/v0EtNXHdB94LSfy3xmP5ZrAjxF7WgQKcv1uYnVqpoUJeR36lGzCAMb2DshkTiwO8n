/*
 * PNBTR+JELLIE DSP - Comprehensive Logging System
 * Phase 3: Comprehensive Logging of PNBTR/JELLIE Output
 * 
 * Robust logging mechanism to capture every aspect of system behavior
 * during transmission for offline analysis and training data generation
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <queue>

namespace pnbtr_jellie {

// Comprehensive data structures for logging different aspects
struct AudioLogEntry {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
    
    // Ground truth data
    std::vector<float> original_audio;
    uint32_t original_sample_rate;
    uint32_t original_channels;
    
    // Received/processed data
    std::vector<float> received_audio;
    std::vector<float> reconstructed_audio;
    bool was_packet_lost = false;
    bool required_pnbtr_reconstruction = false;
    
    // Quality metrics
    double snr_db = 0.0;
    double thd_n_db = 0.0;
    double frequency_response_error_db = 0.0;
    double phase_error_degrees = 0.0;
};

struct PnbtrActionLogEntry {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
    
    // PNBTR decision context
    enum ActionType {
        NO_ACTION,
        LINEAR_INTERPOLATION,
        SPLINE_INTERPOLATION, 
        HARMONIC_PREDICTION,
        NEURAL_EXTRAPOLATION,
        ZERO_NOISE_DITHER,
        TIMING_ADJUSTMENT
    } action_type = NO_ACTION;
    
    // Input conditions that triggered action
    double packet_loss_percentage_recent = 0.0;
    double network_jitter_ms = 0.0;
    double signal_continuity_score = 0.0;
    uint32_t gap_duration_samples = 0;
    
    // PNBTR internal metrics
    double prediction_confidence = 0.0;
    double reconstruction_quality_estimate = 0.0;
    std::vector<float> prediction_input_context;
    std::vector<float> prediction_output;
    
    // Actual vs predicted comparison (filled after processing)
    std::vector<float> actual_ground_truth;
    double prediction_accuracy_score = 0.0;
    
    // Performance metrics
    double processing_time_us = 0.0;
    double cpu_usage_percent = 0.0;
    uint64_t memory_usage_bytes = 0;
};

struct NetworkLogEntry {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
    
    // Network timing
    std::chrono::high_resolution_clock::time_point sent_time;
    std::chrono::high_resolution_clock::time_point received_time;
    double latency_ms = 0.0;
    double jitter_ms = 0.0;
    
    // Network conditions
    bool was_lost = false;
    bool was_reordered = false;
    uint32_t reorder_distance = 0;
    double bandwidth_utilization_percent = 0.0;
    
    // Packet details
    size_t packet_size_bytes = 0;
    uint32_t redundancy_streams_received = 0;
    uint32_t redundancy_streams_valid = 0;
    
    // Network state estimates
    double estimated_packet_loss_rate = 0.0;
    double estimated_bandwidth_kbps = 0.0;
    double estimated_rtt_ms = 0.0;
};

struct QualityLogEntry {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
    
    // Overall system quality metrics
    double end_to_end_snr_db = 0.0;
    double end_to_end_latency_ms = 0.0;
    double reconstruction_success_rate = 0.0;
    
    // Component-specific metrics
    double jellie_encoding_quality_score = 0.0;
    double jellie_decoding_quality_score = 0.0;
    double pnbtr_reconstruction_quality_score = 0.0;
    
    // Detailed analysis
    std::vector<double> frequency_domain_error_db; // Per-band analysis
    std::vector<double> temporal_alignment_error_ms;
    bool passes_quality_threshold = false;
    
    // User-perceivable quality estimates
    double mos_score = 0.0; // Mean Opinion Score equivalent
    double artifacts_count = 0.0;
    double audible_glitch_count = 0.0;
};

// Main logging coordinator
class ComprehensiveLogger {
public:
    struct LoggingConfig {
        // Enable/disable specific logging components
        bool enable_audio_logging = true;
        bool enable_pnbtr_logging = true;
        bool enable_network_logging = true;
        bool enable_quality_logging = true;
        
        // Storage configuration
        std::string log_directory = "logs/";
        std::string session_prefix = "pnbtr_session";
        bool use_binary_format = true; // More efficient than text
        bool enable_real_time_compression = true;
        
        // Performance configuration
        bool use_background_thread = true; // Non-blocking writes
        size_t buffer_size_entries = 10000;
        double flush_interval_seconds = 5.0;
        
        // Data retention
        size_t max_log_files = 100;
        uint64_t max_file_size_mb = 100;
        bool auto_archive_old_logs = true;
        
        // Training data preparation
        bool enable_training_data_export = true;
        std::string training_data_format = "numpy"; // numpy, csv, json
    };
    
    ComprehensiveLogger(const LoggingConfig& config);
    ~ComprehensiveLogger();
    
    // Lifecycle management
    bool initialize();
    void shutdown();
    bool isRunning() const { return is_running_.load(); }
    
    // Configuration updates
    void updateConfig(const LoggingConfig& config);
    const LoggingConfig& getConfig() const { return config_; }
    
    // Logging interface
    void logAudioData(const AudioLogEntry& entry);
    void logPnbtrAction(const PnbtrActionLogEntry& entry);
    void logNetworkEvent(const NetworkLogEntry& entry);
    void logQualityMetrics(const QualityLogEntry& entry);
    
    // Batch logging for performance
    void logAudioDataBatch(const std::vector<AudioLogEntry>& entries);
    void logPnbtrActionBatch(const std::vector<PnbtrActionLogEntry>& entries);
    
    // Session management
    void startNewSession(const std::string& session_name = "");
    void endCurrentSession();
    std::string getCurrentSessionId() const { return current_session_id_; }
    
    // Log integrity and verification
    bool verifyLogIntegrity() const;
    size_t getTotalLoggedEntries() const;
    double getLoggedDataSizeGB() const;
    
    // Data export for training
    bool exportTrainingData(const std::string& output_path, 
                          const std::string& format = "numpy");
    bool exportSessionSummary(const std::string& output_path);
    
    // Analysis and statistics
    struct LoggingStats {
        uint64_t total_audio_entries = 0;
        uint64_t total_pnbtr_entries = 0;
        uint64_t total_network_entries = 0;
        uint64_t total_quality_entries = 0;
        
        double average_logging_latency_us = 0.0;
        double peak_logging_latency_us = 0.0;
        uint64_t total_bytes_logged = 0;
        
        std::chrono::high_resolution_clock::time_point session_start_time;
        std::chrono::high_resolution_clock::time_point last_log_time;
    };
    
    const LoggingStats& getStats() const { return stats_; }
    void resetStats();

private:
    LoggingConfig config_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_initialized_{false};
    
    // Session management
    std::string current_session_id_;
    std::chrono::high_resolution_clock::time_point session_start_time_;
    
    // Background logging thread
    std::thread logging_thread_;
    void loggingThreadFunction();
    
    // Thread-safe queues for different log types
    std::queue<AudioLogEntry> audio_queue_;
    std::queue<PnbtrActionLogEntry> pnbtr_queue_;
    std::queue<NetworkLogEntry> network_queue_;
    std::queue<QualityLogEntry> quality_queue_;
    
    mutable std::mutex audio_mutex_;
    mutable std::mutex pnbtr_mutex_;
    mutable std::mutex network_mutex_;
    mutable std::mutex quality_mutex_;
    mutable std::mutex stats_mutex_;
    
    // File writers
    std::unique_ptr<std::ofstream> audio_file_;
    std::unique_ptr<std::ofstream> pnbtr_file_;
    std::unique_ptr<std::ofstream> network_file_;
    std::unique_ptr<std::ofstream> quality_file_;
    
    // Statistics tracking
    LoggingStats stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    
    // File management
    void createLogFiles();
    void closeLogFiles();
    void rotateLogFiles();
    std::string generateSessionId() const;
    std::string generateLogFilePath(const std::string& type) const;
    
    // Data serialization
    void serializeAudioEntry(const AudioLogEntry& entry, std::ostream& stream);
    void serializePnbtrEntry(const PnbtrActionLogEntry& entry, std::ostream& stream);
    void serializeNetworkEntry(const NetworkLogEntry& entry, std::ostream& stream);
    void serializeQualityEntry(const QualityLogEntry& entry, std::ostream& stream);
    
    // Training data preparation
    void processForTraining();
    void exportNumpyFormat(const std::string& output_path);
    void exportCSVFormat(const std::string& output_path);
    void exportJSONFormat(const std::string& output_path);
    
    // Performance optimization
    void flushBuffers();
    void compressOldLogs();
    void cleanupOldFiles();
    
    static constexpr size_t MAX_QUEUE_SIZE = 50000;
    static constexpr double STATS_UPDATE_INTERVAL_SEC = 1.0;
};

// Helper classes for specific analysis tasks

class LogAnalyzer {
public:
    LogAnalyzer(const std::string& log_directory);
    
    // Load and parse log files
    bool loadSession(const std::string& session_id);
    std::vector<std::string> getAvailableSessions() const;
    
    // Analysis functions
    struct AnalysisReport {
        // Overall performance
        double average_end_to_end_latency_ms = 0.0;
        double average_reconstruction_accuracy = 0.0;
        double total_packet_loss_rate = 0.0;
        
        // PNBTR effectiveness
        double pnbtr_prediction_accuracy = 0.0;
        double pnbtr_processing_efficiency = 0.0;
        uint32_t pnbtr_activations_count = 0;
        
        // Network characteristics
        double network_latency_mean_ms = 0.0;
        double network_latency_std_ms = 0.0;
        double network_jitter_mean_ms = 0.0;
        std::vector<double> latency_distribution;
        
        // Quality assessment
        double overall_quality_score = 0.0;
        std::vector<double> quality_over_time;
        uint32_t quality_degradation_events = 0;
        
        // Recommendations for improvement
        std::vector<std::string> recommendations;
    };
    
    AnalysisReport generateReport() const;
    
    // Specific analysis functions
    std::vector<std::pair<double, double>> analyzePnbtrPerformance() const; // (conditions, accuracy)
    std::vector<NetworkLogEntry> identifyWorstNetworkPeriods() const;
    std::vector<AudioLogEntry> findQualityDegradationEvents() const;
    
    // Training data preparation
    bool prepareTrainingDataset(const std::string& output_path, 
                              const std::vector<std::string>& session_ids);

private:
    std::string log_directory_;
    std::vector<AudioLogEntry> audio_entries_;
    std::vector<PnbtrActionLogEntry> pnbtr_entries_;
    std::vector<NetworkLogEntry> network_entries_;
    std::vector<QualityLogEntry> quality_entries_;
    
    // Analysis helper functions
    void correlateNetworkAndPnbtrEvents();
    double calculateOverallQualityScore() const;
    std::vector<std::string> generateRecommendations() const;
};

// Real-time log monitor for debugging and validation
class LogMonitor {
public:
    LogMonitor(ComprehensiveLogger* logger);
    
    // Real-time monitoring
    void startMonitoring();
    void stopMonitoring();
    
    // Callback registration for real-time events
    using AudioCallback = std::function<void(const AudioLogEntry&)>;
    using PnbtrCallback = std::function<void(const PnbtrActionLogEntry&)>;
    using NetworkCallback = std::function<void(const NetworkLogEntry&)>;
    using QualityCallback = std::function<void(const QualityLogEntry&)>;
    
    void registerAudioCallback(AudioCallback callback);
    void registerPnbtrCallback(PnbtrCallback callback);
    void registerNetworkCallback(NetworkCallback callback);
    void registerQualityCallback(QualityCallback callback);
    
    // Real-time statistics
    struct MonitorStats {
        uint32_t entries_per_second = 0;
        double average_quality_score = 0.0;
        uint32_t pnbtr_activations_per_minute = 0;
        double current_packet_loss_rate = 0.0;
    };
    
    MonitorStats getCurrentStats() const;

private:
    ComprehensiveLogger* logger_;
    std::atomic<bool> is_monitoring_{false};
    
    std::vector<AudioCallback> audio_callbacks_;
    std::vector<PnbtrCallback> pnbtr_callbacks_;
    std::vector<NetworkCallback> network_callbacks_;
    std::vector<QualityCallback> quality_callbacks_;
    
    std::thread monitor_thread_;
    void monitoringThreadFunction();
    
    MonitorStats current_stats_;
    mutable std::mutex callback_mutex_;
};

} // namespace pnbtr_jellie
