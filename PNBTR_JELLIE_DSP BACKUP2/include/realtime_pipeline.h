#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <functional>

/**
 * @brief Real-time Audio Processing Pipeline for PNBTR
 * 
 * Addresses 250708_093109_System_Audit.md requirements for:
 * - Low-latency real-time processing
 * - Streaming capability
 * - Buffer management
 * - Performance monitoring
 */
class RealTimeAudioPipeline {
public:
    /**
     * @brief Audio buffer structure
     */
    struct AudioBuffer {
        std::vector<float> data;
        uint32_t sample_rate;
        uint16_t channels;
        uint64_t timestamp_us;
        uint32_t buffer_id;
        
        AudioBuffer(size_t size = 0) : data(size), sample_rate(48000), 
                                      channels(1), timestamp_us(0), buffer_id(0) {}
    };
    
    /**
     * @brief Processing configuration
     */
    struct Config {
        uint32_t sample_rate = 48000;
        uint16_t channels = 2;
        uint32_t buffer_size = 256;         // Samples per buffer
        uint32_t num_buffers = 8;           // Ring buffer size
        double target_latency_ms = 10.0;    // Target latency
        bool enable_monitoring = true;      // Performance monitoring
        
        // Processing thread configuration
        int processing_thread_priority = -1; // -1 = default, >0 = real-time priority
        bool use_dedicated_thread = true;
    };
    
    /**
     * @brief Performance metrics
     */
    struct PerformanceMetrics {
        std::atomic<double> current_latency_ms{0.0};
        std::atomic<double> average_latency_ms{0.0};
        std::atomic<double> max_latency_ms{0.0};
        std::atomic<double> cpu_load_percent{0.0};
        std::atomic<uint64_t> buffers_processed{0};
        std::atomic<uint64_t> buffers_dropped{0};
        std::atomic<uint64_t> underruns{0};
        std::atomic<uint64_t> overruns{0};
        
        // Reset all metrics
        void reset() {
            current_latency_ms = 0.0;
            average_latency_ms = 0.0;
            max_latency_ms = 0.0;
            cpu_load_percent = 0.0;
            buffers_processed = 0;
            buffers_dropped = 0;
            underruns = 0;
            overruns = 0;
        }
    };
    
    /**
     * @brief Processing function signature
     */
    using ProcessingFunction = std::function<bool(const AudioBuffer&, AudioBuffer&)>;

public:
    /**
     * @brief Constructor
     */
    RealTimeAudioPipeline(const Config& config = {});

    /**
     * @brief Destructor
     */
    ~RealTimeAudioPipeline();

    /**
     * @brief Initialize the pipeline
     * @param processing_function Function to process audio buffers
     * @return True if initialization successful
     */
    bool initialize(ProcessingFunction processing_function);

    /**
     * @brief Start real-time processing
     * @return True if started successfully
     */
    bool start();

    /**
     * @brief Stop real-time processing
     */
    void stop();

    /**
     * @brief Check if pipeline is running
     * @return True if running
     */
    bool isRunning() const { return is_running_.load(); }

    /**
     * @brief Process a single buffer (synchronous)
     * @param input_buffer Input audio buffer
     * @param output_buffer Output audio buffer
     * @return True if processed successfully
     */
    bool processBuffer(const AudioBuffer& input_buffer, AudioBuffer& output_buffer);

    /**
     * @brief Queue input buffer for asynchronous processing
     * @param buffer Input buffer to process
     * @return True if queued successfully (false if queue full)
     */
    bool queueInputBuffer(const AudioBuffer& buffer);

    /**
     * @brief Get processed output buffer
     * @param buffer Output buffer (if available)
     * @param timeout_ms Timeout in milliseconds
     * @return True if buffer available
     */
    bool getOutputBuffer(AudioBuffer& buffer, uint32_t timeout_ms = 0);

    /**
     * @brief Get current performance metrics
     * @return Current performance metrics
     */
    PerformanceMetrics getMetrics() const { return metrics_; }

    /**
     * @brief Reset performance metrics
     */
    void resetMetrics() { metrics_.reset(); }

    /**
     * @brief Validate real-time capability
     * @param test_duration_sec Duration of test in seconds
     * @return True if meets real-time requirements
     */
    bool validateRealTimeCapability(double test_duration_sec = 10.0);

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Update configuration (only when stopped)
     * @param new_config New configuration
     * @return True if updated successfully
     */
    bool updateConfig(const Config& new_config);

private:
    /**
     * @brief Processing thread main loop
     */
    void processingThreadMain();

    /**
     * @brief Update performance metrics
     */
    void updateMetrics(double processing_time_ms);

    /**
     * @brief Set thread priority for real-time processing
     */
    bool setThreadPriority(std::thread& thread, int priority);

    /**
     * @brief Get current timestamp in microseconds
     */
    uint64_t getCurrentTimestampUs() const;

private:
    Config config_;
    ProcessingFunction processing_function_;
    
    // Threading
    std::atomic<bool> is_running_{false};
    std::atomic<bool> should_stop_{false};
    std::thread processing_thread_;
    
    // Buffer queues
    std::queue<AudioBuffer> input_queue_;
    std::queue<AudioBuffer> output_queue_;
    std::mutex input_mutex_;
    std::mutex output_mutex_;
    std::condition_variable input_cv_;
    std::condition_variable output_cv_;
    
    // Performance monitoring
    mutable PerformanceMetrics metrics_;
    std::vector<double> latency_history_;
    std::mutex metrics_mutex_;
    
    // Buffer pool for memory efficiency
    std::vector<std::unique_ptr<AudioBuffer>> buffer_pool_;
    std::mutex pool_mutex_;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_process_time_;
};

/**
 * @brief PNBTR-specific real-time processor
 * 
 * Integrates PNBTR processing with real-time pipeline
 */
class PNBTRRealTimeProcessor {
public:
    /**
     * @brief PNBTR processing configuration
     */
    struct PNBTRConfig {
        bool enable_dither_replacement = true;
        bool enable_packet_loss_recovery = true;
        uint16_t target_bit_depth = 24;
        double processing_gain_db = 0.0;
        
        // JELLIE integration
        bool enable_jellie_encoding = false;
        uint16_t jellie_channels = 8;
        double packet_loss_simulation = 0.0;  // 0-100%
    };

public:
    /**
     * @brief Constructor
     */
    PNBTRRealTimeProcessor(const PNBTRConfig& pnbtr_config = {});

    /**
     * @brief Initialize PNBTR processor
     * @param pipeline_config Real-time pipeline configuration
     * @return True if initialized successfully
     */
    bool initialize(const RealTimeAudioPipeline::Config& pipeline_config);

    /**
     * @brief Start real-time PNBTR processing
     * @return True if started successfully
     */
    bool start();

    /**
     * @brief Stop real-time processing
     */
    void stop();

    /**
     * @brief Process audio with PNBTR
     * @param input_buffer Input audio
     * @param output_buffer Processed audio
     * @return True if processed successfully
     */
    bool processPNBTR(const RealTimeAudioPipeline::AudioBuffer& input_buffer,
                      RealTimeAudioPipeline::AudioBuffer& output_buffer);

    /**
     * @brief Get real-time performance metrics
     * @return Performance metrics including PNBTR-specific metrics
     */
    struct PNBTRMetrics {
        RealTimeAudioPipeline::PerformanceMetrics pipeline_metrics;
        double pnbtr_processing_time_ms = 0.0;
        double quality_score = 0.0;
        uint64_t packets_recovered = 0;
        uint64_t dither_replacements = 0;
    };
    
    PNBTRMetrics getMetrics() const;

    /**
     * @brief Validate PNBTR real-time performance
     * @param test_duration_sec Test duration
     * @return True if meets real-time requirements with quality standards
     */
    bool validatePNBTRRealTime(double test_duration_sec = 30.0);

private:
    /**
     * @brief PNBTR processing function for pipeline
     */
    bool pnbtrProcessingFunction(const RealTimeAudioPipeline::AudioBuffer& input,
                                RealTimeAudioPipeline::AudioBuffer& output);

    /**
     * @brief Apply PNBTR dither replacement
     */
    bool applyPNBTRDitherReplacement(std::vector<float>& audio_data);

    /**
     * @brief Simulate and recover from packet loss
     */
    bool simulateAndRecoverPacketLoss(std::vector<float>& audio_data);

private:
    PNBTRConfig pnbtr_config_;
    std::unique_ptr<RealTimeAudioPipeline> pipeline_;
    
    // PNBTR-specific metrics
    mutable PNBTRMetrics pnbtr_metrics_;
    std::mutex pnbtr_metrics_mutex_;
    
    // PNBTR processing state
    std::vector<float> previous_buffer_;
    uint32_t buffer_count_;
    
    // Quality monitoring
    std::vector<double> quality_history_;
};

/**
 * @brief Streaming audio validator for continuous testing
 */
class StreamingAudioValidator {
public:
    /**
     * @brief Validation configuration
     */
    struct ValidationConfig {
        double validation_interval_sec = 1.0;  // How often to validate
        bool continuous_monitoring = true;      // Continuous vs burst validation
        
        // Quality thresholds
        double min_snr_db = 60.0;
        double max_thd_n_percent = 0.1;
        double max_latency_ms = 10.0;
        
        // Test signal generation
        bool inject_test_signals = true;
        double test_signal_interval_sec = 10.0;
        std::vector<std::string> test_signal_types = {"sine", "white_noise", "chirp"};
    };

public:
    /**
     * @brief Constructor
     */
    StreamingAudioValidator(const ValidationConfig& config = {});

    /**
     * @brief Start continuous validation
     * @param processor PNBTR processor to validate
     * @return True if started successfully
     */
    bool startValidation(PNBTRRealTimeProcessor& processor);

    /**
     * @brief Stop validation
     */
    void stopValidation();

    /**
     * @brief Get validation results
     * @return Current validation status and metrics
     */
    struct ValidationResults {
        bool meets_realtime_requirements = false;
        bool meets_quality_standards = false;
        double current_snr_db = 0.0;
        double current_thd_n = 0.0;
        double current_latency_ms = 0.0;
        uint64_t total_validations = 0;
        uint64_t passed_validations = 0;
        std::string status_message;
    };
    
    ValidationResults getResults() const;

private:
    /**
     * @brief Validation thread main loop
     */
    void validationThreadMain(PNBTRRealTimeProcessor& processor);

    /**
     * @brief Generate test signal
     */
    RealTimeAudioPipeline::AudioBuffer generateTestSignal(const std::string& type,
                                                          uint32_t sample_rate,
                                                          uint32_t buffer_size);

    /**
     * @brief Validate single buffer
     */
    bool validateBuffer(const RealTimeAudioPipeline::AudioBuffer& input,
                       const RealTimeAudioPipeline::AudioBuffer& output);

private:
    ValidationConfig config_;
    std::atomic<bool> is_validating_{false};
    std::thread validation_thread_;
    
    mutable ValidationResults results_;
    std::mutex results_mutex_;
};
