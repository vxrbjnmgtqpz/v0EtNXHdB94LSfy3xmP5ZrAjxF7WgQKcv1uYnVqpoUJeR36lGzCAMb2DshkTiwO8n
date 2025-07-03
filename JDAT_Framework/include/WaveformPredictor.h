#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <mutex>

namespace jdat {

/**
 * @brief PNTBTR (Predictive Network Temporal Buffered Transmission Recovery) waveform predictor
 * 
 * This class implements intelligent audio waveform prediction for recovering missing audio segments
 * when packets are lost or delayed in network transmission.
 */
class WaveformPredictor {
public:
    /**
     * @brief Prediction configuration
     */
    struct Config {
        uint32_t sample_rate = 96000;
        uint32_t max_prediction_samples = 4800;  // 50ms at 96kHz
        uint32_t history_buffer_ms = 100;        // History to analyze for patterns
        double prediction_confidence_threshold = 0.7;
        bool enable_harmonic_analysis = true;
        bool enable_pattern_matching = true;
        bool enable_zero_crossing_optimization = true;
        uint32_t analysis_window_samples = 1024;
    };

    /**
     * @brief Prediction quality metrics
     */
    struct PredictionQuality {
        double confidence_score = 0.0;    // 0.0 to 1.0
        double harmonic_match = 0.0;      // Harmonic content similarity
        double pattern_match = 0.0;       // Pattern repetition match
        double spectral_continuity = 0.0; // Spectral smoothness
        bool is_reliable = false;         // Overall reliability assessment
    };

    /**
     * @brief Prediction result
     */
    struct PredictionResult {
        std::vector<float> predicted_samples;
        PredictionQuality quality;
        uint32_t samples_predicted = 0;
        uint64_t prediction_time_us = 0;
        std::string method_used;
    };

private:
    Config config_;
    std::vector<float> history_buffer_;
    std::atomic<bool> is_initialized_{false};
    std::atomic<uint64_t> total_predictions_{0};
    
    // Analysis state
    struct AnalysisState {
        std::vector<float> windowed_samples;
        std::vector<float> frequency_domain;
        std::vector<float> autocorrelation;
        double fundamental_frequency = 0.0;
        double energy_level = 0.0;
        bool is_periodic = false;
        uint32_t period_samples = 0;
    };
    
    AnalysisState last_analysis_;
    
    // Statistics
    struct Statistics {
        uint64_t predictions_made = 0;
        uint64_t successful_predictions = 0;
        double average_confidence = 0.0;
        double average_prediction_time_us = 0.0;
        uint32_t max_samples_predicted = 0;
    };
    
    mutable std::mutex stats_mutex_;
    Statistics stats_;

public:
    /**
     * @brief Constructor
     * @param config Predictor configuration
     */
    explicit WaveformPredictor(const Config& config);

    /**
     * @brief Destructor
     */
    ~WaveformPredictor();

    /**
     * @brief Initialize the predictor
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Update the history buffer with new audio samples
     * @param samples New audio samples to add to history
     */
    void updateHistory(const std::vector<float>& samples);

    /**
     * @brief Predict missing audio samples
     * @param num_samples Number of samples to predict
     * @param gap_start_timestamp Timestamp when gap started (optional)
     * @return Prediction result
     */
    PredictionResult predictSamples(uint32_t num_samples, uint64_t gap_start_timestamp = 0);

    /**
     * @brief Predict with specific method
     * @param num_samples Number of samples to predict
     * @param method Prediction method to use
     * @return Prediction result
     */
    PredictionResult predictWithMethod(uint32_t num_samples, const std::string& method);

    /**
     * @brief Clear history buffer and reset state
     */
    void reset();

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Update configuration
     * @param config New configuration
     * @return True if updated successfully
     */
    bool updateConfig(const Config& config);

    /**
     * @brief Get current statistics
     * @return Prediction statistics
     */
    Statistics getStatistics() const;

    /**
     * @brief Reset statistics
     */
    void resetStatistics();

    /**
     * @brief Check if predictor has sufficient history for prediction
     * @param required_samples Number of samples needed for prediction
     * @return True if sufficient history available
     */
    bool hasSufficientHistory(uint32_t required_samples) const;

    /**
     * @brief Get current history buffer size
     * @return Number of samples in history buffer
     */
    uint32_t getHistorySize() const;

    /**
     * @brief Analyze audio characteristics of current history
     * @return Analysis results
     */
    AnalysisState analyzeCurrentHistory() const;

    /**
     * @brief Estimate confidence for predicting given number of samples
     * @param num_samples Number of samples to predict
     * @return Estimated confidence (0.0 to 1.0)
     */
    double estimateConfidence(uint32_t num_samples) const;

    /**
     * @brief Available prediction methods
     */
    enum class Method {
        AUTO,              // Automatically select best method
        LINEAR_PREDICTION, // Linear predictive coding
        HARMONIC_SYNTHESIS,// Harmonic continuation
        PATTERN_REPETITION,// Pattern-based repetition
        SPECTRAL_MATCHING, // Spectral domain prediction
        ZERO_PADDING,      // Simple zero padding (fallback)
        FADE_TO_SILENCE    // Gradual fade to silence
    };

    /**
     * @brief Get available prediction methods
     * @return Vector of method names
     */
    static std::vector<std::string> getAvailableMethods();

    /**
     * @brief Convert method enum to string
     * @param method Method enum
     * @return Method name string
     */
    static std::string methodToString(Method method);

    /**
     * @brief Convert string to method enum
     * @param method_str Method name string
     * @return Method enum
     */
    static Method stringToMethod(const std::string& method_str);

private:
    /**
     * @brief Linear prediction using LPC coefficients
     * @param num_samples Number of samples to predict
     * @return Predicted samples
     */
    std::vector<float> linearPredict(uint32_t num_samples);

    /**
     * @brief Harmonic synthesis prediction
     * @param num_samples Number of samples to predict
     * @return Predicted samples
     */
    std::vector<float> harmonicSynthesize(uint32_t num_samples);

    /**
     * @brief Pattern repetition prediction
     * @param num_samples Number of samples to predict
     * @return Predicted samples
     */
    std::vector<float> patternRepeat(uint32_t num_samples);

    /**
     * @brief Spectral domain prediction
     * @param num_samples Number of samples to predict
     * @return Predicted samples
     */
    std::vector<float> spectralPredict(uint32_t num_samples);

    /**
     * @brief Zero padding fallback
     * @param num_samples Number of samples to generate
     * @return Zero samples
     */
    std::vector<float> zeroPad(uint32_t num_samples);

    /**
     * @brief Fade to silence
     * @param num_samples Number of samples to generate
     * @return Faded samples
     */
    std::vector<float> fadeToSilence(uint32_t num_samples);

    /**
     * @brief Perform autocorrelation analysis
     * @param samples Input samples
     * @return Autocorrelation coefficients
     */
    std::vector<float> computeAutocorrelation(const std::vector<float>& samples);

    /**
     * @brief Find fundamental frequency
     * @param samples Input samples
     * @return Fundamental frequency in Hz
     */
    double findFundamentalFrequency(const std::vector<float>& samples);

    /**
     * @brief Detect periodicity in signal
     * @param samples Input samples
     * @return Period in samples (0 if not periodic)
     */
    uint32_t detectPeriod(const std::vector<float>& samples);

    /**
     * @brief Calculate RMS energy
     * @param samples Input samples
     * @return RMS energy level
     */
    double calculateRMSEnergy(const std::vector<float>& samples);

    /**
     * @brief Apply window function to samples
     * @param samples Input samples
     * @param window_type Window type ("hann", "hamming", etc.)
     * @return Windowed samples
     */
    std::vector<float> applyWindow(const std::vector<float>& samples, 
                                  const std::string& window_type = "hann");

    /**
     * @brief Optimize for zero crossings
     * @param samples Input samples to optimize
     * @return Optimized samples
     */
    std::vector<float> optimizeZeroCrossings(const std::vector<float>& samples);

    /**
     * @brief Select best prediction method based on analysis
     * @param num_samples Number of samples to predict
     * @return Best method for current signal characteristics
     */
    Method selectBestMethod(uint32_t num_samples) const;

    /**
     * @brief Calculate prediction quality metrics
     * @param predicted_samples Predicted samples
     * @param method Method used
     * @return Quality metrics
     */
    PredictionQuality calculateQuality(const std::vector<float>& predicted_samples, 
                                     Method method) const;

    /**
     * @brief Update statistics after prediction
     * @param result Prediction result
     */
    void updateStatistics(const PredictionResult& result);

    /**
     * @brief Ensure history buffer doesn't exceed maximum size
     */
    void maintainHistoryBuffer();
};

/**
 * @brief Create a default waveform predictor
 * @param sample_rate Sample rate in Hz
 * @return Unique pointer to WaveformPredictor
 */
std::unique_ptr<WaveformPredictor> createDefaultPredictor(uint32_t sample_rate = 96000);

/**
 * @brief Create a high-quality waveform predictor
 * @param sample_rate Sample rate in Hz
 * @return Unique pointer to WaveformPredictor configured for high quality
 */
std::unique_ptr<WaveformPredictor> createHighQualityPredictor(uint32_t sample_rate = 96000);

} // namespace jdat 