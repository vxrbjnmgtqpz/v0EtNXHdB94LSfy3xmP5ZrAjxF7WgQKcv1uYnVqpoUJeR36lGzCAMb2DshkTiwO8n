#pragma once

#include <vector>
#include <string>
#include <complex>
#include <memory>
#include <map>

/**
 * @brief Comprehensive Audio Quality Analyzer for PNBTR System
 * 
 * Implements all metrics required by the 250708_093109_System_Audit.md:
 * - Frequency Response Retention Testing
 * - Dynamic Range and Distortion Measurement (THD+N)
 * - Spectral Analysis Tools
 * - Phase Linearity Testing
 * - Real-time Processing Validation
 * - Coloration Measurement ("Color %")
 */
class AudioQualityAnalyzer {
public:
    /**
     * @brief Comprehensive quality metrics structure
     */
    struct QualityMetrics {
        // Basic Metrics
        double snr_db = 0.0;                    // Signal-to-Noise Ratio
        double thd_plus_n_percent = 0.0;        // Total Harmonic Distortion + Noise
        double thd_percent = 0.0;               // THD only (without noise)
        double dynamic_range_db = 0.0;          // Dynamic range
        double noise_floor_dbfs = 0.0;          // Noise floor level
        
        // Frequency Response Metrics
        double freq_response_flatness_db = 0.0; // Maximum deviation from flat response
        double freq_response_20hz_db = 0.0;     // Response at 20 Hz
        double freq_response_20khz_db = 0.0;    // Response at 20 kHz
        double high_freq_rolloff_db = 0.0;      // High frequency rolloff
        
        // Phase Metrics
        double phase_linearity_score = 0.0;     // Phase linearity (0-1, 1=linear)
        double group_delay_variation_ms = 0.0;  // Group delay variation
        double phase_coherence = 0.0;           // Phase coherence (stereo)
        
        // Distortion & Coloration
        double coloration_percent = 0.0;        // Overall "color" percentage
        double intermodulation_distortion_db = 0.0; // IMD
        double harmonic_content_deviation = 0.0; // Harmonic structure change
        
        // Dynamic Metrics
        double transient_preservation = 0.0;    // Transient response (0-1)
        double dynamic_compression_ratio = 0.0; // Unintended compression
        double peak_to_rms_ratio_db = 0.0;      // Crest factor
        
        // Stereo Metrics (if applicable)
        double channel_crosstalk_db = 0.0;      // L/R channel separation
        double stereo_imaging_correlation = 0.0; // Stereo imaging preservation
        
        // Processing Metrics
        double latency_ms = 0.0;                // Processing latency
        double cpu_load_percent = 0.0;          // CPU utilization
        
        // Overall Quality Score
        double overall_quality_score = 0.0;     // Composite quality (0-1)
        bool meets_hifi_standards = false;      // Meets hi-fi quality standards
    };
    
    /**
     * @brief Frequency response analysis configuration
     */
    struct FrequencyAnalysisConfig {
        double start_freq_hz = 20.0;            // Start frequency for analysis
        double end_freq_hz = 20000.0;           // End frequency for analysis
        uint32_t num_points = 100;              // Number of measurement points
        double sweep_duration_sec = 5.0;        // Duration of frequency sweep
        bool logarithmic_spacing = true;        // Log or linear frequency spacing
        double tolerance_db = 1.0;              // Acceptable deviation for "flat"
    };
    
    /**
     * @brief Test signal generation options
     */
    enum class TestSignalType {
        SINE_WAVE,
        FREQUENCY_SWEEP,
        WHITE_NOISE,
        PINK_NOISE,
        MULTI_TONE,
        IMPULSE,
        CHIRP,
        COMPLEX_HARMONIC
    };

public:
    /**
     * @brief Constructor
     */
    AudioQualityAnalyzer();

    /**
     * @brief Destructor
     */
    ~AudioQualityAnalyzer();

    /**
     * @brief Analyze comprehensive audio quality
     * @param input_signal Original input signal
     * @param output_signal Processed output signal
     * @param sample_rate Audio sample rate
     * @param channels Number of channels
     * @return Comprehensive quality metrics
     */
    QualityMetrics analyzeQuality(const std::vector<float>& input_signal,
                                  const std::vector<float>& output_signal,
                                  uint32_t sample_rate,
                                  uint16_t channels = 1);

    /**
     * @brief Test frequency response retention
     * @param process_function Function that processes audio (input -> output)
     * @param sample_rate Audio sample rate
     * @param config Frequency analysis configuration
     * @return Frequency response measurements
     */
    std::vector<std::pair<double, double>> testFrequencyResponse(
        std::function<std::vector<float>(const std::vector<float>&)> process_function,
        uint32_t sample_rate,
        const FrequencyAnalysisConfig& config);
    
    std::vector<std::pair<double, double>> testFrequencyResponse(
        std::function<std::vector<float>(const std::vector<float>&)> process_function,
        uint32_t sample_rate);

    /**
     * @brief Measure phase linearity
     * @param input_signal Input signal
     * @param output_signal Output signal
     * @param sample_rate Audio sample rate
     * @return Phase linearity metrics
     */
    struct PhaseMetrics {
        double linearity_score = 0.0;          // 0-1, 1=perfectly linear
        double group_delay_variation_ms = 0.0;  // Group delay variation
        std::vector<double> phase_response;     // Phase vs frequency
        std::vector<double> group_delay;        // Group delay vs frequency
    };
    
    PhaseMetrics analyzePhaseLinearity(const std::vector<float>& input_signal,
                                       const std::vector<float>& output_signal,
                                       uint32_t sample_rate);

    /**
     * @brief Calculate Total Harmonic Distortion + Noise
     * @param signal Audio signal
     * @param sample_rate Audio sample rate
     * @param fundamental_freq Fundamental frequency (if known, 0 for auto-detect)
     * @return THD+N percentage
     */
    double calculateTHDN(const std::vector<float>& signal,
                         uint32_t sample_rate,
                         double fundamental_freq = 0.0);

    /**
     * @brief Calculate "coloration" percentage (overall signal alteration)
     * @param input_signal Original signal
     * @param output_signal Processed signal
     * @param sample_rate Audio sample rate
     * @return Coloration percentage (0% = transparent, higher = more colored)
     */
    double calculateColoration(const std::vector<float>& input_signal,
                               const std::vector<float>& output_signal,
                               uint32_t sample_rate);

    /**
     * @brief Perform spectral analysis
     * @param signal Audio signal
     * @param sample_rate Audio sample rate
     * @param window_type FFT window type ("hann", "blackman", "kaiser")
     * @return Frequency spectrum (frequency, magnitude pairs)
     */
    std::vector<std::pair<double, double>> performSpectralAnalysis(
        const std::vector<float>& signal,
        uint32_t sample_rate,
        const std::string& window_type = "hann");

    /**
     * @brief Generate test signal
     * @param type Test signal type
     * @param duration_sec Duration in seconds
     * @param sample_rate Audio sample rate
     * @param frequency_hz Frequency for sine/sweep signals
     * @param amplitude Amplitude (0.0-1.0)
     * @return Generated test signal
     */
    std::vector<float> generateTestSignal(TestSignalType type,
                                          double duration_sec,
                                          uint32_t sample_rate,
                                          double frequency_hz = 1000.0,
                                          double amplitude = 0.5);

    /**
     * @brief Measure processing latency
     * @param process_function Function to measure
     * @param input_signal Test signal
     * @return Latency in milliseconds
     */
    double measureProcessingLatency(
        std::function<std::vector<float>(const std::vector<float>&)> process_function,
        const std::vector<float>& input_signal);

    /**
     * @brief Validate real-time processing capability
     * @param process_function Function to test
     * @param sample_rate Audio sample rate
     * @param buffer_size Processing buffer size
     * @param target_latency_ms Target latency in milliseconds
     * @return True if meets real-time requirements
     */
    bool validateRealTimeProcessing(
        std::function<std::vector<float>(const std::vector<float>&)> process_function,
        uint32_t sample_rate,
        uint32_t buffer_size,
        double target_latency_ms = 10.0);

    /**
     * @brief Generate comprehensive quality report
     * @param metrics Quality metrics
     * @param test_name Test name/description
     * @return HTML formatted report
     */
    std::string generateQualityReport(const QualityMetrics& metrics,
                                      const std::string& test_name = "Audio Quality Test");

    /**
     * @brief Check if metrics meet high-fidelity standards
     * @param metrics Quality metrics to evaluate
     * @return True if meets hi-fi standards
     */
    bool meetsHiFiStandards(const QualityMetrics& metrics);

private:
    // Internal calculation methods
    double calculateSNR(const std::vector<float>& signal,
                       const std::vector<float>& noise);
    
    double calculateDynamicRange(const std::vector<float>& signal);
    
    double calculateNoiseFloor(const std::vector<float>& signal);
    
    std::vector<std::complex<double>> performFFT(const std::vector<float>& signal);
    
    std::vector<float> applyWindow(const std::vector<float>& signal,
                                   const std::string& window_type);
    
    double findFundamentalFrequency(const std::vector<float>& signal,
                                    uint32_t sample_rate);
    
    std::vector<double> calculateGroupDelay(const std::vector<std::complex<double>>& input_fft,
                                           const std::vector<std::complex<double>>& output_fft);
    
    double measureTransientPreservation(const std::vector<float>& input_signal,
                                        const std::vector<float>& output_signal);
    
    double calculateChannelCrosstalk(const std::vector<float>& left_channel,
                                    const std::vector<float>& right_channel);

    // Internal state
    struct AnalyzerState;
    std::unique_ptr<AnalyzerState> state_;
};
