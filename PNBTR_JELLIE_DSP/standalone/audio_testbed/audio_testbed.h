#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "MetalBridge.h"
#include "RingBuffer.h"
#include "audio_quality_analyzer.h"

// Forward declarations
class AudioQualityAnalyzer;

namespace pnbtr {
    class PNBTRFramework;
    struct AudioBuffer;
    struct AudioContext;
}

namespace jdat {
    class PNBTR_JDAT_Bridge;
}

/**
 * @brief Audio Testbed for PNBTR dither replacement validation
 * 
 * This testbed validates the revolutionary PNBTR dither replacement technology
 * by comparing traditional dithering with mathematical LSB reconstruction.
 * 
 * NEW: Includes JELLIE 8-channel encoding with PNBTR packet loss recovery
 */
class AudioTestbed {
public:
    /**
     * @brief Audio processing configuration
     */
    struct Config {
        uint32_t sample_rate;
        uint16_t bit_depth;
        uint16_t channels;
        bool enable_pnbtr;
        bool enable_traditional_dither;
        bool enable_quality_analysis;
        std::string output_directory;
        std::string report_directory;
        
        Config() : sample_rate(96000), bit_depth(24), channels(2), 
                  enable_pnbtr(true), enable_traditional_dither(true), 
                  enable_quality_analysis(true), output_directory("output/"), 
                  report_directory("reports/") {}
    };

    /**
     * @brief JELLIE + PNBTR test configuration
     */
    struct JellieConfig {
        uint32_t input_sample_rate;
        uint32_t jellie_sample_rate;     // 192kHz oversampling
        uint16_t jdat_channels;          // 8 channels for redundancy
        double packet_loss_percentage;   // Packet loss simulation
        double burst_loss_percentage;    // Burst packet loss
        uint32_t max_burst_length_ms;    // Maximum burst duration
        
        JellieConfig() : input_sample_rate(44100), jellie_sample_rate(192000), 
                        jdat_channels(8), packet_loss_percentage(5.0),
                        burst_loss_percentage(15.0), max_burst_length_ms(50) {}
    };

    /**
     * @brief Quality analysis results
     */
    using QualityMetrics = AudioQualityAnalyzer::QualityMetrics;

    /**
     * @brief Comparison results between processing methods
     */
    struct ComparisonResult {
        QualityMetrics original;
        QualityMetrics traditional_dither;
        QualityMetrics pnbtr_processed;
        
        double quality_improvement_db = 0.0;
        double noise_reduction_db = 0.0;
        std::string quality_analysis;
    };

    /**
     * @brief JELLIE + PNBTR integration test results
     */
    struct JellieTestResult {
        QualityMetrics original_quality;
        QualityMetrics jellie_only_quality;
        QualityMetrics pnbtr_enhanced_quality;
        
        double packets_lost_percent = 0.0;
        double reconstruction_success_rate = 0.0;
        uint32_t clicks_detected = 0;
        uint32_t pops_detected = 0;
        
        double jellie_encoding_time_ms = 0.0;
        double pnbtr_reconstruction_time_ms = 0.0;
        double total_processing_time_ms = 0.0;
        
        double jellie_vs_original_db = 0.0;
        double pnbtr_vs_jellie_db = 0.0;
        double pnbtr_vs_original_db = 0.0;
        
        std::string analysis_summary;
        bool test_passed = false;
    };

private:
    Config config_;
    std::unique_ptr<AudioQualityAnalyzer> quality_analyzer_;
    std::unique_ptr<MetalBridge> metal_bridge_;
    std::unique_ptr<RingBuffer<float>> input_ring_buffer_;
    std::unique_ptr<RingBuffer<float>> output_ring_buffer_;
    
    // Test statistics
    struct TestStats {
        uint32_t tests_run = 0;
        uint32_t tests_passed = 0;
        uint32_t tests_failed = 0;
        double average_quality_improvement = 0.0;
        double average_processing_time_ms = 0.0;
    } stats_;

public:
    /**
     * @brief Constructor
     */
    AudioTestbed(const Config& config = {});

    /**
     * @brief Destructor
     */
    ~AudioTestbed();

    /**
     * @brief Initialize the testbed
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();

    /**
     * @brief Run comprehensive audio processing tests
     * @return True if all tests passed
     */
    bool runAllTests();

    /**
     * @brief 🎯 Run JELLIE 8-channel + PNBTR integration test
     * @param jellie_config JELLIE test configuration
     * @return JELLIE test results
     */
    JellieTestResult runJelliePnbtrTest(const JellieConfig& jellie_config = {});

    /**
     * @brief Test JELLIE encoding with packet loss and PNBTR recovery
     * @param signal_type Signal type ("sine", "white_noise", "complex")
     * @param jellie_config JELLIE configuration
     * @return Test results
     */
    JellieTestResult testJellieWithPacketLoss(const std::string& signal_type,
                                            const JellieConfig& jellie_config);

    /**
     * @brief Test PNBTR dither replacement on a single audio file
     * @param input_file Path to input audio file
     * @param output_prefix Prefix for output files
     * @return Comparison results
     */
    ComparisonResult testSingleFile(const std::string& input_file, 
                                   const std::string& output_prefix = "test");

    /**
     * @brief Test PNBTR vs traditional dithering A/B comparison
     * @param audio_data Input audio samples
     * @param sample_rate Audio sample rate
     * @return Comparison results
     */
    ComparisonResult compareProcessingMethods(const std::vector<float>& audio_data,
                                            uint32_t sample_rate);

    /**
     * @brief Generate test audio signal
     * @param type Signal type ("sine", "white_noise", "complex")
     * @param frequency_hz Frequency for sine wave
     * @param duration_sec Duration in seconds
     * @param sample_rate Sample rate
     * @return Generated audio samples
     */
    std::vector<float> generateTestSignal(const std::string& type,
                                        double frequency_hz = 1000.0,
                                        double duration_sec = 5.0,
                                        uint32_t sample_rate = 44100);

    /**
     * @brief Simulate JELLIE 8-channel encoding
     * @param audio_data Input audio samples
     * @param jellie_config JELLIE configuration
     * @return 8-channel encoded streams
     */
    std::vector<std::vector<uint8_t>> simulateJellieEncoding(const std::vector<float>& audio_data,
                                                           const JellieConfig& jellie_config);

    /**
     * @brief Simulate packet loss on encoded streams
     * @param encoded_streams JELLIE encoded streams
     * @param loss_percentage Packet loss percentage
     * @param burst_percentage Burst loss percentage
     * @return Damaged streams
     */
    std::vector<std::vector<uint8_t>> simulatePacketLoss(
        const std::vector<std::vector<uint8_t>>& encoded_streams,
        double loss_percentage,
        double burst_percentage = 0.0);

    /**
     * @brief Decode JELLIE streams without PNBTR
     * @param damaged_streams Damaged JELLIE streams
     * @param jellie_config JELLIE configuration
     * @return Decoded audio
     */
    std::vector<float> decodeJellieOnly(const std::vector<std::vector<uint8_t>>& damaged_streams,
                                      const JellieConfig& jellie_config);

    /**
     * @brief Decode JELLIE streams with PNBTR reconstruction
     * @param damaged_streams Damaged JELLIE streams
     * @param jellie_config JELLIE configuration
     * @return PNBTR-enhanced audio
     */
    std::vector<float> decodeWithPnbtrReconstruction(
        const std::vector<std::vector<uint8_t>>& damaged_streams,
        const JellieConfig& jellie_config);

    /**
     * @brief Process audio with traditional dithering
     * @param audio_data Input audio samples
     * @param target_bit_depth Target bit depth for quantization
     * @return Processed audio with traditional dithering
     */
    std::vector<float> processWithTraditionalDither(const std::vector<float>& audio_data,
                                                   uint16_t target_bit_depth);

    /**
     * @brief Process audio with PNBTR dither replacement
     * @param audio_data Input audio samples
     * @param sample_rate Audio sample rate
     * @return Processed audio with PNBTR
     */
    std::vector<float> processWithPNBTR(const std::vector<float>& audio_data,
                                       uint32_t sample_rate);

    /**
     * @brief Analyze audio quality metrics
     * @param audio_data Audio samples to analyze
     * @param sample_rate Audio sample rate
     * @return Quality metrics
     */
    QualityMetrics analyzeQuality(const std::vector<float>& audio_data,
                                 uint32_t sample_rate);

    /**
     * @brief Detect clicks and pops in audio
     * @param audio_data Audio samples
     * @param sample_rate Sample rate
     * @param clicks Output click count
     * @param pops Output pop count
     * @return Total artifacts detected
     */
    uint32_t detectClicksAndPops(const std::vector<float>& audio_data,
                                uint32_t sample_rate,
                                uint32_t& clicks,
                                uint32_t& pops);

    /**
     * @brief Generate detailed quality report
     * @param result Comparison results
     * @param output_path Path for report file
     * @return True if report generated successfully
     */
    bool generateQualityReport(const ComparisonResult& result,
                              const std::string& output_path);

    /**
     * @brief Generate JELLIE test report
     * @param result JELLIE test results
     * @param output_path Path for report file
     * @return True if report generated successfully
     */
    bool generateJellieTestReport(const JellieTestResult& result,
                                const std::string& output_path);

    /**
     * @brief Get test statistics
     * @return Current test statistics
     */
    const TestStats& getStatistics() const { return stats_; }

    /**
     * @brief Test synthetic sine wave processing
     * @return True if test passed
     */
    bool testSyntheticWave();

    /**
     * @brief Test white noise processing
     * @return True if test passed
     */
    bool testWhiteNoise();

    /**
     * @brief Test complex harmonic content processing
     * @return True if test passed
     */
    bool testComplexHarmonics();

    /**
     * @brief Test bit depth reduction processing
     * @return True if test passed
     */
    bool testBitDepthReduction();

    /**
     * @brief Load test audio file
     * @param file_path Path to audio file
     * @param audio_data Output audio samples
     * @param sample_rate Output sample rate
     * @param channels Output channel count
     * @return True if loaded successfully
     */
    bool loadAudioFile(const std::string& file_path,
                      std::vector<float>& audio_data,
                      uint32_t& sample_rate,
                      uint16_t& channels);

    /**
     * @brief Save audio file
     * @param file_path Output file path
     * @param audio_data Audio samples
     * @param sample_rate Audio sample rate
     * @param channels Channel count
     * @param bit_depth Bit depth
     * @return True if saved successfully
     */
    bool saveAudioFile(const std::string& file_path,
                      const std::vector<float>& audio_data,
                      uint32_t sample_rate,
                      uint16_t channels,
                      uint16_t bit_depth = 24);

private:
    /**
     * @brief Initialize PNBTR framework
     */
    bool initializePNBTR();

    /**
     * @brief Apply traditional triangular dithering
     */
    std::vector<float> applyTriangularDither(const std::vector<float>& audio_data,
                                           uint16_t target_bits);

    /**
     * @brief Calculate SNR (Signal-to-Noise Ratio)
     */
    double calculateSNR(const std::vector<float>& signal,
                       const std::vector<float>& reference);

    /**
     * @brief Calculate THD+N (Total Harmonic Distortion + Noise)
     */
    double calculateTHDN(const std::vector<float>& audio_data,
                        uint32_t sample_rate);

    /**
     * @brief Calculate LUFS (Loudness Units Full Scale)
     */
    double calculateLUFS(const std::vector<float>& audio_data,
                        uint32_t sample_rate);



    /**
     * @brief Update test statistics
     */
    void updateStatistics(const ComparisonResult& result, double processing_time_ms);

    /**
     * @brief Upsample audio to higher sample rate
     */
    std::vector<float> upsampleAudio(const std::vector<float>& input_audio,
                                   uint32_t input_rate,
                                   uint32_t output_rate);

    /**
     * @brief Downsample audio to lower sample rate
     */
    std::vector<float> downsampleAudio(const std::vector<float>& input_audio,
                                     uint32_t input_rate,
                                     uint32_t output_rate);
}; 