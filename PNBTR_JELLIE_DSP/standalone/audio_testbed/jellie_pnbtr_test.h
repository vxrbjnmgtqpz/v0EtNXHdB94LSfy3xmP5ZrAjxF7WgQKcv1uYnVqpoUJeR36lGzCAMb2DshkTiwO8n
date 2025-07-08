#pragma once

#include "audio_testbed.h"
#include "../../../JDAT_Framework/include/JELLIEEncoder.h"
#include "../../../JDAT_Framework/include/JELLIEDecoder.h"
#include "../../../JDAT_Framework/include/PNBTR_JDAT_Bridge.h"
#include <vector>
#include <string>
#include <memory>

/**
 * @brief JELLIE + PNBTR Integration Testbed
 * 
 * Tests the complete pipeline:
 * 1. JELLIE 8-channel redundant encoding at 192kHz
 * 2. Packet loss simulation
 * 3. PNBTR reconstruction to maintain 44.1kHz+ quality
 * 4. Quality comparison and analysis
 */
class JelliePnbtrTest {
public:
    /**
     * @brief Test configuration for JELLIE + PNBTR pipeline
     */
    struct Config {
        uint32_t input_sample_rate;
        uint32_t jellie_sample_rate;
        uint16_t jellie_channels;
        uint16_t redundant_streams;
        
        // Packet loss simulation
        double packet_loss_percentage;
        double burst_loss_percentage;
        uint32_t max_burst_length_ms;
        
        // Quality targets
        uint32_t target_output_rate;
        double target_snr_db;
        bool require_no_clicks;
        
        std::string test_signal_type;
        double test_frequency_hz;
        double test_duration_sec;
        
        Config() : input_sample_rate(44100), jellie_sample_rate(192000), jellie_channels(8),
                  redundant_streams(2), packet_loss_percentage(5.0), burst_loss_percentage(15.0),
                  max_burst_length_ms(50), target_output_rate(44100), target_snr_db(60.0),
                  require_no_clicks(true), test_signal_type("sine"), test_frequency_hz(1000.0),
                  test_duration_sec(5.0) {}
    };

    /**
     * @brief Comprehensive test results
     */
    struct TestResults {
        // Quality metrics
        AudioTestbed::QualityMetrics original_quality;
        AudioTestbed::QualityMetrics jellie_only_quality;
        AudioTestbed::QualityMetrics pnbtr_enhanced_quality;
        
        // Reconstruction performance
        double packets_lost_percent = 0.0;
        double reconstruction_success_rate = 0.0;
        uint32_t clicks_detected = 0;
        uint32_t pops_detected = 0;
        
        // Timing performance
        double jellie_encoding_time_ms = 0.0;
        double packet_loss_simulation_ms = 0.0;
        double pnbtr_reconstruction_time_ms = 0.0;
        double total_processing_time_ms = 0.0;
        
        // Quality improvements
        double jellie_vs_original_db = 0.0;
        double pnbtr_vs_jellie_db = 0.0;
        double pnbtr_vs_original_db = 0.0;
        
        std::string analysis_summary;
        bool test_passed = false;
    };

private:
    Config config_;
    std::unique_ptr<JELLIEEncoder> encoder_;
    std::unique_ptr<JELLIEDecoder> decoder_;
    std::unique_ptr<PNBTR_JDAT_Bridge> pnbtr_bridge_;

public:
    /**
     * @brief Constructor
     */
    JelliePnbtrTest(const Config& config = {});

    /**
     * @brief Destructor
     */
    ~JelliePnbtrTest();

    /**
     * @brief Initialize the test system
     */
    bool initialize();

    /**
     * @brief Shutdown and cleanup
     */
    void shutdown();

    /**
     * @brief Run comprehensive JELLIE + PNBTR test suite
     */
    TestResults runComprehensiveTests();

    /**
     * @brief Test specific signal type with packet loss scenarios
     */
    TestResults testSignalWithPacketLoss(const std::string& signal_type,
                                       double packet_loss_percent);

    /**
     * @brief Generate test audio signal
     */
    std::vector<float> generateTestSignal(const std::string& type,
                                        double frequency_hz,
                                        double duration_sec,
                                        uint32_t sample_rate);

    /**
     * @brief Encode audio using JELLIE 8-channel redundancy
     */
    std::vector<std::vector<uint8_t>> encodeWithJELLIE(const std::vector<float>& audio_data);

    /**
     * @brief Simulate packet loss on JELLIE streams
     */
    std::vector<std::vector<uint8_t>> simulatePacketLoss(
        const std::vector<std::vector<uint8_t>>& jellie_streams,
        double loss_percentage,
        double burst_percentage = 0.0,
        uint32_t max_burst_ms = 0);

    /**
     * @brief Decode JELLIE streams (without PNBTR)
     */
    std::vector<float> decodeJELLIEOnly(const std::vector<std::vector<uint8_t>>& damaged_streams);

    /**
     * @brief Decode JELLIE streams with PNBTR reconstruction
     */
    std::vector<float> decodeWithPNBTRReconstruction(
        const std::vector<std::vector<uint8_t>>& damaged_streams);

    /**
     * @brief Analyze audio quality and detect clicks/pops
     */
    AudioTestbed::QualityMetrics analyzeAudioQuality(const std::vector<float>& audio_data,
                                                    uint32_t sample_rate,
                                                    uint32_t& clicks_detected,
                                                    uint32_t& pops_detected);

    /**
     * @brief Compare reconstruction methods
     */
    TestResults compareReconstructionMethods(const std::vector<float>& original_audio,
                                           const std::vector<float>& jellie_only,
                                           const std::vector<float>& pnbtr_enhanced);

    /**
     * @brief Generate detailed test report
     */
    bool generateTestReport(const TestResults& results, const std::string& output_path);

    /**
     * @brief Save audio samples for analysis
     */
    bool saveTestAudio(const std::vector<float>& original,
                      const std::vector<float>& jellie_only,
                      const std::vector<float>& pnbtr_enhanced,
                      const std::string& test_name);

private:
    /**
     * @brief Create 8-channel redundant encoding strategy
     */
    void setupJELLIERedundancy();

    /**
     * @brief Detect audio clicks and pops
     */
    uint32_t detectClicksAndPops(const std::vector<float>& audio_data,
                                uint32_t sample_rate,
                                uint32_t& clicks,
                                uint32_t& pops);

    /**
     * @brief Calculate reconstruction success rate
     */
    double calculateReconstructionRate(const std::vector<std::vector<uint8_t>>& original_streams,
                                     const std::vector<std::vector<uint8_t>>& damaged_streams);

    /**
     * @brief Upsample audio to JELLIE rate
     */
    std::vector<float> upsampleToJELLIERate(const std::vector<float>& input_audio,
                                          uint32_t input_rate,
                                          uint32_t output_rate);

    /**
     * @brief Downsample from JELLIE rate  
     */
    std::vector<float> downsampleFromJELLIERate(const std::vector<float>& jellie_audio,
                                              uint32_t jellie_rate,
                                              uint32_t target_rate);
}; 