#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

namespace jdat {

/**
 * @brief Information about packet loss events for PNBTR recovery
 */
struct PacketLossInfo {
    bool has_packet_loss = false;
    uint64_t sequence_number = 0;
    uint64_t timestamp_us = 0;
    float gap_duration_ms = 0.0f;
    uint8_t lost_streams = 0;  // Bitmask of lost streams
};

/**
 * @brief Bridge between PNBTR Framework and JDAT Framework
 * 
 * This class integrates PNBTR (Predictive Neural Buffered Transient Recovery)
 * with JDAT audio streaming for VST3 plugin development. It provides:
 * 
 * - Packet loss recovery using PNBTR neural prediction
 * - Audio quality enhancement via LSB reconstruction  
 * - Continuous learning from audio streams
 * - Real-time performance monitoring
 */
class PNBTR_JDAT_Bridge {
public:
    /**
     * @brief Performance and quality statistics
     */
    struct Statistics {
        uint64_t total_predictions = 0;
        uint64_t successful_predictions = 0;
        uint64_t packet_loss_events = 0;
        double average_prediction_time_ms = 0.0;
        double prediction_accuracy = 0.0;
        double packet_loss_recovery_rate = 0.0;
    };

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;

public:
    /**
     * @brief Constructor
     */
    PNBTR_JDAT_Bridge();

    /**
     * @brief Destructor
     */
    ~PNBTR_JDAT_Bridge();

    /**
     * @brief Initialize the bridge with PNBTR framework
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Shutdown the bridge and cleanup resources
     */
    void shutdown();

    /**
     * @brief Enhance JELLIE decoded audio using PNBTR
     * 
     * This is the main integration point between JELLIE decoder and PNBTR.
     * Uses PNBTR for packet loss recovery or quality enhancement.
     * 
     * @param decoded_audio Raw decoded audio from JELLIE
     * @param stream_availability Map of stream IDs to availability
     * @param sample_rate Audio sample rate
     * @param loss_info Information about packet loss
     * @return Enhanced/recovered audio
     */
    std::vector<float> enhanceJELLIEDecoding(
        const std::vector<float>& decoded_audio,
        const std::map<uint8_t, bool>& stream_availability,
        uint32_t sample_rate,
        PacketLossInfo loss_info = {}
    );

    /**
     * @brief Submit learning data to PNBTR for continuous improvement
     * @param pnbtr_output PNBTR processed audio
     * @param reference_audio Reference/ground truth audio
     */
    void submitLearningData(
        const std::vector<float>& pnbtr_output,
        const std::vector<float>& reference_audio
    );

    /**
     * @brief Get performance and quality statistics
     * @return Current statistics
     */
    Statistics getStatistics() const;

    /**
     * @brief Enable/disable PNBTR processing
     * @param enable True to enable PNBTR
     * @return True if successful
     */
    bool enablePNBTR(bool enable);

    /**
     * @brief Enable/disable continuous learning
     * @param enable True to enable learning
     * @return True if successful
     */
    bool enableLearning(bool enable);

private:
    // Make Impl a friend to access private members
    friend class Impl;

    /**
     * @brief Generate unique session ID
     */
    std::string generateSessionId();

    /**
     * @brief Get current timestamp in microseconds
     */
    uint64_t getCurrentTimestamp();
};

/**
 * @brief Create PNBTR-JDAT bridge with default settings
 * @return Initialized bridge or nullptr on failure
 */
std::unique_ptr<PNBTR_JDAT_Bridge> createPNBTRJDATBridge();

/**
 * @brief Create PNBTR-JDAT bridge configured for VST3 plugin
 * @param sample_rate VST3 host sample rate
 * @param bit_depth Audio bit depth (16, 24, 32)
 * @param channels Number of audio channels
 * @return Initialized bridge or nullptr on failure
 */
std::unique_ptr<PNBTR_JDAT_Bridge> createPNBTRJDATBridgeForVST3(
    uint32_t sample_rate = 96000,
    uint16_t bit_depth = 24,
    uint16_t channels = 2
);

} // namespace jdat 