#pragma once

#include "JSONADATMessage.h"
#include <vector>
#include <array>
#include <memory>
#include <atomic>
#include <mutex>

namespace jsonadat {

/**
 * @brief ADAT Lightpipe simulator for implementing the 192k strategy
 * 
 * This class simulates ADAT's 4-channel capability to implement the innovative
 * 192kHz strategy using interleaved streams and redundancy.
 */
class ADATSimulator {
public:
    /**
     * @brief ADAT configuration
     */
    struct Config {
        SampleRate base_sample_rate = SampleRate::SR_96000;  // Base rate for each stream
        bool enable_192k_mode = true;     // Enable 192k simulation
        uint8_t redundancy_streams = 2;   // Number of redundancy streams (0-2)
        bool enable_parity = true;        // Enable parity calculation
        uint32_t sync_word = 0x7E7E7E7E;  // ADAT sync word
        bool enable_error_correction = true;
    };

    /**
     * @brief ADAT channel state
     */
    struct ChannelState {
        bool is_active = false;
        bool is_interleaved = false;
        uint32_t offset_samples = 0;
        std::vector<float> buffer;
        uint64_t last_timestamp_us = 0;
        uint32_t sync_errors = 0;
    };

    /**
     * @brief ADAT frame structure (4 channels)
     */
    struct ADATFrame {
        std::array<std::vector<float>, 4> channels;  // 4 ADAT channels
        uint32_t sync_word = 0;
        uint64_t timestamp_us = 0;
        uint32_t frame_id = 0;
        bool is_valid = true;
        
        ADATFrame() {
            for (auto& channel : channels) {
                channel.clear();
            }
        }
    };

    /**
     * @brief Reconstruction result for 192k mode
     */
    struct ReconstructionResult {
        std::vector<float> reconstructed_samples;  // 192k reconstructed audio
        double confidence_score = 0.0;            // Reconstruction confidence
        uint32_t errors_corrected = 0;            // Number of errors corrected
        uint32_t channels_used = 0;               // Number of channels used
        bool reconstruction_successful = true;
    };

private:
    Config config_;
    std::array<ChannelState, 4> channel_states_;
    std::atomic<uint32_t> frame_counter_{0};
    std::atomic<bool> is_synchronized_{false};
    
    // Sync and timing
    uint64_t sync_reference_time_us_ = 0;
    uint32_t samples_per_frame_ = 0;
    uint64_t frame_duration_us_ = 0;
    
    // Error correction state
    struct ErrorCorrectionState {
        std::vector<float> parity_buffer;
        uint32_t correction_events = 0;
        uint32_t unrecoverable_errors = 0;
    };
    
    ErrorCorrectionState error_correction_;

public:
    /**
     * @brief Constructor
     * @param config ADAT configuration
     */
    explicit ADATSimulator(const Config& config);

    /**
     * @brief Destructor
     */
    ~ADATSimulator();

    /**
     * @brief Initialize the ADAT simulator
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Encode mono audio into 4-channel ADAT frame for 192k strategy
     * @param mono_samples Input mono audio samples
     * @param timestamp_us Sample timestamp
     * @return ADAT frame with interleaved and redundancy streams
     */
    ADATFrame encodeTo192k(const std::vector<float>& mono_samples, uint64_t timestamp_us);

    /**
     * @brief Decode ADAT frame back to 192k mono audio
     * @param adat_frame Input ADAT frame
     * @return Reconstruction result with 192k audio
     */
    ReconstructionResult decodeFrom192k(const ADATFrame& adat_frame);

    /**
     * @brief Process individual channel data
     * @param channel_id Channel identifier (0-3)
     * @param samples Channel audio samples
     * @param timestamp_us Sample timestamp
     * @return True if processed successfully
     */
    bool processChannelData(uint8_t channel_id, 
                          const std::vector<float>& samples, 
                          uint64_t timestamp_us);

    /**
     * @brief Reconstruct 192k audio from available channels
     * @return Reconstruction result
     */
    ReconstructionResult reconstruct192kAudio();

    /**
     * @brief Check if enough channels are available for reconstruction
     * @return True if reconstruction is possible
     */
    bool canReconstruct() const;

    /**
     * @brief Get current channel states
     * @return Array of channel states
     */
    const std::array<ChannelState, 4>& getChannelStates() const { return channel_states_; }

    /**
     * @brief Get channel state for specific channel
     * @param channel_id Channel identifier (0-3)
     * @return Channel state
     */
    const ChannelState& getChannelState(uint8_t channel_id) const;

    /**
     * @brief Reset all channel states
     */
    void resetChannels();

    /**
     * @brief Reset specific channel state
     * @param channel_id Channel identifier (0-3)
     */
    void resetChannel(uint8_t channel_id);

    /**
     * @brief Check synchronization status
     * @return True if synchronized
     */
    bool isSynchronized() const { return is_synchronized_.load(); }

    /**
     * @brief Force synchronization
     * @param reference_time_us Reference timestamp for sync
     * @return True if sync successful
     */
    bool forceSynchronization(uint64_t reference_time_us);

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
     * @brief Statistics for ADAT simulation
     */
    struct Statistics {
        uint64_t frames_encoded = 0;
        uint64_t frames_decoded = 0;
        uint64_t sync_errors = 0;
        uint64_t reconstruction_errors = 0;
        uint64_t error_corrections = 0;
        double average_confidence = 0.0;
        uint32_t active_channels = 0;
        double reconstruction_success_rate = 0.0;
    };

    /**
     * @brief Get current statistics
     * @return ADAT simulation statistics
     */
    Statistics getStatistics() const;

    /**
     * @brief Reset statistics
     */
    void resetStatistics();

    /**
     * @brief Create JSONADAT messages from ADAT frame
     * @param adat_frame ADAT frame to convert
     * @param session_id Session identifier
     * @param sequence_base Base sequence number
     * @return Vector of JSONADAT messages (one per active channel)
     */
    std::vector<JSONADATMessage> createJSONADATMessages(const ADATFrame& adat_frame,
                                                        const std::string& session_id,
                                                        uint64_t sequence_base);

    /**
     * @brief Reconstruct ADAT frame from JSONADAT messages
     * @param messages Vector of JSONADAT messages
     * @return Reconstructed ADAT frame
     */
    ADATFrame reconstructFromJSONADAT(const std::vector<JSONADATMessage>& messages);

    /**
     * @brief Calculate optimal interleaving strategy
     * @param target_rate Target sample rate
     * @param base_rate Base sample rate per stream
     * @return Optimal offset values for each stream
     */
    static std::array<uint32_t, 4> calculateOptimalOffsets(SampleRate target_rate, 
                                                           SampleRate base_rate);

    /**
     * @brief Generate sync word for ADAT frame
     * @param frame_id Frame identifier
     * @return Generated sync word
     */
    static uint32_t generateSyncWord(uint32_t frame_id);

    /**
     * @brief Validate sync word
     * @param sync_word Sync word to validate
     * @param expected_frame_id Expected frame ID
     * @return True if sync word is valid
     */
    static bool validateSyncWord(uint32_t sync_word, uint32_t expected_frame_id);

private:
    /**
     * @brief Calculate interleaved samples for 192k mode
     * @param mono_samples Input mono samples
     * @param stream_id Stream identifier (0-1 for interleaved streams)
     * @return Interleaved samples for this stream
     */
    std::vector<float> calculateInterleavedSamples(const std::vector<float>& mono_samples,
                                                  uint8_t stream_id);

    /**
     * @brief Generate redundancy data
     * @param primary_samples Primary audio samples
     * @param secondary_samples Secondary audio samples
     * @param redundancy_type Type of redundancy (parity, etc.)
     * @return Redundancy data
     */
    std::vector<float> generateRedundancy(const std::vector<float>& primary_samples,
                                         const std::vector<float>& secondary_samples,
                                         uint8_t redundancy_type);

    /**
     * @brief Merge interleaved streams back to 192k
     * @param stream1 First interleaved stream
     * @param stream2 Second interleaved stream
     * @return Merged 192k samples
     */
    std::vector<float> mergeInterleavedStreams(const std::vector<float>& stream1,
                                              const std::vector<float>& stream2);

    /**
     * @brief Perform error correction using redundancy data
     * @param primary_data Primary channel data
     * @param redundancy_data Redundancy channel data
     * @return Error-corrected data
     */
    std::vector<float> performErrorCorrection(const std::vector<float>& primary_data,
                                             const std::vector<float>& redundancy_data);

    /**
     * @brief Calculate reconstruction confidence
     * @param channels_available Number of channels available
     * @param sync_quality Synchronization quality
     * @return Confidence score (0.0 to 1.0)
     */
    double calculateConfidence(uint32_t channels_available, double sync_quality) const;

    /**
     * @brief Update frame timing parameters
     */
    void updateFrameTiming();

    /**
     * @brief Check frame synchronization
     * @param frame ADAT frame to check
     * @return True if frame is properly synchronized
     */
    bool checkFrameSync(const ADATFrame& frame);

    // Statistics tracking
    mutable std::mutex stats_mutex_;
    Statistics stats_;
};

/**
 * @brief Create default ADAT simulator for 192k mode
 * @param base_sample_rate Base sample rate for streams
 * @return Unique pointer to ADATSimulator
 */
std::unique_ptr<ADATSimulator> createDefault192kSimulator(SampleRate base_sample_rate = SampleRate::SR_96000);

/**
 * @brief Create ADAT simulator with maximum redundancy
 * @param base_sample_rate Base sample rate for streams
 * @return Unique pointer to ADATSimulator with full redundancy
 */
std::unique_ptr<ADATSimulator> createMaxRedundancySimulator(SampleRate base_sample_rate = SampleRate::SR_96000);

} // namespace jsonadat 