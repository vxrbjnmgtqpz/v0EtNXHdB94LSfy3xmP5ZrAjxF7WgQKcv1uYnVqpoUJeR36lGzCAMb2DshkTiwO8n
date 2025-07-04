#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"
#include "ADATSimulator.h"
#include "AudioBufferManager.h"

using namespace jdat;

/**
 * @brief Comprehensive JELLIE Framework Test Suite
 * 
 * Tests all core functionality including:
 * - JELLIE encoding/decoding accuracy
 * - ADAT-inspired 4-stream redundancy
 * - Packet loss recovery
 * - PNBTR integration
 * - Performance benchmarks
 */
class JELLIETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Configure encoder
        encoder_config_.sample_rate = SampleRate::SR_96000;
        encoder_config_.quality = AudioQuality::STUDIO;
        encoder_config_.frame_size_samples = 960;  // 10ms at 96kHz
        encoder_config_.redundancy_level = 2;
        encoder_config_.session_id = "test-session";
        
        // Configure decoder
        decoder_config_.expected_sample_rate = SampleRate::SR_96000;
        decoder_config_.expected_frame_size = 960;
        decoder_config_.redundancy_level = 2;
        decoder_config_.enable_pnbtr = true;
        
        encoder_ = std::make_unique<JELLIEEncoder>(encoder_config_);
        decoder_ = std::make_unique<JELLIEDecoder>(decoder_config_);
    }
    
    void TearDown() override {
        encoder_.reset();
        decoder_.reset();
    }
    
    std::vector<float> generateTestSineWave(double frequency, uint32_t samples, double& phase) {
        std::vector<float> result(samples);
        const uint32_t sample_rate = static_cast<uint32_t>(encoder_config_.sample_rate);
        
        for (uint32_t i = 0; i < samples; ++i) {
            result[i] = 0.5f * std::sin(phase * 2.0 * M_PI * frequency / sample_rate);
            phase++;
        }
        
        return result;
    }
    
    double calculateSNR(const std::vector<float>& original, const std::vector<float>& processed) {
        if (original.size() != processed.size()) {
            return -1.0;
        }
        
        double signal_power = 0.0;
        double noise_power = 0.0;
        
        for (size_t i = 0; i < original.size(); ++i) {
            signal_power += original[i] * original[i];
            double error = original[i] - processed[i];
            noise_power += error * error;
        }
        
        if (noise_power == 0.0) return 120.0;  // Perfect reconstruction
        
        return 10.0 * std::log10(signal_power / noise_power);
    }
    
    JELLIEEncoder::Config encoder_config_;
    JELLIEDecoder::Config decoder_config_;
    std::unique_ptr<JELLIEEncoder> encoder_;
    std::unique_ptr<JELLIEDecoder> decoder_;
};

TEST_F(JELLIETest, BasicEncodingDecoding) {
    // Generate test audio
    double phase = 0.0;
    auto original_audio = generateTestSineWave(440.0, 960, phase);
    
    // Encode
    auto messages = encoder_->encodeFrame(original_audio);
    
    // Verify we get 4 messages (ADAT-inspired streams)
    EXPECT_EQ(messages.size(), 4);
    
    // Verify message structure
    for (const auto& message : messages) {
        EXPECT_EQ(message.header.version, 2);
        EXPECT_EQ(message.header.message_type, MessageType::AUDIO_DATA);
        EXPECT_EQ(message.header.sample_rate, SampleRate::SR_96000);
        EXPECT_GT(message.audio_data.num_samples, 0);
        EXPECT_EQ(message.audio_data.bit_depth, 24);
    }
    
    // Decode
    auto decoded_audio = decoder_->decodeMessages(messages);
    
    // Verify output size
    EXPECT_EQ(decoded_audio.size(), original_audio.size());
    
    // Calculate SNR (should be > 100 dB for lossless encoding)
    double snr = calculateSNR(original_audio, decoded_audio);
    EXPECT_GT(snr, 100.0) << "SNR too low: " << snr << " dB";
}

TEST_F(JELLIETest, ADATRedundancyRecovery) {
    ADATSimulator adat_sim(2);
    
    // Generate test audio
    std::vector<float> original_audio(960);
    for (size_t i = 0; i < original_audio.size(); ++i) {
        original_audio[i] = 0.5f * std::sin(i * 2.0 * M_PI * 440.0 / 96000.0);
    }
    
    // Split into streams
    auto streams = adat_sim.splitToStreams(original_audio);
    EXPECT_EQ(streams.size(), 4);
    
    // Test reconstruction with all streams
    auto reconstructed_full = adat_sim.reconstructFromStreams(streams);
    double snr_full = calculateSNR(original_audio, reconstructed_full);
    EXPECT_GT(snr_full, 100.0) << "Full reconstruction SNR: " << snr_full << " dB";
    
    // Test reconstruction with missing stream 1 (simulate packet loss)
    streams[1].clear();  // Remove odd samples stream
    auto reconstructed_partial = adat_sim.reconstructFromStreams(streams);
    double snr_partial = calculateSNR(original_audio, reconstructed_partial);
    EXPECT_GT(snr_partial, 60.0) << "Partial reconstruction SNR: " << snr_partial << " dB";
    
    // Test reconstruction with only redundancy streams
    streams[0].clear();  // Remove even samples too
    auto reconstructed_redundancy = adat_sim.reconstructFromStreams(streams);
    double snr_redundancy = calculateSNR(original_audio, reconstructed_redundancy);
    EXPECT_GT(snr_redundancy, 40.0) << "Redundancy-only SNR: " << snr_redundancy << " dB";
}

TEST_F(JELLIETest, PacketLossRecovery) {
    // Generate test audio
    double phase = 0.0;
    auto original_audio = generateTestSineWave(440.0, 960, phase);
    
    // Encode
    auto messages = encoder_->encodeFrame(original_audio);
    
    // Simulate packet loss - remove 50% of messages
    std::vector<JDATMessage> partial_messages;
    for (size_t i = 0; i < messages.size(); i += 2) {
        partial_messages.push_back(messages[i]);
    }
    
    // Decode with packet loss
    auto decoded_audio = decoder_->decodeMessages(partial_messages);
    
    // Verify output size is maintained
    EXPECT_EQ(decoded_audio.size(), original_audio.size());
    
    // Calculate SNR (should still be reasonable with redundancy)
    double snr = calculateSNR(original_audio, decoded_audio);
    EXPECT_GT(snr, 30.0) << "Packet loss recovery SNR: " << snr << " dB";
}

TEST_F(JELLIETest, AudioBufferManager) {
    AudioBufferManager::Config buffer_config;
    buffer_config.sample_rate = 96000;
    buffer_config.frame_size_samples = 960;
    buffer_config.buffer_size_ms = 20;
    
    AudioBufferManager buffer_manager(buffer_config);
    
    // Test buffer operations
    EXPECT_TRUE(buffer_manager.start());
    
    // Generate test frames
    std::vector<std::vector<float>> test_frames;
    for (int i = 0; i < 5; ++i) {
        std::vector<float> frame(960);
        for (size_t j = 0; j < frame.size(); ++j) {
            frame[j] = std::sin((i * 960 + j) * 2.0 * M_PI * 440.0 / 96000.0);
        }
        test_frames.push_back(frame);
    }
    
    // Add frames to buffer
    for (const auto& frame : test_frames) {
        EXPECT_TRUE(buffer_manager.addFrame(frame));
    }
    
    // Check buffer usage
    EXPECT_GT(buffer_manager.getAvailableFrames(), 0);
    
    // Retrieve frames
    for (size_t i = 0; i < test_frames.size(); ++i) {
        auto retrieved_frame = buffer_manager.getNextFrame();
        EXPECT_EQ(retrieved_frame.size(), test_frames[i].size());
        
        // Verify frame content
        double snr = calculateSNR(test_frames[i], retrieved_frame);
        EXPECT_GT(snr, 100.0) << "Buffer frame " << i << " SNR: " << snr << " dB";
    }
    
    buffer_manager.stop();
}

TEST_F(JELLIETest, LatencyBenchmark) {
    const int num_iterations = 1000;
    std::vector<uint64_t> encode_times, decode_times;
    
    // Generate test audio
    double phase = 0.0;
    auto test_audio = generateTestSineWave(440.0, 960, phase);
    
    for (int i = 0; i < num_iterations; ++i) {
        // Measure encoding time
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto messages = encoder_->encodeFrame(test_audio);
        auto encode_end = std::chrono::high_resolution_clock::now();
        
        auto encode_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            encode_end - encode_start
        ).count();
        encode_times.push_back(encode_duration);
        
        // Measure decoding time
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto decoded_audio = decoder_->decodeMessages(messages);
        auto decode_end = std::chrono::high_resolution_clock::now();
        
        auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            decode_end - decode_start
        ).count();
        decode_times.push_back(decode_duration);
    }
    
    // Calculate statistics
    auto avg_encode = std::accumulate(encode_times.begin(), encode_times.end(), 0ULL) / num_iterations;
    auto avg_decode = std::accumulate(decode_times.begin(), decode_times.end(), 0ULL) / num_iterations;
    
    std::cout << "Average encoding time: " << avg_encode << " μs\n";
    std::cout << "Average decoding time: " << avg_decode << " μs\n";
    std::cout << "Total round-trip time: " << (avg_encode + avg_decode) << " μs\n";
    
    // Performance targets for professional audio (10ms frame = 10000μs)
    EXPECT_LT(avg_encode, 100) << "Encoding too slow: " << avg_encode << " μs";
    EXPECT_LT(avg_decode, 100) << "Decoding too slow: " << avg_decode << " μs";
    EXPECT_LT(avg_encode + avg_decode, 200) << "Total latency too high: " << (avg_encode + avg_decode) << " μs";
}

TEST_F(JELLIETest, QualityWithDifferentSampleRates) {
    std::vector<SampleRate> sample_rates = {
        SampleRate::SR_48000,
        SampleRate::SR_96000,
        SampleRate::SR_192000
    };
    
    for (auto sr : sample_rates) {
        encoder_config_.sample_rate = sr;
        decoder_config_.expected_sample_rate = sr;
        
        // Adjust frame size for sample rate
        uint32_t base_frame_size = 480;  // 10ms at 48kHz
        uint32_t multiplier = (sr == SampleRate::SR_96000) ? 2 : 
                             (sr == SampleRate::SR_192000) ? 4 : 1;
        encoder_config_.frame_size_samples = base_frame_size * multiplier;
        decoder_config_.expected_frame_size = base_frame_size * multiplier;
        
        // Recreate encoder/decoder with new settings
        encoder_ = std::make_unique<JELLIEEncoder>(encoder_config_);
        decoder_ = std::make_unique<JELLIEDecoder>(decoder_config_);
        
        // Test encoding/decoding
        double phase = 0.0;
        auto original_audio = generateTestSineWave(1000.0, encoder_config_.frame_size_samples, phase);
        
        auto messages = encoder_->encodeFrame(original_audio);
        auto decoded_audio = decoder_->decodeMessages(messages);
        
        double snr = calculateSNR(original_audio, decoded_audio);
        
        std::cout << "Sample rate " << static_cast<uint32_t>(sr) 
                  << " Hz SNR: " << snr << " dB\n";
        
        EXPECT_GT(snr, 100.0) << "Poor quality at " << static_cast<uint32_t>(sr) << " Hz";
    }
}

TEST_F(JELLIETest, MessageIntegrity) {
    // Generate test audio
    double phase = 0.0;
    auto original_audio = generateTestSineWave(440.0, 960, phase);
    
    // Encode
    auto messages = encoder_->encodeFrame(original_audio);
    
    // Test message integrity
    for (const auto& message : messages) {
        // Verify checksum
        EXPECT_NE(message.header.checksum, 0);
        
        // Verify session ID
        EXPECT_STREQ(message.header.session_id, "test-session");
        
        // Verify timestamp is reasonable
        EXPECT_GT(message.header.timestamp_us, 0);
        
        // Verify audio data integrity
        EXPECT_EQ(message.audio_data.bit_depth, 24);
        EXPECT_FALSE(message.audio_data.is_compressed);
        EXPECT_EQ(message.audio_data.compression_ratio, 1.0f);
        EXPECT_GT(message.audio_data.num_samples, 0);
        EXPECT_EQ(message.audio_data.samples.size(), message.audio_data.num_samples);
    }
    
    // Test that corrupted messages are rejected
    auto corrupted_message = messages[0];
    corrupted_message.header.checksum = 0xDEADBEEF;  // Invalid checksum
    
    std::vector<JDATMessage> corrupted_messages = {corrupted_message};
    auto decoded_corrupted = decoder_->decodeMessages(corrupted_messages);
    
    // Should fall back to silence or PNBTR prediction
    EXPECT_EQ(decoded_corrupted.size(), original_audio.size());
}

// Main test runner
int main(int argc, char** argv) {
    std::cout << "JELLIE Framework Test Suite\n";
    std::cout << "===========================\n\n";
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
