#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "ADATSimulator.h"
#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"

using namespace jsonadat;

/**
 * @brief Generate a complex test signal with multiple frequencies
 * @param sample_rate Sample rate in Hz
 * @param duration_ms Duration in milliseconds
 * @return Vector of audio samples
 */
std::vector<float> generateComplexSignal(uint32_t sample_rate, uint32_t duration_ms) {
    const uint32_t num_samples = (sample_rate * duration_ms) / 1000;
    std::vector<float> samples(num_samples);
    
    // Multiple frequency components to test 192k reconstruction
    const std::vector<double> frequencies = {440.0, 880.0, 1760.0, 3520.0};
    const std::vector<double> amplitudes = {0.3, 0.2, 0.15, 0.1};
    
    for (uint32_t i = 0; i < num_samples; ++i) {
        double sample = 0.0;
        for (size_t f = 0; f < frequencies.size(); ++f) {
            const double phase = 2.0 * M_PI * frequencies[f] * i / sample_rate;
            sample += amplitudes[f] * std::sin(phase);
        }
        samples[i] = static_cast<float>(sample);
    }
    
    return samples;
}

/**
 * @brief Demonstrate 192k ADAT simulation
 */
int main() {
    std::cout << "JSONADAT 192k ADAT Simulation Demo\n";
    std::cout << "===================================\n\n";

    // Create ADAT simulator for 192k mode
    ADATSimulator::Config adat_config;
    adat_config.base_sample_rate = SampleRate::SR_96000;
    adat_config.enable_192k_mode = true;
    adat_config.redundancy_streams = 2;
    adat_config.enable_parity = true;
    adat_config.enable_error_correction = true;

    auto adat_simulator = std::make_unique<ADATSimulator>(adat_config);
    
    if (!adat_simulator->initialize()) {
        std::cerr << "Failed to initialize ADAT simulator" << std::endl;
        return 1;
    }

    std::cout << "Initialized ADAT simulator for 192k mode\n";

    // Configure JELLIE for 192k mode
    JELLIEEncoder::Config encoder_config;
    encoder_config.sample_rate = SampleRate::SR_96000;  // Base rate
    encoder_config.quality = AudioQuality::HIGH_PRECISION;
    encoder_config.frame_size_samples = 480;  // 5ms at 96kHz
    encoder_config.enable_192k_mode = true;
    encoder_config.enable_adat_mapping = true;
    encoder_config.redundancy_level = 2;
    encoder_config.session_id = "adat-192k-demo";

    JELLIEDecoder::Config decoder_config;
    decoder_config.expected_sample_rate = SampleRate::SR_96000;
    decoder_config.quality = AudioQuality::HIGH_PRECISION;
    decoder_config.expect_192k_mode = true;
    decoder_config.expect_adat_mapping = true;
    decoder_config.enable_pntbtr = true;
    decoder_config.max_recovery_gap_ms = 10;

    auto encoder = std::make_unique<JELLIEEncoder>(encoder_config);
    auto decoder = std::make_unique<JELLIEDecoder>(decoder_config);

    std::cout << "Created JELLIE encoder/decoder for 192k mode\n";

    // Message queues for each ADAT channel
    std::array<std::vector<JSONADATMessage>, 4> channel_queues;

    // Set up encoder callback
    encoder->setMessageCallback([&](const JSONADATMessage& message) {
        const auto* audio_data = message.getAudioData();
        if (audio_data) {
            const uint8_t channel = audio_data->channel;
            if (channel < 4) {
                channel_queues[channel].push_back(message);
                std::cout << "Encoded to ADAT channel " << static_cast<int>(channel) 
                          << ": " << audio_data->samples.size() << " samples" << std::endl;
            }
        }
    });

    // Set up decoder callback
    std::vector<float> reconstructed_audio;
    decoder->setAudioOutputCallback([&](const std::vector<float>& samples, uint64_t timestamp) {
        reconstructed_audio.insert(reconstructed_audio.end(), samples.begin(), samples.end());
        std::cout << "Reconstructed " << samples.size() << " samples (192k)" << std::endl;
    });

    // Start components
    if (!encoder->start() || !decoder->start()) {
        std::cerr << "Failed to start encoder/decoder" << std::endl;
        return 1;
    }

    // Generate test signal (complex waveform for 192k testing)
    std::cout << "\nGenerating complex test signal...\n";
    auto test_signal = generateComplexSignal(192000, 500);  // 0.5 seconds at 192k
    std::cout << "Generated " << test_signal.size() << " samples at 192kHz\n\n";

    // Process signal through ADAT 192k strategy
    std::cout << "Processing through ADAT 192k strategy...\n";
    
    const uint32_t chunk_size = 960;  // 10ms at 96kHz base rate
    uint64_t timestamp = 0;
    
    for (size_t offset = 0; offset < test_signal.size(); offset += chunk_size * 2) {
        // Take chunk at 192k rate (double the base rate samples)
        const size_t remaining = test_signal.size() - offset;
        const size_t current_chunk_size = std::min(static_cast<size_t>(chunk_size * 2), remaining);
        
        std::vector<float> chunk(test_signal.begin() + offset,
                               test_signal.begin() + offset + current_chunk_size);
        
        // Encode to ADAT frame
        auto adat_frame = adat_simulator->encodeTo192k(chunk, timestamp);
        
        if (!adat_frame.is_valid) {
            std::cerr << "Failed to encode ADAT frame" << std::endl;
            continue;
        }
        
        std::cout << "ADAT frame " << adat_frame.frame_id << " encoded with " 
                  << adat_frame.channels.size() << " channels\n";
        
        // Convert ADAT frame to JSONADAT messages
        auto messages = adat_simulator->createJSONADATMessages(adat_frame, 
                                                             encoder_config.session_id,
                                                             timestamp);
        
        // Process messages through decoder
        for (const auto& message : messages) {
            if (!decoder->processMessage(message)) {
                std::cerr << "Failed to process JSONADAT message" << std::endl;
            }
        }
        
        timestamp += current_chunk_size;
        
        // Simulate network delay
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    std::cout << "\nFlushing remaining data...\n";
    encoder->flush();
    decoder->flush();

    // Wait for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Analyze results
    std::cout << "\nAnalysis Results:\n";
    std::cout << "Original signal: " << test_signal.size() << " samples\n";
    std::cout << "Reconstructed signal: " << reconstructed_audio.size() << " samples\n";
    
    // Calculate reconstruction ratio
    const double reconstruction_ratio = static_cast<double>(reconstructed_audio.size()) / 
                                      static_cast<double>(test_signal.size());
    std::cout << "Reconstruction ratio: " << (reconstruction_ratio * 100.0) << "%\n";

    // Get ADAT simulator statistics
    auto adat_stats = adat_simulator->getStatistics();
    std::cout << "\nADAT Simulator Statistics:\n";
    std::cout << "  Frames encoded: " << adat_stats.frames_encoded << std::endl;
    std::cout << "  Frames decoded: " << adat_stats.frames_decoded << std::endl;
    std::cout << "  Sync errors: " << adat_stats.sync_errors << std::endl;
    std::cout << "  Error corrections: " << adat_stats.error_corrections << std::endl;
    std::cout << "  Average confidence: " << (adat_stats.average_confidence * 100.0) << "%" << std::endl;
    std::cout << "  Reconstruction success rate: " << (adat_stats.reconstruction_success_rate * 100.0) << "%" << std::endl;

    // Get encoder/decoder statistics
    auto encoder_stats = encoder->getStatistics();
    auto decoder_stats = decoder->getStatistics();

    std::cout << "\nJELLIE Statistics:\n";
    std::cout << "  Encoder messages sent: " << encoder_stats.messages_sent << std::endl;
    std::cout << "  Decoder messages processed: " << decoder_stats.messages_processed << std::endl;
    std::cout << "  Packet loss rate: " << (decoder_stats.packet_loss_rate * 100.0) << "%" << std::endl;

    // Stop components
    encoder->stop();
    decoder->stop();

    std::cout << "\n192k ADAT demo completed successfully!\n";
    std::cout << "The 192k strategy effectively doubles the resolution using interleaved streams.\n";
    
    return 0;
} 