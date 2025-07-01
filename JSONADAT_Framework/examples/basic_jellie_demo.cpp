#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"
#include "JSONADATMessage.h"

using namespace jsonadat;

/**
 * @brief Generate a simple sine wave for testing
 * @param frequency Frequency in Hz
 * @param sample_rate Sample rate in Hz
 * @param duration_ms Duration in milliseconds
 * @return Vector of audio samples
 */
std::vector<float> generateSineWave(double frequency, uint32_t sample_rate, uint32_t duration_ms) {
    const uint32_t num_samples = (sample_rate * duration_ms) / 1000;
    std::vector<float> samples(num_samples);
    
    const double phase_increment = 2.0 * M_PI * frequency / sample_rate;
    
    for (uint32_t i = 0; i < num_samples; ++i) {
        samples[i] = static_cast<float>(0.5 * std::sin(i * phase_increment));
    }
    
    return samples;
}

/**
 * @brief Basic JELLIE demonstration
 */
int main() {
    std::cout << "JSONADAT JELLIE Basic Demo\n";
    std::cout << "==========================\n\n";

    // Configuration
    JELLIEEncoder::Config encoder_config;
    encoder_config.sample_rate = SampleRate::SR_96000;
    encoder_config.quality = AudioQuality::HIGH_PRECISION;
    encoder_config.frame_size_samples = 960;  // 10ms at 96kHz
    encoder_config.redundancy_level = 1;
    encoder_config.enable_192k_mode = false;
    encoder_config.session_id = "demo-session-001";

    JELLIEDecoder::Config decoder_config;
    decoder_config.expected_sample_rate = SampleRate::SR_96000;
    decoder_config.quality = AudioQuality::HIGH_PRECISION;
    decoder_config.buffer_size_ms = 50;
    decoder_config.enable_pntbtr = true;

    // Create encoder and decoder
    auto encoder = std::make_unique<JELLIEEncoder>(encoder_config);
    auto decoder = std::make_unique<JELLIEDecoder>(decoder_config);

    // Message transport simulation
    std::vector<JSONADATMessage> message_queue;
    std::mutex queue_mutex;

    // Set up encoder callback to capture messages
    encoder->setMessageCallback([&](const JSONADATMessage& message) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        message_queue.push_back(message);
        std::cout << "Encoded message: seq=" << message.getSequenceNumber() 
                  << ", samples=" << (message.getAudioData() ? message.getAudioData()->samples.size() : 0)
                  << std::endl;
    });

    // Set up decoder callback to capture output
    uint32_t samples_received = 0;
    decoder->setAudioOutputCallback([&](const std::vector<float>& samples, uint64_t timestamp) {
        samples_received += samples.size();
        std::cout << "Decoded " << samples.size() << " samples (total: " << samples_received << ")" << std::endl;
    });

    // Start encoder and decoder
    if (!encoder->start()) {
        std::cerr << "Failed to start encoder" << std::endl;
        return 1;
    }

    if (!decoder->start()) {
        std::cerr << "Failed to start decoder" << std::endl;
        return 1;
    }

    std::cout << "Started JELLIE encoder and decoder\n\n";

    // Generate test audio (440Hz sine wave for 1 second)
    std::cout << "Generating test audio (440Hz sine wave)...\n";
    auto test_audio = generateSineWave(440.0, 96000, 1000);
    std::cout << "Generated " << test_audio.size() << " samples\n\n";

    // Process audio in chunks
    const uint32_t chunk_size = encoder_config.frame_size_samples;
    uint32_t samples_processed = 0;

    std::cout << "Processing audio through JELLIE...\n";
    
    for (size_t offset = 0; offset < test_audio.size(); offset += chunk_size) {
        const size_t remaining = test_audio.size() - offset;
        const size_t current_chunk_size = std::min(static_cast<size_t>(chunk_size), remaining);
        
        std::vector<float> chunk(test_audio.begin() + offset, 
                               test_audio.begin() + offset + current_chunk_size);
        
        // Encode the chunk
        if (!encoder->processAudio(chunk)) {
            std::cerr << "Failed to process audio chunk" << std::endl;
            break;
        }
        
        samples_processed += current_chunk_size;
        
        // Process any queued messages through decoder
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            for (const auto& message : message_queue) {
                if (!decoder->processMessage(message)) {
                    std::cerr << "Failed to process message in decoder" << std::endl;
                }
            }
            message_queue.clear();
        }
        
        // Small delay to simulate real-time processing
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "\nProcessed " << samples_processed << " samples total\n";

    // Flush any remaining data
    encoder->flush();
    
    // Process final messages
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (const auto& message : message_queue) {
            decoder->processMessage(message);
        }
    }

    decoder->flush();

    // Get statistics
    auto encoder_stats = encoder->getStatistics();
    auto decoder_stats = decoder->getStatistics();

    std::cout << "\nEncoder Statistics:\n";
    std::cout << "  Messages sent: " << encoder_stats.messages_sent << std::endl;
    std::cout << "  Samples processed: " << encoder_stats.samples_processed << std::endl;
    std::cout << "  Encoding errors: " << encoder_stats.encoding_errors << std::endl;
    std::cout << "  Average encoding time: " << encoder_stats.average_encoding_time_us << " μs" << std::endl;

    std::cout << "\nDecoder Statistics:\n";
    std::cout << "  Messages received: " << decoder_stats.messages_received << std::endl;
    std::cout << "  Messages processed: " << decoder_stats.messages_processed << std::endl;
    std::cout << "  Packets lost: " << decoder_stats.packets_lost << std::endl;
    std::cout << "  Samples output: " << decoder_stats.samples_output << std::endl;
    std::cout << "  Average decoding time: " << decoder_stats.average_decoding_time_us << " μs" << std::endl;

    // Stop components
    encoder->stop();
    decoder->stop();

    std::cout << "\nDemo completed successfully!\n";
    return 0;
} 