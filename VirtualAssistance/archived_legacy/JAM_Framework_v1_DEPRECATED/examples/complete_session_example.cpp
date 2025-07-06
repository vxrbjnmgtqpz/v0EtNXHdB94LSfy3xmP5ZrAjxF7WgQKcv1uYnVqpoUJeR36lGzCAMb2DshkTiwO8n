#include "../include/jam_framework.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

using namespace jam;

int main() {
    std::cout << "JAM Framework - Complete Session Example\n";
    std::cout << "=========================================\n\n";
    
    // Initialize JAM Framework
    JAMFramework jam;
    
    JAMConfig config;
    config.multicast_group = "239.255.77.77";
    config.port = 7777;
    config.gpu_backend = JAMConfig::JAM_GPU_AUTO;
    config.compression_level = JAMConfig::JAM_TOAST_OPTIMIZED;
    config.enable_burst_logic = true;
    config.target_latency_ms = 3.0f;
    
    if (!jam.initialize(config)) {
        std::cerr << "Failed to initialize JAM Framework\n";
        return -1;
    }
    
    std::cout << "JAM Framework initialized successfully\n";
    std::cout << "GPU Available: " << (jam.is_gpu_available() ? "Yes" : "No") << "\n\n";
    
    // Set up callbacks
    jam.on_audio_stream([](const JAMAudioData& data) {
        std::cout << "Received audio: " << data.samples.size() << " samples, "
                  << data.sample_rate << "Hz, " << data.channels << " channels\n";
        if (data.pnbtr_processed) {
            std::cout << "  PNBTR processed with confidence: " << data.prediction_confidence << "\n";
        }
    });
    
    jam.on_midi_stream([](const JAMMIDIData& data) {
        std::cout << "Received MIDI: Status=0x" << std::hex << (int)data.status 
                  << " Data1=" << std::dec << (int)data.data1 
                  << " Data2=" << (int)data.data2;
        if (data.burst_count > 1) {
            std::cout << " (burst " << (int)data.burst_count << ")";
        }
        std::cout << "\n";
    });
    
    jam.on_video_stream([](const JAMVideoData& data) {
        std::cout << "Received video: " << data.width << "x" << data.height 
                  << " (" << data.pixel_data.size() << " bytes)\n";
        if (data.gpu_processed) {
            std::cout << "  GPU processed with shader ID: " << data.shader_id << "\n";
        }
    });
    
    jam.on_session_event([](const std::string& session_id, bool joined) {
        std::cout << "Session " << session_id << " " << (joined ? "joined" : "left") << "\n";
    });
    
    // Start session
    std::string session_id = utils::generate_session_id();
    std::cout << "Starting session: " << session_id << "\n\n";
    
    if (!jam.start_session(session_id)) {
        std::cerr << "Failed to start session\n";
        return -1;
    }
    
    // Send some test data
    std::cout << "Sending test data...\n";
    
    // Send audio data
    JAMAudioData audio_data;
    audio_data.session_id = session_id;
    audio_data.timestamp_ns = utils::get_timestamp_ns();
    audio_data.sample_rate = 48000;
    audio_data.channels = 2;
    audio_data.bit_depth = 24;
    
    // Generate a 440Hz sine wave
    const int samples_per_channel = 480; // 10ms at 48kHz
    audio_data.samples.resize(samples_per_channel * audio_data.channels);
    
    for (int i = 0; i < samples_per_channel; ++i) {
        float sample = std::sin(2.0f * M_PI * 440.0f * i / audio_data.sample_rate) * 0.5f;
        audio_data.samples[i * 2] = sample;     // Left channel
        audio_data.samples[i * 2 + 1] = sample; // Right channel
    }
    
    jam.send_audio(audio_data);
    std::cout << "Sent audio data: " << audio_data.samples.size() << " samples\n";
    
    // Send MIDI data
    JAMMIDIData midi_data;
    midi_data.session_id = session_id;
    midi_data.timestamp_ns = utils::get_timestamp_ns();
    midi_data.status = 0x90; // Note On
    midi_data.data1 = 60;    // Middle C
    midi_data.data2 = 127;   // Full velocity
    
    jam.send_midi(midi_data);
    std::cout << "Sent MIDI Note On: C4\n";
    
    // Send video data (simple test pattern)
    JAMVideoData video_data;
    video_data.session_id = session_id;
    video_data.timestamp_ns = utils::get_timestamp_ns();
    video_data.width = 320;
    video_data.height = 240;
    
    // Create a simple test pattern (RGBA)
    video_data.pixel_data.resize(video_data.width * video_data.height * 4);
    for (uint32_t y = 0; y < video_data.height; ++y) {
        for (uint32_t x = 0; x < video_data.width; ++x) {
            uint32_t index = (y * video_data.width + x) * 4;
            video_data.pixel_data[index + 0] = (x * 255) / video_data.width;  // R
            video_data.pixel_data[index + 1] = (y * 255) / video_data.height; // G
            video_data.pixel_data[index + 2] = 128;                           // B
            video_data.pixel_data[index + 3] = 255;                           // A
        }
    }
    
    jam.send_video(video_data);
    std::cout << "Sent video frame: " << video_data.width << "x" << video_data.height << "\n\n";
    
    // Run for a few seconds to receive data
    std::cout << "Running session for 5 seconds...\n";
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // Print statistics
        auto stats = jam.get_statistics();
        std::cout << "Stats: " << stats.packets_received << " rx, " 
                  << stats.packets_sent << " tx, "
                  << stats.average_latency_ms << "ms latency, "
                  << (stats.compression_ratio * 100.0f) << "% compression\n";
    }
    
    // Stop session
    std::cout << "\nStopping session...\n";
    jam.stop_session(session_id);
    
    // Final statistics
    auto final_stats = jam.get_statistics();
    std::cout << "\nFinal Statistics:\n";
    std::cout << "  Packets Received: " << final_stats.packets_received << "\n";
    std::cout << "  Packets Sent: " << final_stats.packets_sent << "\n";
    std::cout << "  Bytes Processed: " << final_stats.bytes_processed << "\n";
    std::cout << "  Average Latency: " << final_stats.average_latency_ms << "ms\n";
    std::cout << "  Compression Ratio: " << (final_stats.compression_ratio * 100.0f) << "%\n";
    std::cout << "  Active Streams: " << final_stats.active_streams << "\n";
    std::cout << "  GPU Utilization: " << (final_stats.gpu_utilization * 100.0f) << "%\n";
    
    std::cout << "\nJAM Framework session complete!\n";
    return 0;
}
