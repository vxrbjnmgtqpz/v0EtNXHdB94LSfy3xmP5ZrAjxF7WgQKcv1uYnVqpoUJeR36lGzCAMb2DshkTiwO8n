#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"
#include "AudioBufferManager.h"

using namespace jdat;

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully...\n";
    g_running = false;
}

/**
 * @brief Professional studio monitoring demonstration
 * 
 * This example demonstrates real-time audio processing with JELLIE encoding/decoding
 * suitable for professional studio monitoring applications.
 * 
 * Features:
 * - Real-time audio input/output simulation
 * - 4-stream ADAT-inspired redundancy
 * - PNBTR integration for packet loss recovery
 * - Performance monitoring and statistics
 * - Studio-quality latency targets (<200μs)
 */
class StudioMonitor {
public:
    struct Config {
        SampleRate sample_rate = SampleRate::SR_96000;
        uint32_t frame_size_samples = 480;  // 5ms frames at 96kHz
        uint32_t buffer_size_ms = 10;       // Ultra-low latency buffer
        uint8_t redundancy_level = 2;       // Full redundancy
        bool enable_pnbtr = true;
        bool enable_gpu = false;
        uint32_t monitoring_duration_sec = 60;
    };
    
    StudioMonitor(const Config& config) : config_(config) {}
    
    bool initialize() {
        std::cout << "Initializing JELLIE Studio Monitor...\n";
        std::cout << "=====================================\n";
        
        // Configure encoder
        JELLIEEncoder::Config encoder_config;
        encoder_config.sample_rate = config_.sample_rate;
        encoder_config.quality = AudioQuality::STUDIO;
        encoder_config.frame_size_samples = config_.frame_size_samples;
        encoder_config.redundancy_level = config_.redundancy_level;
        encoder_config.buffer_size_ms = config_.buffer_size_ms;
        encoder_config.session_id = "studio-monitor-001";
        
        encoder_ = std::make_unique<JELLIEEncoder>(encoder_config);
        
        // Configure decoder
        JELLIEDecoder::Config decoder_config;
        decoder_config.expected_sample_rate = config_.sample_rate;
        decoder_config.expected_frame_size = config_.frame_size_samples;
        decoder_config.redundancy_level = config_.redundancy_level;
        decoder_config.buffer_size_ms = config_.buffer_size_ms;
        decoder_config.enable_pnbtr = config_.enable_pnbtr;
        
        decoder_ = std::make_unique<JELLIEDecoder>(decoder_config);
        
        // Enable GPU acceleration if requested
        if (config_.enable_gpu) {
            std::cout << "Enabling GPU acceleration...\n";
            encoder_->enableGPUAcceleration("shaders/jellie_encoder.comp");
            decoder_->enableGPUAcceleration("shaders/jellie_decoder.comp");
        }
        
        // Set up callbacks
        setupCallbacks();
        
        std::cout << "JELLIE Studio Monitor initialized successfully!\n\n";
        return true;
    }
    
    void run() {
        std::cout << "Starting studio monitoring session...\n";
        std::cout << "Press Ctrl+C to stop\n\n";
        
        // Start audio processing
        encoder_->start();
        decoder_->start();
        
        // Start audio simulation thread
        audio_thread_ = std::thread([this]() { audioSimulationLoop(); });
        
        // Start monitoring thread
        monitor_thread_ = std::thread([this]() { monitoringLoop(); });
        
        // Main loop
        auto start_time = std::chrono::steady_clock::now();
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Check for timeout
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= config_.monitoring_duration_sec) {
                std::cout << "\nMonitoring session complete.\n";
                break;
            }
        }
        
        // Clean shutdown
        shutdown();
    }
    
private:
    void setupCallbacks() {
        // Encoder message callback - simulates network transmission
        encoder_->setMessageCallback([this](const JDATMessage& message) {
            // Simulate network transmission with optional packet loss
            if (simulatePacketLoss()) {
                return; // Packet lost
            }
            
            // Add to decoder processing queue
            decoder_->processMessage(message);
            
            // Update statistics
            packets_sent_++;
        });
        
        // Decoder output callback - simulates audio output
        decoder_->setOutputCallback([this](const std::vector<float>& samples, uint64_t timestamp) {
            // Calculate latency
            auto current_time = getCurrentTimestamp();
            uint64_t latency_us = current_time - timestamp;
            
            // Update latency statistics
            updateLatencyStats(latency_us);
            
            // Simulate audio output processing
            processAudioOutput(samples);
            
            frames_processed_++;
        });
    }
    
    void audioSimulationLoop() {
        const uint32_t sample_rate = static_cast<uint32_t>(config_.sample_rate);
        const auto frame_duration = std::chrono::microseconds(
            (config_.frame_size_samples * 1000000) / sample_rate
        );
        
        std::cout << "Audio simulation started (Frame duration: " 
                  << frame_duration.count() << "μs)\n";
        
        uint32_t frame_counter = 0;
        double phase = 0.0;
        
        while (g_running) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // Generate test audio (mix of sine waves for realistic testing)
            std::vector<float> audio_frame = generateTestAudio(frame_counter, phase);
            
            // Process through encoder
            try {
                auto messages = encoder_->encodeFrame(audio_frame);
                // Messages are sent via callback
                
            } catch (const std::exception& e) {
                std::cerr << "Encoding error: " << e.what() << std::endl;
            }
            
            frame_counter++;
            
            // Maintain precise timing
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto elapsed = frame_end - frame_start;
            
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
        }
        
        std::cout << "Audio simulation stopped.\n";
    }
    
    void monitoringLoop() {
        const auto update_interval = std::chrono::seconds(5);
        
        while (g_running) {
            std::this_thread::sleep_for(update_interval);
            
            // Print performance statistics
            printStatistics();
        }
    }
    
    std::vector<float> generateTestAudio(uint32_t frame_counter, double& phase) {
        std::vector<float> samples(config_.frame_size_samples);
        const uint32_t sample_rate = static_cast<uint32_t>(config_.sample_rate);
        
        // Generate complex test signal (multiple frequency components)
        for (size_t i = 0; i < samples.size(); ++i) {
            // Fundamental: 440 Hz (A4)
            double fundamental = 0.3 * std::sin(phase * 2.0 * M_PI * 440.0 / sample_rate);
            
            // Harmonics for realistic instrument simulation
            double harmonic2 = 0.1 * std::sin(phase * 2.0 * M_PI * 880.0 / sample_rate);
            double harmonic3 = 0.05 * std::sin(phase * 2.0 * M_PI * 1320.0 / sample_rate);
            
            // Add some musical modulation
            double modulation = 0.1 * std::sin(phase * 2.0 * M_PI * 5.0 / sample_rate);
            
            samples[i] = static_cast<float>((fundamental + harmonic2 + harmonic3) * (1.0 + modulation));
            
            phase++;
        }
        
        return samples;
    }
    
    bool simulatePacketLoss() {
        // Simulate realistic network conditions (1% packet loss)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        
        bool packet_lost = dis(gen) < 0.01; // 1% loss rate
        if (packet_lost) {
            packets_lost_++;
        }
        
        return packet_lost;
    }
    
    void processAudioOutput(const std::vector<float>& samples) {
        // Simulate audio output processing (DAC, monitoring, etc.)
        // In a real application, this would send to audio hardware
        
        // Calculate audio quality metrics
        updateAudioQualityStats(samples);
    }
    
    void updateLatencyStats(uint64_t latency_us) {
        latency_sum_ += latency_us;
        latency_count_++;
        
        if (latency_us < min_latency_) {
            min_latency_ = latency_us;
        }
        
        if (latency_us > max_latency_) {
            max_latency_ = latency_us;
        }
    }
    
    void updateAudioQualityStats(const std::vector<float>& samples) {
        // Calculate RMS level
        double rms_sum = 0.0;
        for (float sample : samples) {
            rms_sum += sample * sample;
        }
        double rms = std::sqrt(rms_sum / samples.size());
        
        // Update running average
        total_rms_ += rms;
        rms_count_++;
    }
    
    void printStatistics() {
        double avg_latency = latency_count_ > 0 ? 
            static_cast<double>(latency_sum_) / latency_count_ : 0.0;
        
        double avg_rms = rms_count_ > 0 ? total_rms_ / rms_count_ : 0.0;
        
        double packet_loss_rate = packets_sent_ > 0 ? 
            static_cast<double>(packets_lost_) / packets_sent_ * 100.0 : 0.0;
        
        std::cout << "\n=== STUDIO MONITOR STATISTICS ===\n";
        std::cout << "Frames Processed: " << frames_processed_ << "\n";
        std::cout << "Packets Sent: " << packets_sent_ << "\n";
        std::cout << "Packets Lost: " << packets_lost_ << " (" 
                  << std::fixed << std::setprecision(2) << packet_loss_rate << "%)\n";
        std::cout << "Average Latency: " << std::fixed << std::setprecision(1) 
                  << avg_latency << "μs\n";
        std::cout << "Latency Range: " << min_latency_ << "μs - " << max_latency_ << "μs\n";
        std::cout << "Average RMS Level: " << std::fixed << std::setprecision(4) 
                  << avg_rms << "\n";
        std::cout << "==================================\n\n";
    }
    
    void shutdown() {
        g_running = false;
        
        if (audio_thread_.joinable()) {
            audio_thread_.join();
        }
        
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
        
        encoder_->stop();
        decoder_->stop();
        
        std::cout << "\nFinal Statistics:\n";
        printStatistics();
    }
    
    uint64_t getCurrentTimestamp() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    
    Config config_;
    std::unique_ptr<JELLIEEncoder> encoder_;
    std::unique_ptr<JELLIEDecoder> decoder_;
    
    std::thread audio_thread_;
    std::thread monitor_thread_;
    
    // Statistics
    std::atomic<uint64_t> frames_processed_{0};
    std::atomic<uint64_t> packets_sent_{0};
    std::atomic<uint64_t> packets_lost_{0};
    
    uint64_t latency_sum_ = 0;
    uint64_t latency_count_ = 0;
    uint64_t min_latency_ = UINT64_MAX;
    uint64_t max_latency_ = 0;
    
    double total_rms_ = 0.0;
    uint64_t rms_count_ = 0;
};

int main(int argc, char* argv[]) {
    // Install signal handler for clean shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "JELLIE Studio Monitor Demo\n";
    std::cout << "==========================\n";
    std::cout << "Professional audio monitoring with ultra-low latency\n\n";
    
    // Configure studio monitor
    StudioMonitor::Config config;
    config.sample_rate = SampleRate::SR_96000;
    config.frame_size_samples = 480;        // 5ms frames
    config.buffer_size_ms = 10;             // 10ms buffer
    config.redundancy_level = 2;            // Full redundancy
    config.enable_pnbtr = true;             // Enable prediction
    config.enable_gpu = false;              // Set to true if GPU available
    config.monitoring_duration_sec = 30;    // 30 second demo
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--gpu") {
            config.enable_gpu = true;
        } else if (arg == "--duration" && i + 1 < argc) {
            config.monitoring_duration_sec = std::stoi(argv[++i]);
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            int sr = std::stoi(argv[++i]);
            if (sr == 48000) config.sample_rate = SampleRate::SR_48000;
            else if (sr == 96000) config.sample_rate = SampleRate::SR_96000;
            else if (sr == 192000) config.sample_rate = SampleRate::SR_192000;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --gpu                Enable GPU acceleration\n";
            std::cout << "  --duration SECONDS   Monitoring duration (default: 30)\n";
            std::cout << "  --sample-rate RATE   Sample rate: 48000, 96000, 192000 (default: 96000)\n";
            std::cout << "  --help               Show this help\n";
            return 0;
        }
    }
    
    // Create and run studio monitor
    StudioMonitor monitor(config);
    
    if (!monitor.initialize()) {
        std::cerr << "Failed to initialize studio monitor\n";
        return 1;
    }
    
    monitor.run();
    
    std::cout << "\nStudio monitoring session completed successfully!\n";
    return 0;
}
