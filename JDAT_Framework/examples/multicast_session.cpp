#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <signal.h>
#include <cstring>

#include "JELLIEEncoder.h"
#include "JELLIEDecoder.h"

using namespace jdat;

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully...\n";
    g_running = false;
}

/**
 * @brief TOAST Multicast Session Demo
 * 
 * This example demonstrates JELLIE audio streaming over UDP multicast
 * using the TOAST protocol. Shows both sender and receiver modes.
 * 
 * Features:
 * - UDP multicast transmission
 * - TOAST protocol integration
 * - Session management
 * - Real-time audio streaming
 * - Network resilience testing
 */
class MulticastSession {
public:
    enum class Mode { SENDER, RECEIVER };
    
    struct Config {
        Mode mode = Mode::SENDER;
        std::string session_id = "default-session";
        std::string multicast_address = "239.255.1.1";
        uint16_t port = 7777;
        SampleRate sample_rate = SampleRate::SR_96000;
        uint32_t frame_size_samples = 960;  // 10ms at 96kHz
        uint8_t redundancy_level = 2;
        bool enable_pnbtr = true;
        uint32_t duration_sec = 300;  // 5 minutes
    };
    
    MulticastSession(const Config& config) : config_(config) {}
    
    bool initialize() {
        std::cout << "Initializing JELLIE Multicast Session...\n";
        std::cout << "========================================\n";
        std::cout << "Mode: " << (config_.mode == Mode::SENDER ? "SENDER" : "RECEIVER") << "\n";
        std::cout << "Session ID: " << config_.session_id << "\n";
        std::cout << "Multicast: " << config_.multicast_address << ":" << config_.port << "\n\n";
        
        if (config_.mode == Mode::SENDER) {
            return initializeSender();
        } else {
            return initializeReceiver();
        }
    }
    
    void run() {
        if (config_.mode == Mode::SENDER) {
            runSender();
        } else {
            runReceiver();
        }
    }
    
private:
    bool initializeSender() {
        // Configure encoder
        JELLIEEncoder::Config encoder_config;
        encoder_config.sample_rate = config_.sample_rate;
        encoder_config.quality = AudioQuality::STUDIO;
        encoder_config.frame_size_samples = config_.frame_size_samples;
        encoder_config.redundancy_level = config_.redundancy_level;
        encoder_config.session_id = config_.session_id;
        
        encoder_ = std::make_unique<JELLIEEncoder>(encoder_config);
        
        // Initialize network transport (simulated)
        transport_ = std::make_unique<TOASTTransport>(
            config_.multicast_address, 
            config_.port
        );
        
        // Set up encoder callback for network transmission
        encoder_->setMessageCallback([this](const JDATMessage& message) {
            sendMessageOverNetwork(message);
        });
        
        std::cout << "Sender initialized successfully!\n";
        return true;
    }
    
    bool initializeReceiver() {
        // Configure decoder
        JELLIEDecoder::Config decoder_config;
        decoder_config.expected_sample_rate = config_.sample_rate;
        decoder_config.expected_frame_size = config_.frame_size_samples;
        decoder_config.redundancy_level = config_.redundancy_level;
        decoder_config.enable_pnbtr = config_.enable_pnbtr;
        
        decoder_ = std::make_unique<JELLIEDecoder>(decoder_config);
        
        // Initialize network transport (simulated)
        transport_ = std::make_unique<TOASTTransport>(
            config_.multicast_address, 
            config_.port
        );
        
        // Set up decoder callback for audio output
        decoder_->setOutputCallback([this](const std::vector<float>& samples, uint64_t timestamp) {
            processReceivedAudio(samples, timestamp);
        });
        
        std::cout << "Receiver initialized successfully!\n";
        return true;
    }
    
    void runSender() {
        std::cout << "Starting audio transmission...\n";
        std::cout << "Generating test audio signal...\n\n";
        
        encoder_->start();
        
        // Start audio generation thread
        audio_thread_ = std::thread([this]() { audioGenerationLoop(); });
        
        // Start statistics thread
        stats_thread_ = std::thread([this]() { statisticsLoop(); });
        
        // Main loop
        auto start_time = std::chrono::steady_clock::now();
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= config_.duration_sec) {
                std::cout << "\nTransmission session complete.\n";
                break;
            }
        }
        
        shutdown();
    }
    
    void runReceiver() {
        std::cout << "Starting audio reception...\n";
        std::cout << "Listening for incoming streams...\n\n";
        
        decoder_->start();
        
        // Start network listening thread
        network_thread_ = std::thread([this]() { networkListenLoop(); });
        
        // Start statistics thread
        stats_thread_ = std::thread([this]() { statisticsLoop(); });
        
        // Main loop
        auto start_time = std::chrono::steady_clock::now();
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= config_.duration_sec) {
                std::cout << "\nReception session complete.\n";
                break;
            }
        }
        
        shutdown();
    }
    
    void audioGenerationLoop() {
        const uint32_t sample_rate = static_cast<uint32_t>(config_.sample_rate);
        const auto frame_duration = std::chrono::microseconds(
            (config_.frame_size_samples * 1000000) / sample_rate
        );
        
        uint32_t frame_counter = 0;
        double phase = 0.0;
        
        while (g_running) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // Generate musical test signal
            std::vector<float> audio_frame = generateMusicalTestSignal(frame_counter, phase);
            
            try {
                // Encode and transmit
                auto messages = encoder_->encodeFrame(audio_frame);
                frames_generated_++;
                
            } catch (const std::exception& e) {
                std::cerr << "Audio generation error: " << e.what() << std::endl;
            }
            
            frame_counter++;
            
            // Maintain precise timing
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto elapsed = frame_end - frame_start;
            
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
        }
    }
    
    void networkListenLoop() {
        std::cout << "Network listener started...\n";
        
        while (g_running) {
            // Simulate receiving network messages
            auto messages = receiveMessagesFromNetwork();
            
            for (const auto& message : messages) {
                try {
                    decoder_->processMessage(message);
                    messages_received_++;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Message processing error: " << e.what() << std::endl;
                }
            }
            
            // Small delay to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void statisticsLoop() {
        const auto update_interval = std::chrono::seconds(10);
        
        while (g_running) {
            std::this_thread::sleep_for(update_interval);
            printStatistics();
        }
    }
    
    std::vector<float> generateMusicalTestSignal(uint32_t frame_counter, double& phase) {
        std::vector<float> samples(config_.frame_size_samples);
        const uint32_t sample_rate = static_cast<uint32_t>(config_.sample_rate);
        
        // Generate chord progression (C major - F major - G major - C major)
        double time_sec = static_cast<double>(frame_counter * config_.frame_size_samples) / sample_rate;
        int chord_index = static_cast<int>(time_sec / 4.0) % 4;  // 4 second chords
        
        // Chord frequencies
        std::vector<double> chord_freqs;
        switch (chord_index) {
            case 0: chord_freqs = {261.63, 329.63, 392.00}; break;  // C major
            case 1: chord_freqs = {349.23, 440.00, 523.25}; break;  // F major
            case 2: chord_freqs = {392.00, 493.88, 587.33}; break;  // G major
            case 3: chord_freqs = {261.63, 329.63, 392.00}; break;  // C major
        }
        
        // Generate samples
        for (size_t i = 0; i < samples.size(); ++i) {
            float sample = 0.0f;
            
            // Mix chord frequencies
            for (double freq : chord_freqs) {
                sample += 0.2f * std::sin(phase * 2.0 * M_PI * freq / sample_rate);
            }
            
            // Add subtle envelope for musicality
            double envelope = 0.5 + 0.5 * std::sin(phase * 2.0 * M_PI * 2.0 / sample_rate);
            sample *= envelope;
            
            samples[i] = sample;
            phase++;
        }
        
        return samples;
    }
    
    void sendMessageOverNetwork(const JDATMessage& message) {
        // Simulate network transmission
        messages_sent_++;
        
        // Simulate packet loss (2% loss rate for testing)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        
        if (dis(gen) < 0.02) {  // 2% packet loss
            packets_lost_++;
            return;  // Packet lost
        }
        
        // In real implementation, send via UDP multicast
        // transport_->sendMulticast(message);
        
        // For simulation, add to simulated network buffer
        {
            std::lock_guard<std::mutex> lock(network_mutex_);
            network_buffer_.push_back(message);
        }
    }
    
    std::vector<JDATMessage> receiveMessagesFromNetwork() {
        std::vector<JDATMessage> messages;
        
        {
            std::lock_guard<std::mutex> lock(network_mutex_);
            
            // Simulate network latency and jitter
            if (!network_buffer_.empty() && shouldDeliverMessage()) {
                messages.push_back(network_buffer_.front());
                network_buffer_.erase(network_buffer_.begin());
            }
        }
        
        return messages;
    }
    
    bool shouldDeliverMessage() {
        // Simulate realistic network delivery timing
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        
        return dis(gen) < 0.8;  // 80% chance of immediate delivery
    }
    
    void processReceivedAudio(const std::vector<float>& samples, uint64_t timestamp) {
        // Process received audio (would normally go to audio output)
        frames_processed_++;
        
        // Calculate quality metrics
        double rms = calculateRMS(samples);
        total_rms_ += rms;
        rms_count_++;
        
        // Simulate audio output latency
        output_latency_sum_ += 50;  // 50μs simulated output latency
        output_latency_count_++;
    }
    
    double calculateRMS(const std::vector<float>& samples) {
        double sum = 0.0;
        for (float sample : samples) {
            sum += sample * sample;
        }
        return std::sqrt(sum / samples.size());
    }
    
    void printStatistics() {
        std::cout << "\n=== MULTICAST SESSION STATISTICS ===\n";
        
        if (config_.mode == Mode::SENDER) {
            std::cout << "Frames Generated: " << frames_generated_ << "\n";
            std::cout << "Messages Sent: " << messages_sent_ << "\n";
            std::cout << "Packets Lost: " << packets_lost_ << "\n";
            
            if (messages_sent_ > 0) {
                double loss_rate = static_cast<double>(packets_lost_) / messages_sent_ * 100.0;
                std::cout << "Packet Loss Rate: " << std::fixed << std::setprecision(2) 
                          << loss_rate << "%\n";
            }
        } else {
            std::cout << "Messages Received: " << messages_received_ << "\n";
            std::cout << "Frames Processed: " << frames_processed_ << "\n";
            
            if (rms_count_ > 0) {
                double avg_rms = total_rms_ / rms_count_;
                std::cout << "Average RMS Level: " << std::fixed << std::setprecision(4) 
                          << avg_rms << "\n";
            }
            
            if (output_latency_count_ > 0) {
                double avg_latency = static_cast<double>(output_latency_sum_) / output_latency_count_;
                std::cout << "Average Output Latency: " << std::fixed << std::setprecision(1) 
                          << avg_latency << "μs\n";
            }
        }
        
        std::cout << "====================================\n\n";
    }
    
    void shutdown() {
        g_running = false;
        
        if (audio_thread_.joinable()) {
            audio_thread_.join();
        }
        
        if (network_thread_.joinable()) {
            network_thread_.join();
        }
        
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }
        
        if (encoder_) {
            encoder_->stop();
        }
        
        if (decoder_) {
            decoder_->stop();
        }
        
        std::cout << "\nFinal Statistics:\n";
        printStatistics();
    }
    
    // Mock TOAST Transport class for simulation
    class TOASTTransport {
    public:
        TOASTTransport(const std::string& address, uint16_t port) 
            : multicast_address_(address), port_(port) {}
        
        void sendMulticast(const JDATMessage& message) {
            // UDP multicast implementation would go here
        }
        
    private:
        std::string multicast_address_;
        uint16_t port_;
    };
    
    Config config_;
    std::unique_ptr<JELLIEEncoder> encoder_;
    std::unique_ptr<JELLIEDecoder> decoder_;
    std::unique_ptr<TOASTTransport> transport_;
    
    std::thread audio_thread_;
    std::thread network_thread_;
    std::thread stats_thread_;
    
    // Network simulation
    std::vector<JDATMessage> network_buffer_;
    std::mutex network_mutex_;
    
    // Statistics
    std::atomic<uint64_t> frames_generated_{0};
    std::atomic<uint64_t> frames_processed_{0};
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> packets_lost_{0};
    
    double total_rms_ = 0.0;
    uint64_t rms_count_ = 0;
    uint64_t output_latency_sum_ = 0;
    uint64_t output_latency_count_ = 0;
};

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "JELLIE Multicast Session Demo\n";
    std::cout << "=============================\n";
    std::cout << "Professional audio streaming over UDP multicast\n\n";
    
    MulticastSession::Config config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "sender") {
                config.mode = MulticastSession::Mode::SENDER;
            } else if (mode == "receiver") {
                config.mode = MulticastSession::Mode::RECEIVER;
            }
        } else if (arg == "--session" && i + 1 < argc) {
            config.session_id = argv[++i];
        } else if (arg == "--address" && i + 1 < argc) {
            config.multicast_address = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration_sec = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --mode MODE          sender or receiver (default: sender)\n";
            std::cout << "  --session ID         Session identifier (default: default-session)\n";
            std::cout << "  --address ADDR       Multicast address (default: 239.255.1.1)\n";
            std::cout << "  --port PORT          UDP port (default: 7777)\n";
            std::cout << "  --duration SECONDS   Session duration (default: 300)\n";
            std::cout << "  --help               Show this help\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " --mode sender --session studio-001\n";
            std::cout << "  " << argv[0] << " --mode receiver --session studio-001\n";
            return 0;
        }
    }
    
    MulticastSession session(config);
    
    if (!session.initialize()) {
        std::cerr << "Failed to initialize multicast session\n";
        return 1;
    }
    
    session.run();
    
    std::cout << "\nMulticast session completed successfully!\n";
    return 0;
}
