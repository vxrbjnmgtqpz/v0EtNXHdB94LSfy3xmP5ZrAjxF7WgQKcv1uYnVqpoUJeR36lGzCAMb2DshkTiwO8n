/**
 * @file basic_jamcam_demo.cpp
 * @brief Basic JAMCam demonstration of JVID ultra-low latency video streaming
 * 
 * This example demonstrates:
 * - Video capture and encoding with JAMCamEncoder
 * - Video decoding and rendering with JAMCamDecoder  
 * - Frame prediction and recovery using PNTBTR
 * - Integration with TOAST transport protocol
 * - Performance monitoring and statistics
 */

#include "JVIDMessage.h"
#include "JAMCamEncoder.h"
#include "JAMCamDecoder.h"
#include "FramePredictor.h"
#include "VideoBufferManager.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>

using namespace jvid;

/**
 * @brief Demo configuration
 */
struct DemoConfig {
    VideoResolution resolution = VideoResolution::LOW_144P;
    VideoQuality quality = VideoQuality::FAST;
    uint32_t target_fps = 15;
    uint32_t duration_seconds = 30;
    bool enable_face_detection = true;
    bool enable_auto_framing = true;
    bool enable_frame_prediction = true;
    bool simulate_packet_loss = true;
    float packet_loss_rate = 0.05f;  // 5% packet loss simulation
};

/**
 * @brief Demo statistics tracking
 */
struct DemoStats {
    std::atomic<uint64_t> frames_captured{0};
    std::atomic<uint64_t> frames_encoded{0};
    std::atomic<uint64_t> frames_transmitted{0};
    std::atomic<uint64_t> frames_received{0};
    std::atomic<uint64_t> frames_decoded{0};
    std::atomic<uint64_t> frames_dropped{0};
    std::atomic<uint64_t> frames_predicted{0};
    
    std::atomic<double> average_encode_latency{0.0};
    std::atomic<double> average_decode_latency{0.0};
    std::atomic<double> average_end_to_end_latency{0.0};
    std::atomic<double> current_bitrate_kbps{0.0};
    
    void printSummary() const {
        std::cout << "\n=== JVID Demo Statistics ===" << std::endl;
        std::cout << "Frames captured: " << frames_captured.load() << std::endl;
        std::cout << "Frames encoded: " << frames_encoded.load() << std::endl;
        std::cout << "Frames transmitted: " << frames_transmitted.load() << std::endl;
        std::cout << "Frames received: " << frames_received.load() << std::endl;
        std::cout << "Frames decoded: " << frames_decoded.load() << std::endl;
        std::cout << "Frames dropped: " << frames_dropped.load() << std::endl;
        std::cout << "Frames predicted: " << frames_predicted.load() << std::endl;
        
        std::cout << "\nLatency Performance:" << std::endl;
        std::cout << "Average encode latency: " << average_encode_latency.load() << " μs" << std::endl;
        std::cout << "Average decode latency: " << average_decode_latency.load() << " μs" << std::endl;
        std::cout << "Average end-to-end latency: " << average_end_to_end_latency.load() << " μs" << std::endl;
        std::cout << "Current bitrate: " << current_bitrate_kbps.load() << " kbps" << std::endl;
        
        if (frames_transmitted.load() > 0) {
            double packet_loss = 1.0 - (double(frames_received.load()) / double(frames_transmitted.load()));
            std::cout << "Packet loss rate: " << (packet_loss * 100.0) << "%" << std::endl;
        }
        
        if (frames_predicted.load() > 0) {
            double prediction_rate = double(frames_predicted.load()) / double(frames_decoded.load());
            std::cout << "Frame prediction rate: " << (prediction_rate * 100.0) << "%" << std::endl;
        }
    }
};

/**
 * @brief Mock transport layer for simulating TOAST protocol
 */
class MockTransport {
private:
    std::atomic<bool> running_{false};
    std::function<void(const JVIDMessage&)> receiver_callback_;
    float packet_loss_rate_;
    
public:
    explicit MockTransport(float packet_loss_rate = 0.0f) 
        : packet_loss_rate_(packet_loss_rate) {}
    
    void setReceiverCallback(std::function<void(const JVIDMessage&)> callback) {
        receiver_callback_ = callback;
    }
    
    bool transmit(const JVIDMessage& message) {
        if (!running_.load()) return false;
        
        // Simulate packet loss
        if (packet_loss_rate_ > 0.0f) {
            float random_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (random_val < packet_loss_rate_) {
                return false; // Simulate packet dropped
            }
        }
        
        // Simulate network transmission delay (50-200 microseconds)
        auto delay_us = 50 + (rand() % 150);
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        
        // Deliver to receiver
        if (receiver_callback_) {
            receiver_callback_(message);
        }
        
        return true;
    }
    
    void start() { running_.store(true); }
    void stop() { running_.store(false); }
    bool isRunning() const { return running_.load(); }
};

/**
 * @brief JAMCam video streaming demo
 */
class JAMCamDemo {
private:
    DemoConfig config_;
    std::unique_ptr<JAMCamEncoder> encoder_;
    std::unique_ptr<JAMCamDecoder> decoder_;
    std::unique_ptr<MockTransport> transport_;
    std::atomic<bool> running_{false};
    
    DemoStats stats_;
    
public:
    explicit JAMCamDemo(const DemoConfig& config) : config_(config) {
        initializeComponents();
    }
    
    ~JAMCamDemo() {
        stop();
    }
    
    bool start() {
        std::cout << "Starting JAMCam ultra-low latency video demo..." << std::endl;
        std::cout << "Resolution: " << getResolutionName(config_.resolution) << std::endl;
        std::cout << "Quality: " << getQualityName(config_.quality) << std::endl;
        std::cout << "Target FPS: " << config_.target_fps << std::endl;
        std::cout << "Duration: " << config_.duration_seconds << " seconds" << std::endl;
        
        if (config_.simulate_packet_loss) {
            std::cout << "Simulating " << (config_.packet_loss_rate * 100.0f) << "% packet loss" << std::endl;
        }
        
        running_.store(true);
        transport_->start();
        
        if (!encoder_->start()) {
            std::cerr << "Failed to start encoder" << std::endl;
            return false;
        }
        
        if (!decoder_->start()) {
            std::cerr << "Failed to start decoder" << std::endl;
            return false;
        }
        
        // Run demo for specified duration
        std::this_thread::sleep_for(std::chrono::seconds(config_.duration_seconds));
        
        stop();
        return true;
    }
    
    void stop() {
        if (!running_.load()) return;
        
        std::cout << "\nStopping demo..." << std::endl;
        running_.store(false);
        
        encoder_->stop();
        decoder_->stop();
        transport_->stop();
        
        stats_.printSummary();
    }
    
private:
    void initializeComponents() {
        // Configure encoder
        JAMCamEncoder::Config encoder_config;
        encoder_config.resolution = config_.resolution;
        encoder_config.quality = config_.quality;
        encoder_config.target_fps = config_.target_fps;
        encoder_config.target_latency_us = 300;  // 300μs target
        encoder_config.enable_face_detection = config_.enable_face_detection;
        encoder_config.enable_auto_framing = config_.enable_auto_framing;
        encoder_config.enable_frame_dropping = true;
        encoder_config.max_encode_time_us = 500;
        
        encoder_ = std::make_unique<JAMCamEncoder>(encoder_config);
        
        // Configure decoder
        JAMCamDecoder::Config decoder_config;
        decoder_config.target_latency_us = 300;  // 300μs target
        decoder_config.enable_frame_prediction = config_.enable_frame_prediction;
        decoder_config.enable_gpu_decoding = true;
        decoder_config.enable_adaptive_quality = true;
        decoder_config.max_decode_time_us = 500;
        
        auto resolution_dims = JVIDMessage::getResolutionDimensions(config_.resolution);
        decoder_config.display_width = resolution_dims.first;
        decoder_config.display_height = resolution_dims.second;
        
        decoder_ = std::make_unique<JAMCamDecoder>(decoder_config);
        
        // Configure transport
        transport_ = std::make_unique<MockTransport>(
            config_.simulate_packet_loss ? config_.packet_loss_rate : 0.0f
        );
        
        setupCallbacks();
    }
    
    void setupCallbacks() {
        // Encoder frame callback - transmit encoded frames
        encoder_->setFrameCallback([this](const JVIDMessage& message) {
            stats_.frames_encoded.fetch_add(1);
            
            // Calculate encode latency
            auto now = std::chrono::high_resolution_clock::now();
            auto encode_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count() - message.timing_info.encode_timestamp_us;
            
            stats_.average_encode_latency.store(
                (stats_.average_encode_latency.load() + encode_latency) / 2.0
            );
            
            // Transmit via mock transport
            if (transport_->transmit(message)) {
                stats_.frames_transmitted.fetch_add(1);
            }
        });
        
        // Encoder frame drop callback
        encoder_->setFrameDropCallback([this](uint64_t dropped_seq, uint32_t encode_time) {
            stats_.frames_dropped.fetch_add(1);
            std::cout << "Frame " << dropped_seq << " dropped (encode time: " 
                      << encode_time << "μs)" << std::endl;
        });
        
        // Transport receiver callback - deliver frames to decoder
        transport_->setReceiverCallback([this](const JVIDMessage& message) {
            stats_.frames_received.fetch_add(1);
            decoder_->processMessage(message);
        });
        
        // Decoder frame ready callback - frame is ready for display
        decoder_->setFrameReadyCallback([this](const VideoBufferManager::FrameBuffer* frame) {
            stats_.frames_decoded.fetch_add(1);
            
            // Calculate end-to-end latency
            auto now = std::chrono::high_resolution_clock::now();
            auto end_to_end_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count() - frame->timestamp_us;
            
            stats_.average_end_to_end_latency.store(
                (stats_.average_end_to_end_latency.load() + end_to_end_latency) / 2.0
            );
            
            // Simulate frame display
            displayFrame(frame);
        });
        
        // Decoder frame drop callback
        decoder_->setFrameDropCallback([this](const JVIDMessage& message, const std::string& reason) {
            stats_.frames_dropped.fetch_add(1);
            std::cout << "Frame " << message.sequence_number << " dropped: " << reason << std::endl;
        });
        
        // Statistics callbacks
        encoder_->setStatsCallback([this](const VideoStreamStats& stats) {
            updateEncoderStats(stats);
        });
        
        decoder_->setStatsCallback([this](const VideoStreamStats& stats) {
            updateDecoderStats(stats);
        });
    }
    
    void displayFrame(const VideoBufferManager::FrameBuffer* frame) {
        // Simulate frame display processing
        // In a real application, this would render to screen/canvas
        
        static uint64_t last_display_time = 0;
        static uint32_t display_count = 0;
        
        display_count++;
        
        // Print progress every second
        auto now = std::chrono::high_resolution_clock::now();
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        if (now_ms - last_display_time >= 1000) {
            std::cout << "Displaying frame " << frame->sequence_number 
                      << " (" << frame->width << "x" << frame->height << ") "
                      << "FPS: " << display_count << std::endl;
            display_count = 0;
            last_display_time = now_ms;
        }
    }
    
    void updateEncoderStats(const VideoStreamStats& encoder_stats) {
        stats_.current_bitrate_kbps.store(encoder_stats.bandwidth_kbps);
        stats_.frames_captured.store(encoder_stats.frames_sent);
    }
    
    void updateDecoderStats(const VideoStreamStats& decoder_stats) {
        stats_.frames_predicted.store(decoder_stats.frames_predicted);
        
        // Update decode latency
        stats_.average_decode_latency.store(decoder_stats.average_end_to_end_latency_us);
    }
    
    std::string getResolutionName(VideoResolution resolution) const {
        switch (resolution) {
            case VideoResolution::ULTRA_LOW_72P: return "ULTRA_LOW_72P (128x72)";
            case VideoResolution::LOW_144P: return "LOW_144P (256x144)";
            case VideoResolution::MEDIUM_240P: return "MEDIUM_240P (426x240)";
            case VideoResolution::HIGH_360P: return "HIGH_360P (640x360)";
            default: return "Unknown";
        }
    }
    
    std::string getQualityName(VideoQuality quality) const {
        switch (quality) {
            case VideoQuality::ULTRA_FAST: return "ULTRA_FAST";
            case VideoQuality::FAST: return "FAST";
            case VideoQuality::BALANCED: return "BALANCED";
            case VideoQuality::HIGH_QUALITY: return "HIGH_QUALITY";
            default: return "Unknown";
        }
    }
};

/**
 * @brief Parse command line arguments
 */
DemoConfig parseArguments(int argc, char* argv[]) {
    DemoConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--resolution" && i + 1 < argc) {
            std::string res = argv[++i];
            if (res == "72p") config.resolution = VideoResolution::ULTRA_LOW_72P;
            else if (res == "144p") config.resolution = VideoResolution::LOW_144P;
            else if (res == "240p") config.resolution = VideoResolution::MEDIUM_240P;
            else if (res == "360p") config.resolution = VideoResolution::HIGH_360P;
        }
        else if (arg == "--quality" && i + 1 < argc) {
            std::string qual = argv[++i];
            if (qual == "ultra_fast") config.quality = VideoQuality::ULTRA_FAST;
            else if (qual == "fast") config.quality = VideoQuality::FAST;
            else if (qual == "balanced") config.quality = VideoQuality::BALANCED;
            else if (qual == "high") config.quality = VideoQuality::HIGH_QUALITY;
        }
        else if (arg == "--fps" && i + 1 < argc) {
            config.target_fps = std::stoi(argv[++i]);
        }
        else if (arg == "--duration" && i + 1 < argc) {
            config.duration_seconds = std::stoi(argv[++i]);
        }
        else if (arg == "--no-face-detection") {
            config.enable_face_detection = false;
        }
        else if (arg == "--no-auto-framing") {
            config.enable_auto_framing = false;
        }
        else if (arg == "--no-prediction") {
            config.enable_frame_prediction = false;
        }
        else if (arg == "--no-packet-loss") {
            config.simulate_packet_loss = false;
        }
        else if (arg == "--packet-loss" && i + 1 < argc) {
            config.packet_loss_rate = std::stof(argv[++i]);
            config.simulate_packet_loss = true;
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "JVID JAMCam Demo Usage:" << std::endl;
            std::cout << "  --resolution [72p|144p|240p|360p]  Video resolution" << std::endl;
            std::cout << "  --quality [ultra_fast|fast|balanced|high]  Video quality" << std::endl;
            std::cout << "  --fps <number>                      Target framerate" << std::endl;
            std::cout << "  --duration <seconds>                Demo duration" << std::endl;
            std::cout << "  --no-face-detection                 Disable face detection" << std::endl;
            std::cout << "  --no-auto-framing                   Disable auto framing" << std::endl;
            std::cout << "  --no-prediction                     Disable frame prediction" << std::endl;
            std::cout << "  --no-packet-loss                    Disable packet loss simulation" << std::endl;
            std::cout << "  --packet-loss <rate>                Set packet loss rate (0.0-1.0)" << std::endl;
            std::cout << "  --help                              Show this help" << std::endl;
            exit(0);
        }
    }
    
    return config;
}

/**
 * @brief Main demo entry point
 */
int main(int argc, char* argv[]) {
    std::cout << "JVID Framework - JAMCam Ultra-Low Latency Demo" << std::endl;
    std::cout << "Target: <300μs video latency via TOAST transport" << std::endl;
    std::cout << "Features: Face detection, auto-framing, PNTBTR recovery" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    try {
        auto config = parseArguments(argc, argv);
        JAMCamDemo demo(config);
        
        if (!demo.start()) {
            std::cerr << "Demo failed to start" << std::endl;
            return 1;
        }
        
        std::cout << "\nDemo completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }
} 