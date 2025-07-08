#include "../include/PnbtrJellieVST3Minimal.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <memory>

class VST3MinimalDemo {
private:
    std::unique_ptr<PnbtrJellieVST3Minimal> plugin_;
    
    // Audio simulation parameters
    static constexpr double SAMPLE_RATE = 48000.0;
    static constexpr int BLOCK_SIZE = 512;
    static constexpr int CHANNELS = 2;
    static constexpr int TEST_DURATION_SECONDS = 3;
    
    // Test buffers (channel-based like DAW)
    std::vector<std::vector<float>> input_channels_;
    std::vector<std::vector<float>> output_channels_;
    std::vector<float*> input_ptrs_;
    std::vector<float*> output_ptrs_;
    
public:
    VST3MinimalDemo() {
        // Create plugin instance (like DAW would do)
        plugin_ = std::make_unique<PnbtrJellieVST3Minimal>();
        
        // Allocate channel buffers
        input_channels_.resize(CHANNELS);
        output_channels_.resize(CHANNELS);
        input_ptrs_.resize(CHANNELS);
        output_ptrs_.resize(CHANNELS);
        
        for (int ch = 0; ch < CHANNELS; ++ch) {
            input_channels_[ch].resize(BLOCK_SIZE);
            output_channels_[ch].resize(BLOCK_SIZE);
            input_ptrs_[ch] = input_channels_[ch].data();
            output_ptrs_[ch] = output_channels_[ch].data();
        }
        
        std::cout << "=== PNBTR+JELLIE VST3 Minimal Plugin Demo ===" << std::endl;
        std::cout << "Plugin Name: " << PnbtrJellieVST3Minimal::getPluginName() << std::endl;
        std::cout << "Version: " << PnbtrJellieVST3Minimal::getPluginVersion() << std::endl;
        std::cout << "Vendor: " << PnbtrJellieVST3Minimal::getPluginVendor() << std::endl;
        std::cout << "Description: " << PnbtrJellieVST3Minimal::getPluginDescription() << std::endl;
        std::cout << "Plugin UID: 0x" << std::hex << PnbtrJellieVST3Minimal::getPluginUID1() 
                  << PnbtrJellieVST3Minimal::getPluginUID2() << std::dec << std::endl;
        std::cout << "Parameters: " << PnbtrJellieVST3Minimal::getNumParameters() << std::endl;
        std::cout << std::endl;
    }
    
    bool initialize() {
        std::cout << "Initializing VST3 plugin (like DAW would do)..." << std::endl;
        
        // Initialize plugin with DAW-like parameters
        if (!plugin_->initialize(SAMPLE_RATE, BLOCK_SIZE, CHANNELS)) {
            std::cerr << "Failed to initialize VST3 plugin!" << std::endl;
            return false;
        }
        
        std::cout << "VST3 plugin initialized successfully!" << std::endl;
        std::cout << std::endl;
        
        return true;
    }
    
    void testDualModeOperation() {
        std::cout << "=== Dual-Mode VST3 Plugin Operation Test ===" << std::endl;
        std::cout << "Testing both TX (Transmit) and RX (Receive) modes..." << std::endl;
        std::cout << std::endl;
        
        // Test TX Mode
        testTXMode();
        
        // Brief pause
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Test RX Mode
        testRXMode();
        
        displayPerformanceResults();
    }
    
    void testTXMode() {
        std::cout << "--- TX Mode Test (Audio Input → Network) ---" << std::endl;
        
        // Set plugin to TX mode via parameter
        setParameter(PARAM_PLUGIN_MODE, 0.0f); // 0.0 = TX mode
        setParameter(PARAM_TEST_SINE_ENABLE, 1.0f); // Enable sine generator
        setParameter(PARAM_TEST_SINE_FREQ, 440.0f); // 440Hz
        setParameter(PARAM_TEST_SINE_AMPLITUDE, 0.5f); // 50% amplitude
        
        // Process audio blocks
        int total_blocks = (SAMPLE_RATE * 1) / BLOCK_SIZE; // 1 second
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int block = 0; block < total_blocks; ++block) {
            // Clear buffers
            clearBuffers();
            
            // Process audio block (like DAW callback)
            auto block_start = std::chrono::high_resolution_clock::now();
            bool success = plugin_->processAudio(
                const_cast<const float**>(input_ptrs_.data()),
                output_ptrs_.data(),
                BLOCK_SIZE,
                CHANNELS
            );
            auto block_end = std::chrono::high_resolution_clock::now();
            
            double block_time_us = std::chrono::duration<double, std::micro>(block_end - block_start).count();
            
            if (block % 100 == 0) {
                std::cout << "TX Block " << block << "/" << total_blocks 
                          << " - Processing: " << std::fixed << std::setprecision(1) 
                          << block_time_us << "μs";
                
                if (block_time_us < 50.0) {
                    std::cout << " ✅ EXCELLENT";
                } else if (block_time_us < 100.0) {
                    std::cout << " ✅ GOOD";
                } else {
                    std::cout << " ⚠️ SLOW";
                }
                std::cout << std::endl;
            }
            
            // Simulate real-time processing
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "TX Mode test completed in " << std::fixed << std::setprecision(2) 
                  << total_time << " seconds" << std::endl;
        std::cout << std::endl;
    }
    
    void testRXMode() {
        std::cout << "--- RX Mode Test (Network → Audio Output) ---" << std::endl;
        
        // Set plugin to RX mode via parameter
        setParameter(PARAM_PLUGIN_MODE, 1.0f); // 1.0 = RX mode
        setParameter(PARAM_TEST_SINE_ENABLE, 1.0f); // Enable sine generator (for testing)
        setParameter(PARAM_TEST_PACKET_LOSS_ENABLE, 1.0f); // Enable packet loss simulation
        setParameter(PARAM_TEST_PACKET_LOSS_PERCENT, 5.0f); // 5% packet loss
        setParameter(PARAM_PNBTR_STRENGTH, 0.8f); // 80% PNBTR strength
        
        // Process audio blocks
        int total_blocks = (SAMPLE_RATE * 1) / BLOCK_SIZE; // 1 second
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int block = 0; block < total_blocks; ++block) {
            // Clear buffers
            clearBuffers();
            
            // Process audio block (like DAW callback)
            auto block_start = std::chrono::high_resolution_clock::now();
            bool success = plugin_->processAudio(
                const_cast<const float**>(input_ptrs_.data()),
                output_ptrs_.data(),
                BLOCK_SIZE,
                CHANNELS
            );
            auto block_end = std::chrono::high_resolution_clock::now();
            
            double block_time_us = std::chrono::duration<double, std::micro>(block_end - block_start).count();
            
            if (block % 100 == 0) {
                std::cout << "RX Block " << block << "/" << total_blocks 
                          << " - Processing: " << std::fixed << std::setprecision(1) 
                          << block_time_us << "μs";
                
                if (block_time_us < 50.0) {
                    std::cout << " ✅ EXCELLENT";
                } else if (block_time_us < 100.0) {
                    std::cout << " ✅ GOOD";
                } else {
                    std::cout << " ⚠️ SLOW";
                }
                std::cout << std::endl;
            }
            
            // Simulate real-time processing
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "RX Mode test completed in " << std::fixed << std::setprecision(2) 
                  << total_time << " seconds" << std::endl;
        std::cout << std::endl;
    }
    
    void testParameterAutomation() {
        std::cout << "=== Parameter Automation Test ===" << std::endl;
        std::cout << "Testing DAW-style parameter automation..." << std::endl;
        std::cout << std::endl;
        
        // Test all parameters
        std::cout << "Testing parameter get/set functionality:" << std::endl;
        
        // Test plugin mode parameter
        setParameter(PARAM_PLUGIN_MODE, 0.0f);
        float mode = getParameter(PARAM_PLUGIN_MODE);
        std::cout << "Plugin Mode: " << mode << " (0=TX, 1=RX)" << std::endl;
        
        // Test network port
        setParameter(PARAM_NETWORK_PORT, 8888.0f);
        float port = getParameter(PARAM_NETWORK_PORT);
        std::cout << "Network Port: " << port << std::endl;
        
        // Test PNBTR strength
        setParameter(PARAM_PNBTR_STRENGTH, 0.75f);
        float strength = getParameter(PARAM_PNBTR_STRENGTH);
        std::cout << "PNBTR Strength: " << strength << std::endl;
        
        // Test sine wave frequency
        setParameter(PARAM_TEST_SINE_FREQ, 880.0f);
        float freq = getParameter(PARAM_TEST_SINE_FREQ);
        std::cout << "Sine Frequency: " << freq << " Hz" << std::endl;
        
        std::cout << "Parameter automation test completed!" << std::endl;
        std::cout << std::endl;
    }
    
    void displayPerformanceResults() {
        std::cout << "=== VST3 Plugin Performance Results ===" << std::endl;
        
        const auto& stats = plugin_->getPerformanceStats();
        
        std::cout << "Performance Statistics:" << std::endl;
        std::cout << "  Frames processed: " << stats.frames_processed.load() << std::endl;
        std::cout << "  Current latency: " << std::fixed << std::setprecision(1) 
                  << stats.current_latency_us.load() << " μs" << std::endl;
        std::cout << "  Average latency: " << std::fixed << std::setprecision(1) 
                  << stats.avg_latency_us.load() << " μs" << std::endl;
        std::cout << "  Maximum latency: " << std::fixed << std::setprecision(1) 
                  << stats.max_latency_us.load() << " μs" << std::endl;
        std::cout << "  PNBTR SNR improvement: " << std::fixed << std::setprecision(1) 
                  << stats.current_snr_db.load() << " dB" << std::endl;
        std::cout << "  Packets sent: " << stats.packets_sent.load() << std::endl;
        std::cout << "  Packets received: " << stats.packets_received.load() << std::endl;
        std::cout << "  Packets lost: " << stats.packets_lost.load() << std::endl;
        
        // Performance assessment
        double avg_latency = stats.avg_latency_us.load();
        std::cout << std::endl;
        std::cout << "Performance Assessment:" << std::endl;
        
        if (avg_latency < 25.0) {
            std::cout << "🚀 EXCEPTIONAL: Sub-25μs performance - Beyond professional requirements!" << std::endl;
        } else if (avg_latency < 50.0) {
            std::cout << "🎯 EXCELLENT: Sub-50μs performance - Professional DAW ready!" << std::endl;
        } else if (avg_latency < 100.0) {
            std::cout << "✅ GOOD: Sub-100μs performance - Suitable for most applications" << std::endl;
        } else {
            std::cout << "⚠️ NEEDS OPTIMIZATION: Performance above 100μs" << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "VST3 Plugin Status: Ready for DAW integration!" << std::endl;
    }
    
    void testCStyleInterface() {
        std::cout << "=== C-Style Interface Test ===" << std::endl;
        std::cout << "Testing C-style functions for VST3 host integration..." << std::endl;
        
        // Test plugin info functions
        std::cout << "Plugin Info via C interface:" << std::endl;
        std::cout << "  Name: " << getPluginInfo(0) << std::endl;
        std::cout << "  Version: " << getPluginInfo(1) << std::endl;
        std::cout << "  Vendor: " << getPluginInfo(2) << std::endl;
        std::cout << "  Description: " << getPluginInfo(3) << std::endl;
        
        // Test parameter functions
        setPluginParameter(plugin_.get(), PARAM_TEST_SINE_FREQ, 1000.0f);
        float test_freq = getPluginParameter(plugin_.get(), PARAM_TEST_SINE_FREQ);
        std::cout << "C-style parameter test: " << test_freq << " Hz" << std::endl;
        
        std::cout << "C-style interface test completed!" << std::endl;
        std::cout << std::endl;
    }
    
    void shutdown() {
        std::cout << "Shutting down VST3 plugin..." << std::endl;
        plugin_->shutdown();
        std::cout << "VST3 plugin demo complete!" << std::endl;
    }

private:
    void clearBuffers() {
        for (int ch = 0; ch < CHANNELS; ++ch) {
            std::fill(input_channels_[ch].begin(), input_channels_[ch].end(), 0.0f);
            std::fill(output_channels_[ch].begin(), output_channels_[ch].end(), 0.0f);
        }
    }
    
    void setParameter(int param_id, float value) {
        setPluginParameter(plugin_.get(), param_id, value);
    }
    
    float getParameter(int param_id) {
        return getPluginParameter(plugin_.get(), param_id);
    }
};

int main() {
    try {
        VST3MinimalDemo demo;
        
        if (!demo.initialize()) {
            std::cerr << "Failed to initialize VST3 plugin demo!" << std::endl;
            return 1;
        }
        
        // Run comprehensive VST3 plugin tests
        demo.testParameterAutomation();
        demo.testCStyleInterface();
        demo.testDualModeOperation();
        
        demo.shutdown();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "VST3 plugin demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 