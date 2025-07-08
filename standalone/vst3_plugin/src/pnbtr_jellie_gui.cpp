/*
 * PNBTR+JELLIE VST3 Plugin Interactive GUI Application
 * Revolutionary zero-noise dither replacement + 8-channel redundant streaming
 */

#include "PnbtrJelliePlugin.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <csignal>

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

class PnbtrJellieGUI {
private:
    pnbtr_jellie::PnbtrJellieEngine txEngine;
    pnbtr_jellie::PnbtrJellieEngine rxEngine;
    
    // State
    std::atomic<bool> isRunning{false};
    std::atomic<bool> isTxMode{true};
    std::atomic<float> pnbtrStrength{0.75f};
    std::atomic<float> packetLoss{5.0f};
    std::atomic<float> sineFreq{440.0f};
    
    // Metrics
    std::atomic<float> currentLatency{0.0f};
    std::atomic<float> avgLatency{0.0f};
    std::atomic<float> snrImprovement{7.0f};
    std::atomic<int> packetsProcessed{0};
    
    // Threading
    std::thread audioThread;
    std::thread displayThread;
    std::atomic<bool> shouldStop{false};
    
    // Terminal control
    struct termios oldTermios;
    bool terminalSetup = false;
    
public:
    PnbtrJellieGUI() {
        setupTerminal();
        initializeEngines();
        setupSignalHandlers();
    }
    
    ~PnbtrJellieGUI() {
        cleanup();
    }
    
    void setupTerminal() {
#ifndef _WIN32
        tcgetattr(STDIN_FILENO, &oldTermios);
        struct termios newTermios = oldTermios;
        newTermios.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newTermios);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
        terminalSetup = true;
#endif
    }
    
    void restoreTerminal() {
#ifndef _WIN32
        if (terminalSetup) {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldTermios);
        }
#endif
    }
    
    void setupSignalHandlers() {
        std::signal(SIGINT, [](int) {
            std::cout << "\n🛑 Shutting down gracefully..." << std::endl;
            exit(0);
        });
    }
    
    void initializeEngines() {
        std::cout << "🎮 Initializing PNBTR+JELLIE engines..." << std::endl;
        
        // Initialize both engines
        txEngine.initialize(48000, 512);
        rxEngine.initialize(48000, 512);
        
        // Configure TX engine
        txEngine.setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::TX_MODE);
        
        // Configure RX engine
        rxEngine.setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::RX_MODE);
        
        updateEngineSettings();
        
        std::cout << "✅ Both TX and RX engines initialized" << std::endl;
    }
    
    void updateEngineSettings() {
        // Network configuration
        pnbtr_jellie::PnbtrJellieEngine::NetworkConfig networkConfig;
        networkConfig.target_ip = "239.255.0.1";
        networkConfig.target_port = 8888;
        networkConfig.enable_multicast = true;
        networkConfig.redundancy_level = 4;
        
        txEngine.setNetworkConfig(networkConfig);
        rxEngine.setNetworkConfig(networkConfig);
        
        // PNBTR configuration
        pnbtr_jellie::PnbtrJellieEngine::PnbtrConfig pnbtrConfig;
        pnbtrConfig.enable_reconstruction = true;
        pnbtrConfig.prediction_strength = pnbtrStrength.load();
        pnbtrConfig.prediction_window_ms = 50;
        pnbtrConfig.enable_zero_noise_dither = true;
        
        txEngine.setPnbtrConfig(pnbtrConfig);
        rxEngine.setPnbtrConfig(pnbtrConfig);
        
        // Test configuration
        pnbtr_jellie::PnbtrJellieEngine::TestConfig testConfig;
        testConfig.enable_sine_generator = true;
        testConfig.sine_frequency_hz = sineFreq.load();
        testConfig.sine_amplitude = 0.5f;
        testConfig.enable_packet_loss_simulation = true;
        testConfig.packet_loss_percentage = packetLoss.load();
        testConfig.enable_latency_monitoring = true;
        
        txEngine.setTestConfig(testConfig);
        rxEngine.setTestConfig(testConfig);
    }
    
    void run() {
        clearScreen();
        showWelcome();
        
        startAudioThread();
        startDisplayThread();
        
        // Main input loop
        while (!shouldStop.load()) {
            handleInput();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Wait for threads
        if (audioThread.joinable()) audioThread.join();
        if (displayThread.joinable()) displayThread.join();
    }
    
    void showWelcome() {
        std::cout << "\n";
        std::cout << "🚀 PNBTR+JELLIE VST3 Plugin Interactive GUI\n";
        std::cout << "═══════════════════════════════════════════════\n";
        std::cout << "Revolutionary Zero-Noise + 8-Channel Streaming\n";
        std::cout << "Sub-100μs Latency • GPU-Accelerated\n";
        std::cout << "═══════════════════════════════════════════════\n";
        std::cout << "\n";
        std::cout << "🎛️  CONTROLS:\n";
        std::cout << "   [SPACE] Start/Stop   [M] TX/RX Mode\n";
        std::cout << "   [↑/↓] PNBTR Strength [←/→] Packet Loss\n";
        std::cout << "   [Q/W] Frequency      [ESC] Exit\n";
        std::cout << "\n";
    }
    
    void startAudioThread() {
        audioThread = std::thread([this]() {
            const int bufferSize = 512;
            const int channels = 2;
            float inputBuffer[bufferSize * channels] = {0};
            float outputBuffer[bufferSize * channels];
            
            while (!shouldStop.load()) {
                if (isRunning.load()) {
                    if (isTxMode.load()) {
                        txEngine.processAudio(inputBuffer, outputBuffer, bufferSize, channels);
                    } else {
                        rxEngine.processAudio(inputBuffer, outputBuffer, bufferSize, channels);
                    }
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        });
    }
    
    void startDisplayThread() {
        displayThread = std::thread([this]() {
            while (!shouldStop.load()) {
                updateDisplay();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    
    void updateDisplay() {
        clearScreen();
        
        std::cout << "🚀 PNBTR+JELLIE VST3 Plugin - Live Status\n";
        std::cout << "════════════════════════════════════════════\n";
        
        // Status line
        std::string status = isRunning.load() ? "🔴 PROCESSING" : "⏸️  STOPPED";
        std::string mode = isTxMode.load() ? "🔊 TX (Transmit)" : "🔉 RX (Receive)";
        std::cout << "Status: " << status << "  |  Mode: " << mode << "\n\n";
        
        // Get performance stats
        auto currentEngine = isTxMode.load() ? &txEngine : &rxEngine;
        const auto& stats = currentEngine->getPerformanceStats();
        
        // Performance
        std::cout << "📊 PERFORMANCE:\n";
        std::cout << "   Current Latency: " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << stats.current_latency_us.load() << " μs";
        
        if (stats.current_latency_us.load() < 100.0) {
            std::cout << " ✅";
        } else {
            std::cout << " ⚠️";
        }
        std::cout << "\n";
        
        std::cout << "   Average Latency: " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << stats.avg_processing_time_us.load() << " μs\n";
        std::cout << "   Max Latency:     " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << stats.max_processing_time_us.load() << " μs\n";
        std::cout << "   SNR Gain:        " << std::setw(8) << std::fixed << std::setprecision(1) 
                  << stats.current_snr_db.load() << " dB\n";
        std::cout << "   Frames:          " << std::setw(8) << stats.frames_processed.load() << "\n";
        std::cout << "   Packets Sent:    " << std::setw(8) << stats.packets_sent.load() << "\n";
        std::cout << "   Packets Lost:    " << std::setw(8) << stats.packets_lost.load() << "\n\n";
        
        // Settings
        std::cout << "🎛️  SETTINGS:\n";
        std::cout << "   PNBTR Strength:  " << std::setw(6) << std::fixed << std::setprecision(2) 
                  << pnbtrStrength.load() << "\n";
        std::cout << "   Packet Loss:     " << std::setw(6) << std::fixed << std::setprecision(1) 
                  << packetLoss.load() << " %\n";
        std::cout << "   Sine Frequency:  " << std::setw(6) << std::fixed << std::setprecision(0) 
                  << sineFreq.load() << " Hz\n\n";
        
        // Network
        std::cout << "🌐 NETWORK:\n";
        std::cout << "   Multicast: 239.255.0.1:8888\n";
        std::cout << "   JELLIE 8-Channel: Active\n";
        std::cout << "   ADAT Redundancy: Enabled\n\n";
        
        std::cout << "🎯 [SPACE] Start/Stop  [M] Mode  [↑↓] PNBTR  [←→] Loss  [ESC] Exit\n";
        std::cout << std::flush;
    }
    
    void handleInput() {
        char key = getKey();
        if (key == 0) return;
        
        switch (key) {
            case ' ':
                isRunning.store(!isRunning.load());
                break;
            case 'm':
            case 'M':
                isTxMode.store(!isTxMode.load());
                break;
            case 'A': // Up arrow
                adjustPnbtrStrength(0.05f);
                break;
            case 'B': // Down arrow
                adjustPnbtrStrength(-0.05f);
                break;
            case 'C': // Right arrow
                adjustPacketLoss(1.0f);
                break;
            case 'D': // Left arrow
                adjustPacketLoss(-1.0f);
                break;
            case 'q':
            case 'Q':
                adjustSineFreq(-50.0f);
                break;
            case 'w':
            case 'W':
                adjustSineFreq(50.0f);
                break;
            case 27: // ESC
                shouldStop.store(true);
                break;
        }
        
        updateEngineSettings();
    }
    
    char getKey() {
#ifdef _WIN32
        if (_kbhit()) {
            return _getch();
        }
        return 0;
#else
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            if (ch == 27) { // ESC sequence
                char seq[2];
                if (read(STDIN_FILENO, &seq, 2) == 2) {
                    if (seq[0] == '[') {
                        return seq[1]; // Return A, B, C, D for arrows
                    }
                }
                return 27; // Just ESC
            }
            return ch;
        }
        return 0;
#endif
    }
    
    void adjustPnbtrStrength(float delta) {
        float newValue = std::max(0.0f, std::min(1.0f, pnbtrStrength.load() + delta));
        pnbtrStrength.store(newValue);
    }
    
    void adjustPacketLoss(float delta) {
        float newValue = std::max(0.0f, std::min(50.0f, packetLoss.load() + delta));
        packetLoss.store(newValue);
    }
    
    void adjustSineFreq(float delta) {
        float newValue = std::max(100.0f, std::min(2000.0f, sineFreq.load() + delta));
        sineFreq.store(newValue);
    }
    
    void clearScreen() {
        std::cout << "\033[2J\033[H" << std::flush;
    }
    
    void cleanup() {
        shouldStop.store(true);
        
        if (audioThread.joinable()) {
            audioThread.join();
        }
        
        if (displayThread.joinable()) {
            displayThread.join();
        }
        
        restoreTerminal();
    }
};

int main() {
    try {
        std::cout << "🚀 PNBTR+JELLIE VST3 Plugin GUI Starting..." << std::endl;
        
        PnbtrJellieGUI gui;
        gui.run();
        
        std::cout << "\n👋 Shutting down..." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
} 