#pragma once

#include "network_simulator.h"
#include "signal_transmission.h"
#include "comprehensive_logging.h"
#include "training_preparation.h"
#include <string>
#include <memory>
#include <vector>
#include <atomic>

/**
 * @brief Simple Training Testbed for PNBTR+JELLIE DSP
 * Uses proven core components for training data collection
 */
class TrainingTestbed {
public:
    struct Config {
        uint32_t sample_rate = 48000;
        uint16_t channels = 2;
        std::string input_device;
        std::string output_directory = "training_output/";
        bool enable_logging = true;
        bool enable_training_data = true;
        bool enable_network_simulation = true;
        double packet_loss_percentage = 5.0;
    };

    TrainingTestbed(const Config& config);
    ~TrainingTestbed();

    bool initialize();
    void run();
    void shutdown();

    // Training control
    bool startDataCollection();
    void stopDataCollection();
    bool isCollecting() const { return is_collecting_.load(); }

    // Access to components for GUI
    pnbtr_jellie::NetworkSimulator* getNetworkSimulator() { return network_simulator_.get(); }
    pnbtr_jellie::RealSignalTransmission* getSignalTransmission() { return signal_transmission_.get(); }
    pnbtr_jellie::ComprehensiveLogger* getLogger() { return logger_.get(); }
    
    // Status and metrics
    struct Status {
        bool initialized = false;
        bool collecting = false;
        uint64_t samples_processed = 0;
        uint64_t packets_sent = 0;
        uint64_t packets_lost = 0;
        double avg_latency_us = 0.0;
        double snr_improvement_db = 0.0;
    };
    
    Status getStatus() const;

private:
    Config config_;
    std::atomic<bool> is_collecting_{false};
    
    // Proven core components
    std::unique_ptr<pnbtr_jellie::NetworkSimulator> network_simulator_;
    std::unique_ptr<pnbtr_jellie::RealSignalTransmission> signal_transmission_;
    std::unique_ptr<pnbtr_jellie::ComprehensiveLogger> logger_;
    std::unique_ptr<pnbtr_jellie::TrainingDataPreparator> data_preparator_;
    
    // Internal state
    bool initialized_ = false;
    uint64_t samples_processed_ = 0;
};
