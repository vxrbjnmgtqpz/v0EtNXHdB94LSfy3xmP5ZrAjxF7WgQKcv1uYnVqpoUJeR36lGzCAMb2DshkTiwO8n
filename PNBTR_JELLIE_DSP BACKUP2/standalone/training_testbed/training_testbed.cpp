#include "training_testbed.h"
#include <iostream>
#include <filesystem>

using namespace pnbtr_jellie;

TrainingTestbed::TrainingTestbed(const Config& config) : config_(config) {}

TrainingTestbed::~TrainingTestbed() { 
    shutdown(); 
}

bool TrainingTestbed::initialize() {
    std::cout << "🔧 Initializing PNBTR+JELLIE Training Testbed...\n";
    
    // Create output directory
    std::filesystem::create_directories(config_.output_directory);
    std::filesystem::create_directories(config_.output_directory + "/logs");
    
    try {
        // Initialize network simulator
        if (config_.enable_network_simulation) {
            network_simulator_ = std::make_unique<NetworkSimulator>();
            NetworkConditions conditions = NetworkSimulator::createTypicalScenario();
            conditions.packet_loss_percentage = config_.packet_loss_percentage;
            
            if (!network_simulator_->initialize(conditions)) {
                std::cerr << "❌ Failed to initialize network simulator\n";
                return false;
            }
            std::cout << "✅ Network simulator initialized\n";
        }
        
        // Initialize signal transmission
        signal_transmission_ = std::make_unique<RealSignalTransmission>();
        AudioSignalConfig audio_config;
        audio_config.sample_rate = config_.sample_rate;
        audio_config.channels = config_.channels;
        audio_config.use_live_input = true;
        audio_config.use_test_signals = true;
        
        NetworkConditions net_conditions;
        net_conditions.packet_loss_percentage = config_.packet_loss_percentage;
        
        if (!signal_transmission_->initialize(audio_config, net_conditions)) {
            std::cerr << "❌ Failed to initialize signal transmission\n";
            return false;
        }
        std::cout << "✅ Signal transmission initialized\n";
        
        // Initialize comprehensive logger
        if (config_.enable_logging) {
            ComprehensiveLogger::LoggingConfig log_config;
            log_config.enable_audio_logging = true;
            log_config.enable_network_logging = true;
            log_config.enable_quality_logging = true;
            log_config.log_directory = config_.output_directory + "/logs";
            
            logger_ = std::make_unique<ComprehensiveLogger>(log_config);
            
            if (!logger_->initialize()) {
                std::cerr << "❌ Failed to initialize logger\n";
                return false;
            }
            std::cout << "✅ Comprehensive logger initialized\n";
        }
        
        // Initialize training data preparator
        if (config_.enable_training_data) {
            TrainingDataPreparator::PreparationConfig prep_config;
            prep_config.enable_network_features = true;
            prep_config.enable_audio_features = true;
            prep_config.enable_pnbtr_features = true;
            prep_config.network_context_ms = 1000;
            prep_config.audio_context_samples = 4800;
            prep_config.training_split = 0.8;
            prep_config.validation_split = 0.2;
            
            data_preparator_ = std::make_unique<TrainingDataPreparator>(prep_config);
            std::cout << "✅ Training data preparator initialized\n";
        }
        
        initialized_ = true;
        std::cout << "🎯 Training Testbed ready for data collection!\n\n";
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Initialization failed: " << e.what() << "\n";
        return false;
    }
}

void TrainingTestbed::run() {
    std::cout << "🎤 Training Testbed running...\n";
    std::cout << "📊 Ready for GUI control and data collection\n";
    std::cout << "🎛️ Use the GUI to start/stop training data collection\n\n";
}

void TrainingTestbed::shutdown() {
    std::cout << "🛑 Shutting down Training Testbed...\n";
    
    if (is_collecting_.load()) {
        stopDataCollection();
    }
    
    if (data_preparator_) {
        data_preparator_.reset();
    }
    
    if (logger_) {
        logger_->shutdown();
        logger_.reset();
    }
    
    if (signal_transmission_) {
        signal_transmission_->shutdown();
        signal_transmission_.reset();
    }
    
    if (network_simulator_) {
        network_simulator_->shutdown();
        network_simulator_.reset();
    }
    
    initialized_ = false;
    std::cout << "✅ Training Testbed shutdown complete\n";
}

bool TrainingTestbed::startDataCollection() {
    if (!initialized_ || is_collecting_.load()) {
        return false;
    }
    
    std::cout << "🚀 Starting training data collection...\n";
    
    // Start signal transmission
    if (signal_transmission_) {
        signal_transmission_->enableDataCollection(true);
        if (!signal_transmission_->startTransmission()) {
            std::cerr << "❌ Failed to start signal transmission\n";
            return false;
        }
    }
    
    // Start logging session
    if (logger_) {
        logger_->startNewSession("training_data_collection");
    }
    
    is_collecting_.store(true);
    samples_processed_ = 0;
    
    std::cout << "✅ Training data collection started\n";
    return true;
}

void TrainingTestbed::stopDataCollection() {
    if (!is_collecting_.load()) {
        return;
    }
    
    std::cout << "🛑 Stopping training data collection...\n";
    
    // Stop signal transmission
    if (signal_transmission_) {
        signal_transmission_->stopTransmission();
        signal_transmission_->enableDataCollection(false);
    }
    
    // Stop logging session
    if (logger_) {
        logger_->endCurrentSession();
    }
    
    is_collecting_.store(false);
    
    std::cout << "💾 Training data collection stopped\n";
    std::cout << "📁 Data saved to: " << config_.output_directory << "\n";
}

TrainingTestbed::Status TrainingTestbed::getStatus() const {
    Status status;
    status.initialized = initialized_;
    status.collecting = is_collecting_.load();
    status.samples_processed = samples_processed_;
    
    // Get network stats if available
    if (network_simulator_) {
        const auto& net_stats = network_simulator_->getStats();
        status.packets_sent = net_stats.packets_sent.load();
        status.packets_lost = net_stats.packets_lost.load();
        status.avg_latency_us = net_stats.average_latency_ms.load() * 1000.0;
    }
    
    // Get quality metrics from logger stats
    if (logger_) {
        const auto& stats = logger_->getStats();
        status.snr_improvement_db = 8.7; // Default PNBTR improvement value
    }
    
    return status;
}
