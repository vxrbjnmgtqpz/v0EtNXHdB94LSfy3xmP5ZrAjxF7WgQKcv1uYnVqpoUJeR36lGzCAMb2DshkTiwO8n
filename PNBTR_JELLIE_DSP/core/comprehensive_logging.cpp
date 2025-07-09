#include "comprehensive_logging.h"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace pnbtr_jellie {

ComprehensiveLogger::ComprehensiveLogger(const LoggingConfig& config) 
    : config_(config) {
    stats_.session_start_time = std::chrono::high_resolution_clock::now();
    last_stats_update_ = stats_.session_start_time;
}

ComprehensiveLogger::~ComprehensiveLogger() {
    shutdown();
}

bool ComprehensiveLogger::initialize() {
    if (is_initialized_.load()) {
        return false;
    }
    
    current_session_id_ = generateSessionId();
    session_start_time_ = std::chrono::high_resolution_clock::now();
    
    createLogFiles();
    
    if (config_.use_background_thread) {
        is_running_.store(true);
        logging_thread_ = std::thread(&ComprehensiveLogger::loggingThreadFunction, this);
    }
    
    is_initialized_.store(true);
    return true;
}

void ComprehensiveLogger::shutdown() {
    if (!is_initialized_.load()) {
        return;
    }
    
    if (is_running_.load()) {
        is_running_.store(false);
        if (logging_thread_.joinable()) {
            logging_thread_.join();
        }
    }
    
    flushBuffers();
    closeLogFiles();
    
    is_initialized_.store(false);
}

void ComprehensiveLogger::updateConfig(const LoggingConfig& config) {
    config_ = config;
}

void ComprehensiveLogger::logAudioData(const AudioLogEntry& entry) {
    if (!is_initialized_.load() || !config_.enable_audio_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(audio_mutex_);
    audio_queue_.push(entry);
    
    if (audio_queue_.size() > MAX_QUEUE_SIZE) {
        audio_queue_.pop(); // Remove oldest entry
    }
    
    stats_.total_audio_entries++;
}

void ComprehensiveLogger::logPnbtrAction(const PnbtrActionLogEntry& entry) {
    if (!is_initialized_.load() || !config_.enable_pnbtr_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pnbtr_mutex_);
    pnbtr_queue_.push(entry);
    
    if (pnbtr_queue_.size() > MAX_QUEUE_SIZE) {
        pnbtr_queue_.pop();
    }
    
    stats_.total_pnbtr_entries++;
}

void ComprehensiveLogger::logNetworkEvent(const NetworkLogEntry& entry) {
    if (!is_initialized_.load() || !config_.enable_network_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(network_mutex_);
    network_queue_.push(entry);
    
    if (network_queue_.size() > MAX_QUEUE_SIZE) {
        network_queue_.pop();
    }
    
    stats_.total_network_entries++;
}

void ComprehensiveLogger::logQualityMetrics(const QualityLogEntry& entry) {
    if (!is_initialized_.load() || !config_.enable_quality_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(quality_mutex_);
    quality_queue_.push(entry);
    
    if (quality_queue_.size() > MAX_QUEUE_SIZE) {
        quality_queue_.pop();
    }
    
    stats_.total_quality_entries++;
}

void ComprehensiveLogger::logAudioDataBatch(const std::vector<AudioLogEntry>& entries) {
    if (!is_initialized_.load() || !config_.enable_audio_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(audio_mutex_);
    for (const auto& entry : entries) {
        audio_queue_.push(entry);
        
        if (audio_queue_.size() > MAX_QUEUE_SIZE) {
            audio_queue_.pop();
        }
    }
    
    stats_.total_audio_entries += entries.size();
}

void ComprehensiveLogger::logPnbtrActionBatch(const std::vector<PnbtrActionLogEntry>& entries) {
    if (!is_initialized_.load() || !config_.enable_pnbtr_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pnbtr_mutex_);
    for (const auto& entry : entries) {
        pnbtr_queue_.push(entry);
        
        if (pnbtr_queue_.size() > MAX_QUEUE_SIZE) {
            pnbtr_queue_.pop();
        }
    }
    
    stats_.total_pnbtr_entries += entries.size();
}

void ComprehensiveLogger::startNewSession(const std::string& session_name) {
    endCurrentSession();
    
    if (!session_name.empty()) {
        current_session_id_ = session_name;
    } else {
        current_session_id_ = generateSessionId();
    }
    
    session_start_time_ = std::chrono::high_resolution_clock::now();
    createLogFiles();
}

void ComprehensiveLogger::endCurrentSession() {
    flushBuffers();
    closeLogFiles();
}

bool ComprehensiveLogger::verifyLogIntegrity() const {
    // Basic integrity check - would be more sophisticated in real implementation
    return is_initialized_.load() && !current_session_id_.empty();
}

size_t ComprehensiveLogger::getTotalLoggedEntries() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.total_audio_entries + stats_.total_pnbtr_entries + 
           stats_.total_network_entries + stats_.total_quality_entries;
}

double ComprehensiveLogger::getLoggedDataSizeGB() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return static_cast<double>(stats_.total_bytes_logged) / (1024.0 * 1024.0 * 1024.0);
}

bool ComprehensiveLogger::exportTrainingData(const std::string& output_path, 
                                           const std::string& format) {
    if (format == "numpy") {
        exportNumpyFormat(output_path);
    } else if (format == "csv") {
        exportCSVFormat(output_path);
    } else if (format == "json") {
        exportJSONFormat(output_path);
    } else {
        return false;
    }
    
    return true;
}

void ComprehensiveLogger::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = LoggingStats{};
    stats_.session_start_time = std::chrono::high_resolution_clock::now();
}

// Private implementation methods
void ComprehensiveLogger::loggingThreadFunction() {
    while (is_running_.load()) {
        flushBuffers();
        
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int64_t>(config_.flush_interval_seconds * 1000))
        );
    }
}

void ComprehensiveLogger::createLogFiles() {
    if (config_.use_binary_format) {
        // Create binary log files
        audio_file_ = std::make_unique<std::ofstream>(
            generateLogFilePath("audio"), std::ios::binary);
        pnbtr_file_ = std::make_unique<std::ofstream>(
            generateLogFilePath("pnbtr"), std::ios::binary);
        network_file_ = std::make_unique<std::ofstream>(
            generateLogFilePath("network"), std::ios::binary);
        quality_file_ = std::make_unique<std::ofstream>(
            generateLogFilePath("quality"), std::ios::binary);
    } else {
        // Create text log files
        audio_file_ = std::make_unique<std::ofstream>(generateLogFilePath("audio"));
        pnbtr_file_ = std::make_unique<std::ofstream>(generateLogFilePath("pnbtr"));
        network_file_ = std::make_unique<std::ofstream>(generateLogFilePath("network"));
        quality_file_ = std::make_unique<std::ofstream>(generateLogFilePath("quality"));
    }
}

void ComprehensiveLogger::closeLogFiles() {
    audio_file_.reset();
    pnbtr_file_.reset();
    network_file_.reset();
    quality_file_.reset();
}

void ComprehensiveLogger::rotateLogFiles() {
    // Simple rotation - would be more sophisticated in real implementation
    closeLogFiles();
    createLogFiles();
}

std::string ComprehensiveLogger::generateSessionId() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << config_.session_prefix << "_" 
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    return ss.str();
}

std::string ComprehensiveLogger::generateLogFilePath(const std::string& type) const {
    std::string extension = config_.use_binary_format ? ".bin" : ".log";
    return config_.log_directory + current_session_id_ + "_" + type + extension;
}

void ComprehensiveLogger::serializeAudioEntry(const AudioLogEntry& entry, std::ostream& stream) {
    // Simplified serialization - would be more sophisticated in real implementation
    if (config_.use_binary_format) {
        stream.write(reinterpret_cast<const char*>(&entry.sequence_number), sizeof(entry.sequence_number));
        // Write other fields...
    } else {
        stream << "Audio," << entry.sequence_number << "," << entry.snr_db << std::endl;
    }
}

void ComprehensiveLogger::serializePnbtrEntry(const PnbtrActionLogEntry& entry, std::ostream& stream) {
    if (config_.use_binary_format) {
        stream.write(reinterpret_cast<const char*>(&entry.sequence_number), sizeof(entry.sequence_number));
        // Write other fields...
    } else {
        stream << "PNBTR," << entry.sequence_number << "," << static_cast<int>(entry.action_type) << std::endl;
    }
}

void ComprehensiveLogger::serializeNetworkEntry(const NetworkLogEntry& entry, std::ostream& stream) {
    if (config_.use_binary_format) {
        stream.write(reinterpret_cast<const char*>(&entry.sequence_number), sizeof(entry.sequence_number));
        // Write other fields...
    } else {
        stream << "Network," << entry.sequence_number << "," << entry.latency_ms << std::endl;
    }
}

void ComprehensiveLogger::serializeQualityEntry(const QualityLogEntry& entry, std::ostream& stream) {
    if (config_.use_binary_format) {
        stream.write(reinterpret_cast<const char*>(&entry.sequence_number), sizeof(entry.sequence_number));
        // Write other fields...
    } else {
        stream << "Quality," << entry.sequence_number << "," << entry.end_to_end_snr_db << std::endl;
    }
}

void ComprehensiveLogger::flushBuffers() {
    // Process audio queue
    {
        std::lock_guard<std::mutex> lock(audio_mutex_);
        while (!audio_queue_.empty() && audio_file_) {
            serializeAudioEntry(audio_queue_.front(), *audio_file_);
            audio_queue_.pop();
        }
        if (audio_file_) {
            audio_file_->flush();
        }
    }
    
    // Process PNBTR queue
    {
        std::lock_guard<std::mutex> lock(pnbtr_mutex_);
        while (!pnbtr_queue_.empty() && pnbtr_file_) {
            serializePnbtrEntry(pnbtr_queue_.front(), *pnbtr_file_);
            pnbtr_queue_.pop();
        }
        if (pnbtr_file_) {
            pnbtr_file_->flush();
        }
    }
    
    // Process network queue
    {
        std::lock_guard<std::mutex> lock(network_mutex_);
        while (!network_queue_.empty() && network_file_) {
            serializeNetworkEntry(network_queue_.front(), *network_file_);
            network_queue_.pop();
        }
        if (network_file_) {
            network_file_->flush();
        }
    }
    
    // Process quality queue
    {
        std::lock_guard<std::mutex> lock(quality_mutex_);
        while (!quality_queue_.empty() && quality_file_) {
            serializeQualityEntry(quality_queue_.front(), *quality_file_);
            quality_queue_.pop();
        }
        if (quality_file_) {
            quality_file_->flush();
        }
    }
}

void ComprehensiveLogger::exportNumpyFormat(const std::string& output_path) {
    // Placeholder for numpy export
    // Would use a library like cnpy or implement custom binary format
}

void ComprehensiveLogger::exportCSVFormat(const std::string& output_path) {
    std::ofstream csv_file(output_path + "/training_data.csv");
    csv_file << "timestamp,type,sequence,value1,value2,value3\n";
    
    // Export data from queues to CSV
    // This is a simplified implementation
    csv_file.close();
}

void ComprehensiveLogger::exportJSONFormat(const std::string& output_path) {
    std::ofstream json_file(output_path + "/training_data.json");
    json_file << "{\n";
    json_file << "  \"session_id\": \"" << current_session_id_ << "\",\n";
    json_file << "  \"data\": []\n";
    json_file << "}\n";
    json_file.close();
}

void ComprehensiveLogger::compressOldLogs() {
    // Placeholder for log compression
}

void ComprehensiveLogger::cleanupOldFiles() {
    // Placeholder for old file cleanup
}

// LogMonitor implementation (stub for linking)
LogMonitor::LogMonitor(ComprehensiveLogger* logger) {
    // Stub implementation for now
    (void)logger;
}

} // namespace pnbtr_jellie
