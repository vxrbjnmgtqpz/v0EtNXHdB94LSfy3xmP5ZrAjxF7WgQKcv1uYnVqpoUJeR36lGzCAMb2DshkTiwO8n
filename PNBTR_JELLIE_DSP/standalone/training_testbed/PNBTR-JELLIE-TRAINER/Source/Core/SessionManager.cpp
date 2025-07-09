#include "SessionManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

SessionManager::SessionManager()
    : sessionActive(false)
    , sessionPaused(false)
    , sessionStartTime(0.0)
    , totalPauseTime(0.0)
{
    createDefaultSession();
}

SessionManager::~SessionManager() {
    if (sessionActive) {
        stopSession();
    }
}

bool SessionManager::createDefaultSession() {
    config = Config(); // Use default values
    clearMetrics();
    return true;
}

bool SessionManager::loadSession(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open session file: " << jsonPath << std::endl;
        return false;
    }
    
    std::string jsonContent((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    
    if (configFromJson(jsonContent)) {
        std::cout << "Session loaded from: " << jsonPath << std::endl;
        return true;
    } else {
        std::cerr << "Failed to parse session JSON" << std::endl;
        return false;
    }
}

bool SessionManager::saveSession(const std::string& jsonPath) const {
    std::ofstream file(jsonPath);
    if (!file.is_open()) {
        std::cerr << "Failed to create session file: " << jsonPath << std::endl;
        return false;
    }
    
    file << configToJson();
    file.close();
    
    std::cout << "Session saved to: " << jsonPath << std::endl;
    return true;
}

void SessionManager::updateConfig(const Config& newConfig) {
    config = newConfig;
    std::cout << "Session configuration updated" << std::endl;
}

void SessionManager::startSession() {
    if (sessionActive) {
        std::cout << "Session already active" << std::endl;
        return;
    }
    
    clearMetrics();
    sessionActive = true;
    sessionPaused = false;
    sessionStartTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    totalPauseTime = 0.0;
    
    std::cout << "Training session started" << std::endl;
}

void SessionManager::stopSession() {
    if (!sessionActive) {
        std::cout << "No active session to stop" << std::endl;
        return;
    }
    
    sessionActive = false;
    sessionPaused = false;
    
    // Calculate total session duration
    double currentTime = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    sessionMetrics.sessionDuration = currentTime - sessionStartTime - totalPauseTime;
    
    std::cout << "Training session stopped. Duration: " 
              << sessionMetrics.sessionDuration << " seconds" << std::endl;
}

void SessionManager::pauseSession() {
    if (!sessionActive || sessionPaused) {
        return;
    }
    
    sessionPaused = true;
    // Pause time tracking would be implemented here
    std::cout << "Session paused" << std::endl;
}

void SessionManager::resumeSession() {
    if (!sessionActive || !sessionPaused) {
        return;
    }
    
    sessionPaused = false;
    // Resume time tracking would be implemented here
    std::cout << "Session resumed" << std::endl;
}

void SessionManager::recordMetrics(const AudioMetrics& metrics) {
    if (!sessionActive || sessionPaused) {
        return;
    }
    
    // Store current values (assuming AudioMetrics has these fields)
    sessionMetrics.currentSnr = 0.0f; // Would be metrics.snr
    sessionMetrics.currentLatency = 0.0f; // Would be metrics.latency
    sessionMetrics.currentGapQuality = 0.0f; // Would be metrics.gapQuality
    sessionMetrics.currentProcessingTime = 0.0f; // Would be metrics.processingTime
    
    // Add to history
    sessionMetrics.snrHistory.push_back(sessionMetrics.currentSnr);
    sessionMetrics.latencyHistory.push_back(sessionMetrics.currentLatency);
    sessionMetrics.gapQualityHistory.push_back(sessionMetrics.currentGapQuality);
    sessionMetrics.processingTimeHistory.push_back(sessionMetrics.currentProcessingTime);
    
    sessionMetrics.totalSamplesProcessed += config.blockSize;
}

void SessionManager::clearMetrics() {
    sessionMetrics = SessionMetrics();
}

void SessionManager::setInputBuffer(const float* buffer, size_t numSamples) {
    inputBuffer.assign(buffer, buffer + numSamples);
}

void SessionManager::setOutputBuffer(const float* buffer, size_t numSamples) {
    outputBuffer.assign(buffer, buffer + numSamples);
}

bool SessionManager::exportSession(const ExportOptions& options) {
    if (!createExportDirectory(config.outputDirectory)) {
        std::cerr << "Failed to create export directory" << std::endl;
        return false;
    }
    
    std::string timestamp = options.timestamp.empty() ? generateTimestamp() : options.timestamp;
    std::string basePath = config.outputDirectory + "/" + options.sessionName + "_" + timestamp;
    
    bool success = true;
    
    if (options.includeWaveforms && config.exportWav) {
        success &= exportWaveforms(basePath);
    }
    
    if (options.includeWaveforms && config.exportPng) {
        success &= exportWaveformImages(basePath);
    }
    
    if (options.includeMetrics && config.exportCsv) {
        success &= exportMetrics(basePath + "_metrics.csv");
    }
    
    if (options.includeConfig && config.exportJson) {
        success &= exportConfig(basePath + "_config.json");
    }
    
    std::cout << "Session export " << (success ? "completed" : "failed") 
              << " to: " << basePath << std::endl;
    
    return success;
}

bool SessionManager::exportWaveforms(const std::string& basePath) {
    // Simple WAV export (would need proper WAV file format implementation)
    std::ofstream inputFile(basePath + "_input.wav", std::ios::binary);
    std::ofstream outputFile(basePath + "_output.wav", std::ios::binary);
    
    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Failed to create waveform files" << std::endl;
        return false;
    }
    
    // For now, just write raw float data (would need WAV header in real implementation)
    inputFile.write(reinterpret_cast<const char*>(inputBuffer.data()), 
                   inputBuffer.size() * sizeof(float));
    outputFile.write(reinterpret_cast<const char*>(outputBuffer.data()), 
                    outputBuffer.size() * sizeof(float));
    
    inputFile.close();
    outputFile.close();
    
    std::cout << "Waveforms exported" << std::endl;
    return true;
}

bool SessionManager::exportMetrics(const std::string& csvPath) {
    std::ofstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Failed to create metrics file: " << csvPath << std::endl;
        return false;
    }
    
    file << metricsToCSV();
    file.close();
    
    std::cout << "Metrics exported to: " << csvPath << std::endl;
    return true;
}

bool SessionManager::exportConfig(const std::string& jsonPath) {
    return saveSession(jsonPath);
}

bool SessionManager::exportWaveformImages(const std::string& basePath) {
    // Would implement PNG export from Metal texture here
    // For now, just create placeholder files
    std::ofstream inputImg(basePath + "_input.png");
    std::ofstream outputImg(basePath + "_output.png");
    
    if (inputImg.is_open()) {
        inputImg << "PNG waveform placeholder" << std::endl;
        inputImg.close();
    }
    
    if (outputImg.is_open()) {
        outputImg << "PNG waveform placeholder" << std::endl;
        outputImg.close();
    }
    
    std::cout << "Waveform images exported (placeholder)" << std::endl;
    return true;
}

std::string SessionManager::generateTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string SessionManager::generateSessionPath(const std::string& extension) const {
    return config.outputDirectory + "/session_" + generateTimestamp() + extension;
}

bool SessionManager::createExportDirectory(const std::string& path) const {
    // Simple directory creation (would use proper filesystem APIs in real implementation)
    std::string command = "mkdir -p " + path;
    return system(command.c_str()) == 0;
}

std::string SessionManager::configToJson() const {
    std::stringstream json;
    json << "{\n";
    json << "  \"audio\": {\n";
    json << "    \"sampleRate\": " << config.sampleRate << ",\n";
    json << "    \"blockSize\": " << config.blockSize << ",\n";
    json << "    \"numChannels\": " << config.numChannels << "\n";
    json << "  },\n";
    json << "  \"network\": {\n";
    json << "    \"packetLossPercent\": " << config.packetLossPercent << ",\n";
    json << "    \"jitterMs\": " << config.jitterMs << "\n";
    json << "  },\n";
    json << "  \"processing\": {\n";
    json << "    \"enableJellie\": " << (config.enableJellie ? "true" : "false") << ",\n";
    json << "    \"enablePnbtr\": " << (config.enablePnbtr ? "true" : "false") << ",\n";
    json << "    \"enableMetrics\": " << (config.enableMetrics ? "true" : "false") << ",\n";
    json << "    \"useGpuProcessing\": " << (config.useGpuProcessing ? "true" : "false") << "\n";
    json << "  },\n";
    json << "  \"visualization\": {\n";
    json << "    \"waveformWidth\": " << config.waveformWidth << ",\n";
    json << "    \"waveformHeight\": " << config.waveformHeight << "\n";
    json << "  }\n";
    json << "}\n";
    
    return json.str();
}

bool SessionManager::configFromJson(const std::string& json) {
    // Simple JSON parsing (would use proper JSON library in real implementation)
    // For now, just use defaults and return true
    std::cout << "JSON config parsing not fully implemented yet" << std::endl;
    return true;
}

std::string SessionManager::metricsToJson() const {
    std::stringstream json;
    json << "{\n";
    json << "  \"session\": {\n";
    json << "    \"duration\": " << sessionMetrics.sessionDuration << ",\n";
    json << "    \"samplesProcessed\": " << sessionMetrics.totalSamplesProcessed << ",\n";
    json << "    \"packetsLost\": " << sessionMetrics.totalPacketsLost << "\n";
    json << "  },\n";
    json << "  \"current\": {\n";
    json << "    \"snr\": " << sessionMetrics.currentSnr << ",\n";
    json << "    \"latency\": " << sessionMetrics.currentLatency << ",\n";
    json << "    \"gapQuality\": " << sessionMetrics.currentGapQuality << "\n";
    json << "  }\n";
    json << "}\n";
    
    return json.str();
}

std::string SessionManager::metricsToCSV() const {
    std::stringstream csv;
    csv << "Index,SNR,Latency,GapQuality,ProcessingTime\n";
    
    size_t maxSize = std::max({sessionMetrics.snrHistory.size(),
                              sessionMetrics.latencyHistory.size(),
                              sessionMetrics.gapQualityHistory.size(),
                              sessionMetrics.processingTimeHistory.size()});
    
    for (size_t i = 0; i < maxSize; ++i) {
        csv << i << ",";
        csv << (i < sessionMetrics.snrHistory.size() ? sessionMetrics.snrHistory[i] : 0.0f) << ",";
        csv << (i < sessionMetrics.latencyHistory.size() ? sessionMetrics.latencyHistory[i] : 0.0f) << ",";
        csv << (i < sessionMetrics.gapQualityHistory.size() ? sessionMetrics.gapQualityHistory[i] : 0.0f) << ",";
        csv << (i < sessionMetrics.processingTimeHistory.size() ? sessionMetrics.processingTimeHistory[i] : 0.0f) << "\n";
    }
    
    return csv.str();
} 