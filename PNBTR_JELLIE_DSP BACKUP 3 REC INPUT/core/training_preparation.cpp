#include "training_preparation.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace pnbtr_jellie {

TrainingDataPreparator::TrainingDataPreparator(const PreparationConfig& config) 
    : config_(config) {
}

TrainingDataPreparator::~TrainingDataPreparator() {
}

bool TrainingDataPreparator::loadLoggedData(const std::vector<std::string>& session_ids, 
                                           const std::string& log_directory) {
    // Load logged data from multiple sessions
    for (const auto& session_id : session_ids) {
        // In a real implementation, this would load actual log files
        // For now, create placeholder data
        
        // Add some sample audio logs
        for (int i = 0; i < 100; ++i) {
            AudioLogEntry audio_log;
            audio_log.timestamp = std::chrono::high_resolution_clock::now();
            audio_log.sequence_number = i;
            audio_log.original_audio.resize(1024, 0.5f);
            audio_log.received_audio.resize(1024, 0.45f);
            audio_log.snr_db = 60.0 + (i % 20);
            audio_logs_.push_back(audio_log);
        }
        
        // Add some sample PNBTR logs
        for (int i = 0; i < 50; ++i) {
            PnbtrActionLogEntry pnbtr_log;
            pnbtr_log.timestamp = std::chrono::high_resolution_clock::now();
            pnbtr_log.sequence_number = i;
            pnbtr_log.action_type = static_cast<PnbtrActionLogEntry::ActionType>(i % 7);
            pnbtr_log.prediction_confidence = 0.8 + (i % 10) * 0.02;
            pnbtr_log.processing_time_us = 25.0 + (i % 15);
            pnbtr_logs_.push_back(pnbtr_log);
        }
        
        // Add some sample network logs
        for (int i = 0; i < 200; ++i) {
            NetworkLogEntry network_log;
            network_log.timestamp = std::chrono::high_resolution_clock::now();
            network_log.sequence_number = i;
            network_log.latency_ms = 20.0 + (i % 50);
            network_log.jitter_ms = 2.0 + (i % 10);
            network_log.was_lost = (i % 20) == 0;
            network_logs_.push_back(network_log);
        }
        
        // Add some sample quality logs
        for (int i = 0; i < 75; ++i) {
            QualityLogEntry quality_log;
            quality_log.timestamp = std::chrono::high_resolution_clock::now();
            quality_log.sequence_number = i;
            quality_log.end_to_end_snr_db = 55.0 + (i % 25);
            quality_log.passes_quality_threshold = (i % 3) != 0;
            quality_logs_.push_back(quality_log);
        }
    }
    
    return !audio_logs_.empty() && !network_logs_.empty();
}

bool TrainingDataPreparator::consolidateMultipleDatasets(const std::vector<std::string>& dataset_paths) {
    // Consolidate datasets from multiple sources
    for (const auto& path : dataset_paths) {
        // In real implementation, would load and merge datasets
        // For now, just indicate success
    }
    
    return true;
}

bool TrainingDataPreparator::extractFeatures() {
    if (audio_logs_.empty() || network_logs_.empty() || pnbtr_logs_.empty()) {
        return false;
    }
    
    training_samples_.clear();
    
    // Correlate log entries and extract features
    correlateLogEntries();
    
    // Extract features for each correlated sample
    for (size_t i = 0; i < std::min({audio_logs_.size(), network_logs_.size(), pnbtr_logs_.size()}); ++i) {
        TrainingSample sample;
        
        // Extract network features
        std::vector<NetworkLogEntry> network_context;
        for (size_t j = (i > 10 ? i - 10 : 0); j <= i && j < network_logs_.size(); ++j) {
            network_context.push_back(network_logs_[j]);
        }
        sample.network_features = extractNetworkFeatures(network_logs_[i], network_context);
        
        // Extract audio features
        sample.audio_features = extractAudioFeatures(audio_logs_[i]);
        
        // Extract PNBTR features
        std::vector<PnbtrActionLogEntry> pnbtr_context;
        for (size_t j = (i > 5 ? i - 5 : 0); j <= i && j < pnbtr_logs_.size(); ++j) {
            pnbtr_context.push_back(pnbtr_logs_[j]);
        }
        sample.pnbtr_features = extractPnbtrFeatures(pnbtr_logs_[i], pnbtr_context);
        
        // Set metadata
        sample.timestamp = audio_logs_[i].timestamp;
        sample.ground_truth_quality_score = audio_logs_[i].snr_db / 100.0;
        
        if (validateSampleQuality(sample)) {
            training_samples_.push_back(sample);
        }
    }
    
    // Apply normalization if enabled
    if (config_.normalize_features) {
        normalizeFeatures();
    }
    
    return !training_samples_.empty();
}

bool TrainingDataPreparator::calculateTrainingTargets() {
    for (auto& sample : training_samples_) {
        // Calculate appropriate training target based on the data
        // For simplicity, create audio gap fill target
        sample.target.type = TrainingTarget::AUDIO_GAP_FILL;
        sample.target.target_audio_samples = sample.audio_features.audio_envelope_after;
        sample.target.target_quality_score = sample.ground_truth_quality_score;
        sample.target.target_meets_threshold = sample.ground_truth_quality_score > config_.minimum_quality_threshold;
    }
    
    return true;
}

bool TrainingDataPreparator::labelDataWithNetworkConditions() {
    for (auto& sample : training_samples_) {
        // Create network conditions based on extracted features
        sample.network_conditions.base_latency_ms = sample.network_features.estimated_rtt_ms;
        sample.network_conditions.packet_loss_percentage = sample.network_features.loss_rate_recent * 100.0;
        sample.network_conditions.bandwidth_kbps = static_cast<uint32_t>(sample.network_features.estimated_bandwidth_kbps);
        sample.network_conditions.jitter_variance_ms = sample.network_features.jitter_magnitude_ms;
    }
    
    return true;
}

TrainingDataPreparator::DataSplit TrainingDataPreparator::createDataSplit() const {
    DataSplit split;
    
    if (training_samples_.empty()) {
        return split;
    }
    
    // Create a copy for shuffling
    std::vector<TrainingSample> shuffled_samples = training_samples_;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shuffled_samples.begin(), shuffled_samples.end(), gen);
    
    // Calculate split sizes
    size_t total_size = shuffled_samples.size();
    size_t train_size = static_cast<size_t>(total_size * config_.training_split);
    size_t val_size = static_cast<size_t>(total_size * config_.validation_split);
    
    // Split the data
    split.training_set.assign(shuffled_samples.begin(), shuffled_samples.begin() + train_size);
    split.validation_set.assign(shuffled_samples.begin() + train_size, 
                               shuffled_samples.begin() + train_size + val_size);
    split.test_set.assign(shuffled_samples.begin() + train_size + val_size, shuffled_samples.end());
    
    return split;
}

bool TrainingDataPreparator::exportDataSplit(const DataSplit& split, const std::string& output_directory) {
    // Export training set
    std::ofstream train_file(output_directory + "/training_set.dat", std::ios::binary);
    for (const auto& sample : split.training_set) {
        serializeTrainingSample(sample, train_file);
    }
    
    // Export validation set
    std::ofstream val_file(output_directory + "/validation_set.dat", std::ios::binary);
    for (const auto& sample : split.validation_set) {
        serializeTrainingSample(sample, val_file);
    }
    
    // Export test set
    std::ofstream test_file(output_directory + "/test_set.dat", std::ios::binary);
    for (const auto& sample : split.test_set) {
        serializeTrainingSample(sample, test_file);
    }
    
    return true;
}

TrainingDataPreparator::DatasetAnalysis TrainingDataPreparator::analyzeDataset() const {
    DatasetAnalysis analysis;
    
    analysis.total_samples = training_samples_.size();
    
    if (training_samples_.empty()) {
        return analysis;
    }
    
    // Calculate average target quality
    double total_quality = 0.0;
    for (const auto& sample : training_samples_) {
        total_quality += sample.ground_truth_quality_score;
    }
    analysis.average_target_quality = total_quality / training_samples_.size();
    
    // Calculate feature coverage score (simplified)
    analysis.feature_coverage_score = 0.85; // Placeholder
    
    // Add recommendations
    if (analysis.total_samples < 1000) {
        analysis.data_collection_recommendations.push_back("Collect more training samples (minimum 1000 recommended)");
    }
    
    if (analysis.average_target_quality < 0.7) {
        analysis.data_collection_recommendations.push_back("Focus on higher quality audio scenarios");
    }
    
    return analysis;
}

bool TrainingDataPreparator::validateDatasetQuality() const {
    if (training_samples_.empty()) {
        return false;
    }
    
    // Check minimum sample count
    if (training_samples_.size() < 100) {
        return false;
    }
    
    // Validate each sample
    for (const auto& sample : training_samples_) {
        if (!validateSampleQuality(sample)) {
            return false;
        }
    }
    
    return true;
}

bool TrainingDataPreparator::exportTensorFlowFormat(const std::string& output_path) const {
    // Placeholder for TensorFlow export
    std::ofstream tf_file(output_path + "/tensorflow_dataset.tf");
    tf_file << "# TensorFlow dataset export placeholder\n";
    tf_file << "# Total samples: " << training_samples_.size() << "\n";
    return true;
}

bool TrainingDataPreparator::exportPyTorchFormat(const std::string& output_path) const {
    // Placeholder for PyTorch export
    std::ofstream pt_file(output_path + "/pytorch_dataset.pt");
    pt_file << "# PyTorch dataset export placeholder\n";
    pt_file << "# Total samples: " << training_samples_.size() << "\n";
    return true;
}

bool TrainingDataPreparator::exportNumpyFormat(const std::string& output_path) const {
    // Placeholder for NumPy export
    std::ofstream np_file(output_path + "/numpy_dataset.npy");
    np_file << "# NumPy dataset export placeholder\n";
    np_file << "# Total samples: " << training_samples_.size() << "\n";
    return true;
}

bool TrainingDataPreparator::exportCSVFormat(const std::string& output_path) const {
    std::ofstream csv_file(output_path + "/dataset.csv");
    
    // Write header
    csv_file << "timestamp,network_latency,network_loss_rate,audio_energy,pnbtr_confidence,target_quality\n";
    
    // Write data
    for (const auto& sample : training_samples_) {
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            sample.timestamp.time_since_epoch()).count();
        
        csv_file << timestamp << ","
                << sample.network_features.estimated_rtt_ms << ","
                << sample.network_features.loss_rate_recent << ","
                << sample.audio_features.signal_energy << ","
                << sample.pnbtr_features.prediction_confidence << ","
                << sample.ground_truth_quality_score << "\n";
    }
    
    return true;
}

// Private implementation methods
NetworkFeatures TrainingDataPreparator::extractNetworkFeatures(const NetworkLogEntry& network_log,
                                                               const std::vector<NetworkLogEntry>& context) const {
    NetworkFeatures features;
    
    features.estimated_rtt_ms = network_log.latency_ms;
    features.jitter_magnitude_ms = network_log.jitter_ms;
    
    // Calculate loss rate from context
    if (!context.empty()) {
        int lost_packets = 0;
        for (const auto& entry : context) {
            if (entry.was_lost) lost_packets++;
        }
        features.loss_rate_recent = static_cast<double>(lost_packets) / context.size();
    }
    
    // Estimate bandwidth and connection stability
    features.estimated_bandwidth_kbps = estimateBandwidth(context);
    features.connection_stability_score = calculateConnectionStability(context);
    
    return features;
}

AudioFeatures TrainingDataPreparator::extractAudioFeatures(const AudioLogEntry& audio_log) const {
    AudioFeatures features;
    
    if (!audio_log.original_audio.empty()) {
        // Calculate basic audio features
        features.signal_energy = 0.0;
        for (float sample : audio_log.original_audio) {
            features.signal_energy += sample * sample;
        }
        features.signal_energy /= audio_log.original_audio.size();
        
        features.spectral_centroid_hz = calculateSpectralCentroid(audio_log.original_audio);
        features.zero_crossing_rate = calculateZeroCrossingRate(audio_log.original_audio);
        features.harmonic_amplitudes = extractHarmonicContent(audio_log.original_audio);
        features.harmonicity_score = calculateHarmonicity(audio_log.original_audio);
    }
    
    if (!audio_log.received_audio.empty()) {
        features.signal_continuity_score = calculateSignalContinuity(
            audio_log.original_audio, audio_log.received_audio);
    }
    
    return features;
}

PnbtrInternalFeatures TrainingDataPreparator::extractPnbtrFeatures(const PnbtrActionLogEntry& pnbtr_log,
                                                                   const std::vector<PnbtrActionLogEntry>& context) const {
    PnbtrInternalFeatures features;
    
    features.prediction_confidence = pnbtr_log.prediction_confidence;
    features.reconstruction_quality_estimate = pnbtr_log.reconstruction_quality_estimate;
    
    // Calculate recent accuracy from context
    if (!context.empty()) {
        double total_accuracy = 0.0;
        for (const auto& entry : context) {
            total_accuracy += entry.prediction_accuracy_score;
        }
        features.recent_accuracy_score = total_accuracy / context.size();
    }
    
    // Count consecutive successful predictions
    features.consecutive_successful_predictions = 0;
    for (auto it = context.rbegin(); it != context.rend(); ++it) {
        if (it->prediction_accuracy_score > 0.8) {
            features.consecutive_successful_predictions++;
        } else {
            break;
        }
    }
    
    return features;
}

void TrainingDataPreparator::normalizeFeatures() {
    if (training_samples_.empty()) {
        return;
    }
    
    // Normalize network features
    double max_latency = 0.0;
    double max_bandwidth = 0.0;
    for (const auto& sample : training_samples_) {
        max_latency = std::max(max_latency, sample.network_features.estimated_rtt_ms);
        max_bandwidth = std::max(max_bandwidth, sample.network_features.estimated_bandwidth_kbps);
    }
    
    for (auto& sample : training_samples_) {
        if (max_latency > 0) {
            sample.network_features.estimated_rtt_ms /= max_latency;
        }
        if (max_bandwidth > 0) {
            sample.network_features.estimated_bandwidth_kbps /= max_bandwidth;
        }
    }
}

void TrainingDataPreparator::removeOutliers() {
    if (!config_.exclude_extreme_outliers) {
        return;
    }
    
    // Simple outlier removal based on quality score
    std::vector<double> quality_scores;
    for (const auto& sample : training_samples_) {
        quality_scores.push_back(sample.ground_truth_quality_score);
    }
    
    if (quality_scores.empty()) {
        return;
    }
    
    // Calculate mean and std dev
    double mean = std::accumulate(quality_scores.begin(), quality_scores.end(), 0.0) / quality_scores.size();
    double sq_sum = 0.0;
    for (double score : quality_scores) {
        sq_sum += (score - mean) * (score - mean);
    }
    double std_dev = std::sqrt(sq_sum / quality_scores.size());
    
    // Remove outliers
    training_samples_.erase(
        std::remove_if(training_samples_.begin(), training_samples_.end(),
            [mean, std_dev, this](const TrainingSample& sample) {
                return std::abs(sample.ground_truth_quality_score - mean) > 
                       config_.outlier_threshold_sigma * std_dev;
            }),
        training_samples_.end()
    );
}

void TrainingDataPreparator::correlateLogEntries() {
    // Sort all logs by timestamp for correlation
    std::sort(audio_logs_.begin(), audio_logs_.end(),
        [](const AudioLogEntry& a, const AudioLogEntry& b) {
            return a.timestamp < b.timestamp;
        });
    
    std::sort(network_logs_.begin(), network_logs_.end(),
        [](const NetworkLogEntry& a, const NetworkLogEntry& b) {
            return a.timestamp < b.timestamp;
        });
    
    std::sort(pnbtr_logs_.begin(), pnbtr_logs_.end(),
        [](const PnbtrActionLogEntry& a, const PnbtrActionLogEntry& b) {
            return a.timestamp < b.timestamp;
        });
}

bool TrainingDataPreparator::validateSampleQuality(const TrainingSample& sample) const {
    // Check if sample meets minimum quality thresholds
    return sample.ground_truth_quality_score >= config_.minimum_quality_threshold &&
           sample.network_features.estimated_rtt_ms > 0 &&
           sample.audio_features.signal_energy > 0;
}

// Audio signal processing methods
double TrainingDataPreparator::calculateSpectralCentroid(const std::vector<float>& audio) const {
    if (audio.empty()) return 0.0;
    
    // Simplified spectral centroid calculation
    double weighted_sum = 0.0;
    double magnitude_sum = 0.0;
    
    for (size_t i = 0; i < audio.size(); ++i) {
        double magnitude = std::abs(audio[i]);
        weighted_sum += i * magnitude;
        magnitude_sum += magnitude;
    }
    
    return magnitude_sum > 0 ? weighted_sum / magnitude_sum : 0.0;
}

double TrainingDataPreparator::calculateZeroCrossingRate(const std::vector<float>& audio) const {
    if (audio.size() < 2) return 0.0;
    
    int zero_crossings = 0;
    for (size_t i = 1; i < audio.size(); ++i) {
        if ((audio[i-1] >= 0 && audio[i] < 0) || (audio[i-1] < 0 && audio[i] >= 0)) {
            zero_crossings++;
        }
    }
    
    return static_cast<double>(zero_crossings) / (audio.size() - 1);
}

std::vector<double> TrainingDataPreparator::extractHarmonicContent(const std::vector<float>& audio) const {
    // Simplified harmonic content extraction
    std::vector<double> harmonics(10, 0.0); // 10 harmonics
    
    if (audio.empty()) return harmonics;
    
    // Basic peak detection (placeholder for actual FFT)
    for (size_t i = 0; i < harmonics.size() && i < 10; ++i) {
        harmonics[i] = 0.1 * (i + 1); // Placeholder values
    }
    
    return harmonics;
}

double TrainingDataPreparator::calculateHarmonicity(const std::vector<float>& audio) const {
    if (audio.empty()) return 0.0;
    
    // Simplified harmonicity score
    auto harmonics = extractHarmonicContent(audio);
    double fundamental = harmonics.empty() ? 0.0 : harmonics[0];
    double total_energy = std::accumulate(harmonics.begin(), harmonics.end(), 0.0);
    
    return total_energy > 0 ? fundamental / total_energy : 0.0;
}

double TrainingDataPreparator::calculateSignalContinuity(const std::vector<float>& before, 
                                                        const std::vector<float>& after) const {
    if (before.empty() || after.empty()) return 0.0;
    
    // Calculate correlation between end of before and start of after
    size_t compare_length = std::min(before.size(), after.size()) / 4; // Compare last/first quarter
    if (compare_length == 0) return 0.0;
    
    double correlation = 0.0;
    for (size_t i = 0; i < compare_length; ++i) {
        size_t before_idx = before.size() - compare_length + i;
        correlation += before[before_idx] * after[i];
    }
    
    return correlation / compare_length;
}

// Network analysis methods
double TrainingDataPreparator::calculateConnectionStability(const std::vector<NetworkLogEntry>& history) const {
    if (history.empty()) return 0.0;
    
    // Calculate stability based on latency variance
    std::vector<double> latencies;
    for (const auto& entry : history) {
        latencies.push_back(entry.latency_ms);
    }
    
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double variance = 0.0;
    for (double latency : latencies) {
        variance += (latency - mean) * (latency - mean);
    }
    variance /= latencies.size();
    
    // Lower variance = higher stability
    return 1.0 / (1.0 + variance / 100.0);
}

double TrainingDataPreparator::estimateBandwidth(const std::vector<NetworkLogEntry>& history) const {
    if (history.empty()) return 1000.0; // Default 1 Mbps
    
    // Simplified bandwidth estimation based on packet loss and latency
    double avg_latency = 0.0;
    int lost_packets = 0;
    
    for (const auto& entry : history) {
        avg_latency += entry.latency_ms;
        if (entry.was_lost) lost_packets++;
    }
    
    avg_latency /= history.size();
    double loss_rate = static_cast<double>(lost_packets) / history.size();
    
    // Higher latency and loss = lower estimated bandwidth
    double bandwidth = 1000.0 / (1.0 + avg_latency / 50.0 + loss_rate * 10.0);
    return std::max(100.0, bandwidth); // Minimum 100 kbps
}

double TrainingDataPreparator::estimateRTT(const std::vector<NetworkLogEntry>& history) const {
    if (history.empty()) return 50.0; // Default 50ms
    
    double total_latency = 0.0;
    for (const auto& entry : history) {
        total_latency += entry.latency_ms;
    }
    
    return total_latency / history.size();
}

void TrainingDataPreparator::serializeTrainingSample(const TrainingSample& sample, std::ostream& stream) const {
    // Simplified binary serialization
    stream.write(reinterpret_cast<const char*>(&sample.ground_truth_quality_score), 
                sizeof(sample.ground_truth_quality_score));
    stream.write(reinterpret_cast<const char*>(&sample.network_features.estimated_rtt_ms), 
                sizeof(sample.network_features.estimated_rtt_ms));
    // Write other fields as needed...
}

bool TrainingDataPreparator::deserializeTrainingSample(TrainingSample& sample, std::istream& stream) const {
    // Simplified binary deserialization
    stream.read(reinterpret_cast<char*>(&sample.ground_truth_quality_score), 
               sizeof(sample.ground_truth_quality_score));
    stream.read(reinterpret_cast<char*>(&sample.network_features.estimated_rtt_ms), 
               sizeof(sample.network_features.estimated_rtt_ms));
    // Read other fields as needed...
    
    return !stream.fail();
}

} // namespace pnbtr_jellie
