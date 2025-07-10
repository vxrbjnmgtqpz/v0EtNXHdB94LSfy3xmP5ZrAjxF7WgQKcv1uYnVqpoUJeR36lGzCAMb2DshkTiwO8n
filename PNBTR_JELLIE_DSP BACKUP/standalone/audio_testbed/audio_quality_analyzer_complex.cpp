#include "audio_quality_analyzer.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <complex>
#include <functional>
#include <sstream>
#include <iomanip>

// Internal state structure (simplified without FFTW)
struct AudioQualityAnalyzer::AnalyzerState {
    // Simple FFT buffers
    std::vector<std::complex<double>> fft_buffer;
    std::vector<double> window_cache;
    std::map<std::string, std::vector<double>> windows_cache;
    
    ~AnalyzerState() {
        if (forward_plan) fftw_destroy_plan(forward_plan);
        if (inverse_plan) fftw_destroy_plan(inverse_plan);
        if (fft_input) fftw_free(fft_input);
        if (fft_output) fftw_free(fft_output);
    }
};

AudioQualityAnalyzer::AudioQualityAnalyzer() 
    : state_(std::make_unique<AnalyzerState>()) {
}

AudioQualityAnalyzer::~AudioQualityAnalyzer() = default;

AudioQualityAnalyzer::QualityMetrics AudioQualityAnalyzer::analyzeQuality(
    const std::vector<float>& input_signal,
    const std::vector<float>& output_signal,
    uint32_t sample_rate,
    uint16_t channels) {
    
    QualityMetrics metrics;
    
    // Ensure signals are the same length
    size_t min_length = std::min(input_signal.size(), output_signal.size());
    if (min_length == 0) return metrics;
    
    std::vector<float> input_aligned(input_signal.begin(), input_signal.begin() + min_length);
    std::vector<float> output_aligned(output_signal.begin(), output_signal.begin() + min_length);
    
    // Calculate basic metrics
    std::vector<float> error_signal(min_length);
    for (size_t i = 0; i < min_length; ++i) {
        error_signal[i] = input_aligned[i] - output_aligned[i];
    }
    
    metrics.snr_db = calculateSNR(output_aligned, error_signal);
    metrics.thd_plus_n_percent = calculateTHDN(output_aligned, sample_rate);
    metrics.dynamic_range_db = calculateDynamicRange(output_aligned);
    metrics.noise_floor_dbfs = calculateNoiseFloor(output_aligned);
    
    // Calculate coloration
    metrics.coloration_percent = calculateColoration(input_aligned, output_aligned, sample_rate);
    
    // Frequency response analysis
    auto freq_response = testFrequencyResponse(
        [&](const std::vector<float>& input) -> std::vector<float> {
            // For this analysis, we'll use the ratio between input and output
            return output_aligned;
        }, sample_rate);
    
    if (!freq_response.empty()) {
        // Calculate frequency response flatness
        std::vector<double> magnitudes;
        for (const auto& point : freq_response) {
            magnitudes.push_back(point.second);
        }
        
        double mean_magnitude = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();
        double max_deviation = 0.0;
        for (double mag : magnitudes) {
            max_deviation = std::max(max_deviation, std::abs(20.0 * std::log10(mag / mean_magnitude)));
        }
        metrics.freq_response_flatness_db = max_deviation;
        
        // Find specific frequency responses
        auto find_response_at_freq = [&](double target_freq) -> double {
            double closest_freq_diff = std::numeric_limits<double>::max();
            double response = 0.0;
            for (const auto& point : freq_response) {
                double freq_diff = std::abs(point.first - target_freq);
                if (freq_diff < closest_freq_diff) {
                    closest_freq_diff = freq_diff;
                    response = 20.0 * std::log10(point.second);
                }
            }
            return response;
        };
        
        metrics.freq_response_20hz_db = find_response_at_freq(20.0);
        metrics.freq_response_20khz_db = find_response_at_freq(20000.0);
    }
    
    // Phase linearity analysis
    auto phase_metrics = analyzePhaseLinearity(input_aligned, output_aligned, sample_rate);
    metrics.phase_linearity_score = phase_metrics.linearity_score;
    metrics.group_delay_variation_ms = phase_metrics.group_delay_variation_ms;
    
    // Transient preservation
    metrics.transient_preservation = measureTransientPreservation(input_aligned, output_aligned);
    
    // Calculate overall quality score (weighted combination)
    double snr_weight = 0.25;
    double thd_weight = 0.25;
    double freq_weight = 0.20;
    double phase_weight = 0.15;
    double transient_weight = 0.15;
    
    // Normalize each metric to 0-1 scale
    double snr_score = std::min(1.0, std::max(0.0, metrics.snr_db / 60.0)); // 60dB = perfect
    double thd_score = std::max(0.0, 1.0 - metrics.thd_plus_n_percent / 1.0); // <1% = good
    double freq_score = std::max(0.0, 1.0 - metrics.freq_response_flatness_db / 3.0); // <3dB = good
    double phase_score = metrics.phase_linearity_score;
    double transient_score = metrics.transient_preservation;
    
    metrics.overall_quality_score = 
        snr_weight * snr_score +
        thd_weight * thd_score +
        freq_weight * freq_score +
        phase_weight * phase_score +
        transient_weight * transient_score;
    
    // Check if meets hi-fi standards
    metrics.meets_hifi_standards = meetsHiFiStandards(metrics);
    
    return metrics;
}

std::vector<std::pair<double, double>> AudioQualityAnalyzer::testFrequencyResponse(
    std::function<std::vector<float>(const std::vector<float>&)> process_function,
    uint32_t sample_rate,
    const FrequencyAnalysisConfig& config) {
    
    std::vector<std::pair<double, double>> response_points;
    
    // Generate frequency points (logarithmic or linear spacing)
    std::vector<double> test_frequencies;
    for (uint32_t i = 0; i < config.num_points; ++i) {
        double ratio = static_cast<double>(i) / (config.num_points - 1);
        double freq;
        
        if (config.logarithmic_spacing) {
            double log_start = std::log10(config.start_freq_hz);
            double log_end = std::log10(config.end_freq_hz);
            freq = std::pow(10.0, log_start + ratio * (log_end - log_start));
        } else {
            freq = config.start_freq_hz + ratio * (config.end_freq_hz - config.start_freq_hz);
        }
        
        test_frequencies.push_back(freq);
    }
    
    // Test each frequency
    for (double freq : test_frequencies) {
        // Generate test sine wave
        auto test_signal = generateTestSignal(
            TestSignalType::SINE_WAVE,
            config.sweep_duration_sec,
            sample_rate,
            freq,
            0.5 // 50% amplitude to avoid clipping
        );
        
        // Process signal
        auto processed_signal = process_function(test_signal);
        
        if (processed_signal.size() >= test_signal.size()) {
            // Calculate RMS of input and output at this frequency
            double input_rms = 0.0;
            double output_rms = 0.0;
            
            for (size_t i = 0; i < test_signal.size(); ++i) {
                input_rms += test_signal[i] * test_signal[i];
                output_rms += processed_signal[i] * processed_signal[i];
            }
            
            input_rms = std::sqrt(input_rms / test_signal.size());
            output_rms = std::sqrt(output_rms / test_signal.size());
            
            // Calculate magnitude response
            double magnitude_response = (input_rms > 1e-10) ? output_rms / input_rms : 0.0;
            
            response_points.emplace_back(freq, magnitude_response);
        }
    }
    
    return response_points;
}

AudioQualityAnalyzer::PhaseMetrics AudioQualityAnalyzer::analyzePhaseLinearity(
    const std::vector<float>& input_signal,
    const std::vector<float>& output_signal,
    uint32_t sample_rate) {
    
    PhaseMetrics metrics;
    
    if (input_signal.size() != output_signal.size() || input_signal.empty()) {
        return metrics;
    }
    
    // Perform FFT on both signals
    auto input_fft = performFFT(input_signal);
    auto output_fft = performFFT(output_signal);
    
    if (input_fft.size() != output_fft.size()) {
        return metrics;
    }
    
    // Calculate phase response
    std::vector<double> phase_diff;
    std::vector<double> frequencies;
    
    size_t half_size = input_fft.size() / 2;
    for (size_t i = 1; i < half_size; ++i) { // Skip DC
        double freq = static_cast<double>(i * sample_rate) / (2 * half_size);
        frequencies.push_back(freq);
        
        // Calculate phase difference
        double input_phase = std::arg(input_fft[i]);
        double output_phase = std::arg(output_fft[i]);
        double phase_difference = output_phase - input_phase;
        
        // Unwrap phase
        while (phase_difference > M_PI) phase_difference -= 2 * M_PI;
        while (phase_difference < -M_PI) phase_difference += 2 * M_PI;
        
        phase_diff.push_back(phase_difference);
    }
    
    metrics.phase_response = phase_diff;
    
    // Calculate group delay
    metrics.group_delay = calculateGroupDelay(input_fft, output_fft);
    
    // Calculate linearity score (how close to linear the phase response is)
    if (phase_diff.size() > 2) {
        // Fit a line to the phase response and measure deviation
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        size_t n = phase_diff.size();
        
        for (size_t i = 0; i < n; ++i) {
            double x = frequencies[i];
            double y = phase_diff[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        // Linear regression
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;
        
        // Calculate R-squared (coefficient of determination)
        double ss_tot = 0, ss_res = 0;
        double mean_y = sum_y / n;
        
        for (size_t i = 0; i < n; ++i) {
            double y_pred = slope * frequencies[i] + intercept;
            ss_tot += (phase_diff[i] - mean_y) * (phase_diff[i] - mean_y);
            ss_res += (phase_diff[i] - y_pred) * (phase_diff[i] - y_pred);
        }
        
        metrics.linearity_score = std::max(0.0, 1.0 - ss_res / ss_tot);
    }
    
    // Calculate group delay variation
    if (!metrics.group_delay.empty()) {
        auto minmax = std::minmax_element(metrics.group_delay.begin(), metrics.group_delay.end());
        metrics.group_delay_variation_ms = (*minmax.second - *minmax.first) * 1000.0;
    }
    
    return metrics;
}

double AudioQualityAnalyzer::calculateTHDN(const std::vector<float>& signal,
                                           uint32_t sample_rate,
                                           double fundamental_freq) {
    
    if (signal.empty()) return 100.0; // Maximum distortion for invalid input
    
    // If fundamental frequency not provided, try to detect it
    if (fundamental_freq <= 0.0) {
        fundamental_freq = findFundamentalFrequency(signal, sample_rate);
    }
    
    if (fundamental_freq <= 0.0) return 100.0;
    
    // Perform FFT
    auto spectrum = performFFT(signal);
    size_t fft_size = spectrum.size();
    
    // Find fundamental frequency bin
    double freq_resolution = static_cast<double>(sample_rate) / (2 * fft_size);
    size_t fundamental_bin = static_cast<size_t>(fundamental_freq / freq_resolution + 0.5);
    
    if (fundamental_bin >= fft_size / 2) return 100.0;
    
    // Calculate fundamental power
    double fundamental_magnitude = std::abs(spectrum[fundamental_bin]);
    double fundamental_power = fundamental_magnitude * fundamental_magnitude;
    
    // Calculate harmonic powers
    double harmonic_power = 0.0;
    for (int harmonic = 2; harmonic <= 10; ++harmonic) { // Up to 10th harmonic
        size_t harmonic_bin = fundamental_bin * harmonic;
        if (harmonic_bin < fft_size / 2) {
            double harmonic_magnitude = std::abs(spectrum[harmonic_bin]);
            harmonic_power += harmonic_magnitude * harmonic_magnitude;
        }
    }
    
    // Calculate noise power (everything except fundamental and harmonics)
    double noise_power = 0.0;
    std::vector<bool> harmonic_bins(fft_size / 2, false);
    
    // Mark harmonic bins
    harmonic_bins[fundamental_bin] = true;
    for (int harmonic = 2; harmonic <= 10; ++harmonic) {
        size_t harmonic_bin = fundamental_bin * harmonic;
        if (harmonic_bin < fft_size / 2) {
            harmonic_bins[harmonic_bin] = true;
        }
    }
    
    // Sum non-harmonic content as noise
    for (size_t i = 1; i < fft_size / 2; ++i) { // Skip DC
        if (!harmonic_bins[i]) {
            double magnitude = std::abs(spectrum[i]);
            noise_power += magnitude * magnitude;
        }
    }
    
    // Calculate THD+N percentage
    double total_distortion_power = harmonic_power + noise_power;
    
    if (fundamental_power <= 0.0) return 100.0;
    
    double thd_n = std::sqrt(total_distortion_power / fundamental_power) * 100.0;
    
    return std::min(100.0, thd_n);
}

double AudioQualityAnalyzer::calculateColoration(const std::vector<float>& input_signal,
                                                 const std::vector<float>& output_signal,
                                                 uint32_t sample_rate) {
    
    if (input_signal.size() != output_signal.size() || input_signal.empty()) {
        return 100.0; // Maximum coloration for invalid input
    }
    
    // Calculate spectral difference
    auto input_spectrum = performSpectralAnalysis(input_signal, sample_rate);
    auto output_spectrum = performSpectralAnalysis(output_signal, sample_rate);
    
    if (input_spectrum.size() != output_spectrum.size()) {
        return 100.0;
    }
    
    // Calculate spectral deviation
    double spectral_deviation = 0.0;
    double total_energy = 0.0;
    
    for (size_t i = 0; i < input_spectrum.size(); ++i) {
        double input_mag = input_spectrum[i].second;
        double output_mag = output_spectrum[i].second;
        double difference = std::abs(input_mag - output_mag);
        
        spectral_deviation += difference * difference;
        total_energy += input_mag * input_mag;
    }
    
    if (total_energy <= 0.0) return 100.0;
    
    // Calculate THD contribution
    double thd_contribution = calculateTHDN(output_signal, sample_rate) / 100.0;
    
    // Calculate dynamic range difference
    double input_dr = calculateDynamicRange(input_signal);
    double output_dr = calculateDynamicRange(output_signal);
    double dr_difference = std::abs(input_dr - output_dr) / 60.0; // Normalize to 60dB
    
    // Combine metrics for overall coloration percentage
    double spectral_coloration = std::sqrt(spectral_deviation / total_energy) * 100.0;
    double combined_coloration = std::min(100.0, 
        0.5 * spectral_coloration + 
        0.3 * thd_contribution * 100.0 + 
        0.2 * dr_difference * 100.0);
    
    return combined_coloration;
}

std::vector<std::pair<double, double>> AudioQualityAnalyzer::performSpectralAnalysis(
    const std::vector<float>& signal,
    uint32_t sample_rate,
    const std::string& window_type) {
    
    if (signal.empty()) return {};
    
    // Apply window function
    auto windowed_signal = applyWindow(signal, window_type);
    
    // Perform FFT
    auto spectrum = performFFT(windowed_signal);
    
    // Convert to frequency-magnitude pairs
    std::vector<std::pair<double, double>> result;
    double freq_resolution = static_cast<double>(sample_rate) / (2 * spectrum.size());
    
    for (size_t i = 0; i < spectrum.size() / 2; ++i) {
        double frequency = i * freq_resolution;
        double magnitude = std::abs(spectrum[i]);
        result.emplace_back(frequency, magnitude);
    }
    
    return result;
}

std::vector<float> AudioQualityAnalyzer::generateTestSignal(TestSignalType type,
                                                           double duration_sec,
                                                           uint32_t sample_rate,
                                                           double frequency_hz,
                                                           double amplitude) {
    
    size_t num_samples = static_cast<size_t>(duration_sec * sample_rate);
    std::vector<float> signal(num_samples);
    
    switch (type) {
        case TestSignalType::SINE_WAVE: {
            double omega = 2.0 * M_PI * frequency_hz / sample_rate;
            for (size_t i = 0; i < num_samples; ++i) {
                signal[i] = amplitude * std::sin(omega * i);
            }
            break;
        }
        
        case TestSignalType::FREQUENCY_SWEEP: {
            double start_freq = 20.0;
            double end_freq = frequency_hz; // Use frequency_hz as end frequency
            for (size_t i = 0; i < num_samples; ++i) {
                double t = static_cast<double>(i) / sample_rate;
                double freq = start_freq * std::pow(end_freq / start_freq, t / duration_sec);
                double omega = 2.0 * M_PI * freq / sample_rate;
                signal[i] = amplitude * std::sin(omega * i);
            }
            break;
        }
        
        case TestSignalType::WHITE_NOISE: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, amplitude);
            for (size_t i = 0; i < num_samples; ++i) {
                signal[i] = dist(gen);
            }
            break;
        }
        
        case TestSignalType::PINK_NOISE: {
            // Generate white noise first
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            // Apply pink noise filter (simple approximation)
            std::vector<float> white_noise(num_samples);
            for (size_t i = 0; i < num_samples; ++i) {
                white_noise[i] = dist(gen);
            }
            
            // Simple pink noise filter
            float b0 = 0.02109238f, b1 = 0.07113478f, b2 = 0.68873558f;
            float a1 = -0.62436687f, a2 = 0.29794407f;
            float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
            
            for (size_t i = 0; i < num_samples; ++i) {
                float x0 = white_noise[i];
                float y0 = b0 * x0 + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;
                signal[i] = amplitude * y0;
                
                x2 = x1; x1 = x0;
                y2 = y1; y1 = y0;
            }
            break;
        }
        
        case TestSignalType::IMPULSE: {
            std::fill(signal.begin(), signal.end(), 0.0f);
            if (num_samples > 0) {
                signal[0] = amplitude;
            }
            break;
        }
        
        default:
            // Default to sine wave
            double omega = 2.0 * M_PI * frequency_hz / sample_rate;
            for (size_t i = 0; i < num_samples; ++i) {
                signal[i] = amplitude * std::sin(omega * i);
            }
            break;
    }
    
    return signal;
}

double AudioQualityAnalyzer::measureProcessingLatency(
    std::function<std::vector<float>(const std::vector<float>&)> process_function,
    const std::vector<float>& input_signal) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto output = process_function(input_signal);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return duration.count() / 1000.0; // Convert to milliseconds
}

bool AudioQualityAnalyzer::validateRealTimeProcessing(
    std::function<std::vector<float>(const std::vector<float>&)> process_function,
    uint32_t sample_rate,
    uint32_t buffer_size,
    double target_latency_ms) {
    
    // Generate test buffer
    std::vector<float> test_buffer = generateTestSignal(
        TestSignalType::WHITE_NOISE,
        static_cast<double>(buffer_size) / sample_rate,
        sample_rate,
        1000.0,
        0.5
    );
    
    // Measure processing time for multiple iterations
    const int num_iterations = 100;
    double total_time_ms = 0.0;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto latency = measureProcessingLatency(process_function, test_buffer);
        total_time_ms += latency;
    }
    
    double average_latency_ms = total_time_ms / num_iterations;
    
    return average_latency_ms <= target_latency_ms;
}

std::string AudioQualityAnalyzer::generateQualityReport(const QualityMetrics& metrics,
                                                        const std::string& test_name) {
    
    std::ostringstream report;
    
    report << "<!DOCTYPE html>\n<html>\n<head>\n";
    report << "<title>" << test_name << " - Audio Quality Report</title>\n";
    report << "<style>\n";
    report << "body { font-family: Arial, sans-serif; margin: 40px; }\n";
    report << ".metric-good { color: green; font-weight: bold; }\n";
    report << ".metric-warning { color: orange; font-weight: bold; }\n";
    report << ".metric-bad { color: red; font-weight: bold; }\n";
    report << "table { border-collapse: collapse; width: 100%; }\n";
    report << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
    report << "th { background-color: #f2f2f2; }\n";
    report << "</style>\n</head>\n<body>\n";
    
    report << "<h1>" << test_name << " - Audio Quality Analysis</h1>\n";
    report << "<p>Generated on: " << std::chrono::system_clock::now().time_since_epoch().count() << "</p>\n";
    
    // Overall quality indicator
    std::string overall_class = "metric-good";
    std::string overall_status = "EXCELLENT";
    if (metrics.overall_quality_score < 0.8) {
        overall_class = "metric-warning";
        overall_status = "GOOD";
    }
    if (metrics.overall_quality_score < 0.6) {
        overall_class = "metric-bad";
        overall_status = "POOR";
    }
    
    report << "<h2>Overall Quality: <span class=\"" << overall_class << "\">";
    report << overall_status << " (" << std::fixed << std::setprecision(1) 
           << metrics.overall_quality_score * 100.0 << "%)</span></h2>\n";
    
    if (metrics.meets_hifi_standards) {
        report << "<p class=\"metric-good\">✓ Meets Hi-Fi Audio Standards</p>\n";
    } else {
        report << "<p class=\"metric-bad\">✗ Does not meet Hi-Fi Audio Standards</p>\n";
    }
    
    // Metrics table
    report << "<table>\n<tr><th>Metric</th><th>Value</th><th>Status</th><th>Target</th></tr>\n";
    
    auto add_metric = [&](const std::string& name, double value, const std::string& unit,
                         double good_threshold, double warning_threshold, bool higher_is_better = true) {
        std::string status_class = "metric-bad";
        std::string status_text = "POOR";
        
        if (higher_is_better) {
            if (value >= good_threshold) {
                status_class = "metric-good";
                status_text = "EXCELLENT";
            } else if (value >= warning_threshold) {
                status_class = "metric-warning";
                status_text = "GOOD";
            }
        } else {
            if (value <= good_threshold) {
                status_class = "metric-good";
                status_text = "EXCELLENT";
            } else if (value <= warning_threshold) {
                status_class = "metric-warning";
                status_text = "GOOD";
            }
        }
        
        report << "<tr><td>" << name << "</td>";
        report << "<td>" << std::fixed << std::setprecision(2) << value << " " << unit << "</td>";
        report << "<td class=\"" << status_class << "\">" << status_text << "</td>";
        report << "<td>" << (higher_is_better ? ">= " : "<= ") << good_threshold << " " << unit << "</td></tr>\n";
    };
    
    add_metric("Signal-to-Noise Ratio", metrics.snr_db, "dB", 60.0, 40.0, true);
    add_metric("THD+N", metrics.thd_plus_n_percent, "%", 0.1, 0.5, false);
    add_metric("Coloration", metrics.coloration_percent, "%", 0.1, 1.0, false);
    add_metric("Frequency Response Flatness", metrics.freq_response_flatness_db, "dB", 1.0, 3.0, false);
    add_metric("Phase Linearity", metrics.phase_linearity_score * 100, "%", 90.0, 70.0, true);
    add_metric("Dynamic Range", metrics.dynamic_range_db, "dB", 90.0, 60.0, true);
    add_metric("Transient Preservation", metrics.transient_preservation * 100, "%", 90.0, 70.0, true);
    
    if (metrics.latency_ms > 0) {
        add_metric("Processing Latency", metrics.latency_ms, "ms", 10.0, 20.0, false);
    }
    
    report << "</table>\n";
    
    // Additional details
    report << "<h3>Detailed Measurements</h3>\n";
    report << "<ul>\n";
    report << "<li>Noise Floor: " << std::fixed << std::setprecision(1) << metrics.noise_floor_dbfs << " dBFS</li>\n";
    report << "<li>Group Delay Variation: " << std::fixed << std::setprecision(2) << metrics.group_delay_variation_ms << " ms</li>\n";
    
    if (metrics.channel_crosstalk_db != 0.0) {
        report << "<li>Channel Crosstalk: " << std::fixed << std::setprecision(1) << metrics.channel_crosstalk_db << " dB</li>\n";
    }
    
    report << "</ul>\n";
    
    report << "</body>\n</html>";
    
    return report.str();
}

bool AudioQualityAnalyzer::meetsHiFiStandards(const QualityMetrics& metrics) {
    // High-fidelity audio standards (conservative thresholds)
    return (metrics.snr_db >= 60.0 &&                      // At least 60dB SNR
            metrics.thd_plus_n_percent <= 0.1 &&           // Less than 0.1% THD+N
            metrics.freq_response_flatness_db <= 1.0 &&     // Within ±1dB frequency response
            metrics.phase_linearity_score >= 0.9 &&         // 90% phase linearity
            metrics.dynamic_range_db >= 90.0 &&             // At least 90dB dynamic range
            metrics.coloration_percent <= 0.1);             // Less than 0.1% coloration
}

// Private implementation methods

double AudioQualityAnalyzer::calculateSNR(const std::vector<float>& signal,
                                          const std::vector<float>& noise) {
    if (signal.size() != noise.size() || signal.empty()) return 0.0;
    
    double signal_power = 0.0;
    double noise_power = 0.0;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        signal_power += signal[i] * signal[i];
        noise_power += noise[i] * noise[i];
    }
    
    signal_power /= signal.size();
    noise_power /= signal.size();
    
    if (noise_power <= 1e-15) return 120.0; // Very high SNR for virtually no noise
    
    return 10.0 * std::log10(signal_power / noise_power);
}

double AudioQualityAnalyzer::calculateDynamicRange(const std::vector<float>& signal) {
    if (signal.empty()) return 0.0;
    
    // Find peak and RMS
    float peak = 0.0;
    double rms_sum = 0.0;
    
    for (float sample : signal) {
        peak = std::max(peak, std::abs(sample));
        rms_sum += sample * sample;
    }
    
    double rms = std::sqrt(rms_sum / signal.size());
    
    if (rms <= 1e-10) return 0.0;
    
    return 20.0 * std::log10(peak / rms);
}

double AudioQualityAnalyzer::calculateNoiseFloor(const std::vector<float>& signal) {
    if (signal.empty()) return -120.0;
    
    // Calculate RMS of the signal
    double rms_sum = 0.0;
    for (float sample : signal) {
        rms_sum += sample * sample;
    }
    
    double rms = std::sqrt(rms_sum / signal.size());
    
    if (rms <= 1e-10) return -120.0; // Very low noise floor
    
    return 20.0 * std::log10(rms); // Assuming full scale is 1.0
}

std::vector<std::complex<double>> AudioQualityAnalyzer::performFFT(const std::vector<float>& signal) {
    if (signal.empty()) return {};
    
    size_t fft_size = 1;
    while (fft_size < signal.size()) fft_size <<= 1; // Next power of 2
    
    // Prepare FFTW if needed
    if (state_->fft_size != fft_size) {
        if (state_->forward_plan) fftw_destroy_plan(state_->forward_plan);
        if (state_->fft_input) fftw_free(state_->fft_input);
        if (state_->fft_output) fftw_free(state_->fft_output);
        
        state_->fft_size = fft_size;
        state_->fft_input = fftw_alloc_real(fft_size);
        state_->fft_output = fftw_alloc_complex(fft_size / 2 + 1);
        state_->forward_plan = fftw_plan_dft_r2c_1d(fft_size, state_->fft_input, state_->fft_output, FFTW_ESTIMATE);
    }
    
    // Copy signal to input buffer (zero-pad if necessary)
    std::fill(state_->fft_input, state_->fft_input + fft_size, 0.0);
    for (size_t i = 0; i < std::min(signal.size(), fft_size); ++i) {
        state_->fft_input[i] = signal[i];
    }
    
    // Execute FFT
    fftw_execute(state_->forward_plan);
    
    // Convert to std::complex
    std::vector<std::complex<double>> result(fft_size / 2 + 1);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = std::complex<double>(state_->fft_output[i][0], state_->fft_output[i][1]);
    }
    
    return result;
}

std::vector<float> AudioQualityAnalyzer::applyWindow(const std::vector<float>& signal,
                                                    const std::string& window_type) {
    if (signal.empty()) return signal;
    
    // Check cache for window function
    auto cache_key = window_type + "_" + std::to_string(signal.size());
    auto cache_it = state_->windows_cache.find(cache_key);
    
    std::vector<double> window;
    if (cache_it != state_->windows_cache.end()) {
        window = cache_it->second;
    } else {
        // Generate window function
        window.resize(signal.size());
        size_t N = signal.size();
        
        if (window_type == "hann") {
            for (size_t i = 0; i < N; ++i) {
                window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (N - 1)));
            }
        } else if (window_type == "blackman") {
            for (size_t i = 0; i < N; ++i) {
                window[i] = 0.42 - 0.5 * std::cos(2.0 * M_PI * i / (N - 1)) + 
                           0.08 * std::cos(4.0 * M_PI * i / (N - 1));
            }
        } else {
            // Default to rectangular window
            std::fill(window.begin(), window.end(), 1.0);
        }
        
        // Cache the window
        state_->windows_cache[cache_key] = window;
    }
    
    // Apply window
    std::vector<float> windowed_signal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        windowed_signal[i] = signal[i] * window[i];
    }
    
    return windowed_signal;
}

double AudioQualityAnalyzer::findFundamentalFrequency(const std::vector<float>& signal,
                                                      uint32_t sample_rate) {
    if (signal.empty()) return 0.0;
    
    auto spectrum = performFFT(signal);
    if (spectrum.empty()) return 0.0;
    
    // Find the peak in the spectrum
    double max_magnitude = 0.0;
    size_t peak_bin = 0;
    
    // Only look in audible range (20 Hz to 5 kHz for fundamental detection)
    double freq_resolution = static_cast<double>(sample_rate) / (2 * spectrum.size());
    size_t start_bin = static_cast<size_t>(20.0 / freq_resolution);
    size_t end_bin = static_cast<size_t>(5000.0 / freq_resolution);
    end_bin = std::min(end_bin, spectrum.size() / 2);
    
    for (size_t i = start_bin; i < end_bin; ++i) {
        double magnitude = std::abs(spectrum[i]);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            peak_bin = i;
        }
    }
    
    return peak_bin * freq_resolution;
}

std::vector<double> AudioQualityAnalyzer::calculateGroupDelay(
    const std::vector<std::complex<double>>& input_fft,
    const std::vector<std::complex<double>>& output_fft) {
    
    std::vector<double> group_delay;
    if (input_fft.size() != output_fft.size() || input_fft.size() < 3) {
        return group_delay;
    }
    
    size_t N = input_fft.size();
    group_delay.resize(N - 2); // Can't calculate for first and last bin
    
    for (size_t i = 1; i < N - 1; ++i) {
        // Calculate phase difference
        auto h_prev = output_fft[i-1] / input_fft[i-1];
        auto h_curr = output_fft[i] / input_fft[i];
        auto h_next = output_fft[i+1] / input_fft[i+1];
        
        double phase_prev = std::arg(h_prev);
        double phase_curr = std::arg(h_curr);
        double phase_next = std::arg(h_next);
        
        // Unwrap phases
        while (phase_curr - phase_prev > M_PI) phase_curr -= 2 * M_PI;
        while (phase_curr - phase_prev < -M_PI) phase_curr += 2 * M_PI;
        while (phase_next - phase_curr > M_PI) phase_next -= 2 * M_PI;
        while (phase_next - phase_curr < -M_PI) phase_next += 2 * M_PI;
        
        // Calculate group delay (negative derivative of phase)
        double d_phase = phase_next - phase_prev;
        group_delay[i-1] = -d_phase / (2.0 * 2.0 * M_PI); // Normalize
    }
    
    return group_delay;
}

double AudioQualityAnalyzer::measureTransientPreservation(const std::vector<float>& input_signal,
                                                          const std::vector<float>& output_signal) {
    if (input_signal.size() != output_signal.size() || input_signal.empty()) {
        return 0.0;
    }
    
    // Calculate envelope correlation as a measure of transient preservation
    size_t window_size = std::max(static_cast<size_t>(1), input_signal.size() / 1000);
    
    std::vector<double> input_envelope, output_envelope;
    
    for (size_t i = 0; i < input_signal.size(); i += window_size) {
        size_t end_idx = std::min(i + window_size, input_signal.size());
        
        double input_max = 0.0, output_max = 0.0;
        for (size_t j = i; j < end_idx; ++j) {
            input_max = std::max(input_max, static_cast<double>(std::abs(input_signal[j])));
            output_max = std::max(output_max, static_cast<double>(std::abs(output_signal[j])));
        }
        
        input_envelope.push_back(input_max);
        output_envelope.push_back(output_max);
    }
    
    // Calculate correlation coefficient
    if (input_envelope.size() < 2) return 0.0;
    
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    size_t n = input_envelope.size();
    
    for (size_t i = 0; i < n; ++i) {
        double x = input_envelope[i];
        double y = output_envelope[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    if (denominator <= 1e-10) return 0.0;
    
    return std::max(0.0, numerator / denominator);
}

double AudioQualityAnalyzer::calculateChannelCrosstalk(const std::vector<float>& left_channel,
                                                       const std::vector<float>& right_channel) {
    if (left_channel.size() != right_channel.size() || left_channel.empty()) {
        return 0.0; // No crosstalk measurement possible
    }
    
    // Calculate cross-correlation
    double left_power = 0.0, right_power = 0.0, cross_power = 0.0;
    
    for (size_t i = 0; i < left_channel.size(); ++i) {
        left_power += left_channel[i] * left_channel[i];
        right_power += right_channel[i] * right_channel[i];
        cross_power += left_channel[i] * right_channel[i];
    }
    
    left_power /= left_channel.size();
    right_power /= left_channel.size();
    cross_power /= left_channel.size();
    
    double signal_power = std::max(left_power, right_power);
    
    if (signal_power <= 1e-15) return 120.0; // Very high separation
    
    return 10.0 * std::log10(signal_power / std::abs(cross_power));
}

std::vector<std::pair<double, double>> AudioQualityAnalyzer::testFrequencyResponse(
    std::function<std::vector<float>(const std::vector<float>&)> process_function,
    uint32_t sample_rate) {
    
    FrequencyAnalysisConfig default_config;
    return testFrequencyResponse(process_function, sample_rate, default_config);
}
