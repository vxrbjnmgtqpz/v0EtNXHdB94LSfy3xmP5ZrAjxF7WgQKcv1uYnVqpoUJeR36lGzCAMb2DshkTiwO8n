#pragma once

// PNBTR Framework - Predictive Neural Buffered Transient Recovery
// Revolutionary dither replacement technology - NO traditional dithering
// Complete replacement paradigm: mathematically informed, self-improving, zero-noise
// See PNBTR_refined.md for full technical specification

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace pnbtr {

// Forward declarations
class PNBTREngine;
class PNBTRLearning;
class PNBTRGPU;

// Configuration for PNBTR - Complete Dither Replacement System
struct PNBTRConfig {
    // Core settings - no traditional dither settings needed
    uint32_t sample_rate = 48000;
    uint16_t bit_depth = 24;  // Default 24-bit operation (16/24/32 supported)
    uint16_t channels = 2;
    
    // Prediction settings - the heart of dither replacement
    float prediction_window_ms = 50.0f;  // 50ms contextual extrapolation
    uint32_t lpc_order = 16;             // Autoregressive model order
    uint32_t fft_size = 1024;            // Spectral analysis window
    
    // Neural inference - continuous learning paradigm
    bool enable_neural_inference = true;
    std::string model_path = "models/pnbtr_hybrid.onnx";
    uint32_t max_gpu_streams = 1000;
    
    // Self-improving learning system - replaces static dither patterns
    bool enable_continuous_learning = true;
    bool archive_reference_signals = true;
    std::string learning_data_path = "pnbtr_learning/";
    
    // GPU backend for real-time dither replacement
    enum GPUBackend {
        PNBTR_GPU_AUTO,
        PNBTR_GPU_VULKAN,
        PNBTR_GPU_METAL
    } gpu_backend = PNBTR_GPU_AUTO;
};

// Audio data structures
struct AudioBuffer {
    std::vector<float> samples;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bit_depth;
    uint64_t timestamp_ns;
    
    // PNBTR metadata
    bool pnbtr_processed = false;
    float prediction_confidence = 0.0f;
    bool contains_extrapolated_data = false;
    uint32_t extrapolated_samples = 0;
};

struct AudioContext {
    // Musical context for intelligent prediction
    float fundamental_frequency = 0.0f;  // Hz, 0 = unvoiced/noise
    float pitch_confidence = 0.0f;       // 0-1, how tonal the signal is
    std::vector<float> harmonic_magnitudes; // Harmonic series analysis
    
    // Envelope context
    float attack_time_ms = 0.0f;
    float decay_time_ms = 0.0f;
    float sustain_level = 0.0f;
    float release_time_ms = 0.0f;
    
    // Spectral context
    std::vector<float> spectral_centroid;  // Brightness over time
    std::vector<float> spectral_rolloff;   // High-frequency content
    std::vector<float> mfcc_coefficients;  // Timbral characteristics
    
    // Temporal context
    float tempo_bpm = 0.0f;
    float time_signature_numerator = 4.0f;
    std::string musical_key = "unknown";
};

// Prediction results
struct PredictionResult {
    AudioBuffer predicted_audio;
    AudioContext updated_context;
    
    // Quality metrics
    float prediction_confidence = 0.0f;
    float lsb_reconstruction_quality = 0.0f;
    float neural_inference_confidence = 0.0f;
    
    // Performance metrics
    float processing_time_ms = 0.0f;
    float gpu_utilization = 0.0f;
    uint32_t samples_extrapolated = 0;
    
    // Methodology breakdown
    float lpc_contribution = 0.0f;          // Autoregressive modeling
    float pitch_cycle_contribution = 0.0f;   // Harmonic reconstruction
    float envelope_contribution = 0.0f;      // ADSR modeling
    float neural_contribution = 0.0f;        // RNN/CNN inference
    float spectral_contribution = 0.0f;      // FFT-based reconstruction
};

// Learning data for continuous improvement
struct LearningPair {
    AudioBuffer pnbtr_reconstruction;
    AudioBuffer reference_signal;
    AudioContext context;
    
    float reconstruction_error = 0.0f;
    uint64_t collection_timestamp_ns = 0;
    std::string session_id;
    std::string user_id;
};

// Callback types
using PredictionCallback = std::function<void(const PredictionResult&)>;
using LearningCallback = std::function<void(const LearningPair&)>;
using StatisticsCallback = std::function<void(const struct PNBTRStatistics&)>;

// Main PNBTR Framework class
// PARADIGM: Complete dither replacement, not fallback or supplement
// Traditional dithering is obsolete - PNBTR provides superior results
class PNBTRFramework {
public:
    PNBTRFramework();
    ~PNBTRFramework();
    
    // Core lifecycle
    bool initialize(const PNBTRConfig& config);
    void shutdown();
    
    // PRIMARY FUNCTION: Complete dither replacement
    // This replaces ALL traditional dithering operations
    AudioBuffer replace_dither_with_prediction(const AudioBuffer& quantized_audio,
                                             const AudioContext& context);
    
    // Neural analog extrapolation - 50ms prediction for gap recovery
    AudioBuffer extrapolate_analog_signal(const AudioBuffer& input_audio,
                                         const AudioContext& context,
                                         uint32_t extrapolate_samples);
    
    // LSB reconstruction - mathematically perfect waveform-aware processing
    AudioBuffer reconstruct_lsb_mathematically(const AudioBuffer& input_audio,
                                              const AudioContext& context);
    
    // Real-time processing interface - replaces all dither processing
    PredictionResult process_audio_stream(const AudioBuffer& input,
                                        const AudioContext& context = {});
    
    // Continuous learning interface
    void submit_learning_pair(const AudioBuffer& pnbtr_output,
                            const AudioBuffer& reference,
                            const AudioContext& context);
    
    void enable_reference_archival(bool enabled);
    bool retrain_model_from_accumulated_data();
    
    // Callbacks
    void on_prediction_complete(PredictionCallback callback);
    void on_learning_data_collected(LearningCallback callback);
    void on_statistics_updated(StatisticsCallback callback);
    
    // Performance monitoring
    struct PNBTRStatistics {
        // Processing stats
        uint64_t audio_buffers_processed = 0;
        float average_prediction_confidence = 0.0f;
        float average_processing_time_ms = 0.0f;
        float gpu_utilization_percentage = 0.0f;
        
        // Quality metrics
        float average_lsb_reconstruction_quality = 0.0f;
        float zero_noise_achievement_rate = 1.0f;  // Should always be 1.0
        
        // Learning stats
        uint64_t learning_pairs_collected = 0;
        uint64_t model_retraining_events = 0;
        float model_accuracy_improvement = 0.0f;
        
        // Methodology usage
        float lpc_usage_percentage = 0.0f;
        float neural_inference_usage_percentage = 0.0f;
        float spectral_shaping_usage_percentage = 0.0f;
    };
    
    PNBTRStatistics get_statistics() const;
    
    // GPU control
    bool is_gpu_available() const;
    bool enable_gpu_processing(bool enabled);
    float get_gpu_utilization() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Utility functions
namespace utils {
    // Audio analysis
    AudioContext analyze_audio_context(const AudioBuffer& audio);
    float calculate_fundamental_frequency(const std::vector<float>& samples, uint32_t sample_rate);
    std::vector<float> extract_harmonic_series(const std::vector<float>& samples, 
                                             float fundamental_freq, 
                                             uint32_t sample_rate);
    
    // Quality metrics
    float calculate_prediction_confidence(const AudioBuffer& original, 
                                        const AudioBuffer& predicted);
    float measure_zero_noise_achievement(const AudioBuffer& audio);
    
    // Mathematical utilities
    std::vector<float> compute_lpc_coefficients(const std::vector<float>& samples, uint32_t order);
    std::vector<float> apply_spectral_shaping(const std::vector<float>& input_fft,
                                            const std::vector<float>& target_shape);
    
    // Timestamp utilities
    uint64_t get_timestamp_ns();
    std::string generate_learning_session_id();
}

} // namespace pnbtr
