#pragma once

#include "pnbtr_framework.h"
#include <memory>

namespace pnbtr {

class PNBTRGPU;

class PNBTREngine {
public:
    PNBTREngine(const PNBTRConfig& config, PNBTRGPU& gpu);
    ~PNBTREngine();
    
    bool initialize();
    void shutdown();
    
    // Core PNBTR functions
    AudioBuffer reconstruct_lsb_mathematically(const AudioBuffer& input, 
                                             const AudioContext& context);
    
    AudioBuffer predict_analog_continuation(const AudioBuffer& input,
                                          const AudioContext& context,
                                          uint32_t extrapolate_samples);
    
    PredictionResult run_hybrid_prediction(const AudioBuffer& input,
                                         const AudioContext& context);
    
    // Hybrid prediction methodologies
    std::vector<float> run_lpc_prediction(const std::vector<float>& samples, 
                                        uint32_t predict_samples);
    
    std::vector<float> run_pitch_cycle_reconstruction(const std::vector<float>& samples,
                                                    const AudioContext& context,
                                                    uint32_t predict_samples);
    
    std::vector<float> run_envelope_tracking(const std::vector<float>& samples,
                                           const AudioContext& context,
                                           uint32_t predict_samples);
    
    std::vector<float> run_neural_inference(const std::vector<float>& samples,
                                          const AudioContext& context,
                                          uint32_t predict_samples);
    
    std::vector<float> run_spectral_shaping(const std::vector<float>& samples,
                                          const AudioContext& context,
                                          uint32_t predict_samples);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace pnbtr
