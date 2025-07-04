#include <metal_stdlib>
using namespace metal;

#define SAMPLE_WINDOW 512              // ~10.6 ms @ 48kHz
#define NUM_FORMANTS 3
#define PREDICTION_LENGTH 1024         // ~21 ms

using namespace metal;

struct Formant {
    float centerFreq;   // Hz
    float bandwidth;    // Hz
    float amplitude;    // linear gain
};

// Output format for predicted waveform
kernel void formant_model(
    device const float* recentSamples   [[ buffer(0) ]],
    device float* predictedSamples      [[ buffer(1) ]],
    device Formant* outputFormants      [[ buffer(2) ]],
    uint tid                            [[ thread_position_in_grid ]]
) {
    if (tid > 0) return; // Formant estimation runs once for now

    // Step 1: crude formant extraction via bandpass scanning
    constexpr float sampleRate = 48000.0;
    float energy[NUM_FORMANTS] = {0.0};
    float centerFreqs[NUM_FORMANTS] = {700.0, 1200.0, 2600.0}; // F1–F3 approx for human voice
    float bandwidths[NUM_FORMANTS] = {100.0, 150.0, 200.0};

    for (uint f = 0; f < NUM_FORMANTS; ++f) {
        float fc = centerFreqs[f];
        float bw = bandwidths[f];
        float filtered = 0.0;
        for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
            float t = float(i) / sampleRate;
            filtered += recentSamples[i] * sin(2.0 * M_PI_F * fc * t) * exp(-bw * t);
        }
        energy[f] = fabs(filtered);
    }

    // Normalize and output estimated formants
    for (uint i = 0; i < NUM_FORMANTS; ++i) {
        outputFormants[i].centerFreq = centerFreqs[i];
        outputFormants[i].bandwidth = bandwidths[i];
        outputFormants[i].amplitude = energy[i];
    }

    // Step 2: synthesize using formant sine waves
    for (uint i = 0; i < PREDICTION_LENGTH; ++i) {
        float t = float(i) / sampleRate;
        float sum = 0.0;
        for (uint f = 0; f < NUM_FORMANTS; ++f) {
            float phase = 2.0 * M_PI_F * outputFormants[f].centerFreq * t;
            float decay = exp(-outputFormants[f].bandwidth * t);
            sum += outputFormants[f].amplitude * decay * sin(phase);
        }
        predictedSamples[i] = sum;
    }
}
