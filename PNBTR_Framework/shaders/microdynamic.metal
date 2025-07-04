#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400    // 50 ms @ 48kHz
#define NOISE_SCALE 0.003         // Intensity of micro-noise
#define MOD_FREQ1 1800.0          // Hz
#define MOD_FREQ2 3200.0          // Hz
#define SAMPLE_RATE 48000.0

using namespace metal;

kernel void microdynamic(
    device const float* basePrediction       [[ buffer(0) ]],
    device float* enrichedOutput             [[ buffer(1) ]],
    constant float& seed                     [[ buffer(2) ]],   // Random seed or stream ID
    uint tid                                 [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    float t = float(tid) / SAMPLE_RATE;

    // Subtle modulations — pseudo-randomized high-freq tremble
    float mod1 = sin(2.0 * M_PI_F * MOD_FREQ1 * t + seed);
    float mod2 = sin(2.0 * M_PI_F * MOD_FREQ2 * t + seed * 1.7);

    // Combine with slight high-frequency noise
    float noise = fract(sin((t + seed) * 12345.678) * 54321.123);
    noise = (noise - 0.5) * 2.0 * NOISE_SCALE;

    // Blend
    float base = basePrediction[tid];
    float modulated = base + (mod1 + mod2) * NOISE_SCALE + noise;

    enrichedOutput[tid] = modulated;
}
