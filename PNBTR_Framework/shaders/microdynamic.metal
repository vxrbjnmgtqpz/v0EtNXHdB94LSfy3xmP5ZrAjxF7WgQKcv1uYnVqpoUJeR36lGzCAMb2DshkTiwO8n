#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400
#define SAMPLE_RATE 48000.0

using namespace metal;

// Parameters passed in per-call
struct MicroParams {
    float baseIntensity;      // e.g. 0.003
    float modFreq1;           // Hz, shimmer component 1
    float modFreq2;           // Hz, shimmer component 2
    float grainJitter;        // 0.0 to 1.0
};

kernel void microdynamic(
    device const float* baseInput          [[ buffer(0) ]],
    device const float* confidenceMap      [[ buffer(1) ]], // 1 float per sample (0–1)
    constant MicroParams& params           [[ buffer(2) ]],
    device float* enrichedOutput           [[ buffer(3) ]],
    constant float& seed                   [[ buffer(4) ]],  // stream ID / randomness seed
    uint tid                               [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    float t = float(tid) / SAMPLE_RATE;
    float base = baseInput[tid];
    float conf = clamp(confidenceMap[tid], 0.0, 1.0);

    // Generate shimmer modulation using two sine components
    float mod1 = sin(2.0 * M_PI_F * params.modFreq1 * t + seed);
    float mod2 = sin(2.0 * M_PI_F * params.modFreq2 * t + seed * 1.3);

    // Compute jittered grain noise
    float jitter = fract(sin(t * 1000.0 + seed) * 43758.5453);
    float noise = (jitter - 0.5) * 2.0 * params.grainJitter;

    // Combine shimmer and noise with confidence gating
    float shimmer = (mod1 + mod2) * 0.5;
    float dynamicGrain = (shimmer + noise) * params.baseIntensity * conf;

    enrichedOutput[tid] = base + dynamicGrain;
}
