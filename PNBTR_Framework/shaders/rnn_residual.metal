#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400

using namespace metal;

// Inputs:
// - basePrediction[]: deterministic guess (from LPC, spectral, etc.)
// - residualDelta[]: result of pre-run neural model (injected by CPU/CoreML runtime)
// - correctionScale: dynamic mix scalar (per-model confidence, or session-tuned)

kernel void rnn_residual(
    device const float* basePrediction   [[ buffer(0) ]],
    device const float* residualDelta    [[ buffer(1) ]],
    constant float& correctionScale      [[ buffer(2) ]],
    device float* correctedPrediction    [[ buffer(3) ]],
    uint tid                             [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    float base   = basePrediction[tid];
    float delta  = residualDelta[tid];

    // Apply residual correction with optional weighting
    float corrected = base + (correctionScale * delta);
    correctedPrediction[tid] = corrected;
}
