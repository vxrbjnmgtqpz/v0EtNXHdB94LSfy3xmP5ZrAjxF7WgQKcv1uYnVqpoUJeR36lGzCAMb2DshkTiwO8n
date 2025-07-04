#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400     // 50ms @ 48kHz

using namespace metal;

kernel void rnn_residual(
    device const float* basePrediction     [[ buffer(0) ]],  // From LPC, pitch-cycle, etc.
    device const float* residualDelta      [[ buffer(1) ]],  // From RNN/GRU output (predicted residual)
    device float* correctedPrediction      [[ buffer(2) ]],  // Final output
    uint tid                               [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    // Simple residual addition
    float base = basePrediction[tid];
    float delta = residualDelta[tid];

    correctedPrediction[tid] = base + delta;
}
