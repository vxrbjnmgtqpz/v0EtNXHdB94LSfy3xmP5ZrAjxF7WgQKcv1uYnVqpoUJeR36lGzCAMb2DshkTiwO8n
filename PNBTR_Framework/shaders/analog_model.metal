#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400     // 50 ms @ 48kHz

using namespace metal;

// Analog soft-clipper (tape/saturation-like curve)
inline float analogSaturate(float x) {
    return tanh(1.5 * x);  // Mild saturation curve
}

// Smoothing filter (very basic lowpass decay)
inline float analogLowpass(float x, float prev, float alpha) {
    return (alpha * x) + ((1.0 - alpha) * prev);
}

kernel void analog_model(
    device const float* predictedInput      [[ buffer(0) ]],  // From any model (LPC, RNN, etc.)
    device float* analogSmoothedOutput      [[ buffer(1) ]],
    constant float& alpha                   [[ buffer(2) ]],  // Lowpass coefficient (e.g., 0.25)
    uint tid                                [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    // Read input sample
    float raw = predictedInput[tid];

    // Soft clip to simulate analog wave compression
    float shaped = analogSaturate(raw);

    // Smooth with simple 1-pole filter
    float prev = (tid == 0) ? 0.0 : analogSmoothedOutput[tid - 1];
    float smooth = analogLowpass(shaped, prev, alpha);

    analogSmoothedOutput[tid] = smooth;
}
