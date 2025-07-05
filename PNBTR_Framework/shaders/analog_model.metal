#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400
#define MODE_TANH     0
#define MODE_ATAN     1
#define MODE_SIGMOID  2
#define MODE_SOFTKNEE 3

using namespace metal;

inline float saturate(float x, int mode) {
    switch (mode) {
        case MODE_TANH:     return tanh(1.5 * x);
        case MODE_ATAN:     return atan(x);
        case MODE_SIGMOID:  return (2.0 / (1.0 + exp(-2.0 * x))) - 1.0;
        case MODE_SOFTKNEE: return (x < -1.0) ? -1.0 :
                             (x >  1.0) ?  1.0 :
                             x - (x * x * x) / 3.0; // soft clip approx
        default: return x;
    }
}

inline float adaptiveLowpass(float x, float prev, float dynamicAlpha) {
    return (dynamicAlpha * x) + ((1.0 - dynamicAlpha) * prev);
}

kernel void analog_model(
    device const float* predictedIn       [[ buffer(0) ]],
    device float* analogOut              [[ buffer(1) ]],
    constant int& saturationMode         [[ buffer(2) ]],  // MODE_* constant
    device const float* dynamicAlphaMap  [[ buffer(3) ]],  // Optional: 1 float per sample
    uint tid                             [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    float raw = predictedIn[tid];

    // Choose curve
    float shaped = saturate(raw, saturationMode);

    // Use previous sample for smoothing
    float prev = (tid == 0) ? shaped : analogOut[tid - 1];

    float alpha = dynamicAlphaMap[tid]; // 0.0 → 1.0 smoothing coefficient
    float smoothed = adaptiveLowpass(shaped, prev, alpha);

    analogOut[tid] = smoothed;
}
