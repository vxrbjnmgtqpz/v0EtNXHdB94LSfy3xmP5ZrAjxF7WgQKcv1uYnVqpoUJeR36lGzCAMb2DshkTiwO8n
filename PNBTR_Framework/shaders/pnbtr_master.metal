#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400
#define NUM_MODELS 6  // e.g. LPC, pitch, formant, analog, RNN, microdyn

using namespace metal;

kernel void pnbtr_master(
    device const float* lpcOut         [[ buffer(0) ]],
    device const float* pitchOut       [[ buffer(1) ]],
    device const float* formantOut     [[ buffer(2) ]],
    device const float* analogOut      [[ buffer(3) ]],
    device const float* rnnOut         [[ buffer(4) ]],
    device const float* microdynOut    [[ buffer(5) ]],
    device const float* confidence     [[ buffer(6) ]],  // Per-48-sample block
    device float* finalOut             [[ buffer(7) ]],
    uint tid                           [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    // Block confidence
    uint blockIdx = tid / 48;
    float conf = clamp(confidence[blockIdx], 0.0, 1.0);

    // Fixed blending weights (tunable or learned)
    const float wLPC      = 0.25;
    const float wPitch    = 0.20;
    const float wFormant  = 0.15;
    const float wAnalog   = 0.10;
    const float wRNN      = 0.20;
    const float wMicro    = 0.10;

    // Weighted sum of all predictions
    float blended =
        (lpcOut[tid] * wLPC) +
        (pitchOut[tid] * wPitch) +
        (formantOut[tid] * wFormant) +
        (analogOut[tid] * wAnalog) +
        (rnnOut[tid] * wRNN) +
        (microdynOut[tid] * wMicro);

    // Fade out low-confidence zones
    finalOut[tid] = conf * blended;
}
