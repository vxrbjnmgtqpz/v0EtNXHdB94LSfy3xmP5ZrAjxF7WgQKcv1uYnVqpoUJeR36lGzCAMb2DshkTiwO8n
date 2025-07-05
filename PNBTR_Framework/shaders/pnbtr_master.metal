#include <metal_stdlib>
using namespace metal;

#define PREDICTION_LENGTH 2400
#define CONFIDENCE_BLOCK_SIZE 48     // 1 ms @ 48kHz
#define NUM_MODELS 6

using namespace metal;

// Optional: weights buffer for per-model weighting
struct BlendWeights {
    float lpc;
    float pitch;
    float formant;
    float analog;
    float rnn;
    float micro;
};

// Input buffers for predicted signals
kernel void pnbtr_master(
    device const float* lpcOut         [[ buffer(0) ]],
    device const float* pitchOut       [[ buffer(1) ]],
    device const float* formantOut     [[ buffer(2) ]],
    device const float* analogOut      [[ buffer(3) ]],
    device const float* rnnOut         [[ buffer(4) ]],
    device const float* microdynOut    [[ buffer(5) ]],
    device const float* confidence     [[ buffer(6) ]],
    device const BlendWeights& weights [[ buffer(7) ]],
    device float* finalOut             [[ buffer(8) ]],
    uint tid                           [[ thread_position_in_grid ]]
) {
    if (tid >= PREDICTION_LENGTH) return;

    // Determine block-level confidence
    uint blockIdx = tid / CONFIDENCE_BLOCK_SIZE;
    float conf = clamp(confidence[blockIdx], 0.0, 1.0);

    // Weighted blend of all sources
    float combined =
          lpcOut[tid]      * weights.lpc
        + pitchOut[tid]    * weights.pitch
        + formantOut[tid]  * weights.formant
        + analogOut[tid]   * weights.analog
        + rnnOut[tid]      * weights.rnn
        + microdynOut[tid] * weights.micro;

    // Normalize by total weight
    float totalWeight = weights.lpc + weights.pitch + weights.formant +
                        weights.analog + weights.rnn + weights.micro;
    combined /= max(totalWeight, 1e-5); // prevent div-by-zero

    // Confidence blending: fade out or reduce intensity
    float output = (conf > 0.05) ? combined : combined * conf;

    finalOut[tid] = output;
}
