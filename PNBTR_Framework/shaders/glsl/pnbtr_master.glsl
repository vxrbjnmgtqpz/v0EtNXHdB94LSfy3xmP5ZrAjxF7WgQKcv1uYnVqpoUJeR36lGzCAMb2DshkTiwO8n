#version 450

#define PREDICTION_LENGTH 2400
#define CONFIDENCE_BLOCK_SIZE 48     // 1 ms @ 48kHz
#define NUM_MODELS 6

// Optional: weights buffer for per-model weighting
struct BlendWeights {
    float lpc;
    float pitch;
    float formant;
    float analog;
    float rnn;
    float micro;
};

layout(std430, binding = 0) readonly buffer LpcOut {
    float lpcOut[PREDICTION_LENGTH];
};

layout(std430, binding = 1) readonly buffer PitchOut {
    float pitchOut[PREDICTION_LENGTH];
};

layout(std430, binding = 2) readonly buffer FormantOut {
    float formantOut[PREDICTION_LENGTH];
};

layout(std430, binding = 3) readonly buffer AnalogOut {
    float analogOut[PREDICTION_LENGTH];
};

layout(std430, binding = 4) readonly buffer RnnOut {
    float rnnOut[PREDICTION_LENGTH];
};

layout(std430, binding = 5) readonly buffer MicrodynOut {
    float microdynOut[PREDICTION_LENGTH];
};

layout(std430, binding = 6) readonly buffer Confidence {
    float confidence[PREDICTION_LENGTH / CONFIDENCE_BLOCK_SIZE];
};

layout(std430, binding = 7) readonly buffer Weights {
    BlendWeights weights;
};

layout(std430, binding = 8) writeonly buffer FinalOut {
    float finalOut[PREDICTION_LENGTH];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
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
