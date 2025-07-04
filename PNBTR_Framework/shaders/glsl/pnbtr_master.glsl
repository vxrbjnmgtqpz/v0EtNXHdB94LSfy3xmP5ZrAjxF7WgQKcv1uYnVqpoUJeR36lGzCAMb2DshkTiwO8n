#version 450

#define PREDICTION_LENGTH 2400
#define NUM_MODELS 6  // e.g. LPC, pitch, formant, analog, RNN, microdyn

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
    float confidence[PREDICTION_LENGTH / 48];  // Per-48-sample block
};

layout(std430, binding = 7) writeonly buffer FinalOut {
    float finalOut[PREDICTION_LENGTH];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
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
