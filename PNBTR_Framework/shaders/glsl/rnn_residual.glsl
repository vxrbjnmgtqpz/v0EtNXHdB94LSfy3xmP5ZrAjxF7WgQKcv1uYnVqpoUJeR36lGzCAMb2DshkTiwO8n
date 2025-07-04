#version 450

#define PREDICTION_LENGTH 2400     // 50ms @ 48kHz

layout(std430, binding = 0) readonly buffer BasePrediction {
    float basePrediction[PREDICTION_LENGTH];  // From LPC, pitch-cycle, etc.
};

layout(std430, binding = 1) readonly buffer ResidualDelta {
    float residualDelta[PREDICTION_LENGTH];  // From RNN/GRU output (predicted residual)
};

layout(std430, binding = 2) writeonly buffer CorrectedPrediction {
    float correctedPrediction[PREDICTION_LENGTH];  // Final output
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= PREDICTION_LENGTH) return;

    // Simple residual addition
    float base = basePrediction[tid];
    float delta = residualDelta[tid];

    correctedPrediction[tid] = base + delta;
}
