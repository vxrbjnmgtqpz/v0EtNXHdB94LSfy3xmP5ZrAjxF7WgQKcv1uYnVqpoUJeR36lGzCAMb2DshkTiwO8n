#version 450

#define PREDICTION_LENGTH 2400     // 50 ms @ 48kHz

// Analog soft-clipper (tape/saturation-like curve)
float analogSaturate(float x) {
    return tanh(1.5 * x);  // Mild saturation curve
}

// Smoothing filter (very basic lowpass decay)
float analogLowpass(float x, float prev, float alpha) {
    return (alpha * x) + ((1.0 - alpha) * prev);
}

layout(std430, binding = 0) readonly buffer PredictedInput {
    float predictedInput[PREDICTION_LENGTH];  // From any model (LPC, RNN, etc.)
};

layout(std430, binding = 1) writeonly buffer AnalogSmoothedOutput {
    float analogSmoothedOutput[PREDICTION_LENGTH];
};

layout(std430, binding = 2) readonly buffer AlphaBuffer {
    float alpha;  // Lowpass coefficient (e.g., 0.25)
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
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
