#version 450

#define PREDICTION_LENGTH 2400
#define MODE_TANH     0
#define MODE_ATAN     1
#define MODE_SIGMOID  2
#define MODE_SOFTKNEE 3

float saturate_func(float x, int mode) {
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

float adaptiveLowpass(float x, float prev, float dynamicAlpha) {
    return (dynamicAlpha * x) + ((1.0 - dynamicAlpha) * prev);
}

layout(std430, binding = 0) readonly buffer PredictedIn {
    float predictedIn[PREDICTION_LENGTH];
};

layout(std430, binding = 1) writeonly buffer AnalogOut {
    float analogOut[PREDICTION_LENGTH];
};

layout(std430, binding = 2) readonly buffer SaturationMode {
    int saturationMode;  // MODE_* constant
};

layout(std430, binding = 3) readonly buffer DynamicAlphaMap {
    float dynamicAlphaMap[PREDICTION_LENGTH];  // Optional: 1 float per sample
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= PREDICTION_LENGTH) return;

    float raw = predictedIn[tid];

    // Choose curve
    float shaped = saturate_func(raw, saturationMode);

    // Use previous sample for smoothing
    float prev = (tid == 0) ? shaped : analogOut[tid - 1];

    float alpha = dynamicAlphaMap[tid]; // 0.0 → 1.0 smoothing coefficient
    float smoothed = adaptiveLowpass(shaped, prev, alpha);

    analogOut[tid] = smoothed;
}
