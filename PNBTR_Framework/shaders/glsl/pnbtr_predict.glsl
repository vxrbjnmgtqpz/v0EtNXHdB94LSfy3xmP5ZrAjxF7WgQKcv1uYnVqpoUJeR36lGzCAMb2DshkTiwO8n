#version 450

// Constants
#define SAMPLE_WINDOW 96            // 1 ms @ 96kHz
#define PREDICTION_HORIZON 2400     // 50 ms @ 48kHz
#define SAMPLE_RATE 48000.0
#define PI 3.14159265

// Input sample buffer (last 1 ms of real audio)
layout(std430, binding = 0) readonly buffer InputAudio {
    float recentSamples[SAMPLE_WINDOW];
};

// Output predicted samples (50 ms = 2400 samples @ 48kHz)
layout(std430, binding = 1) writeonly buffer PredictedAudio {
    float predictedSamples[PREDICTION_HORIZON];
};

// Shared analysis results per workgroup
shared float envelope;
shared float baseFreq;
shared float phase;

// Envelope follower
float computeEnvelope() {
    float sum = 0.0;
    for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
        sum += abs(recentSamples[i]);
    }
    return sum / float(SAMPLE_WINDOW);
}

// Crude autocorrelation pitch estimation
float estimatePitch() {
    float peak = 0.0;
    int bestLag = 1;
    for (int lag = 1; lag < SAMPLE_WINDOW / 2; ++lag) {
        float sum = 0.0;
        for (int i = 0; i < SAMPLE_WINDOW - lag; ++i) {
            sum += recentSamples[i] * recentSamples[i + lag];
        }
        if (sum > peak) {
            peak = sum;
            bestLag = lag;
        }
    }
    return SAMPLE_RATE / float(bestLag);
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;

    // Step 1: perform envelope + pitch analysis once (thread 0)
    if (tid == 0) {
        envelope = computeEnvelope();
        float est = estimatePitch();
        baseFreq = clamp(est, 60.0, 2000.0);
        phase = 0.0;
    }

    barrier();

    // Step 2: synthesize predicted waveform
    if (tid < PREDICTION_HORIZON) {
        float t = float(tid) / SAMPLE_RATE;
        float wave = envelope * sin(2.0 * PI * baseFreq * t + phase);
        predictedSamples[tid] = wave;
    }
}
