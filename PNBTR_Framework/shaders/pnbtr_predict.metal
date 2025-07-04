#include <metal_stdlib>
using namespace metal;

// Constants
#define SAMPLE_WINDOW 96            // 1 ms @ 96kHz
#define PREDICTION_HORIZON 2400     // 50 ms @ 48kHz
#define SAMPLE_RATE 48000.0
#define PI 3.14159265

// Input sample buffer (last 1 ms of real audio)
struct InputAudio {
    device const float* recentSamples [[id(0)]];
};

// Output predicted samples (50 ms = 2400 samples @ 48kHz)
struct PredictedAudio {
    device float* predictedSamples [[id(1)]];
};

// Shared analysis results per threadgroup
struct SharedAnalysis {
    float envelope;
    float baseFreq;
    float phase;
};

// Envelope follower
float computeEnvelope(thread const float* samples) {
    float sum = 0.0;
    for (uint i = 0; i < SAMPLE_WINDOW; ++i) {
        sum += fabs(samples[i]);
    }
    return sum / float(SAMPLE_WINDOW);
}

// Crude autocorrelation pitch estimation
float estimatePitch(thread const float* samples) {
    float peak = 0.0;
    int bestLag = 1;
    for (int lag = 1; lag < SAMPLE_WINDOW / 2; ++lag) {
        float sum = 0.0;
        for (int i = 0; i < SAMPLE_WINDOW - lag; ++i) {
            sum += samples[i] * samples[i + lag];
        }
        if (sum > peak) {
            peak = sum;
            bestLag = lag;
        }
    }
    return SAMPLE_RATE / float(bestLag);
}

// Main compute kernel
kernel void pnbtr_predict(
    device const float* recentSamples [[ buffer(0) ]],
    device float* predictedSamples [[ buffer(1) ]],
    uint tid [[thread_position_in_grid]],
    threadgroup SharedAnalysis& shared [[threadgroup(0)]]
) {
    // Step 1: perform envelope + pitch analysis once (thread 0)
    if (tid == 0) {
        shared.envelope = computeEnvelope(recentSamples);
        float est = estimatePitch(recentSamples);
        shared.baseFreq = clamp(est, 60.0, 2000.0);
        shared.phase = 0.0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: synthesize predicted waveform
    if (tid < PREDICTION_HORIZON) {
        float t = float(tid) / SAMPLE_RATE;
        float wave = shared.envelope * sin(2.0 * PI * shared.baseFreq * t + shared.phase);
        predictedSamples[tid] = wave;
    }
}
