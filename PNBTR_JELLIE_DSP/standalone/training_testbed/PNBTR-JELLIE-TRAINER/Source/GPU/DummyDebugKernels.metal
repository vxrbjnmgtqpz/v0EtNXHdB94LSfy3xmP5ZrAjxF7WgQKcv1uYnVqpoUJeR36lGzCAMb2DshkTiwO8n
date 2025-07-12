#include <metal_stdlib>
using namespace metal;

struct DummyParams {
    uint frameOffset;
    uint numSamples;
    float testGain;      // Simple gain to verify data is flowing
    uint frameIndex;     // Frame index for debugging
};

// Dummy pass-through kernel that just copies input to output with optional gain
kernel void audioDummyPassThroughKernel(
    constant DummyParams& params [[buffer(0)]],
    device const float* input     [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {

    uint i = params.frameOffset + gid;
    if (i >= params.numSamples) return;

    // Simple pass-through with gain to verify pipeline is working
    // Add a tiny sine wave at 1kHz to make it audible if working
    float sineWave = 0.01f * sin(2.0f * M_PI_F * 1000.0f * float(i) / 48000.0f);
    output[i] = (input[i] * params.testGain) + sineWave;
}

// Debug kernel that fills output with a test pattern
kernel void audioDebugTestPatternKernel(
    constant DummyParams& params [[buffer(0)]],
    device float*       output    [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {

    uint i = params.frameOffset + gid;
    if (i >= params.numSamples) return;

    // Generate test pattern: alternating positive/negative samples at frame rate
    float pattern = (params.frameIndex % 2 == 0) ? 0.1f : -0.1f;
    output[i] = pattern * sin(2.0f * M_PI_F * float(i) / float(params.numSamples));
}

// Validation kernel that ensures data is non-zero
kernel void audioValidationKernel(
    constant DummyParams& params [[buffer(0)]],
    device const float* input     [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    device atomic_uint* nonZeroCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {

    uint i = params.frameOffset + gid;
    if (i >= params.numSamples) return;

    // Pass through and count non-zero samples
    output[i] = input[i];
    
    if (abs(input[i]) > 0.0001f) {
        atomic_fetch_add_explicit(nonZeroCount, 1, memory_order_relaxed);
    }
}
