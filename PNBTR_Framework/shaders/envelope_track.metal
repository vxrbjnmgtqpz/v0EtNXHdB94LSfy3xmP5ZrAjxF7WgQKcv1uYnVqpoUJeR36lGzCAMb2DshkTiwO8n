#include <metal_stdlib>
using namespace metal;

#define SAMPLE_WINDOW 480       // 10 ms @ 48kHz
#define OUTPUT_POINTS 64        // Envelope output resolution

// Input buffer: recent audio history
kernel void envelope_track(
    device const float* audioHistory        [[ buffer(0) ]],
    device float* envelopeOutput            [[ buffer(1) ]],
    uint tid                                [[ thread_position_in_grid ]]
) {
    if (tid >= OUTPUT_POINTS) return;

    // Compute sample region this thread is responsible for
    const int samplesPerBin = SAMPLE_WINDOW / OUTPUT_POINTS;
    int startIdx = tid * samplesPerBin;
    int endIdx = startIdx + samplesPerBin;

    float sum = 0.0;
    for (int i = startIdx; i < endIdx; ++i) {
        sum += fabs(audioHistory[i]);
    }

    float avg = sum / float(samplesPerBin);
    envelopeOutput[tid] = avg;
}
