#version 450

#define SAMPLE_WINDOW 480       // 10 ms @ 48kHz
#define OUTPUT_POINTS 64        // Envelope output resolution

// Input buffer: recent audio history
layout(std430, binding = 0) readonly buffer AudioHistory {
    float audioHistory[SAMPLE_WINDOW];
};

layout(std430, binding = 1) writeonly buffer EnvelopeOutput {
    float envelopeOutput[OUTPUT_POINTS];
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= OUTPUT_POINTS) return;

    // Compute sample region this thread is responsible for
    const int samplesPerBin = SAMPLE_WINDOW / OUTPUT_POINTS;
    int startIdx = int(tid) * samplesPerBin;
    int endIdx = startIdx + samplesPerBin;

    float sum = 0.0;
    for (int i = startIdx; i < endIdx; ++i) {
        sum += abs(audioHistory[i]);
    }

    float avg = sum / float(samplesPerBin);
    envelopeOutput[tid] = avg;
}
