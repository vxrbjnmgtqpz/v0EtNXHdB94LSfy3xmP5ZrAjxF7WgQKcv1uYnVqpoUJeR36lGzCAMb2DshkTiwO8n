// Compute sample value
// ...existing code...
// Remove debug patch and write actual sample
output[id] = sample;
// Optional: test signal for first few samples
if (id < 4) {
    output[id] = 0.1f * float(id);
}

// In runPipelineStages(), ensure final stage output is bound to reconstructedBuffer
id<MTLBuffer> output = objc_reconstructedBuffer[frameIndex % MAX_FRAMES_IN_FLIGHT];
NSLog(@"[🟩 FINAL STAGE] Writing to reconstructedBuffer[%u]", frameIndex % MAX_FRAMES_IN_FLIGHT);

// In Metal kernel, inject debug pulse
if (id == 0) {
    output[0] = 0.25f;
}