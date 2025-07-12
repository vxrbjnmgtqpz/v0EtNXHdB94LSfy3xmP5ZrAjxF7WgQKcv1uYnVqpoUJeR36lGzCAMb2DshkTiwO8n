kernel void PNBTR_reconstruct(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const bool* packetLossMap [[buffer(2)]],
    device const ProcessingParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float sample = input[id];
    // TEMP BYPASS EVERYTHING
    output[id] = sample;
    // TODO: debug network logic later
}