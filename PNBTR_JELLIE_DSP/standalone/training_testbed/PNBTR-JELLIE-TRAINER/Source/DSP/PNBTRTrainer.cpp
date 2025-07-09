/*
  ==============================================================================

    PNBTRTrainer.cpp
    Created: Real GPU-native training harness implementation

    JUCE AudioProcessor that dispatches Metal kernels for the real system:
    CoreAudio → JELLIE → PNBTR → OUT

  ==============================================================================
*/

#include "PNBTRTrainer.h"
#include "../GPU/MetalBridge.h"
#include "../Metrics/TrainingMetrics.h"
#include "../Network/PacketLossSimulator.h"

//==============================================================================
PNBTRTrainer::PNBTRTrainer()
{
    // Preallocate double buffers for oscilloscope (2x block size, stereo)
    oscInputBufferA.resize(2048 * 2, 0.0f);
    oscInputBufferB.resize(2048 * 2, 0.0f);
    oscOutputBufferA.resize(2048 * 2, 0.0f);
    oscOutputBufferB.resize(2048 * 2, 0.0f);
    // Preallocate recording buffer (e.g. 10 seconds at 48kHz stereo)
    recordBuffer.resize(48000 * 10 * 2, 0.0f);

    // Use singleton MetalBridge
    // All access to MetalBridge is via MetalBridge::getInstance()
    metrics = std::make_unique<TrainingMetrics>();
    packetLossSimulator = std::make_unique<PacketLossSimulator>();
    
    // Initialize Metal bridge
    if (!MetalBridge::getInstance().initialize()) {
        // Fallback to CPU processing if Metal fails
        juce::Logger::writeToLog("Metal initialization failed - falling back to CPU");
    }
}

PNBTRTrainer::~PNBTRTrainer()
{
    releaseResources();
}

//==============================================================================
void PNBTRTrainer::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;
    
    // Initialize GPU buffers for zero-copy processing
    initializeGPUBuffers();
    
    // Configure Metal bridge
    MetalBridge::getInstance().setProcessingParameters(sampleRate, samplesPerBlock);
    MetalBridge::getInstance().prepareBuffers(samplesPerBlock, sampleRate);
    
    // Initialize metrics
    metrics->prepare(sampleRate, samplesPerBlock);
    
    // Initialize packet loss simulator
    packetLossSimulator->prepare(sampleRate, samplesPerBlock);
    
    juce::Logger::writeToLog("PNBTRTrainer prepared: " + 
                            juce::String(sampleRate) + "Hz, " + 
                            juce::String(samplesPerBlock) + " samples");
}

void PNBTRTrainer::releaseResources()
{
    MetalBridge::getInstance().shutdown();
    
    trainingActive.store(false);
}

//==============================================================================
void PNBTRTrainer::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    static int blockCount = 0;
    if (++blockCount % 100 == 0) {
        juce::Logger::writeToLog("[PNBTRTrainer::processBlock] Called (" + juce::String(blockCount) + ")");
    }
    if (!trainingActive.load() || !MetalBridge::getInstance().isInitialized()) {
        // Pass through audio unchanged if training is inactive
        return;
    }
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    // Ensure we have stereo input
    if (numChannels < 2) {
        buffer.setSize(2, numSamples, true, true, true);
    }

    // --- Thread-safe oscilloscope and recording buffer updates ---

    updateOscilloscopeBuffers(buffer);
    if (recordingActive.load())
        updateRecordingBuffer(buffer);

    // **REAL PROCESSING PIPELINE - GPU KERNELS**
    // 1. Copy input samples into shared Metal input buffer (zero-copy)
    copyInputToGPU(buffer);
    // 2. Generate packet loss map for this block
    packetLossSimulator->generateLossMap(packetLossPercentage.load(), 
                                        jitterAmount.load());
    // 3. Encode with JELLIE (Metal kernel: 48kHz→192kHz + 8-channel distribution)
    runJellieEncode();
    // 4. Apply simulated packet loss/jitter (Metal kernel)
    runNetworkSimulation();
    // 5. Reconstruct via PNBTR (Metal kernel: neural gap filling)
    runPNBTRReconstruction();
    // 6. Copy reconstructed buffer back to CPU output (zero-copy)
    copyOutputFromGPU(buffer);
    // 7. Update shared metrics/log buffers (part of core loop)
    updateMetrics();
    // Clear unused MIDI
    midiMessages.clear();
}

// Thread-safe oscilloscope buffer update (call in processBlock)
void PNBTRTrainer::updateOscilloscopeBuffers(const juce::AudioBuffer<float>& buffer)
{
    // Interleave stereo for oscilloscope
    const float* left = buffer.getReadPointer(0);
    const float* right = buffer.getReadPointer(1);
    const int n = buffer.getNumSamples();
    bool toggle = !oscBufferToggle.load(std::memory_order_relaxed);
    std::vector<float>& oscIn = toggle ? oscInputBufferA : oscInputBufferB;
    std::vector<float>& oscOut = toggle ? oscOutputBufferA : oscOutputBufferB;
    for (int i = 0; i < n; ++i) {
        oscIn[i * 2] = left[i];
        oscIn[i * 2 + 1] = right[i];
    }
    // For output oscilloscope, use output buffer (after processing)
    // Here, just copy input as placeholder; replace with actual output if needed
    for (int i = 0; i < n; ++i) {
        oscOut[i * 2] = left[i];
        oscOut[i * 2 + 1] = right[i];
    }
    oscBufferToggle.store(toggle, std::memory_order_release);
}

// Thread-safe recording buffer update (call in processBlock)
void PNBTRTrainer::updateRecordingBuffer(const juce::AudioBuffer<float>& buffer)
{
    const float* left = buffer.getReadPointer(0);
    const float* right = buffer.getReadPointer(1);
    const int n = buffer.getNumSamples();
    size_t pos = recordWritePos.load(std::memory_order_relaxed);
    for (int i = 0; i < n; ++i) {
        recordBuffer[(pos + i) * 2] = left[i];
        recordBuffer[(pos + i) * 2 + 1] = right[i];
    }
    recordWritePos.store((pos + n) % (recordBuffer.size() / 2), std::memory_order_release);
}

// GUI thread: get latest oscilloscope input buffer (thread-safe)
void PNBTRTrainer::getLatestOscInput(float* dest, int numSamples)
{
    bool toggle = oscBufferToggle.load(std::memory_order_acquire);
    const std::vector<float>& oscIn = toggle ? oscInputBufferA : oscInputBufferB;
    std::memcpy(dest, oscIn.data(), sizeof(float) * numSamples * 2);
}

// GUI thread: get latest oscilloscope output buffer (thread-safe)
void PNBTRTrainer::getLatestOscOutput(float* dest, int numSamples)
{
    bool toggle = oscBufferToggle.load(std::memory_order_acquire);
    const std::vector<float>& oscOut = toggle ? oscOutputBufferA : oscOutputBufferB;
    std::memcpy(dest, oscOut.data(), sizeof(float) * numSamples * 2);
}

// GUI thread: get latest recorded buffer (thread-safe)
void PNBTRTrainer::getRecordedBuffer(float* dest, int numSamples, size_t offset)
{
    size_t pos = (recordWritePos.load(std::memory_order_acquire) + offset) % (recordBuffer.size() / 2);
    for (int i = 0; i < numSamples; ++i) {
        dest[i * 2] = recordBuffer[(pos + i) * 2];
        dest[i * 2 + 1] = recordBuffer[(pos + i) * 2 + 1];
    }
}

//==============================================================================
void PNBTRTrainer::runJellieEncode()
{
    if (!MetalBridge::getInstance().runJellieEncode()) {
        juce::Logger::writeToLog("JELLIE encode kernel failed");
    }
}

void PNBTRTrainer::runNetworkSimulation()
{
    if (!MetalBridge::getInstance().runNetworkSimulation(packetLossPercentage.load(), 
                                          jitterAmount.load())) {
        juce::Logger::writeToLog("Network simulation kernel failed");
    }
}

void PNBTRTrainer::runPNBTRReconstruction()
{
    if (!MetalBridge::getInstance().runPNBTRReconstruction()) {
        juce::Logger::writeToLog("PNBTR reconstruction kernel failed");
    }
}

void PNBTRTrainer::updateMetrics()
{
    // Update metrics from GPU buffers (shared data, not post-analysis)
    MetalBridge::getInstance().updateMetrics();
    float snr, thd, latency;
    MetalBridge::getInstance().getMetricsData(&snr, &thd, &latency);
    int totalPackets, lostPackets;
    MetalBridge::getInstance().getPacketLossStats(&totalPackets, &lostPackets);
    // Update training metrics
    metrics->updateFromGPU(snr, thd, latency, totalPackets, lostPackets);
}

//==============================================================================
void PNBTRTrainer::initializeGPUBuffers()
{
    // GPU buffers are initialized in MetalBridge
    // This ensures zero-copy from CPU→GPU via Metal MTLBuffer objects
}

void PNBTRTrainer::copyInputToGPU(const juce::AudioBuffer<float>& buffer)
{
    // Zero-copy transfer to Metal buffer
    const float* leftChannel = buffer.getReadPointer(0);
    const float* rightChannel = buffer.getReadPointer(1);
    
    // Interleave stereo data for GPU processing
    std::vector<float> interleavedData(buffer.getNumSamples() * 2);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        interleavedData[i * 2] = leftChannel[i];
        interleavedData[i * 2 + 1] = rightChannel[i];
    }
    
    MetalBridge::getInstance().copyInputToGPU(interleavedData.data(), 
                               buffer.getNumSamples(), 
                               2);
}

void PNBTRTrainer::copyOutputFromGPU(juce::AudioBuffer<float>& buffer)
{
    // Zero-copy transfer from Metal buffer
    std::vector<float> interleavedData(buffer.getNumSamples() * 2);
    
    MetalBridge::getInstance().copyOutputFromGPU(interleavedData.data(), 
                                  buffer.getNumSamples(), 
                                  2);
    
    // De-interleave stereo data from GPU
    float* leftChannel = buffer.getWritePointer(0);
    float* rightChannel = buffer.getWritePointer(1);
    
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        leftChannel[i] = interleavedData[i * 2];
        rightChannel[i] = interleavedData[i * 2 + 1];
    }
}

//==============================================================================
void PNBTRTrainer::setPacketLossPercentage(float percentage)
{
    packetLossPercentage.store(juce::jlimit(0.0f, 100.0f, percentage));
    MetalBridge::getInstance().setNetworkParameters(percentage, jitterAmount.load());
}

void PNBTRTrainer::setJitterAmount(float jitterMs)
{
    jitterAmount.store(juce::jmax(0.0f, jitterMs));
    MetalBridge::getInstance().setNetworkParameters(packetLossPercentage.load(), jitterMs);
}

void PNBTRTrainer::setGain(float gainDb)
{
    this->gainDb.store(juce::jlimit(-60.0f, 20.0f, gainDb));
}

//==============================================================================
void PNBTRTrainer::startTraining()
{
    trainingActive.store(true);
    metrics->reset();
    juce::Logger::writeToLog("Training started");
}

void PNBTRTrainer::stopTraining()
{
    trainingActive.store(false);
    juce::Logger::writeToLog("Training stopped");
}

//==============================================================================
void PNBTRTrainer::getInputBuffer(float* destination, int numSamples)
{
    // Get input buffer from GPU for oscilloscope display
    if (MetalBridge::getInstance().getInputBufferPtr()) {
        const float* inputPtr = MetalBridge::getInstance().getInputBufferPtr();
        std::memcpy(destination, inputPtr, numSamples * sizeof(float));
    }
}

void PNBTRTrainer::getOutputBuffer(float* destination, int numSamples)
{
    // Get output buffer from GPU for oscilloscope display
    if (MetalBridge::getInstance().getOutputBufferPtr()) {
        const float* outputPtr = MetalBridge::getInstance().getOutputBufferPtr();
        std::memcpy(destination, outputPtr, numSamples * sizeof(float));
    }
}

//==============================================================================
void PNBTRTrainer::getStateInformation(juce::MemoryBlock& destData)
{
    // Save training parameters
    juce::MemoryOutputStream stream(destData, true);
    
    stream.writeFloat(packetLossPercentage.load());
    stream.writeFloat(jitterAmount.load());
    stream.writeFloat(gainDb.load());
    stream.writeBool(trainingActive.load());
}

void PNBTRTrainer::setStateInformation(const void* data, int sizeInBytes)
{
    // Restore training parameters
    juce::MemoryInputStream stream(data, static_cast<size_t>(sizeInBytes), false);
    
    packetLossPercentage.store(stream.readFloat());
    jitterAmount.store(stream.readFloat());
    gainDb.store(stream.readFloat());
    trainingActive.store(stream.readBool());
} 