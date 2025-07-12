#include "FrameSyncCoordinator.h"
#include <cstring>
#include <algorithm>

FrameSyncCoordinator::FrameSyncCoordinator() {
    for (auto& status : frameStates) {
        status.reset();
    }
    
    for (auto& waveform : waveformRegistry) {
        waveform.frameIndex = 0;
        waveform.ready.store(false);
        std::memset(waveform.left, 0, sizeof(waveform.left));
        std::memset(waveform.right, 0, sizeof(waveform.right));
    }
}

FrameSyncCoordinator::~FrameSyncCoordinator() {
    cleanup();
}

bool FrameSyncCoordinator::initializeBuffers(size_t maxSamples) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    maxSamplesPerBuffer = maxSamples;
    
    // Allocate audio buffers
    for (int i = 0; i < MAX_BUFFERED_FRAMES; ++i) {
        bufferStorage[i] = std::make_unique<float[]>(maxSamples * 2); // Stereo
        audioBuffers[i].data = bufferStorage[i].get();
        audioBuffers[i].frameIndex = 0;
        audioBuffers[i].ready = false;
        audioBuffers[i].sampleCount = maxSamples;
        
        // Initialize buffer to silence
        std::memset(audioBuffers[i].data, 0, maxSamples * 2 * sizeof(float));
    }
    
    return true;
}

void FrameSyncCoordinator::cleanup() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    for (int i = 0; i < MAX_BUFFERED_FRAMES; ++i) {
        bufferStorage[i].reset();
        audioBuffers[i].data = nullptr;
        audioBuffers[i].ready = false;
    }
}

void FrameSyncCoordinator::beginNewFrame() {
    auto next = (currentFrameIndex.load() + 1) % MAX_BUFFERED_FRAMES;
    frameStates[next].reset();
    currentFrameIndex.store(currentFrameIndex.load() + 1);
}

uint64_t FrameSyncCoordinator::getCurrentFrameIndex() const {
    return currentFrameIndex.load();
}

// ADDED: Frame access helpers that enforce +1 write / -1 read pattern
// These methods ensure GPU writes to frame N and CoreAudio reads from frame N-1
// This prevents race conditions where CoreAudio tries to read while GPU is writing

uint64_t FrameSyncCoordinator::getWriteFrameIndex() const {
    return currentFrameIndex.load();
}

uint64_t FrameSyncCoordinator::getReadFrameIndex() const {
    uint64_t current = currentFrameIndex.load();
    // Return frame N-1 for read operations, but ensure we don't underflow
    return (current > 0) ? current - 1 : 0;
}

AudioFrameBuffer* FrameSyncCoordinator::getWriteBuffer() {
    uint64_t writeFrame = getWriteFrameIndex();
    return getValidatedOutputBufferForWrite(writeFrame);
}

const AudioFrameBuffer* FrameSyncCoordinator::getReadBuffer() const {
    uint64_t readFrame = getReadFrameIndex();
    return getValidatedOutputBufferForRead(readFrame);
}

const AudioFrameBuffer* FrameSyncCoordinator::getReadBufferForDisplay() const {
    // Same as getReadBuffer() but with clearer intent for GUI/metrics usage
    return getReadBuffer();
}

WaveformFrameData* FrameSyncCoordinator::getWriteWaveform() {
    uint64_t writeFrame = getWriteFrameIndex();
    return getWaveformForWrite(writeFrame);
}

const WaveformFrameData* FrameSyncCoordinator::getReadWaveform() const {
    uint64_t readFrame = getReadFrameIndex();
    return getWaveformForRead(readFrame);
}

const WaveformFrameData* FrameSyncCoordinator::getReadWaveformForDisplay() const {
    // Same as getReadWaveform() but with clearer intent for GUI/metrics usage
    return getReadWaveform();
}

void FrameSyncCoordinator::markStageComplete(SyncRole role, uint64_t frameIndex) {
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    auto& status = frameStates[idx];

    switch (role) {
        case SyncRole::AudioInput:      status.inputReady.store(true); break;
        case SyncRole::GPUProcessor:    status.gpuReady.store(true); break;
        case SyncRole::AudioOutput:     status.outputReady.store(true); break;
        case SyncRole::GUIRenderer:     status.guiReady.store(true); break;
        case SyncRole::WaveformDisplay: status.waveformReady.store(true); break;
        case SyncRole::Recorder:        status.recorderReady.store(true); break;
    }
}

bool FrameSyncCoordinator::isFrameReadyFor(SyncRole role, uint64_t frameIndex) const {
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    const auto& status = frameStates[idx];

    switch (role) {
        case SyncRole::AudioInput:      return status.inputReady.load();
        case SyncRole::GPUProcessor:    return status.gpuReady.load();
        case SyncRole::AudioOutput:     return status.outputReady.load();
        case SyncRole::GUIRenderer:     return status.guiReady.load();
        case SyncRole::WaveformDisplay: return status.waveformReady.load();
        case SyncRole::Recorder:        return status.recorderReady.load();
    }

    return false;
}

bool FrameSyncCoordinator::isFrameFullyComplete(uint64_t frameIndex) const {
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    const auto& status = frameStates[idx];

    return status.inputReady.load() &&
           status.gpuReady.load() &&
           status.outputReady.load() &&
           status.guiReady.load() &&
           status.waveformReady.load() &&
           status.recorderReady.load();
}

AudioFrameBuffer* FrameSyncCoordinator::getValidatedOutputBufferForWrite(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    auto& buf = audioBuffers[idx];
    
    // Always allow buffer allocation for write operations
    // The circular buffer design ensures we can safely overwrite old frames
    buf.frameIndex = frameIndex;
    buf.ready = false; // Mark as being written to
    
    // Add debug logging for the first few frames
    static uint32_t debugCount = 0;
    if (++debugCount <= 10) {
        // This will show up in console logs
        printf("[🔧 FRAME SYNC DEBUG] Allocated buffer %d for frame %llu (debug #%u)\n", 
               (int)idx, frameIndex, debugCount);
    }
    
    return &buf;
}

const AudioFrameBuffer* FrameSyncCoordinator::getValidatedOutputBufferForRead(uint64_t frameIndex) const {
    std::lock_guard<std::mutex> lock(bufferMutex);
    
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    const auto& buf = audioBuffers[idx];
    
    return (buf.ready && buf.frameIndex == frameIndex) ? &buf : nullptr;
}

WaveformFrameData* FrameSyncCoordinator::getWaveformForWrite(uint64_t frameIndex) {
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    auto& wf = waveformRegistry[idx];
    
    wf.frameIndex = frameIndex;
    wf.ready = false;
    return &wf;
}

const WaveformFrameData* FrameSyncCoordinator::getWaveformForRead(uint64_t frameIndex) const {
    auto idx = frameIndex % MAX_BUFFERED_FRAMES;
    const auto& wf = waveformRegistry[idx];
    
    return (wf.ready && wf.frameIndex == frameIndex) ? &wf : nullptr;
}

void FrameSyncCoordinator::advanceIfFrameComplete() {
    uint64_t writeFrame = getWriteFrameIndex();
    
    // For now, we'll mark frames complete when GPU processing is done
    // since not all stages are implemented yet
    auto idx = writeFrame % MAX_BUFFERED_FRAMES;
    const auto& status = frameStates[idx];
    
    if (status.gpuReady.load()) {
        // Mark audio buffer as ready for consumption
        audioBuffers[idx].ready = true;
        
        // Mark waveform data as ready
        waveformRegistry[idx].ready = true;
        
        // Debug logging for successful frame completion
        static uint32_t completeCount = 0;
        if (++completeCount <= 5) {
            printf("[✅ FRAME SYNC DEBUG] Frame %llu completed and marked ready (completion #%u)\n", 
                   writeFrame, completeCount);
        }
    }
}

// DeferredSignalQueue implementation
void DeferredSignalQueue::enqueue(uint64_t frameIndex, Callback cb) {
    std::lock_guard<std::mutex> lock(mutex);
    pending[frameIndex] = std::move(cb);
}

void DeferredSignalQueue::markCompleted(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = pending.find(frameIndex);
    if (it != pending.end()) {
        it->second(frameIndex); // invoke callback
        pending.erase(it);
    }
}

void DeferredSignalQueue::flushUpTo(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = pending.begin(); it != pending.end();) {
        if (it->first <= frameIndex) {
            it->second(it->first);
            it = pending.erase(it);
        } else {
            ++it;
        }
    }
} 