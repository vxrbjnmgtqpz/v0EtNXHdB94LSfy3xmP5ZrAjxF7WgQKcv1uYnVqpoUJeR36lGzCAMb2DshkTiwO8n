#pragma once
#include <cstdint>
#include <atomic>
#include <mutex>
#include <array>
#include <functional>
#include <unordered_map>
#include <memory>
#include <map>

enum class SyncRole {
    AudioInput,
    GPUProcessor,
    AudioOutput,
    GUIRenderer,
    WaveformDisplay,
    Recorder
};

struct FrameStatus {
    std::atomic<bool> inputReady      {false};
    std::atomic<bool> gpuReady        {false};
    std::atomic<bool> outputReady     {false};
    std::atomic<bool> guiReady        {false};
    std::atomic<bool> waveformReady   {false};
    std::atomic<bool> recorderReady   {false};

    void reset() {
        inputReady.store(false);
        gpuReady.store(false);
        outputReady.store(false);
        guiReady.store(false);
        waveformReady.store(false);
        recorderReady.store(false);
    }
    
    // Make non-copyable due to atomic members
    FrameStatus() = default;
    FrameStatus(const FrameStatus&) = delete;
    FrameStatus& operator=(const FrameStatus&) = delete;
};

struct AudioFrameBuffer {
    float* data;
    uint64_t frameIndex;
    bool ready;
    size_t sampleCount;
    
    AudioFrameBuffer() : data(nullptr), frameIndex(0), ready(false), sampleCount(0) {}
};

constexpr int WAVEFORM_SNAPSHOT_SIZE = 128;

struct WaveformFrameData {
    float left[WAVEFORM_SNAPSHOT_SIZE] = {};
    float right[WAVEFORM_SNAPSHOT_SIZE] = {};
    std::atomic<bool> ready = false;
    uint64_t frameIndex = 0;
    
    // Make non-copyable due to atomic members
    WaveformFrameData() = default;
    WaveformFrameData(const WaveformFrameData&) = delete;
    WaveformFrameData& operator=(const WaveformFrameData&) = delete;
};

class FrameSyncCoordinator {
public:
    static constexpr int MAX_BUFFERED_FRAMES = 8;

    FrameSyncCoordinator();
    ~FrameSyncCoordinator();
    
    // Make non-copyable due to atomic members
    FrameSyncCoordinator(const FrameSyncCoordinator&) = delete;
    FrameSyncCoordinator& operator=(const FrameSyncCoordinator&) = delete;

    // Frame lifecycle management
    void beginNewFrame(); // Called once per audio tick (CoreAudio buffer callback)
    uint64_t getCurrentFrameIndex() const;

    // ADDED: Frame access helpers that enforce +1 write / -1 read pattern
    // These methods ensure GPU writes to frame N and CoreAudio reads from frame N-1
    // This prevents race conditions where CoreAudio tries to read while GPU is writing
    
    // Helper for GPU write operations - automatically adds +1 offset
    AudioFrameBuffer* getWriteBuffer();
    WaveformFrameData* getWriteWaveform();
    
    // Helper for CoreAudio read operations - automatically adds -1 offset  
    const AudioFrameBuffer* getReadBuffer() const;
    const WaveformFrameData* getReadWaveform() const;
    
    // Helper for GUI/Metrics read operations - automatically adds -1 offset
    const AudioFrameBuffer* getReadBufferForDisplay() const;
    const WaveformFrameData* getReadWaveformForDisplay() const;
    
    // Get the frame index for write operations (current frame)
    uint64_t getWriteFrameIndex() const;
    
    // Get the frame index for read operations (current frame - 1)
    uint64_t getReadFrameIndex() const;

    // Stage completion tracking
    void markStageComplete(SyncRole role, uint64_t frameIndex);
    bool isFrameReadyFor(SyncRole role, uint64_t frameIndex) const;
    bool isFrameFullyComplete(uint64_t frameIndex) const;

    // Audio buffer management (legacy - prefer helpers above)
    AudioFrameBuffer* getValidatedOutputBufferForWrite(uint64_t frameIndex);
    const AudioFrameBuffer* getValidatedOutputBufferForRead(uint64_t frameIndex) const;

    // Waveform buffer management
    WaveformFrameData* getWaveformForWrite(uint64_t frameIndex);
    const WaveformFrameData* getWaveformForRead(uint64_t frameIndex) const;

    // Frame advancement
    void advanceIfFrameComplete();

    // Initialization
    bool initializeBuffers(size_t maxSamples);
    void cleanup();

private:
    std::atomic<uint64_t> currentFrameIndex {0};
    std::array<FrameStatus, MAX_BUFFERED_FRAMES> frameStates;
    std::array<AudioFrameBuffer, MAX_BUFFERED_FRAMES> audioBuffers;
    std::array<WaveformFrameData, MAX_BUFFERED_FRAMES> waveformRegistry;
    
    // Buffer storage
    std::array<std::unique_ptr<float[]>, MAX_BUFFERED_FRAMES> bufferStorage;
    size_t maxSamplesPerBuffer = 0;
    
    mutable std::mutex bufferMutex;
};

// Template for dynamic resource allocation
template<typename T>
class SignalWindowAllocator {
public:
    using ResourceGenerator = std::function<T(uint64_t)>;

    SignalWindowAllocator(ResourceGenerator gen, size_t maxInflight = 16);

    T getOrCreate(uint64_t frameIndex);
    void markComplete(uint64_t frameIndex);
    void trimCompleted(uint64_t currentFrameIndex);

private:
    std::unordered_map<uint64_t, T> liveResources;
    std::unordered_map<uint64_t, bool> completionFlags;
    ResourceGenerator generator;
    size_t maxFrames;
    std::mutex mutex;
};

// Deferred signal queue for GPU completion callbacks
class DeferredSignalQueue {
public:
    using Callback = std::function<void(uint64_t)>;

    void enqueue(uint64_t frameIndex, Callback cb);
    void markCompleted(uint64_t frameIndex);
    void flushUpTo(uint64_t frameIndex);

private:
    std::map<uint64_t, Callback> pending;
    std::mutex mutex;
};

// Template implementation
template<typename T>
SignalWindowAllocator<T>::SignalWindowAllocator(ResourceGenerator gen, size_t maxInflight)
: generator(std::move(gen)), maxFrames(maxInflight) {}

template<typename T>
T SignalWindowAllocator<T>::getOrCreate(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    if (liveResources.count(frameIndex)) {
        return liveResources[frameIndex];
    }
    T res = generator(frameIndex);
    liveResources[frameIndex] = res;
    completionFlags[frameIndex] = false;
    return res;
}

template<typename T>
void SignalWindowAllocator<T>::markComplete(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    completionFlags[frameIndex] = true;
}

template<typename T>
void SignalWindowAllocator<T>::trimCompleted(uint64_t currentFrameIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = liveResources.begin(); it != liveResources.end();) {
        uint64_t idx = it->first;
        if (idx + maxFrames < currentFrameIndex && completionFlags[idx]) {
            completionFlags.erase(idx);
            it = liveResources.erase(it);
        } else {
            ++it;
        }
    }
} 