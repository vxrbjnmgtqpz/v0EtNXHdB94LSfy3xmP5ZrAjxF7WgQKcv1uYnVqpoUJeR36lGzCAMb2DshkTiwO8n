#pragma once
#include <atomic>
#include <string>
#include <iostream>
#include <iomanip>

struct FrameState {
    std::atomic<bool> gpuReady = false;
    std::atomic<bool> guiReady = false;
    std::atomic<bool> cpuReady = false;
    std::atomic<bool> valid = false;
    std::atomic<bool> dispatched = false;     // NEW: Track if GPU was dispatched
    std::atomic<bool> committed = false;      // NEW: Track if command buffer was committed
    std::atomic<bool> completed = false;      // NEW: Track if completion handler fired
    
    // Reset all states
    void reset() {
        gpuReady.store(false);
        guiReady.store(false);
        cpuReady.store(false);
        valid.store(false);
        dispatched.store(false);
        committed.store(false);
        completed.store(false);
    }
};

class FrameStateTracker {
public:
    static constexpr int kMaxFramesInFlight = 3;
    FrameState frames[kMaxFramesInFlight];
    
    // Enhanced stage marking with detailed GPU pipeline tracking
    void markStageComplete(const std::string& stage, int frameIndex);
    void markGPUDispatched(int frameIndex);
    void markGPUCommitted(int frameIndex);
    void markGPUCompleted(int frameIndex);
    
    void resetFrame(int frameIndex);
    void logFrameState(int frameIndex, const std::string& context = "");
    bool isFrameReady(int frameIndex);
    
    // Debug helpers
    void logDetailed(int frameIndex, const std::string& context = "");
    void logPipelineStatus();
    
private:
    std::string getStatusIcon(bool status) const {
        return status ? "✅" : "❌";
    }
};
