#include "FrameStateTracker.h"
#include <iostream>
#include <sstream>

void FrameStateTracker::markStageComplete(const std::string& stage, int frameIndex) {
    auto& state = frames[frameIndex % kMaxFramesInFlight];
    
    if (stage == "GPU") state.gpuReady = true;
    if (stage == "GUI") state.guiReady = true;
    if (stage == "CPU") state.cpuReady = true;
    
    if (state.gpuReady && state.guiReady && state.cpuReady) {
        state.valid = true;
    }
    
    logFrameState(frameIndex, "[MARK " + stage + "]");
}

void FrameStateTracker::markGPUDispatched(int frameIndex) {
    auto& state = frames[frameIndex % kMaxFramesInFlight];
    state.dispatched = true;
    logDetailed(frameIndex, "[GPU DISPATCH]");
}

void FrameStateTracker::markGPUCommitted(int frameIndex) {
    auto& state = frames[frameIndex % kMaxFramesInFlight];
    state.committed = true;
    logDetailed(frameIndex, "[GPU COMMIT]");
}

void FrameStateTracker::markGPUCompleted(int frameIndex) {
    auto& state = frames[frameIndex % kMaxFramesInFlight];
    state.completed = true;
    state.gpuReady = true;  // Mark GPU stage as ready when completed
    logDetailed(frameIndex, "[GPU COMPLETE]");
}

void FrameStateTracker::resetFrame(int frameIndex) {
    auto& state = frames[frameIndex % kMaxFramesInFlight];
    state.reset();
    logFrameState(frameIndex, "[RESET]");
}

bool FrameStateTracker::isFrameReady(int frameIndex) {
    return frames[frameIndex % kMaxFramesInFlight].valid.load();
}

void FrameStateTracker::logFrameState(int frameIndex, const std::string& context) {
    const auto& s = frames[frameIndex % kMaxFramesInFlight];
    
    std::ostringstream oss;
    oss << context << " Frame " << frameIndex 
        << " | GPU: " << getStatusIcon(s.gpuReady.load())
        << " GUI: " << getStatusIcon(s.guiReady.load())
        << " CPU: " << getStatusIcon(s.cpuReady.load())
        << " VALID: " << getStatusIcon(s.valid.load());
    
    std::cout << oss.str() << std::endl;
}

void FrameStateTracker::logDetailed(int frameIndex, const std::string& context) {
    const auto& s = frames[frameIndex % kMaxFramesInFlight];
    
    std::ostringstream oss;
    oss << context << " Frame " << frameIndex
        << " | DISPATCH: " << getStatusIcon(s.dispatched.load())
        << " COMMIT: " << getStatusIcon(s.committed.load())
        << " COMPLETE: " << getStatusIcon(s.completed.load())
        << " GPU_RDY: " << getStatusIcon(s.gpuReady.load());
    
    std::cout << oss.str() << std::endl;
}

void FrameStateTracker::logPipelineStatus() {
    std::cout << "\n=== GPU PIPELINE STATUS ===" << std::endl;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        const auto& s = frames[i];
        if (s.dispatched.load() || s.committed.load() || s.completed.load()) {
            std::cout << "Slot " << i 
                      << " | D:" << getStatusIcon(s.dispatched.load())
                      << " C:" << getStatusIcon(s.committed.load())
                      << " X:" << getStatusIcon(s.completed.load())
                      << " GPU:" << getStatusIcon(s.gpuReady.load())
                      << std::endl;
        }
    }
    std::cout << "==========================\n" << std::endl;
}
