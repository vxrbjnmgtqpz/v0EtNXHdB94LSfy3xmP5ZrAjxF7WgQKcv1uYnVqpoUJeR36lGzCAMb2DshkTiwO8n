#include "gpu_native/gpu_shared_timeline.h"
#include <memory>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

namespace jam {
namespace gpu_native {

// Static implementation data for GPUSharedTimelineManager
class GPUSharedTimelineManagerImpl {
public:
    bool is_initialized = false;
    void* shared_memory = nullptr;
    size_t memory_size = 0;
    GPUSharedTimeline* timeline = nullptr;
    std::unordered_map<EventType, std::function<bool(const GPUTimelineEvent&)>> event_handlers;
    
    bool initialize_shared_memory() {
        // Allocate shared memory that can be accessed by GPU
        shared_memory = std::aligned_alloc(64, memory_size);
        return shared_memory != nullptr;
    }
};

// Static instance
static std::unique_ptr<GPUSharedTimelineManagerImpl> g_timeline_impl = nullptr;

bool GPUSharedTimelineManager::initialize(size_t timeline_buffer_size) {
    if (g_timeline_impl && g_timeline_impl->is_initialized) {
        return true;
    }
    
    g_timeline_impl = std::make_unique<GPUSharedTimelineManagerImpl>();
    
    g_timeline_impl->memory_size = sizeof(GPUSharedTimeline) + timeline_buffer_size;
    
    if (!g_timeline_impl->initialize_shared_memory()) {
        g_timeline_impl.reset();
        return false;
    }
    
    g_timeline_impl->timeline = reinterpret_cast<GPUSharedTimeline*>(g_timeline_impl->shared_memory);
    memset(g_timeline_impl->timeline, 0, sizeof(GPUSharedTimeline));
    
    g_timeline_impl->timeline->timeline_valid = true;
    g_timeline_impl->timeline->update_counter = 0;
    
    g_timeline_impl->is_initialized = true;
    return true;
}

void GPUSharedTimelineManager::shutdown() {
    if (g_timeline_impl) {
        if (g_timeline_impl->shared_memory) {
            std::free(g_timeline_impl->shared_memory);
        }
        g_timeline_impl.reset();
    }
}

bool GPUSharedTimelineManager::isInitialized() {
    return g_timeline_impl && g_timeline_impl->is_initialized;
}

bool GPUSharedTimelineManager::scheduleEvent(const GPUTimelineEvent& event) {
    if (!g_timeline_impl || !g_timeline_impl->is_initialized) {
        return false;
    }
    
    // For now, immediately process the event
    // In real implementation, this would be added to GPU event queue
    auto it = g_timeline_impl->event_handlers.find(event.type);
    if (it != g_timeline_impl->event_handlers.end()) {
        return it->second(event);
    }
    
    return true;
}

bool GPUSharedTimelineManager::cancelEvent(uint64_t event_id) {
    if (!g_timeline_impl || !g_timeline_impl->is_initialized) {
        return false;
    }
    
    // For now, return true (no queue to remove from)
    (void)event_id; // Suppress unused parameter warning
    return true;
}

bool GPUSharedTimelineManager::registerEventHandler(EventType type, std::function<bool(const GPUTimelineEvent&)> handler) {
    if (!g_timeline_impl) {
        return false;
    }
    
    g_timeline_impl->event_handlers[type] = handler;
    return true;
}

GPUSharedTimeline* GPUSharedTimelineManager::get_timeline() {
    if (g_timeline_impl && g_timeline_impl->timeline) {
        return g_timeline_impl->timeline;
    }
    return nullptr;
}

const GPUSharedTimeline* GPUSharedTimelineManager::get_timeline_const() {
    return get_timeline();
}

void GPUSharedTimelineManager::gpu_update_timeline() {
    if (g_timeline_impl && g_timeline_impl->timeline) {
        g_timeline_impl->timeline->update_counter++;
    }
}

bool GPUSharedTimelineManager::validate_timeline() {
    if (!g_timeline_impl || !g_timeline_impl->timeline) {
        return false;
    }
    return g_timeline_impl->timeline->timeline_valid;
}

void GPUSharedTimelineManager::reset_timeline() {
    if (g_timeline_impl && g_timeline_impl->timeline) {
        memset(g_timeline_impl->timeline, 0, sizeof(GPUSharedTimeline));
        g_timeline_impl->timeline->timeline_valid = true;
        g_timeline_impl->timeline->update_counter = 0;
    }
}

uint64_t GPUSharedTimelineManager::get_update_counter() {
    if (g_timeline_impl && g_timeline_impl->timeline) {
        return g_timeline_impl->timeline->update_counter;
    }
    return 0;
}



GPUSharedTimelineManager::TimelineStats GPUSharedTimelineManager::get_performance_stats() {
    TimelineStats stats = {};
    if (g_timeline_impl && g_timeline_impl->timeline) {
        stats.total_updates = g_timeline_impl->timeline->update_counter;
        stats.timeline_stable = g_timeline_impl->timeline->timeline_valid;
    }
    return stats;
}

void GPUSharedTimelineManager::reset_performance_stats() {
    // Reset performance counters
}

} // namespace gpu_native
} // namespace jam
