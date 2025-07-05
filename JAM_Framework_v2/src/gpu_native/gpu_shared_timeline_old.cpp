#include "gpu_native/gpu_shared_timeline.h"
#include <algorithm>
#include <cassert>
#include <cstring>

#ifdef __APPLE__
#include <sys/mman.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace jam {
namespace gpu_native {

namespace jam {
namespace gpu_native {

// Static implementation data for GPUSharedTimelineManager
class GPUSharedTimelineManagerImpl {
public:
    void* shared_memory = nullptr;
    size_t memory_size = 0;
    GPUSharedTimeline* timeline = nullptr;
    bool is_initialized = false;
    
    // Platform-specific GPU buffer handles (opaque pointers)
    void* gpu_timeline_buffer = nullptr;
    void* gpu_event_buffer = nullptr;

    bool initialize_shared_memory();
    void cleanup_shared_memory();
};

GPUSharedTimelineManager::GPUSharedTimelineManager() : pImpl(std::make_unique<Impl>()) {}

GPUSharedTimelineManager::~GPUSharedTimelineManager() {
    if (pImpl->is_initialized) {
        shutdown();
    }
}

bool GPUSharedTimelineManager::initialize(size_t timeline_capacity, size_t event_capacity) {
    if (pImpl->is_initialized) {
        return true;
    }

    // Calculate required memory size
    pImpl->memory_size = sizeof(GPUSharedTimeline) + 
                        (timeline_capacity * sizeof(TimelineEntry)) +
                        (event_capacity * sizeof(ScheduledEvent));

    if (!pImpl->initialize_shared_memory()) {
        return false;
    }

    // Initialize the timeline structure in shared memory
    pImpl->timeline = reinterpret_cast<GPUSharedTimeline*>(pImpl->shared_memory);
    memset(pImpl->timeline, 0, sizeof(GPUSharedTimeline));
    
    pImpl->timeline->timeline_capacity = timeline_capacity;
    pImpl->timeline->event_capacity = event_capacity;
    pImpl->timeline->timeline_head = 0;
    pImpl->timeline->timeline_tail = 0;
    pImpl->timeline->event_head = 0;
    pImpl->timeline->event_tail = 0;
    pImpl->timeline->current_frame = 0;
    pImpl->timeline->sync_token = 0;

    // Set up pointers to the variable-length arrays
    uint8_t* memory_ptr = reinterpret_cast<uint8_t*>(pImpl->shared_memory);
    size_t offset = sizeof(GPUSharedTimeline);
    
    // Timeline entries follow the main structure
    pImpl->timeline->timeline_entries_offset = offset;
    offset += timeline_capacity * sizeof(TimelineEntry);
    
    // Scheduled events follow the timeline entries
    pImpl->timeline->scheduled_events_offset = offset;

    // TODO: Create platform-specific GPU buffers when needed
    // This would map the shared memory to GPU-accessible buffers

    pImpl->is_initialized = true;
    return true;
}

void GPUSharedTimelineManager::shutdown() {
    if (!pImpl->is_initialized) {
        return;
    }

    // Clean up platform-specific GPU resources
    pImpl->gpu_timeline_buffer = nullptr;
    pImpl->gpu_event_buffer = nullptr;

    pImpl->cleanup_shared_memory();
    pImpl->is_initialized = false;
}

TimelineReader GPUSharedTimelineManager::create_reader() {
    if (!pImpl->is_initialized) {
        return TimelineReader(nullptr);
    }
    return TimelineReader(pImpl->timeline);
}

TimelineWriter GPUSharedTimelineManager::create_writer() {
    if (!pImpl->is_initialized) {
        return TimelineWriter(nullptr);
    }
    return TimelineWriter(pImpl->timeline);
}

EventScheduler GPUSharedTimelineManager::create_scheduler() {
    if (!pImpl->is_initialized) {
        return EventScheduler(nullptr);
    }
    return EventScheduler(pImpl->timeline);
}

GPUSharedTimeline* GPUSharedTimelineManager::get_timeline() const {
    return pImpl->timeline;
}

bool GPUSharedTimelineManager::Impl::initialize_shared_memory() {
#ifdef __APPLE__
    // Use mmap for shared memory on macOS
    shared_memory = mmap(nullptr, memory_size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared_memory == MAP_FAILED) {
        shared_memory = nullptr;
        return false;
    }
    return true;
#elif defined(_WIN32)
    // Use CreateFileMapping on Windows
    HANDLE mapping = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 
                                      0, static_cast<DWORD>(memory_size), nullptr);
    if (!mapping) {
        return false;
    }
    
    shared_memory = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, memory_size);
    CloseHandle(mapping); // We can close this as the mapping will persist
    
    return shared_memory != nullptr;
#else
    // Use POSIX shared memory on Linux
    shared_memory = mmap(nullptr, memory_size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared_memory == MAP_FAILED) {
        shared_memory = nullptr;
        return false;
    }
    return true;
#endif
}

void GPUSharedTimelineManager::Impl::cleanup_shared_memory() {
    if (shared_memory) {
#ifdef __APPLE__
        munmap(shared_memory, memory_size);
#elif defined(_WIN32)
        UnmapViewOfFile(shared_memory);
#else
        munmap(shared_memory, memory_size);
#endif
        shared_memory = nullptr;
    }
}

// TimelineReader implementation
TimelineReader::TimelineReader(GPUSharedTimeline* timeline) : timeline_(timeline) {}

bool TimelineReader::read_entry(uint32_t frame, TimelineEntry& entry) const {
    if (!timeline_) return false;
    
    // Lock-free read using atomic operations
    uint32_t head = __atomic_load_n(&timeline_->timeline_head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&timeline_->timeline_tail, __ATOMIC_ACQUIRE);
    
    if (head == tail) {
        return false; // Timeline is empty
    }
    
    // Get pointer to timeline entries
    TimelineEntry* entries = reinterpret_cast<TimelineEntry*>(
        reinterpret_cast<uint8_t*>(timeline_) + timeline_->timeline_entries_offset);
    
    // Search for the entry with the closest frame <= requested frame
    // Since entries are ordered by frame, we can do a simple scan
    uint32_t best_index = head;
    uint32_t best_frame = UINT32_MAX;
    
    for (uint32_t i = head; i != tail; i = (i + 1) % timeline_->timeline_capacity) {
        uint32_t entry_frame = __atomic_load_n(&entries[i].frame, __ATOMIC_ACQUIRE);
        if (entry_frame <= frame && entry_frame < best_frame) {
            best_frame = entry_frame;
            best_index = i;
        }
    }
    
    if (best_frame == UINT32_MAX) {
        return false; // No suitable entry found
    }
    
    // Copy the entry atomically
    entry.frame = __atomic_load_n(&entries[best_index].frame, __ATOMIC_ACQUIRE);
    entry.bpm = __atomic_load_n(&entries[best_index].bpm, __ATOMIC_ACQUIRE);
    entry.beat_position = __atomic_load_n(&entries[best_index].beat_position, __ATOMIC_ACQUIRE);
    entry.transport_state = __atomic_load_n(&entries[best_index].transport_state, __ATOMIC_ACQUIRE);
    entry.quantum_frames = __atomic_load_n(&entries[best_index].quantum_frames, __ATOMIC_ACQUIRE);
    
    return true;
}

uint32_t TimelineReader::get_current_frame() const {
    if (!timeline_) return 0;
    return __atomic_load_n(&timeline_->current_frame, __ATOMIC_ACQUIRE);
}

uint32_t TimelineReader::get_sync_token() const {
    if (!timeline_) return 0;
    return __atomic_load_n(&timeline_->sync_token, __ATOMIC_ACQUIRE);
}

// TimelineWriter implementation
TimelineWriter::TimelineWriter(GPUSharedTimeline* timeline) : timeline_(timeline) {}

bool TimelineWriter::write_entry(const TimelineEntry& entry) {
    if (!timeline_) return false;
    
    uint32_t head = __atomic_load_n(&timeline_->timeline_head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&timeline_->timeline_tail, __ATOMIC_ACQUIRE);
    uint32_t next_tail = (tail + 1) % timeline_->timeline_capacity;
    
    if (next_tail == head) {
        return false; // Timeline is full
    }
    
    // Get pointer to timeline entries
    TimelineEntry* entries = reinterpret_cast<TimelineEntry*>(
        reinterpret_cast<uint8_t*>(timeline_) + timeline_->timeline_entries_offset);
    
    // Write the entry atomically
    __atomic_store_n(&entries[tail].frame, entry.frame, __ATOMIC_RELEASE);
    __atomic_store_n(&entries[tail].bpm, entry.bpm, __ATOMIC_RELEASE);
    __atomic_store_n(&entries[tail].beat_position, entry.beat_position, __ATOMIC_RELEASE);
    __atomic_store_n(&entries[tail].transport_state, entry.transport_state, __ATOMIC_RELEASE);
    __atomic_store_n(&entries[tail].quantum_frames, entry.quantum_frames, __ATOMIC_RELEASE);
    
    // Update tail pointer
    __atomic_store_n(&timeline_->timeline_tail, next_tail, __ATOMIC_RELEASE);
    
    return true;
}

void TimelineWriter::update_current_frame(uint32_t frame) {
    if (!timeline_) return;
    __atomic_store_n(&timeline_->current_frame, frame, __ATOMIC_RELEASE);
}

void TimelineWriter::increment_sync_token() {
    if (!timeline_) return;
    __atomic_fetch_add(&timeline_->sync_token, 1, __ATOMIC_ACQ_REL);
}

// EventScheduler implementation
EventScheduler::EventScheduler(GPUSharedTimeline* timeline) : timeline_(timeline) {}

bool EventScheduler::schedule_event(const ScheduledEvent& event) {
    if (!timeline_) return false;
    
    uint32_t head = __atomic_load_n(&timeline_->event_head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&timeline_->event_tail, __ATOMIC_ACQUIRE);
    uint32_t next_tail = (tail + 1) % timeline_->event_capacity;
    
    if (next_tail == head) {
        return false; // Event queue is full
    }
    
    // Get pointer to scheduled events
    ScheduledEvent* events = reinterpret_cast<ScheduledEvent*>(
        reinterpret_cast<uint8_t*>(timeline_) + timeline_->scheduled_events_offset);
    
    // Write the event atomically
    __atomic_store_n(&events[tail].frame, event.frame, __ATOMIC_RELEASE);
    __atomic_store_n(&events[tail].event_type, event.event_type, __ATOMIC_RELEASE);
    __atomic_store_n(&events[tail].channel, event.channel, __ATOMIC_RELEASE);
    __atomic_store_n(&events[tail].data1, event.data1, __ATOMIC_RELEASE);
    __atomic_store_n(&events[tail].data2, event.data2, __ATOMIC_RELEASE);
    __atomic_store_n(&events[tail].priority, event.priority, __ATOMIC_RELEASE);
    
    // Update tail pointer
    __atomic_store_n(&timeline_->event_tail, next_tail, __ATOMIC_RELEASE);
    
    return true;
}

bool EventScheduler::get_next_event(uint32_t current_frame, ScheduledEvent& event) {
    if (!timeline_) return false;
    
    uint32_t head = __atomic_load_n(&timeline_->event_head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&timeline_->event_tail, __ATOMIC_ACQUIRE);
    
    if (head == tail) {
        return false; // Event queue is empty
    }
    
    // Get pointer to scheduled events
    ScheduledEvent* events = reinterpret_cast<ScheduledEvent*>(
        reinterpret_cast<uint8_t*>(timeline_) + timeline_->scheduled_events_offset);
    
    // Check if the next event is ready
    uint32_t event_frame = __atomic_load_n(&events[head].frame, __ATOMIC_ACQUIRE);
    if (event_frame > current_frame) {
        return false; // Event is not ready yet
    }
    
    // Copy the event
    event.frame = event_frame;
    event.event_type = __atomic_load_n(&events[head].event_type, __ATOMIC_ACQUIRE);
    event.channel = __atomic_load_n(&events[head].channel, __ATOMIC_ACQUIRE);
    event.data1 = __atomic_load_n(&events[head].data1, __ATOMIC_ACQUIRE);
    event.data2 = __atomic_load_n(&events[head].data2, __ATOMIC_ACQUIRE);
    event.priority = __atomic_load_n(&events[head].priority, __ATOMIC_ACQUIRE);
    
    // Advance head pointer
    uint32_t next_head = (head + 1) % timeline_->event_capacity;
    __atomic_store_n(&timeline_->event_head, next_head, __ATOMIC_RELEASE);
    
    return true;
}

uint32_t EventScheduler::get_event_count() const {
    if (!timeline_) return 0;
    
    uint32_t head = __atomic_load_n(&timeline_->event_head, __ATOMIC_ACQUIRE);
    uint32_t tail = __atomic_load_n(&timeline_->event_tail, __ATOMIC_ACQUIRE);
    
    if (tail >= head) {
        return tail - head;
    } else {
        return timeline_->event_capacity - head + tail;
    }
}

// TimelineSynchronizer implementation
TimelineSynchronizer::TimelineSynchronizer(GPUSharedTimeline* timeline) : timeline_(timeline) {}

bool TimelineSynchronizer::sync_to_timeline(uint32_t external_frame, uint32_t external_token) {
    if (!timeline_) return false;
    
    uint32_t current_token = __atomic_load_n(&timeline_->sync_token, __ATOMIC_ACQUIRE);
    
    if (external_token <= current_token) {
        return false; // External timeline is not ahead of us
    }
    
    // Update our timeline to match the external one
    __atomic_store_n(&timeline_->current_frame, external_frame, __ATOMIC_RELEASE);
    __atomic_store_n(&timeline_->sync_token, external_token, __ATOMIC_RELEASE);
    
    return true;
}

bool TimelineSynchronizer::wait_for_sync(uint32_t timeout_ms) {
    if (!timeline_) return false;
    
    uint32_t initial_token = __atomic_load_n(&timeline_->sync_token, __ATOMIC_ACQUIRE);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto timeout_duration = std::chrono::milliseconds(timeout_ms);
    
    while (std::chrono::high_resolution_clock::now() - start_time < timeout_duration) {
        uint32_t current_token = __atomic_load_n(&timeline_->sync_token, __ATOMIC_ACQUIRE);
        if (current_token != initial_token) {
            return true; // Timeline was updated
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    return false; // Timeout
}

} // namespace gpu_native
} // namespace jam
