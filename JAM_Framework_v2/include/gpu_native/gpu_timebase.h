#pragma once

#include <cstdint>
#include <memory>
#include <functional>

// Forward declarations to avoid platform-specific includes in header
#ifdef __APPLE__
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;  
@class MTLComputePipelineState;
@class MTLBuffer;
#else
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLComputePipelineState;
typedef void* MTLBuffer;
#endif
#endif

#ifdef __linux__
struct VkDevice_T;
struct VkCommandPool_T;
struct VkBuffer_T;
struct VkDeviceMemory_T;
typedef VkDevice_T* VkDevice;
typedef VkCommandPool_T* VkCommandPool;
typedef VkBuffer_T* VkBuffer;
typedef VkDeviceMemory_T* VkDeviceMemory;
#endif

namespace jam {
namespace gpu_native {

/**
 * GPU Timeline Type - High-precision GPU-generated timestamps
 * Measured in nanoseconds from GPU clock initialization
 */
using gpu_timeline_t = uint64_t;

/**
 * Transport State for GPU-native transport control
 */
enum class GPUTransportState : uint8_t {
    STOPPED = 0,
    PLAYING = 1,
    PAUSED = 2,
    RECORDING = 3
};

/**
 * GPU Shared Timeline - Memory-mapped structure accessible by GPU and CPU
 * This structure lives in GPU-accessible memory and provides the master timeline
 */
struct alignas(64) GPUSharedTimeline {
    // Master timing (updated by GPU compute shader)
    gpu_timeline_t master_clock_ns;         // Current GPU time in nanoseconds
    gpu_timeline_t initialization_time_ns;  // GPU clock start time
    
    // Transport state (controlled by GPU)
    GPUTransportState transport_state;      // Current transport state
    gpu_timeline_t transport_start_time_ns; // When transport started playing
    gpu_timeline_t transport_position_ns;   // Current position in timeline
    uint32_t bpm;                          // Current BPM (beats per minute)
    
    // Network synchronization
    gpu_timeline_t network_sync_epoch_ns;  // Network synchronization reference
    gpu_timeline_t last_network_heartbeat_ns; // Last network heartbeat
    
    // Audio/Video frame timing
    gpu_timeline_t current_audio_frame_ns;  // Current audio frame timestamp
    gpu_timeline_t current_video_frame_ns;  // Current video frame timestamp
    uint32_t audio_sample_rate;            // Current audio sample rate
    uint32_t video_frame_rate;             // Current video frame rate
    
    // Status and validation
    bool timeline_valid;                   // GPU timeline is operational
    bool transport_sync_active;           // Transport sync is active
    bool network_sync_active;             // Network sync is active
    uint64_t update_counter;              // Incremented on each GPU update
    
    // Reserved for future expansion
    uint8_t reserved[128];
};

/**
 * GPUTimebase - Master GPU timing controller
 * 
 * Revolutionary approach: The GPU becomes the conductor, providing all timing
 * for the entire multimedia system. CPU only interfaces with legacy DAWs.
 */
class GPUTimebase {
public:
    /**
     * Initialize the GPU-native timebase
     * Creates GPU compute shaders and shared memory timeline
     */
    static bool initialize();
    
    /**
     * Shutdown the GPU timebase and cleanup resources
     */
    static void shutdown();
    
    /**
     * Check if GPU timebase is initialized and operational
     */
    static bool is_initialized();
    
    // Core GPU timing functions
    
    /**
     * Get current GPU time in nanoseconds
     * This is the master clock for all timing decisions
     */
    static gpu_timeline_t get_current_time_ns();
    
    /**
     * Get current GPU time in microseconds (for compatibility)
     */
    static uint64_t get_current_time_us();
    
    /**
     * Sync GPU clock to system hardware clock
     * Called periodically to maintain accuracy
     */
    static void sync_to_hardware();
    
    // Transport timing functions
    
    /**
     * Set transport state (play/stop/pause/record)
     * All transport control originates from GPU timeline
     */
    static void set_transport_state(GPUTransportState state);
    
    /**
     * Get current transport state
     */
    static GPUTransportState get_transport_state();
    
    /**
     * Get current transport position in nanoseconds
     */
    static gpu_timeline_t get_transport_position_ns();
    
    /**
     * Set transport position (seek)
     */
    static void set_transport_position_ns(gpu_timeline_t position_ns);
    
    /**
     * Set/get BPM (beats per minute)
     */
    static void set_bpm(uint32_t bpm);
    static uint32_t get_bpm();
    
    // Network timing functions
    
    /**
     * Get network timestamp for packet transmission
     * All network packets timestamped by GPU
     */
    static gpu_timeline_t get_network_timestamp_ns();
    
    /**
     * Sync network timeline with peer
     */
    static void sync_network_timeline(gpu_timeline_t peer_timestamp_ns);
    
    /**
     * Generate network heartbeat timestamp
     */
    static gpu_timeline_t generate_heartbeat_timestamp();
    
    // Audio/Video timing functions
    
    /**
     * Mark beginning of audio frame processing
     */
    static gpu_timeline_t begin_audio_frame();
    
    /**
     * Mark end of audio frame processing
     */
    static gpu_timeline_t end_audio_frame();
    
    /**
     * Get current audio frame timestamp
     */
    static gpu_timeline_t get_audio_frame_time_ns();
    
    /**
     * Set audio sample rate
     */
    static void set_audio_sample_rate(uint32_t sample_rate);
    
    /**
     * Mark beginning of video frame processing
     */
    static gpu_timeline_t begin_video_frame();
    
    /**
     * Mark end of video frame processing  
     */
    static gpu_timeline_t end_video_frame();
    
    /**
     * Get current video frame timestamp
     */
    static gpu_timeline_t get_video_frame_time_ns();
    
    /**
     * Set video frame rate
     */
    static void set_video_frame_rate(uint32_t frame_rate);
    
    // MIDI timing functions
    
    /**
     * Schedule MIDI event at specific GPU time
     */
    static void schedule_midi_event(uint32_t midi_data, gpu_timeline_t execute_time_ns);
    
    /**
     * Get precise MIDI timestamp for event generation
     */
    static gpu_timeline_t get_midi_timestamp_ns();
    
    // CPU compatibility functions
    
    /**
     * Convert GPU time to CPU std::chrono for DAW interfaces
     * Use sparingly - only for legacy DAW compatibility
     */
    static std::chrono::microseconds gpu_time_to_cpu_time(gpu_timeline_t gpu_time_ns);
    
    /**
     * Convert CPU time to GPU time (for initialization/bridging)
     */
    static gpu_timeline_t cpu_time_to_gpu_time(std::chrono::microseconds cpu_time);
    
    // Advanced functions
    
    /**
     * Get direct access to GPU shared timeline (for performance-critical code)
     * Handle with care - this is the master timeline
     */
    static const GPUSharedTimeline* get_shared_timeline();
    
    /**
     * Register callback for GPU timeline updates
     * Callback runs on GPU timeline, not CPU threads
     */
    using TimelineCallback = std::function<void(const GPUSharedTimeline&)>;
    static void register_timeline_callback(TimelineCallback callback);
    
    /**
     * Get GPU timing precision in nanoseconds
     * Returns the minimum timing resolution the GPU can provide
     */
    static uint64_t get_timing_precision_ns();
    
    /**
     * Benchmark GPU vs CPU timing stability
     * Returns jitter measurements for validation
     */
    struct TimingBenchmark {
        uint64_t gpu_avg_jitter_ns;
        uint64_t gpu_max_jitter_ns;
        uint64_t cpu_avg_jitter_ns;
        uint64_t cpu_max_jitter_ns;
        uint64_t sample_count;
    };
    static TimingBenchmark benchmark_timing_stability(uint32_t duration_seconds = 10);

private:
    // Platform-specific GPU implementation
    class GPUTimebaseImpl;
    static std::unique_ptr<GPUTimebaseImpl> impl_;
    
    // Prevent instantiation - this is a static singleton
    GPUTimebase() = delete;
    ~GPUTimebase() = delete;
    GPUTimebase(const GPUTimebase&) = delete;
    GPUTimebase& operator=(const GPUTimebase&) = delete;
};

/**
 * GPU Timeline Guard - RAII wrapper for GPU timing operations
 * Automatically marks begin/end of timing-critical sections
 */
class GPUTimelineGuard {
public:
    explicit GPUTimelineGuard(const char* operation_name);
    ~GPUTimelineGuard();
    
    gpu_timeline_t get_start_time() const { return start_time_; }
    gpu_timeline_t get_elapsed_ns() const;
    
private:
    gpu_timeline_t start_time_;
    const char* operation_name_;
};

/**
 * Convenience macros for GPU timing
 */
#define GPU_TIMELINE_GUARD(name) ::jam::gpu_native::GPUTimelineGuard _gpu_guard(name)
#define GPU_CURRENT_TIME() ::jam::gpu_native::GPUTimebase::get_current_time_ns()
#define GPU_CURRENT_TIME_US() ::jam::gpu_native::GPUTimebase::get_current_time_us()

} // namespace gpu_native
} // namespace jam
