#include "../include/gpu_transport/gpu_transport_manager.h"
#include "../include/gpu_native/gpu_timebase.h"
#include <iostream>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#endif

// Embedded Metal shader source for GPU transport control
static const std::string kGPUTransportShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

enum class GPUTransportState : uint32_t {
    Stopped = 0,
    Playing = 1,
    Paused = 2,
    Recording = 3
};

struct GPUTransportControlBuffer {
    GPUTransportState current_state;
    uint64_t play_start_frame;
    uint64_t pause_frame;
    uint64_t record_start_frame;
    uint64_t current_frame;
    float bpm;
    uint32_t samples_per_beat;
    uint32_t network_sync_offset;
    uint32_t frame_counter;
    float position_seconds;
    uint32_t is_network_synced;
    uint32_t padding[5];
};

struct GPUTimelineEvent {
    uint64_t timestamp_ns;
    uint32_t event_type;
    uint32_t event_data;
    GPUTransportState new_state;
    uint32_t source_peer_id;
};

inline void transport_control_play(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame) {
    if (buffer.current_state == GPUTransportState::Paused) {
        uint64_t pause_duration = trigger_frame - buffer.pause_frame;
        buffer.play_start_frame = buffer.play_start_frame + pause_duration;
    } else {
        buffer.play_start_frame = trigger_frame;
    }
    buffer.current_state = GPUTransportState::Playing;
    buffer.pause_frame = 0;
}

inline void transport_control_stop(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame) {
    buffer.current_state = GPUTransportState::Stopped;
    buffer.play_start_frame = 0;
    buffer.pause_frame = 0;
    buffer.record_start_frame = 0;
    buffer.position_seconds = 0.0f;
}

inline void transport_control_pause(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame) {
    if (buffer.current_state == GPUTransportState::Playing || 
        buffer.current_state == GPUTransportState::Recording) {
        buffer.current_state = GPUTransportState::Paused;
        buffer.pause_frame = trigger_frame;
    }
}

inline void transport_control_record(device GPUTransportControlBuffer& buffer, uint64_t trigger_frame) {
    buffer.current_state = GPUTransportState::Recording;
    buffer.record_start_frame = trigger_frame;
    buffer.play_start_frame = trigger_frame;
}

inline void update_transport_position(device GPUTransportControlBuffer& buffer) {
    if (buffer.current_state == GPUTransportState::Playing || 
        buffer.current_state == GPUTransportState::Recording) {
        uint64_t elapsed_ns = buffer.current_frame - buffer.play_start_frame;
        buffer.position_seconds = float(elapsed_ns) / 1e9f;
    } else if (buffer.current_state == GPUTransportState::Paused) {
        uint64_t elapsed_ns = buffer.pause_frame - buffer.play_start_frame;
        buffer.position_seconds = float(elapsed_ns) / 1e9f;
    } else {
        buffer.position_seconds = 0.0f;
    }
}

kernel void gpu_transport_control_update(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device uint64_t& master_timebase_ns [[ buffer(1) ]],
    device GPUTimelineEvent* timeline_events [[ buffer(2) ]],
    device uint32_t& event_count [[ buffer(3) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
) {
    if (thread_position_in_grid != 0) return;
    
    transport_buffer.current_frame = master_timebase_ns;
    
    for (uint32_t i = 0; i < event_count; ++i) {
        GPUTimelineEvent event = timeline_events[i];
        
        if (event.timestamp_ns <= master_timebase_ns) {
            switch (event.new_state) {
                case GPUTransportState::Playing:
                    transport_control_play(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Stopped:
                    transport_control_stop(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Paused:
                    transport_control_pause(transport_buffer, event.timestamp_ns);
                    break;
                case GPUTransportState::Recording:
                    transport_control_record(transport_buffer, event.timestamp_ns);
                    break;
            }
            
            timeline_events[i] = timeline_events[event_count - 1];
            event_count--;
            i--;
        }
    }
    
    update_transport_position(transport_buffer);
    transport_buffer.frame_counter++;
}

kernel void gpu_transport_network_sync(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device uint64_t& network_sync_frame [[ buffer(1) ]],
    device uint32_t& sync_command [[ buffer(2) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
) {
    if (thread_position_in_grid != 0) return;
    
    uint64_t adjusted_frame = network_sync_frame + transport_buffer.network_sync_offset;
    
    switch (sync_command) {
        case 1: transport_control_play(transport_buffer, adjusted_frame); break;
        case 2: transport_control_stop(transport_buffer, adjusted_frame); break;
        case 3: transport_control_pause(transport_buffer, adjusted_frame); break;
        case 4: transport_control_record(transport_buffer, adjusted_frame); break;
    }
    
    transport_buffer.is_network_synced = 1;
    update_transport_position(transport_buffer);
}

kernel void gpu_transport_tempo_update(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device float& new_bpm [[ buffer(1) ]],
    device uint64_t& sample_rate [[ buffer(2) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
) {
    if (thread_position_in_grid != 0) return;
    
    transport_buffer.bpm = new_bpm;
    
    float samples_per_second = float(sample_rate);
    float beats_per_second = new_bpm / 60.0f;
    transport_buffer.samples_per_beat = uint32_t(samples_per_second / beats_per_second);
}

// Bars/Beats calculation buffer structure  
struct GPUBarsBeatsBuffer {
    uint32_t bars;              // Current bar number (1-based)
    uint32_t beats;             // Current beat in bar (1-based)
    uint32_t subdivisions;      // Current subdivision (0-based)
    uint32_t beats_per_bar;     // Time signature numerator
    uint32_t beat_unit;         // Time signature denominator
    uint32_t subdivision_count; // Subdivisions per beat
    float total_beats;          // Total beats since start
    float fractional_beat;      // Fractional part of current beat
    uint32_t padding[8];        // Align to 64 bytes
};

kernel void gpu_transport_bars_beats_update(
    device GPUTransportControlBuffer& transport_buffer [[ buffer(0) ]],
    device GPUBarsBeatsBuffer& bars_beats_buffer [[ buffer(1) ]],
    uint thread_position_in_grid [[ thread_position_in_grid ]]
) {
    if (thread_position_in_grid != 0) return;
    
    // Only calculate when playing or recording
    if (transport_buffer.current_state != GPUTransportState::Playing && 
        transport_buffer.current_state != GPUTransportState::Recording) {
        return;
    }
    
    // Calculate elapsed time in seconds from transport position
    float position_seconds = transport_buffer.position_seconds;
    
    // Calculate total beats elapsed
    float beats_per_second = transport_buffer.bpm / 60.0f;
    bars_beats_buffer.total_beats = position_seconds * beats_per_second;
    
    // Extract fractional part
    bars_beats_buffer.fractional_beat = bars_beats_buffer.total_beats - floor(bars_beats_buffer.total_beats);
    
    // Calculate current beat (1-based)
    uint32_t beat_index = uint32_t(floor(bars_beats_buffer.total_beats)) % bars_beats_buffer.beats_per_bar;
    bars_beats_buffer.beats = beat_index + 1;
    
    // Calculate current bar (1-based) 
    bars_beats_buffer.bars = uint32_t(floor(bars_beats_buffer.total_beats / float(bars_beats_buffer.beats_per_bar))) + 1;
    
    // Calculate subdivisions within current beat
    float subdivision_progress = bars_beats_buffer.fractional_beat * float(bars_beats_buffer.subdivision_count);
    bars_beats_buffer.subdivisions = uint32_t(floor(subdivision_progress));
}
)";

namespace jam {
namespace gpu_transport {

// Static flag to track initialization (workaround for member variable corruption)
bool s_gpu_transport_ready = false;

GPUTransportManager& GPUTransportManager::getInstance() {
    static GPUTransportManager instance;
    std::cout << "🔧 getInstance() returning instance at address: " << &instance << std::endl;
    return instance;
}

GPUTransportManager::~GPUTransportManager() {
    std::cout << "🔥 GPU Transport Manager destructor called for instance " << this << " (ID: " << instance_id_ << ")" << std::endl;
    shutdown(); 
}

bool GPUTransportManager::initialize() {
    std::cout << "🚀 initialize() called on instance: " << this << std::endl;
    if (initialized_.load()) {
        std::cout << "✅ GPU Transport Manager already initialized on instance: " << this << std::endl;
        return true; // Already initialized
    }
    
    std::cout << "🚀 Initializing GPU Transport Manager..." << std::endl;
    
#ifdef __OBJC__
    bool result = initializeMetal();
    std::cout << "🔬 initializeMetal() returned: " << (result ? "true" : "false") << std::endl;
    std::cout << "🔬 Just before returning from initialize(), atomic flag: " << (initialized_.load() ? "true" : "false") << ", debug flag: " << (debug_initialized_ ? "true" : "false") << std::endl;
    if (result) {
        std::cout << "✅ GPU Transport Manager Metal initialization successful" << std::endl;
    } else {
        std::cerr << "❌ GPU Transport Manager Metal initialization failed" << std::endl;
    }
    std::cout << "🔬 About to return " << (result ? "true" : "false") << " from initialize()" << std::endl;
    return result;
#else
    return initializeVulkan();
#endif
}

#ifdef __OBJC__
bool GPUTransportManager::initializeMetal() {
    @autoreleasepool {
        // Get Metal device
        metal_device_ = MTLCreateSystemDefaultDevice();
        if (!metal_device_) {
            std::cerr << "❌ Failed to create Metal device for GPU transport" << std::endl;
            return false;
        }
        
        // Create command queue
        command_queue_ = [metal_device_ newCommandQueue];
        if (!command_queue_) {
            std::cerr << "❌ Failed to create Metal command queue" << std::endl;
            return false;
        }
        
        // Load compute shaders from embedded source
        NSError* error = nil;
        
        NSString* shaderSource = [NSString stringWithUTF8String:kGPUTransportShaderSource.c_str()];
        
        if (!shaderSource || [shaderSource length] == 0) {
            std::cerr << "❌ Failed to load embedded transport shader source" << std::endl;
            return false;
        }
        
        id<MTLLibrary> library = [metal_device_ newLibraryWithSource:shaderSource
                                                             options:nil
                                                               error:&error];
        if (!library) {
            std::cerr << "❌ Failed to compile transport shader library: " 
                      << error.localizedDescription.UTF8String << std::endl;
            return false;
        }
        
        // Create compute pipelines
        id<MTLFunction> transportFunction = [library newFunctionWithName:@"gpu_transport_control_update"];
        if (!transportFunction) {
            std::cerr << "❌ Failed to find transport control function" << std::endl;
            return false;
        }
        
        transport_pipeline_ = [metal_device_ newComputePipelineStateWithFunction:transportFunction
                                                                           error:&error];
        if (!transport_pipeline_) {
            std::cerr << "❌ Failed to create transport pipeline: " 
                      << error.localizedDescription.UTF8String << std::endl;
            return false;
        }
        
        // Create network sync pipeline
        id<MTLFunction> networkSyncFunction = [library newFunctionWithName:@"gpu_transport_network_sync"];
        if (networkSyncFunction) {
            network_sync_pipeline_ = [metal_device_ newComputePipelineStateWithFunction:networkSyncFunction
                                                                                   error:&error];
        }
        
        // Create tempo update pipeline
        id<MTLFunction> tempoFunction = [library newFunctionWithName:@"gpu_transport_tempo_update"];
        if (tempoFunction) {
            tempo_update_pipeline_ = [metal_device_ newComputePipelineStateWithFunction:tempoFunction
                                                                                  error:&error];
        }
        
        // Create bars/beats update pipeline
        id<MTLFunction> barsBeatsFunction = [library newFunctionWithName:@"gpu_transport_bars_beats_update"];
        if (barsBeatsFunction) {
            bars_beats_pipeline_ = [metal_device_ newComputePipelineStateWithFunction:barsBeatsFunction
                                                                               error:&error];
        }
        
        // Create GPU buffers
        transport_buffer_ = [metal_device_ newBufferWithLength:sizeof(GPUTransportControlBuffer)
                                                       options:MTLResourceStorageModeShared];
        
        timebase_buffer_ = [metal_device_ newBufferWithLength:sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];
        
        timeline_events_buffer_ = [metal_device_ newBufferWithLength:sizeof(uint32_t) + sizeof(GPUTimelineEvent) * MAX_TIMELINE_EVENTS
                                                             options:MTLResourceStorageModeShared];
        
        network_sync_buffer_ = [metal_device_ newBufferWithLength:sizeof(uint64_t) + sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
        
        tempo_buffer_ = [metal_device_ newBufferWithLength:sizeof(float) + sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
        
        bars_beats_buffer_ = [metal_device_ newBufferWithLength:sizeof(GPUBarsBeatsBuffer)
                                                        options:MTLResourceStorageModeShared];
        
        if (!transport_buffer_ || !timebase_buffer_ || !timeline_events_buffer_ || !bars_beats_buffer_) {
            std::cerr << "❌ Failed to create GPU transport buffers" << std::endl;
            return false;
        }
        
        // Initialize transport state
        GPUTransportControlBuffer* buffer = (GPUTransportControlBuffer*)transport_buffer_.contents;
        memset(buffer, 0, sizeof(GPUTransportControlBuffer));
        buffer->current_state = GPUTransportState::Stopped;
        buffer->bpm = 120.0f;
        buffer->samples_per_beat = 44100 / 2; // Assuming 44.1kHz, 120 BPM
        
        // Initialize event count
        uint32_t* event_count = (uint32_t*)timeline_events_buffer_.contents;
        *event_count = 0;
        
        // Initialize bars/beats buffer with default 4/4 time signature
        GPUBarsBeatsBuffer* bars_beats = (GPUBarsBeatsBuffer*)bars_beats_buffer_.contents;
        memset(bars_beats, 0, sizeof(GPUBarsBeatsBuffer));
        bars_beats->bars = 1;              // Start at bar 1
        bars_beats->beats = 1;             // Start at beat 1
        bars_beats->subdivisions = 0;      // Start at subdivision 0
        bars_beats->beats_per_bar = 4;     // Default 4/4 time
        bars_beats->beat_unit = 4;         // Quarter note gets the beat
        bars_beats->subdivision_count = 4; // 16th note subdivisions
        
        initialized_.store(true, std::memory_order_release);
        debug_initialized_ = true;  // Set debug flag too
        metal_resources_ready_ = true;  // Set simple Metal flag
        s_gpu_transport_ready = true;  // Set static flag
        std::cout << "✅ GPU Transport Manager initialized successfully on instance " << this << std::endl;
        std::cout << "🔧 s_gpu_transport_ready set to: " << s_gpu_transport_ready << std::endl;
        std::cout << "   - Transport pipeline: " << (transport_pipeline_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Network sync pipeline: " << (network_sync_pipeline_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Tempo pipeline: " << (tempo_update_pipeline_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Bars/beats pipeline: " << (bars_beats_pipeline_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Transport buffer: " << (transport_buffer_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Timebase buffer: " << (timebase_buffer_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Timeline events buffer: " << (timeline_events_buffer_ ? "✅" : "❌") << std::endl;
        std::cout << "   - Bars/beats buffer: " << (bars_beats_buffer_ ? "✅" : "❌") << std::endl;
        
        // Verify flag is set correctly
        bool flagState = initialized_.load(std::memory_order_acquire);
        std::cout << "   - Initialized flag state: " << (flagState ? "true" : "false") << std::endl;
        
        // Double-check with direct access
        std::cout << "   - Direct flag check: " << (initialized_.load(std::memory_order_acquire) ? "true" : "false") << std::endl;
        
        // Force memory barrier
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::cout << "   - After memory barrier: " << (initialized_.load(std::memory_order_acquire) ? "true" : "false") << std::endl;
        
        return true;
    }
}
#endif

void GPUTransportManager::play(uint64_t start_frame) {
    if (!isInitialized()) {
        std::cerr << "❌ GPU Transport Manager not initialized for PLAY command" << std::endl;
        return;
    }
    
    // Schedule play event
    uint64_t trigger_frame = start_frame ? start_frame : jam::gpu_native::GPUTimebase::get_current_time_ns();
    scheduleTransportEvent(trigger_frame, GPUTransportState::Playing);
    
    std::cout << "🎵 GPU Transport: PLAY scheduled at frame " << trigger_frame << std::endl;
}

void GPUTransportManager::stop() {
    if (!isInitialized()) return;
    
    uint64_t trigger_frame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    scheduleTransportEvent(trigger_frame, GPUTransportState::Stopped);
    
    std::cout << "⏹️ GPU Transport: STOP scheduled at frame " << trigger_frame << std::endl;
}

void GPUTransportManager::pause() {
    if (!isInitialized()) return;
    
    uint64_t trigger_frame = jam::gpu_native::GPUTimebase::get_current_time_ns();
    scheduleTransportEvent(trigger_frame, GPUTransportState::Paused);
    
    std::cout << "⏸️ GPU Transport: PAUSE scheduled at frame " << trigger_frame << std::endl;
}

void GPUTransportManager::record(uint64_t start_frame) {
    if (!isInitialized()) return;
    
    uint64_t trigger_frame = start_frame ? start_frame : jam::gpu_native::GPUTimebase::get_current_time_ns();
    scheduleTransportEvent(trigger_frame, GPUTransportState::Recording);
    
    std::cout << "🔴 GPU Transport: RECORD scheduled at frame " << trigger_frame << std::endl;
}

void GPUTransportManager::scheduleTransportEvent(uint64_t timestamp_ns, GPUTransportState new_state) {
    if (!initialized_.load()) return;
    
    GPUTimelineEvent event;
    event.timestamp_ns = timestamp_ns;
    event.event_type = 1; // Transport change
    event.event_data = 0;
    event.new_state = new_state;
    event.source_peer_id = 0; // Local
    
    pending_events_.push_back(event);
    
    // Trigger GPU update
    executeTransportUpdate();
}

void GPUTransportManager::update() {
    if (!isInitialized()) return;
    
    executeTransportUpdate();
    readbackTransportState();
    
    // Check for state changes
    if (cached_state_.current_state != last_state_) {
        if (state_callback_) {
            state_callback_(last_state_, cached_state_.current_state);
        }
        last_state_ = cached_state_.current_state;
    }
}

#ifdef __OBJC__
void GPUTransportManager::executeTransportUpdate() {
    if (!transport_pipeline_) return;
    
    @autoreleasepool {
        // Update timebase
        uint64_t current_time = jam::gpu_native::GPUTimebase::get_current_time_ns();
        uint64_t* timebase_ptr = (uint64_t*)timebase_buffer_.contents;
        *timebase_ptr = current_time;
        
        // Update timeline events
        if (!pending_events_.empty()) {
            uint32_t* event_count = (uint32_t*)timeline_events_buffer_.contents;
            GPUTimelineEvent* events = (GPUTimelineEvent*)((uint8_t*)timeline_events_buffer_.contents + sizeof(uint32_t));
            
            size_t events_to_add = std::min(pending_events_.size(), MAX_TIMELINE_EVENTS - *event_count);
            for (size_t i = 0; i < events_to_add; ++i) {
                events[*event_count + i] = pending_events_[i];
            }
            *event_count += events_to_add;
            
            pending_events_.clear();
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:transport_pipeline_];
        [encoder setBuffer:transport_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:timebase_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:timeline_events_buffer_ offset:sizeof(uint32_t) atIndex:2]; // Events array starts after count
        [encoder setBuffer:timeline_events_buffer_ offset:0 atIndex:3]; // Event count at start
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];
        
        // Update bars/beats if pipeline is available
        if (bars_beats_pipeline_) {
            [encoder setComputePipelineState:bars_beats_pipeline_];
            [encoder setBuffer:transport_buffer_ offset:0 atIndex:0];
            [encoder setBuffer:bars_beats_buffer_ offset:0 atIndex:1];
            
            [encoder dispatchThreadgroups:threadgroupsPerGrid
                    threadsPerThreadgroup:threadsPerThreadgroup];
        }
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];  // Wait for GPU work to complete before continuing
    }
}
#endif

void GPUTransportManager::readbackTransportState() {
#ifdef __OBJC__
    if (transport_buffer_) {
        GPUTransportControlBuffer* buffer = (GPUTransportControlBuffer*)transport_buffer_.contents;
        cached_state_ = *buffer;
        state_dirty_.store(false);
    }
    
    // Also read back bars/beats state
    if (bars_beats_buffer_) {
        GPUBarsBeatsBuffer* bars_beats = (GPUBarsBeatsBuffer*)bars_beats_buffer_.contents;
        cached_bars_beats_ = *bars_beats;
    }
#endif
}

// State query methods
GPUTransportState GPUTransportManager::getCurrentState() const {
    return cached_state_.current_state;
}

uint64_t GPUTransportManager::getCurrentFrame() const {
    return cached_state_.current_frame;
}

float GPUTransportManager::getPositionSeconds() const {
    return cached_state_.position_seconds;
}

bool GPUTransportManager::isPlaying() const {
    return cached_state_.current_state == GPUTransportState::Playing ||
           cached_state_.current_state == GPUTransportState::Recording;
}

bool GPUTransportManager::isRecording() const {
    return cached_state_.current_state == GPUTransportState::Recording;
}

bool GPUTransportManager::isPaused() const {
    return cached_state_.current_state == GPUTransportState::Paused;
}

void GPUTransportManager::setBPM(float bpm) {
    if (!initialized_.load()) return;
    
#ifdef __OBJC__
    if (tempo_buffer_ && tempo_update_pipeline_) {
        float* bpm_ptr = (float*)tempo_buffer_.contents;
        *bpm_ptr = bpm;
        
        // Trigger tempo update shader
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:tempo_update_pipeline_];
            [encoder setBuffer:transport_buffer_ offset:0 atIndex:0];
            [encoder setBuffer:tempo_buffer_ offset:0 atIndex:1];
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake(1, 1, 1);
            
            [encoder dispatchThreadgroups:threadgroupsPerGrid
                    threadsPerThreadgroup:threadsPerThreadgroup];
            
            [encoder endEncoding];
            [commandBuffer commit];
        }
    }
#endif
    
    std::cout << "🎼 GPU Transport: BPM set to " << bpm << std::endl;
}

float GPUTransportManager::getBPM() const {
    return cached_state_.bpm;
}

//==============================================================================
// BARS/BEATS CONTROL AND QUERIES
//==============================================================================

void GPUTransportManager::setTimeSignature(uint32_t beatsPerBar, uint32_t beatUnit) {
    if (!initialized_.load()) {
        std::cout << "⚠️ setTimeSignature called but GPU Transport Manager not initialized" << std::endl;
        return;
    }

#ifdef __OBJC__
    if (bars_beats_buffer_ != nil) {
        GPUBarsBeatsBuffer* buffer_ptr = (GPUBarsBeatsBuffer*)[bars_beats_buffer_ contents];
        buffer_ptr->beats_per_bar = beatsPerBar;
        buffer_ptr->beat_unit = beatUnit;
        std::cout << "🎼 GPU Transport: Time signature set to " << beatsPerBar << "/" << beatUnit << std::endl;
    }
#endif
}

void GPUTransportManager::setSubdivision(uint32_t subdivisionCount) {
    if (!initialized_.load()) {
        std::cout << "⚠️ setSubdivision called but GPU Transport Manager not initialized" << std::endl;
        return;
    }

#ifdef __OBJC__
    if (bars_beats_buffer_ != nil) {
        GPUBarsBeatsBuffer* buffer_ptr = (GPUBarsBeatsBuffer*)[bars_beats_buffer_ contents];
        buffer_ptr->subdivision_count = subdivisionCount;
        std::cout << "🎼 GPU Transport: Subdivision count set to " << subdivisionCount << std::endl;
    }
#endif
}

GPUBarsBeatsBuffer GPUTransportManager::getBarsBeatsInfo() const {
    if (!initialized_.load()) {
        std::cout << "⚠️ getBarsBeatsInfo called but GPU Transport Manager not initialized" << std::endl;
        return cached_bars_beats_;
    }

    // Return cached value - this gets updated by the main update() cycle
    return cached_bars_beats_;
}

void GPUTransportManager::setStateChangeCallback(StateChangeCallback callback) {
    state_callback_ = callback;
}

void GPUTransportManager::shutdown() {
    if (!initialized_.load()) {
        std::cout << "⚠️ GPU Transport Manager shutdown called but not initialized" << std::endl;
        return;
    }
    
    std::cout << "🛑 GPU Transport Manager shutdown called" << std::endl;

#ifdef __OBJC__
    // Release Metal resources
    transport_pipeline_ = nil;
    network_sync_pipeline_ = nil;
    tempo_update_pipeline_ = nil;
    bars_beats_pipeline_ = nil;
    transport_buffer_ = nil;
    timebase_buffer_ = nil;
    timeline_events_buffer_ = nil;
    network_sync_buffer_ = nil;
    tempo_buffer_ = nil;
    bars_beats_buffer_ = nil;
    command_queue_ = nil;
    metal_device_ = nil;
#endif
    
    pending_events_.clear();
    initialized_.store(false);
    
    std::cout << "🛑 GPU Transport Manager shutdown complete" << std::endl;
}

} // namespace gpu_transport
} // namespace jam
