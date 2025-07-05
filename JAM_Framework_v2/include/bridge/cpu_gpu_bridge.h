#pragma once

#include "gpu_native/gpu_timebase.h"
#include "gpu_native/gpu_shared_timeline.h"
#include <memory>
#include <functional>
#include <atomic>

namespace JAM::Bridge {

/**
 * CPU-GPU Bridge for Legacy Compatibility
 * 
 * Provides a compatibility layer between legacy CPU-based timing code
 * and the new GPU-native infrastructure. This allows gradual migration
 * of existing components without breaking functionality.
 */
class CPUGPUBridge {
public:
    CPUGPUBridge();
    ~CPUGPUBridge();

    // Initialization
    bool initialize(uint32_t sample_rate = 48000, uint32_t bpm = 120);
    void shutdown();

    // Legacy CPU interface (compatibility layer)
    void start_playback();
    void stop_playback();
    void pause_playback();
    void seek_to_frame(uint32_t frame);
    void set_bpm(uint32_t bpm);
    void set_sample_rate(uint32_t sample_rate);

    // Legacy state queries
    uint32_t get_current_frame() const;
    uint32_t get_sample_rate() const;
    uint32_t get_bpm() const;
    bool is_playing() const;
    bool is_recording() const;

    // Legacy callback system (redirected to GPU events)
    using TransportCallback = std::function<void(bool playing, uint32_t frame)>;
    using BeatCallback = std::function<void(uint32_t beat, uint32_t bpm)>;
    using MIDICallback = std::function<void(uint8_t status, uint8_t data1, uint8_t data2, uint32_t frame)>;

    void set_transport_callback(TransportCallback callback);
    void set_beat_callback(BeatCallback callback);
    void set_midi_callback(MIDICallback callback);

    // Legacy MIDI interface
    void send_midi_note_on(uint8_t channel, uint8_t note, uint8_t velocity, uint32_t frame = 0);
    void send_midi_note_off(uint8_t channel, uint8_t note, uint8_t velocity, uint32_t frame = 0);
    void send_midi_cc(uint8_t channel, uint8_t controller, uint8_t value, uint32_t frame = 0);

    // Legacy audio buffer interface
    void register_audio_callback(std::function<void(float* buffer, size_t frames)> callback);
    void process_audio_buffer(float* buffer, size_t frames);

    // Direct GPU access for performance-critical components
    GPUNative::GPUTimebase* get_gpu_timebase() const;
    GPUNative::GPUSharedTimelineManager* get_timeline_manager() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Legacy Transport Adapter
 * 
 * Adapts the old TransportController interface to work with GPU-native timing.
 * This allows existing DAW integration code to continue working while the
 * GPU becomes the master timebase.
 */
class LegacyTransportAdapter {
public:
    LegacyTransportAdapter(CPUGPUBridge& bridge);
    ~LegacyTransportAdapter();

    // Legacy TransportController API
    void play();
    void stop();
    void pause();
    void record();
    void seek(uint32_t frame);
    void set_tempo(double bpm);
    void set_time_signature(int numerator, int denominator);
    void set_loop_range(uint32_t start, uint32_t end);
    void enable_loop(bool enable);

    // Legacy state queries
    bool is_playing() const;
    bool is_recording() const;
    bool is_paused() const;
    uint32_t get_play_position() const;
    double get_tempo() const;
    void get_time_signature(int& numerator, int& denominator) const;
    void get_loop_range(uint32_t& start, uint32_t& end) const;
    bool is_loop_enabled() const;

    // Legacy sync interface
    void sync_to_external(uint32_t frame, double bpm);
    void set_external_sync_enabled(bool enabled);
    bool is_externally_synced() const;

private:
    CPUGPUBridge& bridge_;
    std::atomic<int> time_sig_numerator_{4};
    std::atomic<int> time_sig_denominator_{4};
    std::atomic<uint32_t> loop_start_{0};
    std::atomic<uint32_t> loop_end_{0};
    std::atomic<bool> loop_enabled_{false};
    std::atomic<bool> external_sync_enabled_{false};
};

/**
 * DAW Interface Bridge
 * 
 * Provides compatibility interfaces for various DAW plugin formats
 * while using GPU-native timing internally.
 */
class DAWInterfaceBridge {
public:
    DAWInterfaceBridge(CPUGPUBridge& bridge);
    ~DAWInterfaceBridge();

    // VST3 compatibility
    struct VST3HostInfo {
        double sample_rate;
        uint32_t max_samples_per_block;
        bool realtime;
    };

    bool initialize_vst3_host(const VST3HostInfo& info);
    void process_vst3_block(float** inputs, float** outputs, uint32_t sample_count);

    // Audio Units (AU) compatibility
    struct AUHostInfo {
        double sample_rate;
        uint32_t frames_per_slice;
        bool realtime;
    };

    bool initialize_au_host(const AUHostInfo& info);
    void process_au_slice(float** inputs, float** outputs, uint32_t frame_count);

    // Max for Live compatibility
    struct M4LHostInfo {
        double sample_rate;
        uint32_t vector_size;
        bool realtime;
    };

    bool initialize_m4l_host(const M4LHostInfo& info);
    void process_m4l_vector(float** inputs, float** outputs, uint32_t vector_size);

    // JSFX compatibility
    struct JSFXHostInfo {
        double sample_rate;
        uint32_t block_size;
        bool realtime;
    };

    bool initialize_jsfx_host(const JSFXHostInfo& info);
    void process_jsfx_block(float** inputs, float** outputs, uint32_t block_size);

    // Common DAW timing interface
    void notify_tempo_change(double bpm);
    void notify_time_signature_change(int numerator, int denominator);
    void notify_transport_state_change(bool playing, bool recording);
    void notify_position_change(uint32_t frame);

private:
    CPUGPUBridge& bridge_;
    VST3HostInfo vst3_info_;
    AUHostInfo au_info_;
    M4LHostInfo m4l_info_;
    JSFXHostInfo jsfx_info_;
    
    bool vst3_initialized_ = false;
    bool au_initialized_ = false;
    bool m4l_initialized_ = false;
    bool jsfx_initialized_ = false;
};

} // namespace JAM::Bridge
