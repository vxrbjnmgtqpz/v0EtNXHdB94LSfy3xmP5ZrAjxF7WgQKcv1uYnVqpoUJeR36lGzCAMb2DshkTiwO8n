#pragma once

#include "jam_framework.h"
#include <vector>

namespace jam {

struct JAMGPUParseResult {
    uint32_t type;
    std::string session_id;
    JAMAudioData audio_data;
    JAMMIDIData midi_data;
    JAMVideoData video_data;
};

class JAMGPU {
public:
    JAMGPU(const JAMConfig& config);
    ~JAMGPU();
    
    bool initialize();
    void shutdown();
    
    bool is_available() const;
    bool set_enabled(bool enabled);
    float get_utilization() const;
    
    // Parser GPU acceleration
    uint32_t create_parser_context();
    void destroy_parser_context(uint32_t context_id);
    JAMGPUParseResult parse_jsonl(uint32_t context_id, const std::vector<uint8_t>& data);
    
    // Audio processing (PNBTR integration)
    JAMAudioData process_audio(const JAMAudioData& input);
    std::vector<float> predict_waveform(const std::vector<float>& samples, uint32_t predict_samples);
    
    // Video processing (JVID integration)
    JAMVideoData process_video(const JAMVideoData& input);
    uint32_t create_video_shader(const std::string& shader_source);
    
    // Memory management
    bool upload_data_to_gpu(const void* data, size_t size, uint32_t& buffer_id);
    bool download_data_from_gpu(uint32_t buffer_id, void* data, size_t size);
    void free_gpu_buffer(uint32_t buffer_id);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace jam
