#include "jam_parser.h"
#include "jam_gpu.h"
#include "jam_toast.h"

#include <json/json.h>
#include <chrono>
#include <sstream>

namespace jam {

class JAMParser::Impl {
public:
    JAMConfig config;
    JAMGPU& gpu;
    JAMToast& toast;
    
    // Statistics
    uint64_t bytes_processed = 0;
    uint64_t packets_parsed = 0;
    float total_parse_time_ms = 0.0f;
    
    // GPU parsing context
    uint32_t gpu_parser_id = 0;
    bool gpu_parsing_enabled = false;
    
    JAMParsedData parse_jsonl_gpu(const std::vector<uint8_t>& data);
    JAMParsedData parse_jsonl_cpu(const std::vector<uint8_t>& data);
    
    JAMAudioData parse_audio_json(const Json::Value& json);
    JAMMIDIData parse_midi_json(const Json::Value& json);
    JAMVideoData parse_video_json(const Json::Value& json);
};

JAMParser::JAMParser(const JAMConfig& config, JAMGPU& gpu, JAMToast& toast)
    : pImpl(std::make_unique<Impl>()) {
    pImpl->config = config;
    pImpl->gpu = gpu;
    pImpl->toast = toast;
}

JAMParser::~JAMParser() {
    shutdown();
}

bool JAMParser::initialize() {
    // Try to enable GPU parsing if available
    if (pImpl->gpu.is_available()) {
        pImpl->gpu_parser_id = pImpl->gpu.create_parser_context();
        pImpl->gpu_parsing_enabled = (pImpl->gpu_parser_id != 0);
    }
    
    return true;
}

void JAMParser::shutdown() {
    if (pImpl->gpu_parsing_enabled) {
        pImpl->gpu.destroy_parser_context(pImpl->gpu_parser_id);
        pImpl->gpu_parsing_enabled = false;
    }
}

JAMParsedData JAMParser::parse_packet(const JAMPacket& packet) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    JAMParsedData result;
    
    // Decompress if needed
    auto decompressed_data = pImpl->toast.decompress(packet.data);
    pImpl->bytes_processed += decompressed_data.size();
    
    // Parse using GPU if available, fallback to CPU
    if (pImpl->gpu_parsing_enabled) {
        result = pImpl->parse_jsonl_gpu(decompressed_data);
    } else {
        result = pImpl->parse_jsonl_cpu(decompressed_data);
    }
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    pImpl->total_parse_time_ms += duration.count() / 1000.0f;
    pImpl->packets_parsed++;
    
    return result;
}

JAMParsedData JAMParser::Impl::parse_jsonl_gpu(const std::vector<uint8_t>& data) {
    // Use GPU compute shader for high-performance JSONL parsing
    auto gpu_result = gpu.parse_jsonl(gpu_parser_id, data);
    
    JAMParsedData result;
    result.type = static_cast<JAMDataType>(gpu_result.type);
    result.session_id = gpu_result.session_id;
    
    // Convert GPU parsing result to appropriate data structure
    switch (result.type) {
        case JAMDataType::Audio:
            result.audio = gpu_result.audio_data;
            break;
        case JAMDataType::MIDI:
            result.midi = gpu_result.midi_data;
            break;
        case JAMDataType::Video:
            result.video = gpu_result.video_data;
            break;
        default:
            break;
    }
    
    return result;
}

JAMParsedData JAMParser::Impl::parse_jsonl_cpu(const std::vector<uint8_t>& data) {
    // Fallback CPU parsing using traditional JSON library
    std::string json_str(data.begin(), data.end());
    
    Json::Value root;
    Json::Reader reader;
    
    JAMParsedData result;
    
    if (!reader.parse(json_str, root)) {
        // Parse error - return empty result
        return result;
    }
    
    // Determine data type from JSON structure
    if (root.isMember("audio_data")) {
        result.type = JAMDataType::Audio;
        result.audio = parse_audio_json(root);
    } else if (root.isMember("midi_data")) {
        result.type = JAMDataType::MIDI;
        result.midi = parse_midi_json(root);
    } else if (root.isMember("video_data")) {
        result.type = JAMDataType::Video;
        result.video = parse_video_json(root);
    }
    
    if (root.isMember("session_id")) {
        result.session_id = root["session_id"].asString();
    }
    
    return result;
}

JAMAudioData JAMParser::Impl::parse_audio_json(const Json::Value& json) {
    JAMAudioData audio;
    
    if (json.isMember("session_id")) {
        audio.session_id = json["session_id"].asString();
    }
    
    if (json.isMember("timestamp")) {
        audio.timestamp_ns = json["timestamp"].asUInt64();
    }
    
    const Json::Value& audio_data = json["audio_data"];
    if (audio_data.isMember("sample_rate")) {
        audio.sample_rate = audio_data["sample_rate"].asUInt();
    }
    
    if (audio_data.isMember("channels")) {
        audio.channels = audio_data["channels"].asUInt();
    }
    
    if (audio_data.isMember("bit_depth")) {
        audio.bit_depth = audio_data["bit_depth"].asUInt();
    }
    
    if (audio_data.isMember("samples")) {
        const Json::Value& samples = audio_data["samples"];
        audio.samples.reserve(samples.size());
        for (const auto& sample : samples) {
            audio.samples.push_back(sample.asFloat());
        }
    }
    
    // PNBTR metadata
    if (audio_data.isMember("pnbtr_processed")) {
        audio.pnbtr_processed = audio_data["pnbtr_processed"].asBool();
    }
    
    if (audio_data.isMember("prediction_confidence")) {
        audio.prediction_confidence = audio_data["prediction_confidence"].asFloat();
    }
    
    return audio;
}

JAMMIDIData JAMParser::Impl::parse_midi_json(const Json::Value& json) {
    JAMMIDIData midi;
    
    if (json.isMember("session_id")) {
        midi.session_id = json["session_id"].asString();
    }
    
    if (json.isMember("timestamp")) {
        midi.timestamp_ns = json["timestamp"].asUInt64();
    }
    
    const Json::Value& midi_data = json["midi_data"];
    if (midi_data.isMember("status")) {
        midi.status = midi_data["status"].asUInt();
    }
    
    if (midi_data.isMember("data1")) {
        midi.data1 = midi_data["data1"].asUInt();
    }
    
    if (midi_data.isMember("data2")) {
        midi.data2 = midi_data["data2"].asUInt();
    }
    
    // Burst logic metadata
    if (midi_data.isMember("burst_count")) {
        midi.burst_count = midi_data["burst_count"].asUInt();
    }
    
    if (midi_data.isMember("sequence_id")) {
        midi.sequence_id = midi_data["sequence_id"].asUInt();
    }
    
    if (midi_data.isMember("is_duplicate")) {
        midi.is_duplicate = midi_data["is_duplicate"].asBool();
    }
    
    return midi;
}

JAMVideoData JAMParser::Impl::parse_video_json(const Json::Value& json) {
    JAMVideoData video;
    
    if (json.isMember("session_id")) {
        video.session_id = json["session_id"].asString();
    }
    
    if (json.isMember("timestamp")) {
        video.timestamp_ns = json["timestamp"].asUInt64();
    }
    
    const Json::Value& video_data = json["video_data"];
    if (video_data.isMember("width")) {
        video.width = video_data["width"].asUInt();
    }
    
    if (video_data.isMember("height")) {
        video.height = video_data["height"].asUInt();
    }
    
    // Direct pixel array (no base64 decoding needed)
    if (video_data.isMember("pixels")) {
        const Json::Value& pixels = video_data["pixels"];
        video.pixel_data.reserve(pixels.size());
        for (const auto& pixel : pixels) {
            video.pixel_data.push_back(pixel.asUInt());
        }
    }
    
    // GPU processing metadata
    if (video_data.isMember("gpu_processed")) {
        video.gpu_processed = video_data["gpu_processed"].asBool();
    }
    
    if (video_data.isMember("shader_id")) {
        video.shader_id = video_data["shader_id"].asUInt();
    }
    
    return video;
}

uint64_t JAMParser::get_bytes_processed() const {
    return pImpl->bytes_processed;
}

uint64_t JAMParser::get_packets_parsed() const {
    return pImpl->packets_parsed;
}

float JAMParser::get_parse_time_avg_ms() const {
    if (pImpl->packets_parsed == 0) return 0.0f;
    return pImpl->total_parse_time_ms / pImpl->packets_parsed;
}

} // namespace jam
