#pragma once

#include "jam_framework.h"
#include <memory>

namespace jam {

class JAMGPU;
class JAMToast;

enum class JAMDataType {
    Audio,
    MIDI,
    Video,
    Session
};

struct JAMPacket {
    std::vector<uint8_t> data;
    std::string source_address;
    uint16_t source_port;
    uint64_t timestamp_ns;
};

struct JAMParsedData {
    JAMDataType type;
    JAMAudioData audio;
    JAMMIDIData midi;
    JAMVideoData video;
    std::string session_id;
};

class JAMParser {
public:
    JAMParser(const JAMConfig& config, JAMGPU& gpu, JAMToast& toast);
    ~JAMParser();
    
    bool initialize();
    void shutdown();
    
    JAMParsedData parse_packet(const JAMPacket& packet);
    
    // Statistics
    uint64_t get_bytes_processed() const;
    uint64_t get_packets_parsed() const;
    float get_parse_time_avg_ms() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace jam
