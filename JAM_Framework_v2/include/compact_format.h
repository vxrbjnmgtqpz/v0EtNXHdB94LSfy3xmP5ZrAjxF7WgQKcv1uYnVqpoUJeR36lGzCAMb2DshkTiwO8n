#pragma once

/**
 * JAM Framework v2: Compact Format Header
 */

#include <vector>
#include <cstdint>

namespace jam {

class CompactFormat {
public:
    CompactFormat();
    ~CompactFormat();
    
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decode(const std::vector<uint8_t>& data);
};

} // namespace jam
