#pragma once

/**
 * JAM Framework v2: JSONL Generator Header
 */

#include <string>
#include <map>

namespace jam {

class JSONLGenerator {
public:
    JSONLGenerator();
    ~JSONLGenerator();
    
    std::string generate_line(const std::map<std::string, std::string>& data);
};

} // namespace jam
