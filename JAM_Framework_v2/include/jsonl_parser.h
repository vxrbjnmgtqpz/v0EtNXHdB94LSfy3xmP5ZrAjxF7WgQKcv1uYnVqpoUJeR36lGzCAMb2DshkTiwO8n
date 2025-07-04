#pragma once

/**
 * JAM Framework v2: JSONL Parser Header
 */

#include <string>

namespace jam {

class JSONLParser {
public:
    JSONLParser();
    ~JSONLParser();
    
    bool parse_line(const std::string& line);
};

} // namespace jam
