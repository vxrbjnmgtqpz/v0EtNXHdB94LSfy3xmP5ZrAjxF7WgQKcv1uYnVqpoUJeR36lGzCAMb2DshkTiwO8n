#pragma once

/**
 * JAM Framework v2: Memory Mapper Header
 */

namespace jam {

class MemoryMapper {
public:
    MemoryMapper();
    ~MemoryMapper();
    
    bool initialize();
    void shutdown();
};

} // namespace jam
