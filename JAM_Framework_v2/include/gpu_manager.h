#pragma once

/**
 * JAM Framework v2: GPU Manager Header
 */

namespace jam {

class GPUManager {
public:
    GPUManager();
    ~GPUManager();
    
    bool initialize();
    void shutdown();
};

} // namespace jam
