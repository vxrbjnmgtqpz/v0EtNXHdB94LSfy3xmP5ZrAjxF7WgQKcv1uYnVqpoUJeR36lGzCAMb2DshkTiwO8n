#pragma once

/**
 * JAM Framework v2: Compute Pipeline Header
 */

namespace jam {

class ComputePipeline {
public:
    ComputePipeline();
    ~ComputePipeline();
    
    bool initialize();
    void shutdown();
};

} // namespace jam
