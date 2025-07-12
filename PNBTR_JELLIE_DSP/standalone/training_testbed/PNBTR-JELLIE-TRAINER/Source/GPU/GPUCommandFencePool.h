#pragma once
#import <Metal/Metal.h>
#include <vector>

class GPUCommandFencePool {
public:
    GPUCommandFencePool(id<MTLDevice> device, int poolSize = 8);
    ~GPUCommandFencePool();
    
    id<MTLSharedEvent> acquireFence(uint64_t frameIndex);
    void releaseFence(id<MTLSharedEvent> fence);
    
private:
    std::vector<id<MTLSharedEvent>> fencePool;
    int poolSize;
    id<MTLDevice> metalDevice;
}; 