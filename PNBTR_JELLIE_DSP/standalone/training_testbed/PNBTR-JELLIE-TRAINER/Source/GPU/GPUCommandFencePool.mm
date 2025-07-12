#import "GPUCommandFencePool.h"
#import <Foundation/Foundation.h>

GPUCommandFencePool::GPUCommandFencePool(id<MTLDevice> device, int poolSize)
: poolSize(poolSize), metalDevice(device)
{
    fencePool.reserve(poolSize);
    for (int i = 0; i < poolSize; ++i) {
        id<MTLSharedEvent> event = [device newSharedEvent];
        event.label = [NSString stringWithFormat:@"Fence_%d", i];
        fencePool.push_back(event);
    }
    
    NSLog(@"[🔗 FENCE POOL] Initialized with %d Metal shared events", poolSize);
}

GPUCommandFencePool::~GPUCommandFencePool() {
    // ARC will handle cleanup of Metal objects
    fencePool.clear();
}

id<MTLSharedEvent> GPUCommandFencePool::acquireFence(uint64_t frameIndex) {
    int index = frameIndex % poolSize;
    return fencePool[index];
}

void GPUCommandFencePool::releaseFence(id<MTLSharedEvent> fence) {
    // In this implementation, fences are reused cyclically
    // No explicit release needed as they're managed by the pool
} 