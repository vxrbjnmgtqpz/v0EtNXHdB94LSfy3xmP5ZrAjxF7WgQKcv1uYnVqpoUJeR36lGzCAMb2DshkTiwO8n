#pragma once

#include <vector>

// Forward declarations for Objective-C types
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
#else
typedef void* id;
#endif

class MetalBridge {
public:
    MetalBridge();
    ~MetalBridge();

    void init();
    void process(const std::vector<float>& input, std::vector<float>& output);

private:
#ifdef __OBJC__
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    std::vector<id<MTLBuffer>> _bufferPool;
#else
    id _device;
    id _commandQueue;
    std::vector<id> _bufferPool;
#endif
    id _semaphore;

    void setupMetal();
    void buildPipelines();
}; 