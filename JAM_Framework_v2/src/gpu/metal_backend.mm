// Metal GPU Backend Implementation
// Placeholder for legacy GPU backend - will be replaced by GPU-native timebase

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Minimal stub for Metal backend to satisfy build requirements
namespace JAM {
namespace GPU {

class MetalBackend {
public:
    MetalBackend() = default;
    ~MetalBackend() = default;
    
    bool initialize() {
        // Stub implementation
        return true;
    }
    
    void shutdown() {
        // Stub implementation
    }
};

} // namespace GPU
} // namespace JAM
