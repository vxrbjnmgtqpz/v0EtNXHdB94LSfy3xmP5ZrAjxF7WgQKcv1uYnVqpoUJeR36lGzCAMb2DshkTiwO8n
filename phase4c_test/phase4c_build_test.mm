#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <iostream>

int main() {
    @autoreleasepool {
        std::cout << "🚀 Phase 4C Build Test Starting..." << std::endl;
        
        // Test Metal GPU availability
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            std::cout << "✅ Metal GPU Device Available" << std::endl;
            NSLog(@"GPU Name: %@", device.name);
            
            // Test basic compute capabilities
            if ([device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]) {
                std::cout << "✅ GPU supports advanced compute features" << std::endl;
            }
            
            // Test memory info
            NSLog(@"GPU Memory: %llu MB", device.recommendedMaxWorkingSetSize / (1024 * 1024));
            
        } else {
            std::cout << "❌ No Metal GPU Device Found" << std::endl;
            return 1;
        }
        
        std::cout << "🎯 Phase 4C Build Test Completed Successfully!" << std::endl;
    }
    return 0;
} 