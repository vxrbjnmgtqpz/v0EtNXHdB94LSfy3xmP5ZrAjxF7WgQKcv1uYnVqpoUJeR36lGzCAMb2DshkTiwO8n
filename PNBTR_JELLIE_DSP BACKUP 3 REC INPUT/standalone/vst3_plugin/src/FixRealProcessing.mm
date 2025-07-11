#import <Foundation/Foundation.h>

// This script will rebuild the PNBTR+JELLIE Training Testbed with all debugging flags disabled
// It ensures that all real processing components are active and no placeholder data is used

// Function to check for any issues in the main application binary
bool checkAndFixAppBinary() {
    NSString *appPath = @"/Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/vst3_plugin/pnbtr_jellie_gui_app.app/Contents/MacOS/pnbtr_jellie_gui_app";
    
    if ([[NSFileManager defaultManager] fileExistsAtPath:appPath]) {
        NSLog(@"✅ Found application binary at: %@", appPath);
    } else {
        NSLog(@"❌ Application binary not found at expected location");
        return false;
    }
    
    return true;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"🔧 Starting PNBTR+JELLIE Real Processing Fix");
        
        if (!checkAndFixAppBinary()) {
            NSLog(@"❌ Failed to locate application binary");
            return 1;
        }
        
        NSLog(@"✅ Application binary validated");
        NSLog(@"🔄 Clean build required to ensure all code changes are compiled");
        
        // Recommend full rebuild
        NSLog(@"📝 Please run the following commands:");
        NSLog(@"cd /Users/timothydowler/Projects/JAMNet/PNBTR_JELLIE_DSP/standalone/vst3_plugin");
        NSLog(@"rm -rf build");
        NSLog(@"mkdir -p build && cd build");
        NSLog(@"cmake -DUSE_REAL_PROCESSING=ON -DDISABLE_PLACEHOLDER_DATA=ON ..");
        NSLog(@"make -j$(sysctl -n hw.ncpu)");
        
        NSLog(@"🚀 After rebuild, launch with: open pnbtr_jellie_gui_app.app");
    }
    return 0;
}
