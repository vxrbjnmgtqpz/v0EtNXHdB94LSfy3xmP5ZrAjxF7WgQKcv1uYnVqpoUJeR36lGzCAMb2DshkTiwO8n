#import <Foundation/Foundation.h>

@interface TestServicePublisher : NSObject <NSNetServiceDelegate>
@property NSNetService *service;
@end

@implementation TestServicePublisher

- (void)publishService {
    self.service = [[NSNetService alloc] initWithDomain:@"" 
                                                   type:@"_toast._tcp." 
                                                   name:@"Test TOAST Server" 
                                                   port:8080];
    
    self.service.delegate = self;
    [self.service publish];
    
    NSLog(@"📡 Publishing TOAST service: Test TOAST Server on port 8080");
}

- (void)netServiceDidPublish:(NSNetService *)service {
    NSLog(@"✅ Service published successfully: %@", service.name);
}

- (void)netService:(NSNetService *)service didNotPublish:(NSDictionary<NSString *, NSNumber *> *)errorDict {
    NSLog(@"❌ Service publication failed: %@", errorDict);
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"🌐 Starting test Bonjour service publisher for TOAST discovery testing");
        
        TestServicePublisher *publisher = [[TestServicePublisher alloc] init];
        [publisher publishService];
        
        // Keep running
        NSLog(@"⏳ Service running. Press Ctrl+C to stop.");
        [[NSRunLoop currentRunLoop] run];
    }
    return 0;
}
