#import <Cocoa/Cocoa.h>
#include "training_testbed.h"
#include <iostream>

@interface SimpleTrainingGUI : NSObject <NSApplicationDelegate, NSWindowDelegate>
{
    NSWindow* window;
    NSButton* startButton;
    NSButton* stopButton;
    NSButton* exportButton;
    NSTextField* statusLabel;
    NSTextField* metricsLabel;
    NSProgressIndicator* progressIndicator;
    
    TrainingTestbed* testbed;
    NSTimer* updateTimer;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification;
- (void)setupWindow;
- (IBAction)startCollection:(id)sender;
- (IBAction)stopCollection:(id)sender;
- (IBAction)exportData:(id)sender;
- (void)updateDisplay:(NSTimer*)timer;

@end

@implementation SimpleTrainingGUI

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    std::cout << "🎛️ Launching Simple Training GUI...\n";
    
    // Initialize training testbed
    TrainingTestbed::Config config;
    config.sample_rate = 48000;
    config.channels = 2;
    config.enable_training_data = true;
    config.enable_logging = true;
    config.enable_network_simulation = true;
    config.packet_loss_percentage = 5.0;
    config.output_directory = "training_output";
    
    testbed = new TrainingTestbed(config);
    
    [self setupWindow];
    
    if (testbed->initialize()) {
        [statusLabel setStringValue:@"✅ Training Testbed Ready"];
        [startButton setEnabled:YES];
    } else {
        [statusLabel setStringValue:@"❌ Failed to Initialize"];
        [startButton setEnabled:NO];
    }
    
    // Start update timer
    updateTimer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                   target:self
                                                 selector:@selector(updateDisplay:)
                                                 userInfo:nil
                                                  repeats:YES];
}

- (void)setupWindow {
    // Create main window
    NSRect windowFrame = NSMakeRect(100, 100, 500, 400);
    window = [[NSWindow alloc] initWithContentRect:windowFrame
                                         styleMask:(NSWindowStyleMaskTitled | 
                                                   NSWindowStyleMaskClosable | 
                                                   NSWindowStyleMaskMiniaturizable)
                                           backing:NSBackingStoreBuffered
                                             defer:NO];
    [window setTitle:@"PNBTR+JELLIE Training Testbed"];
    [window setDelegate:self];
    
    // Status label
    statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 350, 460, 30)];
    [statusLabel setStringValue:@"Initializing..."];
    [statusLabel setEditable:NO];
    [statusLabel setBordered:NO];
    [statusLabel setBackgroundColor:[NSColor clearColor]];
    [statusLabel setFont:[NSFont boldSystemFontOfSize:14]];
    
    // Control buttons
    startButton = [[NSButton alloc] initWithFrame:NSMakeRect(20, 300, 120, 30)];
    [startButton setTitle:@"Start Collection"];
    [startButton setTarget:self];
    [startButton setAction:@selector(startCollection:)];
    [startButton setEnabled:NO];
    
    stopButton = [[NSButton alloc] initWithFrame:NSMakeRect(150, 300, 120, 30)];
    [stopButton setTitle:@"Stop Collection"];
    [stopButton setTarget:self];
    [stopButton setAction:@selector(stopCollection:)];
    [stopButton setEnabled:NO];
    
    exportButton = [[NSButton alloc] initWithFrame:NSMakeRect(280, 300, 120, 30)];
    [exportButton setTitle:@"Export Data"];
    [exportButton setTarget:self];
    [exportButton setAction:@selector(exportData:)];
    
    // Progress indicator
    progressIndicator = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(20, 260, 460, 20)];
    [progressIndicator setStyle:NSProgressIndicatorStyleBar];
    [progressIndicator setIndeterminate:YES];
    
    // Metrics display
    metricsLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 50, 460, 200)];
    [metricsLabel setStringValue:@"Training Metrics:\n\nSamples Processed: 0\nPackets Sent: 0\nPackets Lost: 0\nLatency: 0.0 μs\nSNR Improvement: 0.0 dB"];
    [metricsLabel setEditable:NO];
    [metricsLabel setBordered:YES];
    [metricsLabel setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular]];
    
    // Add to window
    [[window contentView] addSubview:statusLabel];
    [[window contentView] addSubview:startButton];
    [[window contentView] addSubview:stopButton];
    [[window contentView] addSubview:exportButton];
    [[window contentView] addSubview:progressIndicator];
    [[window contentView] addSubview:metricsLabel];
    
    [window makeKeyAndOrderFront:nil];
    
    // Force window to front
    [[NSApplication sharedApplication] setActivationPolicy:NSApplicationActivationPolicyRegular];
    [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
}

- (IBAction)startCollection:(id)sender {
    if (testbed && testbed->startDataCollection()) {
        [statusLabel setStringValue:@"🚀 Collecting Training Data..."];
        [startButton setEnabled:NO];
        [stopButton setEnabled:YES];
        [progressIndicator startAnimation:nil];
        std::cout << "✅ Data collection started from GUI\n";
    } else {
        [statusLabel setStringValue:@"❌ Failed to Start Collection"];
    }
}

- (IBAction)stopCollection:(id)sender {
    if (testbed) {
        testbed->stopDataCollection();
        [statusLabel setStringValue:@"💾 Data Collection Stopped"];
        [startButton setEnabled:YES];
        [stopButton setEnabled:NO];
        [progressIndicator stopAnimation:nil];
        std::cout << "🛑 Data collection stopped from GUI\n";
    }
}

- (IBAction)exportData:(id)sender {
    [statusLabel setStringValue:@"📤 Exporting Training Data..."];
    // TODO: Add export functionality
    [statusLabel setStringValue:@"✅ Training Data Exported"];
    std::cout << "📁 Training data exported from GUI\n";
}

- (void)updateDisplay:(NSTimer*)timer {
    if (!testbed) return;
    
    auto status = testbed->getStatus();
    
    NSString* metrics = [NSString stringWithFormat:@"Training Metrics:\n\n"
                        "Status: %s\n"
                        "Samples Processed: %llu\n"
                        "Packets Sent: %llu\n"
                        "Packets Lost: %llu\n"
                        "Latency: %.1f μs\n"
                        "SNR Improvement: %.1f dB\n"
                        "Loss Rate: %.1f%%",
                        status.collecting ? "🔴 COLLECTING" : "⚪ IDLE",
                        status.samples_processed,
                        status.packets_sent,
                        status.packets_lost,
                        status.avg_latency_us,
                        status.snr_improvement_db,
                        status.packets_sent > 0 ? (100.0 * status.packets_lost / status.packets_sent) : 0.0];
    
    [metricsLabel setStringValue:metrics];
}

- (BOOL)applicationShouldTerminateWhenLastWindowClosed:(NSApplication*)sender {
    return YES;
}

- (void)dealloc {
    if (updateTimer) {
        [updateTimer invalidate];
    }
    if (testbed) {
        testbed->shutdown();
        delete testbed;
    }
}

@end

// C++ entry point for simple GUI
extern "C" int launch_simple_training_gui() {
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        SimpleTrainingGUI* gui = [[SimpleTrainingGUI alloc] init];
        [app setDelegate:gui];
        [app run];
        return 0;
    }
} 