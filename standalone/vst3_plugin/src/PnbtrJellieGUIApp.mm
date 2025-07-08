/*
 * PNBTR+JELLIE VST3 Plugin - REAL GUI APPLICATION
 * Revolutionary zero-noise dither replacement + 8-channel redundant streaming
 * 
 * This is an actual macOS desktop application with windows, buttons, and visual controls
 */

#import <Cocoa/Cocoa.h>
#include "PnbtrJelliePlugin.h"
#include <thread>
#include <atomic>

@interface PnbtrJellieController : NSObject <NSApplicationDelegate>

@property (strong) NSWindow *mainWindow;
@property (strong) NSButton *startStopButton;
@property (strong) NSButton *modeButton;
@property (strong) NSSlider *pnbtrStrengthSlider;
@property (strong) NSSlider *packetLossSlider;
@property (strong) NSSlider *frequencySlider;
@property (strong) NSTextField *latencyLabel;
@property (strong) NSTextField *snrLabel;
@property (strong) NSTextField *packetsLabel;
@property (strong) NSTextField *statusLabel;
@property (strong) NSProgressIndicator *activityIndicator;

// Engine - using instance variables instead of properties for C++ objects
@end

@implementation PnbtrJellieController {
    pnbtr_jellie::PnbtrJellieEngine *_txEngine;
    pnbtr_jellie::PnbtrJellieEngine *_rxEngine;
    std::atomic<bool> _isRunning;
    std::atomic<bool> _isTxMode;
    std::thread *_audioThread;
}

- (void)createMainWindow;
- (void)createControls;
- (void)startAudioProcessing;
- (void)stopAudioProcessing;
- (void)updateDisplay;

@end

@implementation PnbtrJellieController

- (instancetype)init {
    self = [super init];
    if (self) {
        _isRunning = false;
        _isTxMode = true;
        _txEngine = new pnbtr_jellie::PnbtrJellieEngine();
        _rxEngine = new pnbtr_jellie::PnbtrJellieEngine();
        _audioThread = nullptr;
        
        // Initialize engines
        _txEngine->initialize(48000, 512);
        _rxEngine->initialize(48000, 512);
        
        _txEngine->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::TX_MODE);
        _rxEngine->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::RX_MODE);
    }
    return self;
}

- (void)dealloc {
    [self stopAudioProcessing];
    delete _txEngine;
    delete _rxEngine;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [self createMainWindow];
    [self createControls];
    
    // Start display update timer
    [NSTimer scheduledTimerWithTimeInterval:0.1
                                     target:self
                                   selector:@selector(updateDisplay)
                                   userInfo:nil
                                    repeats:YES];
}

- (void)createMainWindow {
    NSRect frame = NSMakeRect(100, 100, 800, 600);
    
    _mainWindow = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled | 
                                                       NSWindowStyleMaskClosable | 
                                                       NSWindowStyleMaskMiniaturizable |
                                                       NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    
    [_mainWindow setTitle:@"PNBTR+JELLIE VST3 Plugin"];
    [_mainWindow setDelegate:(id<NSWindowDelegate>)self];
    [_mainWindow makeKeyAndOrderFront:nil];
    [_mainWindow center];
    
    // Set app icon and make it appear in dock
    NSApplication *app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    [app activateIgnoringOtherApps:YES];
}

- (void)createControls {
    NSView *contentView = [_mainWindow contentView];
    
    // Title Label
    NSTextField *titleLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 550, 760, 30)];
    [titleLabel setStringValue:@"🚀 PNBTR+JELLIE VST3 Plugin - Revolutionary Audio Processing"];
    [titleLabel setBezeled:NO];
    [titleLabel setDrawsBackground:NO];
    [titleLabel setEditable:NO];
    [titleLabel setSelectable:NO];
    [titleLabel setFont:[NSFont boldSystemFontOfSize:16]];
    [titleLabel setAlignment:NSTextAlignmentCenter];
    [contentView addSubview:titleLabel];
    
    // Status Label
    _statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 500, 760, 25)];
    [_statusLabel setStringValue:@"⏸️ STOPPED | 🔊 TX MODE (Transmit)"];
    [_statusLabel setBezeled:NO];
    [_statusLabel setDrawsBackground:NO];
    [_statusLabel setEditable:NO];
    [_statusLabel setSelectable:NO];
    [_statusLabel setFont:[NSFont systemFontOfSize:14]];
    [_statusLabel setAlignment:NSTextAlignmentCenter];
    [contentView addSubview:_statusLabel];
    
    // Start/Stop Button
    _startStopButton = [[NSButton alloc] initWithFrame:NSMakeRect(300, 450, 200, 40)];
    [_startStopButton setTitle:@"▶️ START PROCESSING"];
    [_startStopButton setBezelStyle:NSBezelStyleRounded];
    [_startStopButton setTarget:self];
    [_startStopButton setAction:@selector(toggleProcessing:)];
    [[_startStopButton cell] setControlSize:NSControlSizeRegular];
    [_startStopButton setFont:[NSFont boldSystemFontOfSize:14]];
    [contentView addSubview:_startStopButton];
    
    // Mode Toggle Button
    _modeButton = [[NSButton alloc] initWithFrame:NSMakeRect(520, 450, 150, 40)];
    [_modeButton setTitle:@"🔄 Switch to RX"];
    [_modeButton setBezelStyle:NSBezelStyleRounded];
    [_modeButton setTarget:self];
    [_modeButton setAction:@selector(toggleMode:)];
    [contentView addSubview:_modeButton];
    
    // PNBTR Strength Slider
    NSTextField *pnbtrLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 380, 200, 20)];
    [pnbtrLabel setStringValue:@"PNBTR Strength:"];
    [pnbtrLabel setBezeled:NO];
    [pnbtrLabel setDrawsBackground:NO];
    [pnbtrLabel setEditable:NO];
    [pnbtrLabel setFont:[NSFont boldSystemFontOfSize:12]];
    [contentView addSubview:pnbtrLabel];
    
    _pnbtrStrengthSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(50, 350, 300, 25)];
    [_pnbtrStrengthSlider setMinValue:0.0];
    [_pnbtrStrengthSlider setMaxValue:1.0];
    [_pnbtrStrengthSlider setDoubleValue:0.75];
    [_pnbtrStrengthSlider setTarget:self];
    [_pnbtrStrengthSlider setAction:@selector(pnbtrStrengthChanged:)];
    [contentView addSubview:_pnbtrStrengthSlider];
    
    // Packet Loss Slider
    NSTextField *packetLossLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(450, 380, 200, 20)];
    [packetLossLabel setStringValue:@"Packet Loss Simulation (%):"];
    [packetLossLabel setBezeled:NO];
    [packetLossLabel setDrawsBackground:NO];
    [packetLossLabel setEditable:NO];
    [packetLossLabel setFont:[NSFont boldSystemFontOfSize:12]];
    [contentView addSubview:packetLossLabel];
    
    _packetLossSlider = [[NSSlider alloc] initWithFrame:NSMakeRect(450, 350, 300, 25)];
    [_packetLossSlider setMinValue:0.0];
    [_packetLossSlider setMaxValue:50.0];
    [_packetLossSlider setDoubleValue:5.0];
    [_packetLossSlider setTarget:self];
    [_packetLossSlider setAction:@selector(packetLossChanged:)];
    [contentView addSubview:_packetLossSlider];
    
    // Frequency Slider
    NSTextField *freqLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 300, 200, 20)];
    [freqLabel setStringValue:@"Sine Wave Frequency (Hz):"];
    [freqLabel setBezeled:NO];
    [freqLabel setDrawsBackground:NO];
    [freqLabel setEditable:NO];
    [freqLabel setFont:[NSFont boldSystemFontOfSize:12]];
    [contentView addSubview:freqLabel];
    
    _frequencySlider = [[NSSlider alloc] initWithFrame:NSMakeRect(50, 270, 300, 25)];
    [_frequencySlider setMinValue:100.0];
    [_frequencySlider setMaxValue:2000.0];
    [_frequencySlider setDoubleValue:440.0];
    [_frequencySlider setTarget:self];
    [_frequencySlider setAction:@selector(frequencyChanged:)];
    [contentView addSubview:_frequencySlider];
    
    // Performance Metrics Box
    NSBox *perfBox = [[NSBox alloc] initWithFrame:NSMakeRect(50, 50, 700, 180)];
    [perfBox setTitle:@"📊 Real-Time Performance Metrics"];
    [perfBox setTitleFont:[NSFont boldSystemFontOfSize:14]];
    [contentView addSubview:perfBox];
    
    NSView *perfView = [perfBox contentView];
    
    // Latency Label
    _latencyLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 130, 300, 20)];
    [_latencyLabel setStringValue:@"Current Latency: 0.0 μs"];
    [_latencyLabel setBezeled:NO];
    [_latencyLabel setDrawsBackground:NO];
    [_latencyLabel setEditable:NO];
    [_latencyLabel setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular]];
    [perfView addSubview:_latencyLabel];
    
    // SNR Label
    _snrLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(350, 130, 300, 20)];
    [_snrLabel setStringValue:@"PNBTR SNR Gain: 0.0 dB"];
    [_snrLabel setBezeled:NO];
    [_snrLabel setDrawsBackground:NO];
    [_snrLabel setEditable:NO];
    [_snrLabel setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular]];
    [perfView addSubview:_snrLabel];
    
    // Packets Label
    _packetsLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 100, 650, 20)];
    [_packetsLabel setStringValue:@"Packets: Sent: 0 | Received: 0 | Lost: 0"];
    [_packetsLabel setBezeled:NO];
    [_packetsLabel setDrawsBackground:NO];
    [_packetsLabel setEditable:NO];
    [_packetsLabel setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular]];
    [perfView addSubview:_packetsLabel];
    
    // Activity Indicator
    _activityIndicator = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(20, 50, 32, 32)];
    [_activityIndicator setStyle:NSProgressIndicatorStyleSpinning];
    [_activityIndicator setDisplayedWhenStopped:NO];
    [perfView addSubview:_activityIndicator];
    
    // Network Status
    NSTextField *networkLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(70, 50, 400, 20)];
    [networkLabel setStringValue:@"🌐 Network: 239.255.0.1:8888 | JELLIE 8-Channel | ADAT Redundancy"];
    [networkLabel setBezeled:NO];
    [networkLabel setDrawsBackground:NO];
    [networkLabel setEditable:NO];
    [networkLabel setFont:[NSFont systemFontOfSize:11]];
    [perfView addSubview:networkLabel];
}

- (IBAction)toggleProcessing:(id)sender {
    if (_isRunning.load()) {
        [self stopAudioProcessing];
        [_startStopButton setTitle:@"▶️ START PROCESSING"];
        [_activityIndicator stopAnimation:self];
    } else {
        [self startAudioProcessing];
        [_startStopButton setTitle:@"⏸️ STOP PROCESSING"];
        [_activityIndicator startAnimation:self];
    }
}

- (IBAction)toggleMode:(id)sender {
    _isTxMode = !_isTxMode.load();
    if (_isTxMode.load()) {
        [_modeButton setTitle:@"🔄 Switch to RX"];
    } else {
        [_modeButton setTitle:@"🔄 Switch to TX"];
    }
}

- (IBAction)pnbtrStrengthChanged:(id)sender {
    NSSlider *slider = (NSSlider *)sender;
    double value = [slider doubleValue];
    
    // Update engine configuration
    pnbtr_jellie::PnbtrJellieEngine::PnbtrConfig config;
    config.prediction_strength = value;
    config.enable_reconstruction = true;
    config.prediction_window_ms = 50;
    config.enable_zero_noise_dither = true;
    
    _txEngine->setPnbtrConfig(config);
    _rxEngine->setPnbtrConfig(config);
}

- (IBAction)packetLossChanged:(id)sender {
    NSSlider *slider = (NSSlider *)sender;
    double value = [slider doubleValue];
    
    // Update test configuration
    pnbtr_jellie::PnbtrJellieEngine::TestConfig config;
    config.enable_packet_loss_simulation = true;
    config.packet_loss_percentage = value;
    config.enable_sine_generator = true;
    config.sine_frequency_hz = [_frequencySlider doubleValue];
    config.sine_amplitude = 0.5f;
    
    _txEngine->setTestConfig(config);
    _rxEngine->setTestConfig(config);
}

- (IBAction)frequencyChanged:(id)sender {
    NSSlider *slider = (NSSlider *)sender;
    double value = [slider doubleValue];
    
    // Update test configuration
    pnbtr_jellie::PnbtrJellieEngine::TestConfig config;
    config.enable_sine_generator = true;
    config.sine_frequency_hz = value;
    config.sine_amplitude = 0.5f;
    config.enable_packet_loss_simulation = true;
    config.packet_loss_percentage = [_packetLossSlider doubleValue];
    
    _txEngine->setTestConfig(config);
    _rxEngine->setTestConfig(config);
}

- (void)startAudioProcessing {
    _isRunning = true;
    
    _audioThread = new std::thread([self]() {
        const int bufferSize = 512;
        const int channels = 2;
        float inputBuffer[bufferSize * channels] = {0};
        float outputBuffer[bufferSize * channels];
        
        while (_isRunning.load()) {
            if (_isTxMode.load()) {
                _txEngine->processAudio(inputBuffer, outputBuffer, bufferSize, channels);
            } else {
                _rxEngine->processAudio(inputBuffer, outputBuffer, bufferSize, channels);
            }
            
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
}

- (void)stopAudioProcessing {
    _isRunning = false;
    
    if (_audioThread && _audioThread->joinable()) {
        _audioThread->join();
        delete _audioThread;
        _audioThread = nullptr;
    }
}

- (void)updateDisplay {
    // Update status
    NSString *status = _isRunning.load() ? @"🔴 PROCESSING" : @"⏸️ STOPPED";
    NSString *mode = _isTxMode.load() ? @"🔊 TX MODE (Transmit)" : @"🔉 RX MODE (Receive)";
    [_statusLabel setStringValue:[NSString stringWithFormat:@"%@ | %@", status, mode]];
    
    // Get performance stats
    auto currentEngine = _isTxMode.load() ? _txEngine : _rxEngine;
    const auto& stats = currentEngine->getPerformanceStats();
    
    // Update performance metrics
    [_latencyLabel setStringValue:[NSString stringWithFormat:@"Current Latency: %.1f μs", 
                                   stats.current_latency_us.load()]];
    
    [_snrLabel setStringValue:[NSString stringWithFormat:@"PNBTR SNR Gain: %.1f dB", 
                               stats.current_snr_db.load()]];
    
    [_packetsLabel setStringValue:[NSString stringWithFormat:@"Packets: Sent: %llu | Received: %llu | Lost: %llu",
                                   stats.packets_sent.load(),
                                   stats.packets_received.load(),
                                   stats.packets_lost.load()]];
    
    // Update slider values display
    NSTextField *pnbtrValueLabel = [[NSTextField alloc] init]; // You can add these as properties for better management
}

- (BOOL)applicationShouldTerminateWhenLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (void)windowWillClose:(NSNotification *)notification {
    [self stopAudioProcessing];
    [[NSApplication sharedApplication] terminate:self];
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        
        PnbtrJellieController *controller = [[PnbtrJellieController alloc] init];
        [app setDelegate:controller];
        
        NSLog(@"🚀 PNBTR+JELLIE VST3 Plugin GUI Application Starting...");
        
        [app run];
        
        return 0;
    }
} 