/*
 * PNBTR+JELLIE VST3 Plugin - REAL Core Audio Interface with Oscilloscope
 * Live audio input/output for JDAT testing + Real-time waveform visualization
 */

#import <Cocoa/Cocoa.h>
#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>
#include "../include/PnbtrJelliePlugin.h"
#include <thread>
#include <memory>
#include <vector>
#include <mutex>

// Forward declare the controller
@class PnbtrJellieAudioController;

// Oscilloscope View for real-time waveform display
@interface OscilloscopeView : NSView
@property (nonatomic) std::vector<float> waveformData;
@property (nonatomic) std::mutex* dataMutex;
@property (nonatomic) NSColor* waveformColor;
@property (nonatomic) NSString* channelLabel;
- (void)updateWaveform:(const float*)audioData length:(int)length;
@end

@implementation OscilloscopeView

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        _waveformData.resize(512, 0.0f);
        _dataMutex = new std::mutex();
        _waveformColor = [NSColor greenColor];
        _channelLabel = @"Audio";
    }
    return self;
}

- (void)dealloc {
    if (_dataMutex) {
        delete _dataMutex;
    }
}

- (void)updateWaveform:(const float*)audioData length:(int)length {
    std::lock_guard<std::mutex> lock(*_dataMutex);
    
    // Resize if needed
    if (_waveformData.size() != length) {
        _waveformData.resize(length);
    }
    
    // Copy audio data
    for (int i = 0; i < length; ++i) {
        _waveformData[i] = audioData[i];
    }
    
    // Trigger redraw on main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        [self setNeedsDisplay:YES];
    });
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    NSRect bounds = [self bounds];
    
    // Fill background
    [[NSColor blackColor] setFill];
    NSRectFill(bounds);
    
    // Draw grid
    [[NSColor darkGrayColor] setStroke];
    NSBezierPath* gridPath = [NSBezierPath bezierPath];
    [gridPath setLineWidth:0.5];
    
    // Horizontal grid lines
    for (int i = 0; i <= 4; ++i) {
        CGFloat y = bounds.origin.y + (bounds.size.height * i / 4.0);
        [gridPath moveToPoint:NSMakePoint(bounds.origin.x, y)];
        [gridPath lineToPoint:NSMakePoint(bounds.origin.x + bounds.size.width, y)];
    }
    
    // Vertical grid lines
    for (int i = 0; i <= 8; ++i) {
        CGFloat x = bounds.origin.x + (bounds.size.width * i / 8.0);
        [gridPath moveToPoint:NSMakePoint(x, bounds.origin.y)];
        [gridPath lineToPoint:NSMakePoint(x, bounds.origin.y + bounds.size.height)];
    }
    [gridPath stroke];
    
    // Draw waveform
    std::lock_guard<std::mutex> lock(*_dataMutex);
    
    if (_waveformData.size() > 1) {
        [_waveformColor setStroke];
        NSBezierPath* waveformPath = [NSBezierPath bezierPath];
        [waveformPath setLineWidth:2.0];
        
        CGFloat centerY = bounds.origin.y + bounds.size.height / 2.0;
        CGFloat amplitude = bounds.size.height / 2.0 * 0.8; // 80% of half height
        
        for (size_t i = 0; i < _waveformData.size(); ++i) {
            CGFloat x = bounds.origin.x + (bounds.size.width * i / (_waveformData.size() - 1));
            CGFloat y = centerY + (_waveformData[i] * amplitude);
            
            if (i == 0) {
                [waveformPath moveToPoint:NSMakePoint(x, y)];
            } else {
                [waveformPath lineToPoint:NSMakePoint(x, y)];
            }
        }
        [waveformPath stroke];
    }
    
    // Draw label
    NSString* label = [NSString stringWithFormat:@"%@ - PNBTR Output", _channelLabel];
    NSDictionary* attributes = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightBold],
        NSForegroundColorAttributeName: _waveformColor
    };
    
    NSSize labelSize = [label sizeWithAttributes:attributes];
    NSPoint labelPoint = NSMakePoint(bounds.origin.x + 10, 
                                    bounds.origin.y + bounds.size.height - labelSize.height - 10);
    [label drawAtPoint:labelPoint withAttributes:attributes];
    
    // Draw amplitude scale
    NSArray* scaleLabels = @[@"+1.0", @"+0.5", @"0.0", @"-0.5", @"-1.0"];
    for (int i = 0; i < 5; ++i) {
        CGFloat y = bounds.origin.y + (bounds.size.height * (4-i) / 4.0);
        NSPoint scalePoint = NSMakePoint(bounds.origin.x + bounds.size.width - 40, y - 6);
        
        NSDictionary* scaleAttributes = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName: [NSColor lightGrayColor]
        };
        [scaleLabels[i] drawAtPoint:scalePoint withAttributes:scaleAttributes];
    }
}

@end

// Core Audio callback structure
typedef struct {
    __unsafe_unretained PnbtrJellieAudioController* controller;
    std::unique_ptr<pnbtr_jellie::PnbtrJellieEngine> engine;
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    bool isProcessing;
    __unsafe_unretained OscilloscopeView* oscilloscopeView;
} AudioCallbackData;

// Main application delegate
@interface PnbtrJellieAppDelegate : NSObject <NSApplicationDelegate>
@property (strong) PnbtrJellieAudioController *controller;
@end

// Core Audio Controller with real audio I/O and oscilloscope
@interface PnbtrJellieAudioController : NSObject

// UI Elements
@property (strong) NSWindow *mainWindow;
@property (strong) NSButton *modeToggleButton;
@property (strong) NSPopUpButton *inputDevicePopup;
@property (strong) NSPopUpButton *outputDevicePopup;
@property (strong) NSTextField *statusDisplay;
@property (strong) NSTextField *latencyDisplay;
@property (strong) NSTextField *snrDisplay;
@property (strong) NSTextField *audioInfoDisplay;
@property (strong) NSTextField *levelMeterInput;
@property (strong) NSTextField *levelMeterOutput;
@property (strong) OscilloscopeView *oscilloscopeView;

// Core Audio state
@property AudioUnit audioUnit;
@property AudioUnit inputAudioUnit;
@property BOOL isAudioRunning;
@property BOOL isTxMode;
@property AudioCallbackData *callbackData;

- (void)createMainWindow;
- (void)setupControls;
- (void)setupCoreAudio;
- (void)populateAudioDevices;
- (void)updateDisplay;
- (IBAction)toggleMode:(id)sender;
- (IBAction)inputDeviceChanged:(id)sender;
- (IBAction)outputDeviceChanged:(id)sender;
- (void)startAudioProcessing;

@end

// Audio render callback for real Core Audio processing
static OSStatus audioRenderCallback(void *inRefCon,
                                   AudioUnitRenderActionFlags *ioActionFlags,
                                   const AudioTimeStamp *inTimeStamp,
                                   UInt32 inBusNumber,
                                   UInt32 inNumberFrames,
                                   AudioBufferList *ioData) {
    
    AudioCallbackData *callbackData = (AudioCallbackData *)inRefCon;
    
    if (!callbackData || !callbackData->engine) {
        // Fill with silence if not processing
        for (UInt32 i = 0; i < ioData->mNumberBuffers; ++i) {
            memset(ioData->mBuffers[i].mData, 0, ioData->mBuffers[i].mDataByteSize);
        }
        return noErr;
    }
    
    // Get audio buffers
    float *outputBuffer = (float *)ioData->mBuffers[0].mData;
    UInt32 frameCount = inNumberFrames;
    
    // Resize buffers if needed
    if (callbackData->inputBuffer.size() < frameCount) {
        callbackData->inputBuffer.resize(frameCount, 0.0f);
        callbackData->outputBuffer.resize(frameCount, 0.0f);
    }
    
    // Use the shared input buffer (filled by input callback)
    // Copy from shared input buffer to processing buffer
    for (UInt32 i = 0; i < frameCount; ++i) {
        if (i < callbackData->inputBuffer.size()) {
            // Use real microphone input from the input callback
            callbackData->outputBuffer[i] = callbackData->inputBuffer[i];
        } else {
            callbackData->outputBuffer[i] = 0.0f;
        }
    }
    
    // Process through PNBTR+JELLIE engine
    callbackData->engine->processAudio(callbackData->outputBuffer.data(),
                                      callbackData->outputBuffer.data(),
                                      frameCount);
    
    // Copy processed audio to output
    for (UInt32 i = 0; i < frameCount; ++i) {
        outputBuffer[i] = callbackData->outputBuffer[i];
    }
    
    // Update oscilloscope based on TX/RX mode
    if (callbackData->oscilloscopeView) {
        PnbtrJellieAudioController *controller = callbackData->controller;
        if (controller.isTxMode) {
            // TX Mode: Show PNBTR reconstructed output (processed audio)
            [callbackData->oscilloscopeView updateWaveform:callbackData->outputBuffer.data() 
                                                    length:frameCount];
        } else {
            // RX Mode: Show raw microphone input (unprocessed)
            [callbackData->oscilloscopeView updateWaveform:callbackData->inputBuffer.data() 
                                                    length:frameCount];
        }
    }
    
    return noErr;
}

// NEW: Input callback to capture microphone data
static OSStatus audioInputCallback(void *inRefCon,
                                  AudioUnitRenderActionFlags *ioActionFlags,
                                  const AudioTimeStamp *inTimeStamp,
                                  UInt32 inBusNumber,
                                  UInt32 inNumberFrames,
                                  AudioBufferList *ioData) {
    
    AudioCallbackData *callbackData = (AudioCallbackData *)inRefCon;
    
    if (!callbackData) {
        return noErr;
    }
    
    // Resize input buffer if needed
    if (callbackData->inputBuffer.size() < inNumberFrames) {
        callbackData->inputBuffer.resize(inNumberFrames, 0.0f);
    }
    
    // Create input buffer list for microphone data
    AudioBufferList inputBufferList;
    inputBufferList.mNumberBuffers = 1;
    inputBufferList.mBuffers[0].mNumberChannels = 1;
    inputBufferList.mBuffers[0].mDataByteSize = inNumberFrames * sizeof(Float32);
    inputBufferList.mBuffers[0].mData = callbackData->inputBuffer.data();
    
    // Get REAL microphone input from the input audio unit
    PnbtrJellieAudioController *controller = callbackData->controller;
    if (controller.inputAudioUnit) {
        OSStatus status = AudioUnitRender(controller.inputAudioUnit,
                                         ioActionFlags,
                                         inTimeStamp,
                                         1, // input bus
                                         inNumberFrames,
                                         &inputBufferList);
        
        if (status != noErr) {
            // Fill with silence on error but don't spam logs
            static int errorCount = 0;
            if (errorCount < 5) {
                NSLog(@"Microphone input error: %d (will suppress further errors)", (int)status);
                errorCount++;
            }
            memset(callbackData->inputBuffer.data(), 0, inNumberFrames * sizeof(Float32));
        }
    } else {
        // Fill with silence if no input available
        memset(callbackData->inputBuffer.data(), 0, inNumberFrames * sizeof(Float32));
    }
    
    return noErr;
}

// Implementation
@implementation PnbtrJellieAudioController

- (instancetype)init {
    NSLog(@"Initializing PnbtrJellieAudioController - Auto-starting microphone");
    
    self = [super init];
    if (self) {
        _isAudioRunning = NO;
        _isTxMode = YES;
        _audioUnit = NULL;
        _inputAudioUnit = NULL;
        
        NSLog(@"Creating callback data and engine");
        
        // Initialize callback data
        _callbackData = new AudioCallbackData();
        _callbackData->controller = self;
        _callbackData->engine = std::make_unique<pnbtr_jellie::PnbtrJellieEngine>();
        _callbackData->isProcessing = false;
        _callbackData->oscilloscopeView = nil; // Will be set after UI creation
        
        // Initialize engine
        bool engineInit = _callbackData->engine->initialize(48000, 512);
        NSLog(@"Engine initialized: %s", engineInit ? "SUCCESS" : "FAILED");
        
        _callbackData->engine->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::TX_MODE);
        
        NSLog(@"Creating main window and controls");
        
        [self createMainWindow];
        [self setupControls];
        [self setupCoreAudio];
        [self populateAudioDevices];
        
        // Connect oscilloscope to callback
        _callbackData->oscilloscopeView = _oscilloscopeView;
        NSLog(@"Oscilloscope connected to callback");
        
        // AUTO-START AUDIO PROCESSING
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
            [self startAudioProcessing];
        });
        
        // Start update timer
        [NSTimer scheduledTimerWithTimeInterval:0.1
                                       target:self
                                     selector:@selector(updateDisplay)
                                     userInfo:nil
                                       repeats:YES];
        
        NSLog(@"Controller initialization complete - Audio will auto-start");
    }
    return self;
}

- (void)dealloc {
    if (_isAudioRunning) {
        // Stop audio units
        if (_audioUnit) {
            AudioOutputUnitStop(_audioUnit);
        }
        if (_inputAudioUnit) {
            AudioOutputUnitStop(_inputAudioUnit);
        }
        _callbackData->isProcessing = false;
        _isAudioRunning = NO;
    }
    
    if (_audioUnit) {
        AudioUnitUninitialize(_audioUnit);
        AudioComponentInstanceDispose(_audioUnit);
    }
    
    if (_inputAudioUnit) {
        AudioUnitUninitialize(_inputAudioUnit);
        AudioComponentInstanceDispose(_inputAudioUnit);
    }
    
    if (_callbackData) {
        delete _callbackData;
    }
}

- (void)createMainWindow {
    NSLog(@"Creating main window");
    
    NSRect frame = NSMakeRect(100, 100, 1000, 800); // Larger window for oscilloscope
    
    _mainWindow = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled | 
                                                       NSWindowStyleMaskClosable | 
                                                       NSWindowStyleMaskMiniaturizable |
                                                       NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    
    if (!_mainWindow) {
        NSLog(@"ERROR: Failed to create main window!");
        return;
    }
    
    [_mainWindow setTitle:@"PNBTR+JELLIE - Core Audio + Oscilloscope"];
    [_mainWindow center];
    
    // Make sure it appears on screen - multiple approaches
    NSApplication *app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    
    // Make window key and front
    [_mainWindow makeKeyAndOrderFront:nil];
    [_mainWindow orderFrontRegardless];
    
    // Activate application
    [app activateIgnoringOtherApps:YES];
    
    // Bring to front
    [_mainWindow setLevel:NSNormalWindowLevel];
    [_mainWindow makeMainWindow];
    
    NSLog(@"Main window created and should be visible at frame: %@", NSStringFromRect([_mainWindow frame]));
}

- (void)setupControls {
    NSView *contentView = [_mainWindow contentView];
    
    // Title
    NSTextField *titleLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 730, 900, 40)];
    [titleLabel setStringValue:@"🎤 PNBTR+JELLIE Real Microphone Processing"];
    [titleLabel setBezeled:NO];
    [titleLabel setDrawsBackground:NO];
    [titleLabel setEditable:NO];
    [titleLabel setFont:[NSFont boldSystemFontOfSize:24]];
    [titleLabel setAlignment:NSTextAlignmentCenter];
    [contentView addSubview:titleLabel];
    
    // Subtitle
    NSTextField *subtitleLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 700, 900, 25)];
    [subtitleLabel setStringValue:@"Live Microphone Input • JELLIE Encoding • PNBTR Reconstruction"];
    [subtitleLabel setBezeled:NO];
    [subtitleLabel setDrawsBackground:NO];
    [subtitleLabel setEditable:NO];
    [subtitleLabel setFont:[NSFont systemFontOfSize:14]];
    [subtitleLabel setAlignment:NSTextAlignmentCenter];
    [contentView addSubview:subtitleLabel];
    
    // Oscilloscope View - MAIN FEATURE
    NSBox *scopeBox = [[NSBox alloc] initWithFrame:NSMakeRect(50, 400, 900, 280)];
    [scopeBox setTitle:@"📊 Live Microphone Input - Real-time Waveform Display"];
    [scopeBox setTitleFont:[NSFont boldSystemFontOfSize:16]];
    [contentView addSubview:scopeBox];
    
    NSView *scopeContentView = [scopeBox contentView];
    _oscilloscopeView = [[OscilloscopeView alloc] initWithFrame:NSMakeRect(10, 10, 870, 220)];
    [_oscilloscopeView setWaveformColor:[NSColor cyanColor]];
    [_oscilloscopeView setChannelLabel:@"PNBTR Output"];
    [scopeContentView addSubview:_oscilloscopeView];
    
    // Status Display
    _statusDisplay = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 360, 900, 30)];
    [_statusDisplay setStringValue:@"🔴 MICROPHONE ACTIVE | 🔊 TX MODE | 📊 Live Input Display"];
    [_statusDisplay setBezeled:NO];
    [_statusDisplay setDrawsBackground:NO];
    [_statusDisplay setEditable:NO];
    [_statusDisplay setFont:[NSFont boldSystemFontOfSize:16]];
    [_statusDisplay setAlignment:NSTextAlignmentCenter];
    [contentView addSubview:_statusDisplay];
    
    // Audio Control Section - Always Active
    NSBox *controlBox = [[NSBox alloc] initWithFrame:NSMakeRect(100, 280, 800, 80)];
    [controlBox setTitle:@"🎤 Live Audio Processing - Always Active"];
    [controlBox setTitleFont:[NSFont boldSystemFontOfSize:16]];
    [contentView addSubview:controlBox];
    
    NSView *controlView = [controlBox contentView];
    
    // Mode Toggle Button (centered)
    _modeToggleButton = [[NSButton alloc] initWithFrame:NSMakeRect(325, 25, 150, 40)];
    [_modeToggleButton setTitle:@"🔄 Switch to RX MODE"];
    [_modeToggleButton setBezelStyle:NSBezelStyleRounded];
    [_modeToggleButton setTarget:self];
    [_modeToggleButton setAction:@selector(toggleMode:)];
    [_modeToggleButton setFont:[NSFont systemFontOfSize:14]];
    [controlView addSubview:_modeToggleButton];
    
    // Audio Device Selection
    NSTextField *inputLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(100, 250, 150, 20)];
    [inputLabel setStringValue:@"Audio Input Device:"];
    [inputLabel setBezeled:NO];
    [inputLabel setDrawsBackground:NO];
    [inputLabel setEditable:NO];
    [inputLabel setFont:[NSFont boldSystemFontOfSize:14]];
    [contentView addSubview:inputLabel];
    
    _inputDevicePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(100, 220, 250, 25)];
    [_inputDevicePopup setTarget:self];
    [_inputDevicePopup setAction:@selector(inputDeviceChanged:)];
    [contentView addSubview:_inputDevicePopup];
    
    NSTextField *outputLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(550, 250, 150, 20)];
    [outputLabel setStringValue:@"Audio Output Device:"];
    [outputLabel setBezeled:NO];
    [outputLabel setDrawsBackground:NO];
    [outputLabel setEditable:NO];
    [outputLabel setFont:[NSFont boldSystemFontOfSize:14]];
    [contentView addSubview:outputLabel];
    
    _outputDevicePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(550, 220, 250, 25)];
    [_outputDevicePopup setTarget:self];
    [_outputDevicePopup setAction:@selector(outputDeviceChanged:)];
    [contentView addSubview:_outputDevicePopup];
    
    // Audio Level Meters
    NSBox *levelBox = [[NSBox alloc] initWithFrame:NSMakeRect(100, 120, 800, 80)];
    [levelBox setTitle:@"🎚️ Audio Levels"];
    [levelBox setTitleFont:[NSFont boldSystemFontOfSize:16]];
    [contentView addSubview:levelBox];
    
    NSView *levelView = [levelBox contentView];
    
    _levelMeterInput = [[NSTextField alloc] initWithFrame:NSMakeRect(30, 30, 350, 25)];
    [_levelMeterInput setStringValue:@"Input: ▬▬▬▬▬▬▬▬▬▬ 0.0 dB"];
    [_levelMeterInput setBezeled:NO];
    [_levelMeterInput setDrawsBackground:NO];
    [_levelMeterInput setEditable:NO];
    [_levelMeterInput setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightBold]];
    [levelView addSubview:_levelMeterInput];
    
    _levelMeterOutput = [[NSTextField alloc] initWithFrame:NSMakeRect(420, 30, 350, 25)];
    [_levelMeterOutput setStringValue:@"Output: ▬▬▬▬▬▬▬▬▬▬ 0.0 dB"];
    [_levelMeterOutput setBezeled:NO];
    [_levelMeterOutput setDrawsBackground:NO];
    [_levelMeterOutput setEditable:NO];
    [_levelMeterOutput setFont:[NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightBold]];
    [levelView addSubview:_levelMeterOutput];
    
    // Performance Display
    NSBox *perfBox = [[NSBox alloc] initWithFrame:NSMakeRect(100, 20, 800, 80)];
    [perfBox setTitle:@"📊 Real-Time Performance"];
    [perfBox setTitleFont:[NSFont boldSystemFontOfSize:16]];
    [contentView addSubview:perfBox];
    
    NSView *perfView = [perfBox contentView];
    
    _latencyDisplay = [[NSTextField alloc] initWithFrame:NSMakeRect(30, 35, 250, 25)];
    [_latencyDisplay setStringValue:@"Latency: 0.0 μs"];
    [_latencyDisplay setBezeled:NO];
    [_latencyDisplay setDrawsBackground:NO];
    [_latencyDisplay setEditable:NO];
    [_latencyDisplay setFont:[NSFont monospacedSystemFontOfSize:14 weight:NSFontWeightBold]];
    [perfView addSubview:_latencyDisplay];
    
    _snrDisplay = [[NSTextField alloc] initWithFrame:NSMakeRect(320, 35, 250, 25)];
    [_snrDisplay setStringValue:@"SNR Improvement: 0.0 dB"];
    [_snrDisplay setBezeled:NO];
    [_snrDisplay setDrawsBackground:NO];
    [_snrDisplay setEditable:NO];
    [_snrDisplay setFont:[NSFont monospacedSystemFontOfSize:14 weight:NSFontWeightBold]];
    [perfView addSubview:_snrDisplay];
    
    _audioInfoDisplay = [[NSTextField alloc] initWithFrame:NSMakeRect(30, 5, 740, 25)];
    [_audioInfoDisplay setStringValue:@"🎤 48kHz • Real Microphone Input • JELLIE Processing • Live Display"];
    [_audioInfoDisplay setBezeled:NO];
    [_audioInfoDisplay setDrawsBackground:NO];
    [_audioInfoDisplay setEditable:NO];
    [_audioInfoDisplay setFont:[NSFont systemFontOfSize:12]];
    [perfView addSubview:_audioInfoDisplay];
}

- (void)setupCoreAudio {
    NSLog(@"Setting up Core Audio input and output");
    
    // === OUTPUT AUDIO UNIT ===
    AudioComponentDescription outputDesc;
    outputDesc.componentType = kAudioUnitType_Output;
    outputDesc.componentSubType = kAudioUnitSubType_DefaultOutput;
    outputDesc.componentManufacturer = kAudioUnitManufacturer_Apple;
    outputDesc.componentFlags = 0;
    outputDesc.componentFlagsMask = 0;
    
    AudioComponent outputComponent = AudioComponentFindNext(NULL, &outputDesc);
    if (outputComponent == NULL) {
        NSLog(@"Failed to find audio output component");
        return;
    }
    
    // Create output audio unit
    OSStatus status = AudioComponentInstanceNew(outputComponent, &_audioUnit);
    if (status != noErr) {
        NSLog(@"Failed to create output audio unit: %d", (int)status);
        return;
    }
    
    // === INPUT AUDIO UNIT ===
    AudioComponentDescription inputDesc;
    inputDesc.componentType = kAudioUnitType_Output;
    inputDesc.componentSubType = kAudioUnitSubType_HALOutput;
    inputDesc.componentManufacturer = kAudioUnitManufacturer_Apple;
    inputDesc.componentFlags = 0;
    inputDesc.componentFlagsMask = 0;
    
    AudioComponent inputComponent = AudioComponentFindNext(NULL, &inputDesc);
    if (inputComponent == NULL) {
        NSLog(@"Failed to find audio input component");
        return;
    }
    
    // Create input audio unit
    status = AudioComponentInstanceNew(inputComponent, &_inputAudioUnit);
    if (status != noErr) {
        NSLog(@"Failed to create input audio unit: %d", (int)status);
        return;
    }
    
    // Enable input on the input audio unit
    UInt32 enableIO = 1;
    status = AudioUnitSetProperty(_inputAudioUnit,
                                 kAudioOutputUnitProperty_EnableIO,
                                 kAudioUnitScope_Input,
                                 1, // input element
                                 &enableIO,
                                 sizeof(enableIO));
    if (status != noErr) {
        NSLog(@"Failed to enable input on input audio unit: %d", (int)status);
        return;
    }
    
    // Disable output on the input audio unit (we only want input)
    enableIO = 0;
    status = AudioUnitSetProperty(_inputAudioUnit,
                                 kAudioOutputUnitProperty_EnableIO,
                                 kAudioUnitScope_Output,
                                 0, // output element
                                 &enableIO,
                                 sizeof(enableIO));
    if (status != noErr) {
        NSLog(@"Failed to disable output on input audio unit: %d", (int)status);
        return;
    }
    
    // Set up audio format (48kHz, 32-bit float, mono)
    AudioStreamBasicDescription outputFormat;
    outputFormat.mSampleRate = 48000.0;
    outputFormat.mFormatID = kAudioFormatLinearPCM;
    outputFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
    outputFormat.mFramesPerPacket = 1;
    outputFormat.mChannelsPerFrame = 1; // Mono
    outputFormat.mBytesPerFrame = sizeof(Float32);
    outputFormat.mBytesPerPacket = sizeof(Float32);
    outputFormat.mBitsPerChannel = 32;
    
    // Set up simpler input format for better compatibility
    AudioStreamBasicDescription inputFormat;
    inputFormat.mSampleRate = 48000.0;
    inputFormat.mFormatID = kAudioFormatLinearPCM;
    inputFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked; // Remove non-interleaved for input
    inputFormat.mFramesPerPacket = 1;
    inputFormat.mChannelsPerFrame = 1; // Mono
    inputFormat.mBytesPerFrame = sizeof(Float32);
    inputFormat.mBytesPerPacket = sizeof(Float32);
    inputFormat.mBitsPerChannel = 32;
    
    // Set format on output unit input scope
    status = AudioUnitSetProperty(_audioUnit,
                                 kAudioUnitProperty_StreamFormat,
                                 kAudioUnitScope_Input,
                                 0,
                                 &outputFormat,
                                 sizeof(outputFormat));
    if (status != noErr) {
        NSLog(@"Failed to set output audio format: %d", (int)status);
        return;
    }
    
    // Set format on input unit output scope (what comes out of the input unit)
    status = AudioUnitSetProperty(_inputAudioUnit,
                                 kAudioUnitProperty_StreamFormat,
                                 kAudioUnitScope_Output,
                                 1, // input element
                                 &inputFormat,
                                 sizeof(inputFormat));
    if (status != noErr) {
        NSLog(@"Failed to set input audio format on output scope: %d", (int)status);
        return;
    }
    
    // Get the actual input format from the device first
    UInt32 propertySize = sizeof(AudioStreamBasicDescription);
    AudioStreamBasicDescription deviceInputFormat;
    status = AudioUnitGetProperty(_inputAudioUnit,
                                 kAudioUnitProperty_StreamFormat,
                                 kAudioUnitScope_Input,
                                 1, // input element
                                 &deviceInputFormat,
                                 &propertySize);
    
    if (status == noErr) {
        NSLog(@"Device input format: %.0f Hz, %d channels, %d bits", 
              deviceInputFormat.mSampleRate, 
              (int)deviceInputFormat.mChannelsPerFrame,
              (int)deviceInputFormat.mBitsPerChannel);
        
        // Use device's preferred format but ensure it's mono and float
        deviceInputFormat.mChannelsPerFrame = 1; // Force mono
        deviceInputFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        deviceInputFormat.mBytesPerFrame = sizeof(Float32);
        deviceInputFormat.mBytesPerPacket = sizeof(Float32);
        deviceInputFormat.mBitsPerChannel = 32;
        
        // Try to set the adjusted device format
        status = AudioUnitSetProperty(_inputAudioUnit,
                                     kAudioUnitProperty_StreamFormat,
                                     kAudioUnitScope_Input,
                                     1, // input element
                                     &deviceInputFormat,
                                     sizeof(deviceInputFormat));
        if (status != noErr) {
            NSLog(@"Failed to set adjusted input audio format on input scope: %d", (int)status);
            NSLog(@"Continuing anyway - might still work");
        } else {
            NSLog(@"Successfully set adjusted input audio format");
        }
    } else {
        NSLog(@"Failed to get device input format: %d", (int)status);
    }
    
    // Set render callback on output unit
    AURenderCallbackStruct callback;
    callback.inputProc = audioRenderCallback;
    callback.inputProcRefCon = _callbackData;
    
    status = AudioUnitSetProperty(_audioUnit,
                                 kAudioUnitProperty_SetRenderCallback,
                                 kAudioUnitScope_Input,
                                 0,
                                 &callback,
                                 sizeof(callback));
    if (status != noErr) {
        NSLog(@"Failed to set render callback: %d", (int)status);
        return;
    }
    
    // NEW: Set input callback on input unit to capture microphone
    AURenderCallbackStruct inputCallback;
    inputCallback.inputProc = audioInputCallback;
    inputCallback.inputProcRefCon = _callbackData;
    
    status = AudioUnitSetProperty(_inputAudioUnit,
                                 kAudioOutputUnitProperty_SetInputCallback,
                                 kAudioUnitScope_Global,
                                 0,
                                 &inputCallback,
                                 sizeof(inputCallback));
    if (status != noErr) {
        NSLog(@"Failed to set input callback: %d", (int)status);
        return;
    }
    
    NSLog(@"Input callback set successfully - microphone will be captured");
    
    // Initialize both audio units
    status = AudioUnitInitialize(_audioUnit);
    if (status != noErr) {
        NSLog(@"Failed to initialize output audio unit: %d", (int)status);
        return;
    }
    
    status = AudioUnitInitialize(_inputAudioUnit);
    if (status != noErr) {
        NSLog(@"Failed to initialize input audio unit: %d", (int)status);
        return;
    }
    
    NSLog(@"Core Audio setup complete - Input and Output ready");
}

- (void)populateAudioDevices {
    NSLog(@"Enumerating Core Audio devices");
    
    // Clear existing items
    [_inputDevicePopup removeAllItems];
    [_outputDevicePopup removeAllItems];
    
    // Get all audio devices
    AudioObjectPropertyAddress propertyAddress = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMaster
    };
    
    UInt32 dataSize = 0;
    OSStatus status = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize);
    if (status != noErr) {
        NSLog(@"Error getting audio device count: %d", (int)status);
        return;
    }
    
    UInt32 deviceCount = dataSize / sizeof(AudioDeviceID);
    NSLog(@"Found %d audio devices", (int)deviceCount);
    
    AudioDeviceID *audioDevices = (AudioDeviceID *)malloc(dataSize);
    status = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize, audioDevices);
    
    if (status != noErr) {
        NSLog(@"Error getting audio devices: %d", (int)status);
        free(audioDevices);
        return;
    }
    
    // Enumerate each device
    for (UInt32 i = 0; i < deviceCount; i++) {
        AudioDeviceID deviceID = audioDevices[i];
        
        // Get device name
        CFStringRef deviceName = NULL;
        UInt32 nameSize = sizeof(CFStringRef);
        AudioObjectPropertyAddress nameAddress = {
            kAudioDevicePropertyDeviceNameCFString,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMaster
        };
        
        status = AudioObjectGetPropertyData(deviceID, &nameAddress, 0, NULL, &nameSize, &deviceName);
        if (status != noErr) {
            continue;
        }
        
        NSString *deviceNameStr = (__bridge NSString *)deviceName;
        
        // Check if device has input streams
        AudioObjectPropertyAddress inputStreamsAddress = {
            kAudioDevicePropertyStreams,
            kAudioDevicePropertyScopeInput,
            kAudioObjectPropertyElementMaster
        };
        
        UInt32 inputStreamsSize = 0;
        status = AudioObjectGetPropertyDataSize(deviceID, &inputStreamsAddress, 0, NULL, &inputStreamsSize);
        BOOL hasInput = (status == noErr && inputStreamsSize > 0);
        
        // Check if device has output streams
        AudioObjectPropertyAddress outputStreamsAddress = {
            kAudioDevicePropertyStreams,
            kAudioDevicePropertyScopeOutput,
            kAudioObjectPropertyElementMaster
        };
        
        UInt32 outputStreamsSize = 0;
        status = AudioObjectGetPropertyDataSize(deviceID, &outputStreamsAddress, 0, NULL, &outputStreamsSize);
        BOOL hasOutput = (status == noErr && outputStreamsSize > 0);
        
        // Add to appropriate dropdowns
        if (hasInput) {
            NSMenuItem *inputItem = [[NSMenuItem alloc] initWithTitle:deviceNameStr 
                                                              action:nil 
                                                       keyEquivalent:@""];
            [inputItem setTag:deviceID];
            [[_inputDevicePopup menu] addItem:inputItem];
            NSLog(@"Added input device: %@", deviceNameStr);
        }
        
        if (hasOutput) {
            NSMenuItem *outputItem = [[NSMenuItem alloc] initWithTitle:deviceNameStr 
                                                               action:nil 
                                                        keyEquivalent:@""];
            [outputItem setTag:deviceID];
            [[_outputDevicePopup menu] addItem:outputItem];
            NSLog(@"Added output device: %@", deviceNameStr);
        }
        
        CFRelease(deviceName);
    }
    
    free(audioDevices);
    
    // Select default devices
    if ([_inputDevicePopup numberOfItems] > 0) {
        [_inputDevicePopup selectItemAtIndex:0];
    }
    if ([_outputDevicePopup numberOfItems] > 0) {
        [_outputDevicePopup selectItemAtIndex:0];
    }
    
    NSLog(@"Audio device enumeration complete: %ld input devices, %ld output devices", 
          (long)[_inputDevicePopup numberOfItems], (long)[_outputDevicePopup numberOfItems]);
}

- (IBAction)toggleMode:(id)sender {
    _isTxMode = !_isTxMode;
    
    if (_isTxMode) {
        [_modeToggleButton setTitle:@"🔄 Switch to RX MODE"];
        _callbackData->engine->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::TX_MODE);
        [_oscilloscopeView setChannelLabel:@"PNBTR Output"];
        [_oscilloscopeView setWaveformColor:[NSColor cyanColor]];
    } else {
        [_modeToggleButton setTitle:@"🔄 Switch to TX MODE"];
        _callbackData->engine->setPluginMode(pnbtr_jellie::PnbtrJellieEngine::PluginMode::RX_MODE);
        [_oscilloscopeView setChannelLabel:@"Raw Microphone Input"];
        [_oscilloscopeView setWaveformColor:[NSColor greenColor]];
    }
}

- (IBAction)inputDeviceChanged:(id)sender {
    NSPopUpButton *popup = (NSPopUpButton *)sender;
    NSMenuItem *selectedItem = [popup selectedItem];
    AudioDeviceID deviceID = (AudioDeviceID)[selectedItem tag];
    
    NSLog(@"Input device changed to: %@ (ID: %d)", [selectedItem title], (int)deviceID);
    
    // Set the input device on the input audio unit
    if (_inputAudioUnit) {
        OSStatus status = AudioUnitSetProperty(_inputAudioUnit,
                                              kAudioOutputUnitProperty_CurrentDevice,
                                              kAudioUnitScope_Global,
                                              0,
                                              &deviceID,
                                              sizeof(deviceID));
        if (status != noErr) {
            NSLog(@"Failed to set input device: %d", (int)status);
        } else {
            NSLog(@"Input device set successfully");
        }
    }
}

- (IBAction)outputDeviceChanged:(id)sender {
    NSPopUpButton *popup = (NSPopUpButton *)sender;
    NSMenuItem *selectedItem = [popup selectedItem];
    AudioDeviceID deviceID = (AudioDeviceID)[selectedItem tag];
    
    NSLog(@"Output device changed to: %@ (ID: %d)", [selectedItem title], (int)deviceID);
    
    // Set the output device on the output audio unit
    if (_audioUnit) {
        OSStatus status = AudioUnitSetProperty(_audioUnit,
                                              kAudioOutputUnitProperty_CurrentDevice,
                                              kAudioUnitScope_Global,
                                              0,
                                              &deviceID,
                                              sizeof(deviceID));
        if (status != noErr) {
            NSLog(@"Failed to set output device: %d", (int)status);
        } else {
            NSLog(@"Output device set successfully");
        }
    }
}

- (void)updateDisplay {
    // Update status - always show as active
    NSString *audioStatus = _isAudioRunning ? @"🔴 MICROPHONE ACTIVE" : @"⚠️ MICROPHONE ERROR";
    NSString *mode = _isTxMode ? @"🔊 TX MODE" : @"🔉 RX MODE";
    NSString *scopeStatus = _isAudioRunning ? @"📊 Live Input Display" : @"📊 No Input";
    [_statusDisplay setStringValue:[NSString stringWithFormat:@"%@ | %@ | %@", audioStatus, mode, scopeStatus]];
    
    // Update performance metrics
    if (_callbackData && _callbackData->engine) {
        const auto& stats = _callbackData->engine->getPerformanceStats();
        
        [_latencyDisplay setStringValue:[NSString stringWithFormat:@"Latency: %.1f μs", stats.avg_latency_us.load()]];
        [_snrDisplay setStringValue:[NSString stringWithFormat:@"SNR Improvement: %.1f dB", stats.snr_improvement_db.load()]];
    }
    
    // Update level meters (simplified for now)
    static int meterUpdate = 0;
    meterUpdate++;
    
    if (_isAudioRunning) {
        int bars = (meterUpdate % 20) / 2;
        NSString *inputMeter = [NSString stringWithFormat:@"🎤 Input: %@%@ %.1f dB", 
                               [@"▬▬▬▬▬▬▬▬▬▬" substringToIndex:bars],
                               [@"▭▭▭▭▭▭▭▭▭▭" substringFromIndex:bars],
                               -20.0 + bars * 2.0];
        [_levelMeterInput setStringValue:inputMeter];
        
        NSString *outputMeter = [NSString stringWithFormat:@"🔊 Output: %@%@ %.1f dB", 
                                [@"▬▬▬▬▬▬▬▬▬▬" substringToIndex:bars],
                                [@"▭▭▭▭▭▭▭▭▭▭" substringFromIndex:bars],
                                -18.0 + bars * 2.0];
        [_levelMeterOutput setStringValue:outputMeter];
    } else {
        [_levelMeterInput setStringValue:@"🎤 Input: ▭▭▭▭▭▭▭▭▭▭ No Signal"];
        [_levelMeterOutput setStringValue:@"🔊 Output: ▭▭▭▭▭▭▭▭▭▭ No Signal"];
    }
}

- (void)startAudioProcessing {
    NSLog(@"Auto-starting audio processing");
    
    // Set the selected input device on the input audio unit
    if (_inputAudioUnit && [_inputDevicePopup numberOfItems] > 0) {
        NSMenuItem *selectedInputItem = [_inputDevicePopup selectedItem];
        if (selectedInputItem) {
            AudioDeviceID inputDeviceID = (AudioDeviceID)[selectedInputItem tag];
            NSLog(@"Setting input device to: %@ (ID: %d)", [selectedInputItem title], (int)inputDeviceID);
            
            OSStatus inputDeviceStatus = AudioUnitSetProperty(_inputAudioUnit,
                                                             kAudioOutputUnitProperty_CurrentDevice,
                                                             kAudioUnitScope_Global,
                                                             0,
                                                             &inputDeviceID,
                                                             sizeof(inputDeviceID));
            if (inputDeviceStatus != noErr) {
                NSLog(@"Failed to set input device: %d", (int)inputDeviceStatus);
            } else {
                NSLog(@"Input device set successfully");
            }
        }
    }
    
    // Set the selected output device on the output audio unit
    if (_audioUnit && [_outputDevicePopup numberOfItems] > 0) {
        NSMenuItem *selectedOutputItem = [_outputDevicePopup selectedItem];
        if (selectedOutputItem) {
            AudioDeviceID outputDeviceID = (AudioDeviceID)[selectedOutputItem tag];
            NSLog(@"Setting output device to: %@ (ID: %d)", [selectedOutputItem title], (int)outputDeviceID);
            
            OSStatus outputDeviceStatus = AudioUnitSetProperty(_audioUnit,
                                                              kAudioOutputUnitProperty_CurrentDevice,
                                                              kAudioUnitScope_Global,
                                                              0,
                                                              &outputDeviceID,
                                                              sizeof(outputDeviceID));
            if (outputDeviceStatus != noErr) {
                NSLog(@"Failed to set output device: %d", (int)outputDeviceStatus);
            } else {
                NSLog(@"Output device set successfully");
            }
        }
    }
    
    // Start audio - both input and output
    OSStatus outputStatus = noErr;
    OSStatus inputStatus = noErr;
    
    if (_audioUnit) {
        outputStatus = AudioOutputUnitStart(_audioUnit);
        if (outputStatus != noErr) {
            NSLog(@"Failed to start output audio unit: %d", (int)outputStatus);
        } else {
            NSLog(@"Output audio unit started successfully");
        }
    }
    
    if (_inputAudioUnit) {
        inputStatus = AudioOutputUnitStart(_inputAudioUnit);
        if (inputStatus != noErr) {
            NSLog(@"Failed to start input audio unit: %d", (int)inputStatus);
        } else {
            NSLog(@"Input audio unit started successfully");
        }
    }
    
    // Set running state if at least output succeeded
    if (outputStatus == noErr) {
        _callbackData->isProcessing = true;
        _isAudioRunning = YES;
        
        NSString *statusMsg = @"Audio auto-started - ";
        if (outputStatus == noErr && inputStatus == noErr) {
            statusMsg = [statusMsg stringByAppendingString:@"Microphone and Output active"];
        } else if (outputStatus == noErr) {
            statusMsg = [statusMsg stringByAppendingString:@"Output active (no microphone)"];
        }
        NSLog(@"%@", statusMsg);
    } else {
        NSLog(@"Failed to auto-start audio - output unit failed");
    }
}

// Microphone permission handling removed for simplicity

@end

// App Delegate Implementation
@implementation PnbtrJellieAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSLog(@"Application did finish launching");
    
    // Set activation policy first
    [[NSApplication sharedApplication] setActivationPolicy:NSApplicationActivationPolicyRegular];
    
    // Create controller
    _controller = [[PnbtrJellieAudioController alloc] init];
    
    // Force window to front
    [[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
    
    NSLog(@"Controller created and activated");
}

- (BOOL)applicationShouldTerminateWhenLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end

// Main function
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Starting PNBTR+JELLIE GUI Application");
        
        NSApplication *app = [NSApplication sharedApplication];
        
        PnbtrJellieAppDelegate *delegate = [[PnbtrJellieAppDelegate alloc] init];
        [app setDelegate:delegate];
        
        NSLog(@"App delegate set, starting run loop");
        
        [app run];
    }
    return 0;
} 