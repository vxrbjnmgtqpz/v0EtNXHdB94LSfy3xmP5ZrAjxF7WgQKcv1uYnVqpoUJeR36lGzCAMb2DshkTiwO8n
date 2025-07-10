#import <Cocoa/Cocoa.h>
#import <CoreAudio/CoreAudio.h>
#import <AudioUnit/AudioUnit.h>
#include "training_testbed.h"
#include <iostream>
#include <vector>
#include <string>

// GPU-Native Training Testbed GUI - Core Audio + Cocoa Implementation
// All DSP processing happens in the GPU-native backend, GUI is just for control

@interface TrainingTestbedController : NSObject <NSApplicationDelegate, NSWindowDelegate>
{
@public
    NSWindow* window;
    NSButton* startButton;
    NSButton* stopButton;
    NSButton* exportButton;
    NSPopUpButton* deviceSelector;
    NSTextView* statusLog;
    NSLevelIndicator* levelMeter;
    
    // Integrated VST3 engine components
    TrainingTestbed* testbed;
    bool isCapturing;
    bool isInitialized;
    
    // Core Audio components
    AudioUnit audioUnit;
    AudioDeviceID selectedDevice;
    std::vector<AudioDeviceID> availableDevices;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification;
- (void)setupWindow;
- (void)setupCoreAudio;
- (void)populateDeviceList;
- (IBAction)startCapture:(id)sender;
- (IBAction)stopCapture:(id)sender;
- (IBAction)exportTrainingData:(id)sender;
- (IBAction)deviceChanged:(id)sender;
- (void)logStatus:(NSString*)message;

@end

// Core Audio callback - routes audio to GPU-native backend
OSStatus audioInputCallback(void* inRefCon,
                           AudioUnitRenderActionFlags* ioActionFlags,
                           const AudioTimeStamp* inTimeStamp,
                           UInt32 inBusNumber,
                           UInt32 inNumberFrames,
                           AudioBufferList* ioData) {
    TrainingTestbedController* controller = (__bridge TrainingTestbedController*)inRefCon;
    // Allocate audio buffer
    AudioBufferList* bufferList = (AudioBufferList*)malloc(sizeof(AudioBufferList) + sizeof(AudioBuffer));
    bufferList->mNumberBuffers = 1;
    bufferList->mBuffers[0].mNumberChannels = 2;
    bufferList->mBuffers[0].mDataByteSize = inNumberFrames * 2 * sizeof(float);
    bufferList->mBuffers[0].mData = malloc(bufferList->mBuffers[0].mDataByteSize);
    // Get audio from Core Audio
    OSStatus result = AudioUnitRender(controller->audioUnit, ioActionFlags, inTimeStamp, 
                                     inBusNumber, inNumberFrames, bufferList);
    if (result == noErr && controller->isCapturing && controller->backendPipeline) {
        float* audioData = (float*)bufferList->mBuffers[0].mData;
        // Convert to std::vector<float>
        std::vector<float> audioVec(audioData, audioData + inNumberFrames * 2);
        std::vector<uint8_t> encoded_output;
        // Route audio to backend pipeline (real-time safe)
        controller->backendPipeline->processAudioForTransmission(audioVec, encoded_output);
        // Update level meter on main thread
        dispatch_async(dispatch_get_main_queue(), ^{
            float level = 0.0f;
            for (UInt32 i = 0; i < inNumberFrames * 2; i++) {
                level += fabsf(audioData[i]);
            }
            level /= (inNumberFrames * 2);
            [controller->levelMeter setFloatValue:level * 100.0f];
        });
    }
    free(bufferList->mBuffers[0].mData);
    free(bufferList);
    return result;
}

@implementation TrainingTestbedController

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
    // Initialize with proven VST3 components
    TrainingTestbed::Config config;
    config.sample_rate = 48000;
    config.channels = 2;
    config.enable_training_data = true;
    config.mode = pnbtr_jellie::PnbtrJellieEngine::OperationalMode::TRAINING_DATA_COLLECTION;
    
    testbed = new TrainingTestbed(config);
    if (!testbed->initialize()) {
        [self logStatus:@"❌ Failed to initialize VST3 engine"];
        return;
    }
    
    isCapturing = false;
    isInitialized = false;
    [self setupWindow];
    [self setupCoreAudio];
    [self populateDeviceList];
    [self logStatus:@"🎛️ PNBTR+JELLIE Training Testbed with VST3 Engine Ready"];
    [self logStatus:@"✅ Proven VST3 components integrated"];
    [self logStatus:@"🧠 Mode: Training Data Collection"];
    [self logStatus:@"Select audio input device and start capture..."];
    isInitialized = true;
}

- (void)setupWindow {
    // Create main window
    NSRect windowFrame = NSMakeRect(100, 100, 600, 400);
    window = [[NSWindow alloc] initWithContentRect:windowFrame
                                         styleMask:(NSWindowStyleMaskTitled | 
                                                   NSWindowStyleMaskClosable | 
                                                   NSWindowStyleMaskMiniaturizable | 
                                                   NSWindowStyleMaskResizable)
                                           backing:NSBackingStoreBuffered
                                             defer:NO];
    [window setTitle:@"PNBTR+JELLIE Training Testbed"];
    [window setDelegate:self];
    
    // Create controls
    deviceSelector = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(20, 350, 300, 30)];
    [deviceSelector setTarget:self];
    [deviceSelector setAction:@selector(deviceChanged:)];
    
    startButton = [[NSButton alloc] initWithFrame:NSMakeRect(20, 310, 120, 30)];
    [startButton setTitle:@"Start Capture"];
    [startButton setTarget:self];
    [startButton setAction:@selector(startCapture:)];
    
    stopButton = [[NSButton alloc] initWithFrame:NSMakeRect(150, 310, 120, 30)];
    [stopButton setTitle:@"Stop Capture"];
    [stopButton setTarget:self];
    [stopButton setAction:@selector(stopCapture:)];
    [stopButton setEnabled:NO];
    
    exportButton = [[NSButton alloc] initWithFrame:NSMakeRect(280, 310, 150, 30)];
    [exportButton setTitle:@"Export Training Data"];
    [exportButton setTarget:self];
    [exportButton setAction:@selector(exportTrainingData:)];
    
    levelMeter = [[NSLevelIndicator alloc] initWithFrame:NSMakeRect(450, 310, 120, 30)];
    [levelMeter setMinValue:0.0];
    [levelMeter setMaxValue:100.0];
    [levelMeter setLevelIndicatorStyle:NSLevelIndicatorStyleContinuousCapacity];
    
    // Status log
    NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:NSMakeRect(20, 20, 560, 280)];
    statusLog = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, 560, 280)];
    [statusLog setEditable:NO];
    [statusLog setFont:[NSFont fontWithName:@"Monaco" size:11]];
    [scrollView setDocumentView:statusLog];
    [scrollView setHasVerticalScroller:YES];
    
    // Add to window
    [[window contentView] addSubview:deviceSelector];
    [[window contentView] addSubview:startButton];
    [[window contentView] addSubview:stopButton];
    [[window contentView] addSubview:exportButton];
    [[window contentView] addSubview:levelMeter];
    [[window contentView] addSubview:scrollView];
    
    [window makeKeyAndOrderFront:nil];
}

- (void)setupCoreAudio {
    OSStatus result;
    
    // Create Audio Unit
    AudioComponentDescription desc;
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_HALOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;
    
    AudioComponent component = AudioComponentFindNext(NULL, &desc);
    result = AudioComponentInstanceNew(component, &audioUnit);
    
    if (result != noErr) {
        [self logStatus:@"❌ Failed to create Audio Unit"];
        return;
    }
    
    // Enable input
    UInt32 enableInput = 1;
    result = AudioUnitSetProperty(audioUnit, kAudioOutputUnitProperty_EnableIO,
                                 kAudioUnitScope_Input, 1, &enableInput, sizeof(enableInput));
    
    // Disable output
    UInt32 disableOutput = 0;
    result = AudioUnitSetProperty(audioUnit, kAudioOutputUnitProperty_EnableIO,
                                 kAudioUnitScope_Output, 0, &disableOutput, sizeof(disableOutput));
    
    // Set input callback
    AURenderCallbackStruct callbackStruct;
    callbackStruct.inputProc = audioInputCallback;
    callbackStruct.inputProcRefCon = (__bridge void*)self;
    
    result = AudioUnitSetProperty(audioUnit, kAudioOutputUnitProperty_SetInputCallback,
                                 kAudioUnitScope_Global, 0, &callbackStruct, sizeof(callbackStruct));
    
    // Set audio format
    AudioStreamBasicDescription format;
    format.mSampleRate = 48000.0;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagsNativeFloatPacked;
    format.mBytesPerPacket = 8;
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = 8;
    format.mChannelsPerFrame = 2;
    format.mBitsPerChannel = 32;
    
    result = AudioUnitSetProperty(audioUnit, kAudioUnitProperty_StreamFormat,
                                 kAudioUnitScope_Output, 1, &format, sizeof(format));
    
    result = AudioUnitInitialize(audioUnit);
    
    if (result == noErr) {
        [self logStatus:@"✅ Core Audio initialized"];
    } else {
        [self logStatus:@"❌ Core Audio initialization failed"];
    }
}

- (void)populateDeviceList {
    // Get audio devices
    AudioObjectPropertyAddress propertyAddress;
    propertyAddress.mSelector = kAudioHardwarePropertyDevices;
    propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
    propertyAddress.mElement = kAudioObjectPropertyElementMaster;
    
    UInt32 dataSize = 0;
    AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize);
    
    UInt32 deviceCount = dataSize / sizeof(AudioDeviceID);
    AudioDeviceID* devices = (AudioDeviceID*)malloc(dataSize);
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &dataSize, devices);
    
    [deviceSelector removeAllItems];
    availableDevices.clear();
    
    for (UInt32 i = 0; i < deviceCount; i++) {
        // Get device name
        propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString;
        CFStringRef deviceName;
        dataSize = sizeof(deviceName);
        
        OSStatus result = AudioObjectGetPropertyData(devices[i], &propertyAddress, 0, NULL, &dataSize, &deviceName);
        
        if (result == noErr) {
            // Check if device has input
            propertyAddress.mSelector = kAudioDevicePropertyStreamConfiguration;
            propertyAddress.mScope = kAudioDevicePropertyScopeInput;
            
            AudioObjectGetPropertyDataSize(devices[i], &propertyAddress, 0, NULL, &dataSize);
            AudioBufferList* bufferList = (AudioBufferList*)malloc(dataSize);
            AudioObjectGetPropertyData(devices[i], &propertyAddress, 0, NULL, &dataSize, bufferList);
            
            if (bufferList->mNumberBuffers > 0) {
                NSString* name = (__bridge NSString*)deviceName;
                [deviceSelector addItemWithTitle:name];
                availableDevices.push_back(devices[i]);
            }
            
            free(bufferList);
            CFRelease(deviceName);
        }
    }
    
    free(devices);
    
    if (availableDevices.size() > 0) {
        selectedDevice = availableDevices[0];
        [self logStatus:[NSString stringWithFormat:@"📡 Found %lu input devices", availableDevices.size()]];
    }
}

- (IBAction)startCapture:(id)sender {
    if (!isInitialized || isCapturing) return;
    // Set selected device
    OSStatus result = AudioUnitSetProperty(audioUnit, kAudioOutputUnitProperty_CurrentDevice,
                                          kAudioUnitScope_Global, 0, &selectedDevice, sizeof(selectedDevice));
    if (result != noErr) {
        [self logStatus:@"❌ Failed to set audio device"];
        return;
    }
    // Start backend pipeline
    if (!backendPipeline) {
        backendPipeline = new pnbtr_jellie::RealSignalTransmission();
        pnbtr_jellie::AudioSignalConfig audioConfig;
        audioConfig.sample_rate = 48000;
        audioConfig.channels = 2;
        audioConfig.use_live_input = true;
        audioConfig.use_test_signals = false;
        pnbtr_jellie::NetworkConditions netConfig;
        netConfig.packet_loss_percentage = 5.0;
        backendPipeline->initialize(audioConfig, netConfig);
        backendPipeline->enableDataCollection(true);
        [self logStatus:@"🧠 Backend pipeline initialized for live training data collection"];
    }
    // Start audio unit
    result = AudioOutputUnitStart(audioUnit);
    if (result == noErr) {
        isCapturing = true;
        [startButton setEnabled:NO];
        [stopButton setEnabled:YES];
        [self logStatus:@"🎤 Audio capture started - routing to GPU-native backend"];
    } else {
        [self logStatus:@"❌ Failed to start audio capture"];
    }
}

- (IBAction)stopCapture:(id)sender {
    if (!isCapturing) return;
    AudioOutputUnitStop(audioUnit);
    isCapturing = false;
    [startButton setEnabled:YES];
    [stopButton setEnabled:NO];
    [levelMeter setFloatValue:0.0f];
    [self logStatus:@"🛑 Audio capture stopped"];
    if (backendPipeline) {
        backendPipeline->stopTransmission();
        backendPipeline->shutdown();
        [self logStatus:@"💾 Training session data ready for export"];
    }
}

- (IBAction)exportTrainingData:(id)sender {
    [self logStatus:@"📤 Exporting training data to GPU-native backend..."];
    // Use backend APIs to export training data
    // This assumes logs are written to a known directory, e.g., "training_logs/"
    pnbtr_jellie::TrainingDataPreparator::PreparationConfig prepConfig;
    pnbtr_jellie::TrainingDataPreparator preparator(prepConfig);
    std::vector<std::string> session_ids = {"default_session"}; // TODO: make session id dynamic if needed
    preparator.loadLoggedData(session_ids, "training_logs/");
    preparator.extractFeatures();
    preparator.calculateTrainingTargets();
    preparator.labelDataWithNetworkConditions();
    auto split = preparator.createDataSplit();
    preparator.exportDataSplit(split, "training_output/");
    [self logStatus:@"✅ Training data exported to training_output/"];
    [self logStatus:@"🎯 Ready for Phase 5: Offline Training and Difference Analysis"];
}

- (IBAction)deviceChanged:(id)sender {
    NSInteger selectedIndex = [deviceSelector indexOfSelectedItem];
    if (selectedIndex >= 0 && selectedIndex < availableDevices.size()) {
        selectedDevice = availableDevices[selectedIndex];
        NSString* deviceName = [deviceSelector titleOfSelectedItem];
        [self logStatus:[NSString stringWithFormat:@"🎛️ Selected device: %@", deviceName]];
    }
}

- (void)logStatus:(NSString*)message {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSDate* now = [NSDate date];
        NSDateFormatter* formatter = [[NSDateFormatter alloc] init];
        [formatter setDateFormat:@"HH:mm:ss"];
        NSString* timestamp = [formatter stringFromDate:now];
        
        NSString* logEntry = [NSString stringWithFormat:@"[%@] %@\n", timestamp, message];
        
        NSAttributedString* attributed = [[NSAttributedString alloc] 
                                         initWithString:logEntry 
                                         attributes:@{NSForegroundColorAttributeName: [NSColor textColor]}];
        
        [[statusLog textStorage] appendAttributedString:attributed];
        [statusLog scrollToEndOfDocument:nil];
    });
}

- (void)dealloc {
    if (isCapturing) {
        AudioOutputUnitStop(audioUnit);
    }
    AudioUnitUninitialize(audioUnit);
    AudioComponentInstanceDispose(audioUnit);
    if (backendPipeline) {
        backendPipeline->shutdown();
        delete backendPipeline;
    }
}

@end

// C++ entry point for training testbed GUI
extern "C" {
    int launch_training_testbed_gui() {
        @autoreleasepool {
            NSApplication* app = [NSApplication sharedApplication];
            TrainingTestbedController* controller = [[TrainingTestbedController alloc] init];
            [app setDelegate:controller];
            [app run];
            return 0;
        }
    }
}
