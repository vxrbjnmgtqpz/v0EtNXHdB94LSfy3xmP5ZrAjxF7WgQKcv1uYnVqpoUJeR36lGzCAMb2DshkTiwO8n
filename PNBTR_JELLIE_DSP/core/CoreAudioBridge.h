#pragma once

#include "RingBuffer.h"
#include "MetalBridge.h"

struct CoreAudioBridge {
    RingBuffer<float>* ringBuffer;
    MetalBridge* metalBridge;
    void* internal; // Opaque pointer to internal Obj-C state
};

CoreAudioBridge* createCoreAudioBridge(RingBuffer<float>* rb, MetalBridge* mb);
void destroyCoreAudioBridge(CoreAudioBridge* bridge);

void startAudio(CoreAudioBridge* bridge);
void stopAudio(CoreAudioBridge* bridge); 