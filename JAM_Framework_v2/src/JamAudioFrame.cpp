#include "JamAudioFrame.h"
#include <cstring>

namespace JAMNet {

bool SharedAudioBuffer::pushFrame(const JamAudioFrame& frame) {
    uint32_t currentWrite = writeIndex_.load();
    uint32_t nextWrite = (currentWrite + 1) % RING_BUFFER_SIZE;
    
    // Check if buffer is full
    if (nextWrite == readIndex_.load()) {
        return false; // Buffer full
    }
    
    // Copy frame data
    frames_[currentWrite] = frame;
    
    // Update write index atomically
    writeIndex_.store(nextWrite);
    
    return true;
}

bool SharedAudioBuffer::popFrame(JamAudioFrame& frame) {
    uint32_t currentRead = readIndex_.load();
    
    // Check if buffer is empty
    if (currentRead == writeIndex_.load()) {
        return false; // Buffer empty
    }
    
    // Copy frame data
    frame = frames_[currentRead];
    
    // Update read index atomically
    uint32_t nextRead = (currentRead + 1) % RING_BUFFER_SIZE;
    readIndex_.store(nextRead);
    
    return true;
}

bool SharedAudioBuffer::isEmpty() const {
    return readIndex_.load() == writeIndex_.load();
}

bool SharedAudioBuffer::isFull() const {
    uint32_t nextWrite = (writeIndex_.load() + 1) % RING_BUFFER_SIZE;
    return nextWrite == readIndex_.load();
}

uint32_t SharedAudioBuffer::getAvailableFrames() const {
    uint32_t write = writeIndex_.load();
    uint32_t read = readIndex_.load();
    
    if (write >= read) {
        return write - read;
    } else {
        return (RING_BUFFER_SIZE - read) + write;
    }
}

void SharedAudioBuffer::flush() {
    readIndex_.store(writeIndex_.load());
}

} // namespace JAMNet
