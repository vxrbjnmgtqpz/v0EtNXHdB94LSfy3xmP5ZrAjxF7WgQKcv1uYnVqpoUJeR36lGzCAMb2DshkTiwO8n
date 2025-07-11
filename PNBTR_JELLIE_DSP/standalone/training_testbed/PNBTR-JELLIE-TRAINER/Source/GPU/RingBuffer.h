#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Defines the number of audio frames each buffer in the ring can hold.
constexpr size_t AUDIO_FRAME_SIZE = 512;
// Defines the number of stereo channels.
constexpr size_t AUDIO_CHANNEL_COUNT = 2;

/**
 * @struct AudioFrame
 * @brief Represents a single block of audio data with a timestamp.
 * This struct is used to pass audio data between the real-time audio thread
 * and the GPU processing thread.
 */
struct AudioFrame {
  // The host time at which the audio frame was captured.
  uint64_t hostTime;
  // The audio samples, organized by channel and sample index.
  float samples[AUDIO_CHANNEL_COUNT][AUDIO_FRAME_SIZE];
  // The number of valid samples in this frame.
  uint32_t sample_count;
};

/**
 * @class LockFreeRingBuffer
 * @brief A lock-free, single-producer, single-consumer ring buffer for AudioFrames.
 *
 * This class is designed for real-time audio applications to safely pass data
 * between a high-priority audio thread (the producer) and a lower-priority
 * processing thread (the consumer) without using locks, which could cause priority
 * inversion and audio dropouts.
 */
class LockFreeRingBuffer {
private:
  // The number of AudioFrame buffers in the ring.
  static constexpr size_t BUFFER_FRAME_COUNT = 64;
  // The underlying buffer storing the AudioFrames.
  AudioFrame buffer[BUFFER_FRAME_COUNT];

  // Atomic indices to manage the read and write positions in the ring buffer.
  // memory_order_relaxed is used for performance as synchronization is handled
  // by the acquire/release semantics in the conditional checks.
  std::atomic<size_t> writeIndex{0};
  std::atomic<size_t> readIndex{0};

public:
  /**
   * @brief Pushes a new AudioFrame into the ring buffer. (Producer Only)
   * @param frame The AudioFrame to be added to the buffer.
   * @return True if the push was successful, false if the buffer was full.
   */
  bool push(const AudioFrame& frame) {
    size_t wIndex = writeIndex.load(std::memory_order_relaxed);
    size_t nextIndex = (wIndex + 1) % BUFFER_FRAME_COUNT;

    // Check if the buffer is full.
    if (nextIndex == readIndex.load(std::memory_order_acquire)) {
      return false; 
    }

    buffer[wIndex] = frame;
    writeIndex.store(nextIndex, std::memory_order_release);
    return true;
  }

  /**
   * @brief Pops an AudioFrame from the ring buffer. (Consumer Only)
   * @param frameOut A reference to an AudioFrame to be filled with data from the buffer.
   * @return True if the pop was successful, false if the buffer was empty.
   */
  bool pop(AudioFrame& frameOut) {
    size_t rIndex = readIndex.load(std::memory_order_relaxed);
    
    // Check if the buffer is empty.
    if (rIndex == writeIndex.load(std::memory_order_acquire)) {
      return false;
    }

    frameOut = buffer[rIndex];
    readIndex.store((rIndex + 1) % BUFFER_FRAME_COUNT, std::memory_order_release);
    return true;
  }
}; 