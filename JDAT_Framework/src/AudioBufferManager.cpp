#include "AudioBufferManager.h"
#include <algorithm>
#include <cstring>

namespace jdat {

AudioBufferManager::AudioBufferManager(const Config& config)
    : config_(config)
    , write_index_(0)
    , read_index_(0)
    , is_running_(false) {
    
    // Calculate buffer sizes
    samples_per_frame_ = config_.frame_size_samples;
    frames_in_buffer_ = (config_.buffer_size_ms * config_.sample_rate) / (1000 * samples_per_frame_);
    
    if (frames_in_buffer_ < 4) {
        frames_in_buffer_ = 4; // Minimum buffer size
    }
    
    // Initialize circular buffer
    buffer_.resize(frames_in_buffer_);
    for (auto& frame : buffer_) {
        frame.resize(samples_per_frame_, 0.0f);
    }
    
    // Initialize frame metadata
    frame_metadata_.resize(frames_in_buffer_);
    
    std::cout << "AudioBufferManager initialized:\n";
    std::cout << "  Sample Rate: " << config_.sample_rate << " Hz\n";
    std::cout << "  Frame Size: " << samples_per_frame_ << " samples\n";
    std::cout << "  Buffer Frames: " << frames_in_buffer_ << "\n";
    std::cout << "  Buffer Duration: " << (frames_in_buffer_ * samples_per_frame_ * 1000) / config_.sample_rate << " ms\n";
}

AudioBufferManager::~AudioBufferManager() {
    stop();
}

bool AudioBufferManager::start() {
    if (is_running_) {
        return false;
    }
    
    is_running_ = true;
    return true;
}

void AudioBufferManager::stop() {
    is_running_ = false;
}

bool AudioBufferManager::addFrame(const std::vector<float>& samples) {
    if (samples.size() != samples_per_frame_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Check if buffer is full
    size_t next_write = (write_index_ + 1) % frames_in_buffer_;
    if (next_write == read_index_) {
        // Buffer full - overwrite oldest frame
        read_index_ = (read_index_ + 1) % frames_in_buffer_;
        buffer_overruns_++;
    }
    
    // Copy samples to buffer
    std::copy(samples.begin(), samples.end(), buffer_[write_index_].begin());
    
    // Update metadata
    frame_metadata_[write_index_].timestamp = getCurrentTimestamp();
    frame_metadata_[write_index_].frame_number = frames_written_;
    frame_metadata_[write_index_].valid = true;
    
    // Advance write pointer
    write_index_ = next_write;
    frames_written_++;
    
    return true;
}

std::vector<float> AudioBufferManager::getNextFrame() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Check if data is available
    if (read_index_ == write_index_) {
        // No data available
        buffer_underruns_++;
        return std::vector<float>(samples_per_frame_, 0.0f); // Return silence
    }
    
    // Get frame data
    std::vector<float> result = buffer_[read_index_];
    
    // Mark frame as consumed
    frame_metadata_[read_index_].valid = false;
    
    // Advance read pointer
    read_index_ = (read_index_ + 1) % frames_in_buffer_;
    frames_read_++;
    
    return result;
}

std::vector<float> AudioBufferManager::peekFrame(size_t offset) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Calculate peek position
    size_t peek_index = (read_index_ + offset) % frames_in_buffer_;
    
    // Check if frame is valid
    if (!frame_metadata_[peek_index].valid || peek_index == write_index_) {
        return std::vector<float>(samples_per_frame_, 0.0f); // Return silence
    }
    
    return buffer_[peek_index];
}

size_t AudioBufferManager::getAvailableFrames() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (write_index_ >= read_index_) {
        return write_index_ - read_index_;
    } else {
        return frames_in_buffer_ - read_index_ + write_index_;
    }
}

size_t AudioBufferManager::getBufferCapacity() const {
    return frames_in_buffer_;
}

float AudioBufferManager::getBufferUsage() const {
    return static_cast<float>(getAvailableFrames()) / static_cast<float>(frames_in_buffer_);
}

AudioBufferManager::Statistics AudioBufferManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    Statistics stats;
    stats.frames_written = frames_written_;
    stats.frames_read = frames_read_;
    stats.buffer_overruns = buffer_overruns_;
    stats.buffer_underruns = buffer_underruns_;
    stats.current_buffer_usage = getBufferUsage();
    stats.buffer_capacity_frames = frames_in_buffer_;
    
    return stats;
}

void AudioBufferManager::resetStatistics() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    frames_written_ = 0;
    frames_read_ = 0;
    buffer_overruns_ = 0;
    buffer_underruns_ = 0;
}

bool AudioBufferManager::addMultipleFrames(const std::vector<std::vector<float>>& frames) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Check if all frames fit
    if (frames.size() > frames_in_buffer_ - getAvailableFrames()) {
        return false; // Not enough space
    }
    
    // Add all frames
    for (const auto& frame : frames) {
        if (frame.size() != samples_per_frame_) {
            return false; // Invalid frame size
        }
        
        // Add frame without additional locking (already locked)
        std::copy(frame.begin(), frame.end(), buffer_[write_index_].begin());
        
        // Update metadata
        frame_metadata_[write_index_].timestamp = getCurrentTimestamp();
        frame_metadata_[write_index_].frame_number = frames_written_;
        frame_metadata_[write_index_].valid = true;
        
        // Advance write pointer
        write_index_ = (write_index_ + 1) % frames_in_buffer_;
        frames_written_++;
    }
    
    return true;
}

std::vector<std::vector<float>> AudioBufferManager::getMultipleFrames(size_t count) {
    std::vector<std::vector<float>> result;
    result.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        auto frame = getNextFrame();
        
        // Check if we got valid data
        bool is_silence = std::all_of(frame.begin(), frame.end(), 
                                     [](float sample) { return sample == 0.0f; });
        
        result.push_back(std::move(frame));
        
        // Stop if we hit silence (no more data)
        if (is_silence && getAvailableFrames() == 0) {
            break;
        }
    }
    
    return result;
}

void AudioBufferManager::flush() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Reset pointers
    read_index_ = 0;
    write_index_ = 0;
    
    // Clear all frames
    for (auto& frame : buffer_) {
        std::fill(frame.begin(), frame.end(), 0.0f);
    }
    
    // Reset metadata
    for (auto& metadata : frame_metadata_) {
        metadata.valid = false;
        metadata.timestamp = 0;
        metadata.frame_number = 0;
    }
}

uint64_t AudioBufferManager::getCurrentTimestamp() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

} // namespace jdat
