#pragma once

#include <vector>
#include <atomic>
#include <cstddef>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0) {}

    bool write(const T* data, size_t count) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + count);
        if (next_tail - head_.load(std::memory_order_acquire) > capacity_) {
            return false; // Not enough space
        }

        for (size_t i = 0; i < count; ++i) {
            buffer_[(current_tail + i) % capacity_] = data[i];
        }

        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool read(T* data, size_t count) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (tail_.load(std::memory_order_acquire) - current_head < count) {
            return false; // Not enough data
        }

        for (size_t i = 0; i < count; ++i) {
            data[i] = buffer_[(current_head + i) % capacity_];
        }

        head_.store(current_head + count, std::memory_order_release);
        return true;
    }

    size_t size() const {
        return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire);
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    std::vector<T> buffer_;
    const size_t capacity_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
}; 