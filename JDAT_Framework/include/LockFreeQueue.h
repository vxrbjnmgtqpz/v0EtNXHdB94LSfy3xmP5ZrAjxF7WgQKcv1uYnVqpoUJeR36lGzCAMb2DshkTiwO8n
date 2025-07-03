#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <thread>

namespace jdat {

/**
 * @brief Lock-free single-producer single-consumer queue
 * 
 * High-performance queue implementation for real-time audio processing
 * Based on ring buffer with atomic operations for thread safety.
 */
template<typename T, size_t Size = 1024>
class LockFreeQueue {
    static_assert(Size > 0 && (Size & (Size - 1)) == 0, "Size must be a power of 2");

private:
    struct alignas(64) Node {
        std::atomic<bool> ready{false};
        alignas(64) T data;
    };

    alignas(64) std::array<Node, Size> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

    static constexpr size_t mask_ = Size - 1;

public:
    /**
     * @brief Constructor
     */
    LockFreeQueue() = default;

    /**
     * @brief Destructor
     */
    ~LockFreeQueue() = default;

    /**
     * @brief Copy constructor (deleted)
     */
    LockFreeQueue(const LockFreeQueue&) = delete;

    /**
     * @brief Move constructor (deleted)
     */
    LockFreeQueue(LockFreeQueue&&) = delete;

    /**
     * @brief Copy assignment (deleted)
     */
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    /**
     * @brief Move assignment (deleted)
     */
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;

    /**
     * @brief Try to push an item to the queue (non-blocking)
     * @param item Item to push
     * @return True if push successful, false if queue is full
     */
    bool try_push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail].data = item;
        buffer_[current_tail].ready.store(true, std::memory_order_release);
        tail_.store(next_tail, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Try to push an item to the queue (move version)
     * @param item Item to move into queue
     * @return True if push successful, false if queue is full
     */
    bool try_push(T&& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail].data = std::move(item);
        buffer_[current_tail].ready.store(true, std::memory_order_release);
        tail_.store(next_tail, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Try to pop an item from the queue (non-blocking)
     * @param item Reference to store popped item
     * @return True if pop successful, false if queue is empty
     */
    bool try_pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (!buffer_[current_head].ready.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        item = std::move(buffer_[current_head].data);
        buffer_[current_head].ready.store(false, std::memory_order_release);
        head_.store((current_head + 1) & mask_, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Check if queue is empty
     * @return True if empty
     */
    bool empty() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        return !buffer_[current_head].ready.load(std::memory_order_acquire);
    }

    /**
     * @brief Check if queue is full
     * @return True if full
     */
    bool full() const {
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        const size_t next_tail = (current_tail + 1) & mask_;
        return next_tail == head_.load(std::memory_order_acquire);
    }

    /**
     * @brief Get approximate size of queue
     * @return Approximate number of items in queue
     */
    size_t size() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        return (current_tail - current_head) & mask_;
    }

    /**
     * @brief Get capacity of queue
     * @return Maximum number of items queue can hold
     */
    constexpr size_t capacity() const {
        return Size - 1; // One slot is always kept empty
    }

    /**
     * @brief Clear all items from queue
     */
    void clear() {
        T dummy;
        while (try_pop(dummy)) {
            // Keep popping until empty
        }
    }

    /**
     * @brief Get load factor (0.0 to 1.0)
     * @return Current load as percentage of capacity
     */
    double load_factor() const {
        return static_cast<double>(size()) / static_cast<double>(capacity());
    }
};

/**
 * @brief Multi-producer single-consumer lock-free queue
 * 
 * Thread-safe queue that supports multiple producers but only one consumer.
 * Useful for scenarios where multiple threads need to send data to one processor.
 */
template<typename T, size_t Size = 1024>
class MPSCQueue {
    static_assert(Size > 0 && (Size & (Size - 1)) == 0, "Size must be a power of 2");

private:
    struct alignas(64) Node {
        std::atomic<T*> data{nullptr};
    };

    alignas(64) std::array<Node, Size> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

    static constexpr size_t mask_ = Size - 1;

public:
    /**
     * @brief Constructor
     */
    MPSCQueue() = default;

    /**
     * @brief Destructor
     */
    ~MPSCQueue() {
        clear();
    }

    /**
     * @brief Copy constructor (deleted)
     */
    MPSCQueue(const MPSCQueue&) = delete;

    /**
     * @brief Move constructor (deleted)
     */
    MPSCQueue(MPSCQueue&&) = delete;

    /**
     * @brief Copy assignment (deleted)
     */
    MPSCQueue& operator=(const MPSCQueue&) = delete;

    /**
     * @brief Move assignment (deleted)
     */
    MPSCQueue& operator=(MPSCQueue&&) = delete;

    /**
     * @brief Try to push an item (thread-safe for multiple producers)
     * @param item Item to push
     * @return True if push successful
     */
    bool try_push(const T& item) {
        T* data = new(std::nothrow) T(item);
        if (!data) return false;

        const size_t pos = tail_.fetch_add(1, std::memory_order_acq_rel) & mask_;
        
        T* expected = nullptr;
        while (!buffer_[pos].data.compare_exchange_weak(expected, data, 
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {
            expected = nullptr;
            std::this_thread::yield();
        }
        
        return true;
    }

    /**
     * @brief Try to pop an item (single consumer only)
     * @param item Reference to store popped item
     * @return True if pop successful
     */
    bool try_pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        T* data = buffer_[current_head].data.exchange(nullptr, std::memory_order_acquire);
        if (!data) {
            return false;
        }
        
        item = std::move(*data);
        delete data;
        head_.store((current_head + 1) & mask_, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Check if queue is empty
     * @return True if empty
     */
    bool empty() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        return buffer_[current_head].data.load(std::memory_order_acquire) == nullptr;
    }

    /**
     * @brief Clear all items from queue
     */
    void clear() {
        T dummy;
        while (try_pop(dummy)) {
            // Keep popping until empty
        }
    }

    /**
     * @brief Get approximate size
     * @return Approximate number of items
     */
    size_t size() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        return (current_tail - current_head) & mask_;
    }

    /**
     * @brief Get capacity
     * @return Maximum capacity
     */
    constexpr size_t capacity() const {
        return Size;
    }
};

} // namespace jdat 