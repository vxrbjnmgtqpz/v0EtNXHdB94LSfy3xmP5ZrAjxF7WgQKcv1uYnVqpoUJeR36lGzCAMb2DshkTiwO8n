#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <thread>

namespace jvid {

/**
 * @brief Lock-free single-producer single-consumer queue
 * 
 * High-performance queue implementation for real-time video processing
 * Based on ring buffer with atomic operations for thread safety.
 */
template<typename T>
class LockFreeQueue {
    static_assert(!std::is_reference<T>::value, "T must not be a reference type");
    
public:
    /**
     * @brief Constructor
     * @param capacity Queue capacity (must be power of 2)
     */
    explicit LockFreeQueue(size_t capacity = 1024)
        : capacity_(capacity)
        , mask_(capacity - 1)
        , buffer_(std::make_unique<Slot[]>(capacity))
        , head_(0)
        , tail_(0) {
        
        // Ensure capacity is power of 2
        if ((capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be power of 2");
        }
        
        // Initialize slots
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    
    /**
     * @brief Destructor
     */
    ~LockFreeQueue() = default;
    
    /**
     * @brief Non-copyable
     */
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    
    /**
     * @brief Non-movable (for simplicity)
     */
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
    
    /**
     * @brief Enqueue element (producer side)
     * @param item Item to enqueue
     * @return True if enqueued successfully, false if queue is full
     */
    bool enqueue(T&& item) {
        size_t pos = head_.load(std::memory_order_relaxed);
        Slot* slot = &buffer_[pos & mask_];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
        if (diff == 0) {
            head_.store(pos + 1, std::memory_order_relaxed);
            slot->data = std::move(item);
            slot->sequence.store(pos + 1, std::memory_order_release);
            return true;
        }
        
        return false; // Queue full
    }
    
    /**
     * @brief Enqueue element (copy version)
     * @param item Item to enqueue
     * @return True if enqueued successfully, false if queue is full
     */
    bool enqueue(const T& item) {
        T copy = item;
        return enqueue(std::move(copy));
    }
    
    /**
     * @brief Dequeue element (consumer side)
     * @param item Output parameter for dequeued item
     * @return True if dequeued successfully, false if queue is empty
     */
    bool dequeue(T& item) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        Slot* slot = &buffer_[pos & mask_];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (diff == 0) {
            tail_.store(pos + 1, std::memory_order_relaxed);
            item = std::move(slot->data);
            slot->sequence.store(pos + mask_ + 1, std::memory_order_release);
            return true;
        }
        
        return false; // Queue empty
    }
    
    /**
     * @brief Check if queue is empty
     * @return True if queue is empty
     */
    bool empty() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head == tail;
    }
    
    /**
     * @brief Get approximate size of queue
     * @return Approximate number of elements in queue
     */
    size_t size() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head >= tail ? head - tail : 0;
    }
    
    /**
     * @brief Get queue capacity
     * @return Maximum number of elements queue can hold
     */
    size_t capacity() const {
        return capacity_;
    }
    
    /**
     * @brief Check if queue is full (approximate)
     * @return True if queue appears full
     */
    bool full() const {
        return size() >= capacity_ - 1;
    }

private:
    struct Slot {
        std::atomic<size_t> sequence;
        T data;
    };
    
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    const size_t capacity_;
    const size_t mask_;
    std::unique_ptr<Slot[]> buffer_;
    
    // Align to cache line to avoid false sharing
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
};

/**
 * @brief Multi-producer single-consumer lock-free queue
 * 
 * Allows multiple producers but only one consumer
 * Uses atomic compare-and-swap for thread safety
 */
template<typename T>
class LockFreeMPSCQueue {
public:
    /**
     * @brief Constructor
     * @param capacity Queue capacity (must be power of 2)
     */
    explicit LockFreeMPSCQueue(size_t capacity = 1024)
        : capacity_(capacity)
        , mask_(capacity - 1)
        , buffer_(std::make_unique<Slot[]>(capacity))
        , head_(0)
        , tail_(0) {
        
        if ((capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be power of 2");
        }
        
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    
    /**
     * @brief Enqueue element (multi-producer safe)
     * @param item Item to enqueue
     * @return True if enqueued successfully
     */
    bool enqueue(T&& item) {
        size_t pos = head_.fetch_add(1, std::memory_order_relaxed);
        Slot* slot = &buffer_[pos & mask_];
        
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
        
        if (diff == 0) {
            slot->data = std::move(item);
            slot->sequence.store(pos + 1, std::memory_order_release);
            return true;
        }
        
        return false; // Queue full
    }
    
    /**
     * @brief Enqueue element (copy version)
     * @param item Item to enqueue
     * @return True if enqueued successfully
     */
    bool enqueue(const T& item) {
        T copy = item;
        return enqueue(std::move(copy));
    }
    
    /**
     * @brief Dequeue element (single consumer only)
     * @param item Output parameter for dequeued item
     * @return True if dequeued successfully
     */
    bool dequeue(T& item) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        Slot* slot = &buffer_[pos & mask_];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        
        intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
        if (diff == 0) {
            tail_.store(pos + 1, std::memory_order_relaxed);
            item = std::move(slot->data);
            slot->sequence.store(pos + mask_ + 1, std::memory_order_release);
            return true;
        }
        
        return false; // Queue empty
    }
    
    /**
     * @brief Check if queue is empty
     * @return True if queue is empty
     */
    bool empty() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head == tail;
    }
    
    /**
     * @brief Get approximate size of queue
     * @return Approximate number of elements in queue
     */
    size_t size() const {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return head >= tail ? head - tail : 0;
    }
    
    /**
     * @brief Get queue capacity
     * @return Maximum number of elements queue can hold
     */
    size_t capacity() const {
        return capacity_;
    }

private:
    struct Slot {
        std::atomic<size_t> sequence;
        T data;
    };
    
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    const size_t capacity_;
    const size_t mask_;
    std::unique_ptr<Slot[]> buffer_;
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_;
};

/**
 * @brief Factory functions for common queue configurations
 */

/**
 * @brief Create queue optimized for video frames
 * @param max_frames Maximum number of frames to buffer
 * @return Configured lock-free queue
 */
template<typename T>
std::unique_ptr<LockFreeQueue<T>> createVideoFrameQueue(size_t max_frames = 64) {
    // Round up to next power of 2
    size_t capacity = 1;
    while (capacity < max_frames) {
        capacity <<= 1;
    }
    return std::make_unique<LockFreeQueue<T>>(capacity);
}

/**
 * @brief Create MPSC queue for multiple video sources
 * @param max_frames Maximum number of frames to buffer
 * @return Configured MPSC lock-free queue
 */
template<typename T>
std::unique_ptr<LockFreeMPSCQueue<T>> createMultiSourceVideoQueue(size_t max_frames = 128) {
    size_t capacity = 1;
    while (capacity < max_frames) {
        capacity <<= 1;
    }
    return std::make_unique<LockFreeMPSCQueue<T>>(capacity);
}

} // namespace jvid 