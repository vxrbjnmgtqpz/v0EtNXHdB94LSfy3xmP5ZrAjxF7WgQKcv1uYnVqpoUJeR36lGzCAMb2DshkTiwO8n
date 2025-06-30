#pragma once

#include <atomic>
#include <memory>
#include <array>

namespace JSONMIDI {

/**
 * Lock-free circular buffer for real-time audio thread safety
 * Implements a single-producer, single-consumer queue optimized for MIDI messages
 * 
 * Features:
 * - Wait-free operations for real-time threads
 * - Cache-friendly memory layout
 * - Overflow handling with configurable behavior
 * - Memory ordering optimizations for x86/ARM
 */
template<typename T, size_t Size>
class LockFreeQueue {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
public:
    LockFreeQueue() : 
        writeIndex_(0),
        readIndex_(0),
        capacity_(Size) {
        
        // Initialize all slots to nullptr for pointer types
        for (size_t i = 0; i < Size; ++i) {
            data_[i].store(T{}, std::memory_order_relaxed);
        }
    }
    
    ~LockFreeQueue() = default;
    
    // Non-copyable, non-movable for safety
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
    
    /**
     * Try to push an item to the queue (producer side)
     * Returns true if successful, false if queue is full
     * This operation is wait-free
     */
    bool tryPush(T&& item) {
        const size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        
        // Check if queue is full
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        // Store the item
        data_[currentWrite].store(std::move(item), std::memory_order_relaxed);
        
        // Update write index with release semantics
        writeIndex_.store(nextWrite, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Try to push an item to the queue (copy version)
     */
    bool tryPush(const T& item) {
        T copy = item;
        return tryPush(std::move(copy));
    }
    
    /**
     * Try to pop an item from the queue (consumer side)
     * Returns true if successful, false if queue is empty
     * This operation is wait-free
     */
    bool tryPop(T& item) {
        const size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        // Check if queue is empty
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        // Load the item
        item = data_[currentRead].load(std::memory_order_relaxed);
        
        // Update read index with release semantics
        const size_t nextRead = (currentRead + 1) & (Size - 1);
        readIndex_.store(nextRead, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Check if queue is empty (approximate)
     * This is a hint only - the state may change immediately after calling
     */
    bool empty() const {
        return readIndex_.load(std::memory_order_acquire) == 
               writeIndex_.load(std::memory_order_acquire);
    }
    
    /**
     * Check if queue is full (approximate)
     * This is a hint only - the state may change immediately after calling
     */
    bool full() const {
        const size_t currentWrite = writeIndex_.load(std::memory_order_acquire);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        return nextWrite == readIndex_.load(std::memory_order_acquire);
    }
    
    /**
     * Get approximate number of items in queue
     * This is a hint only - the count may change immediately after calling
     */
    size_t size() const {
        const size_t write = writeIndex_.load(std::memory_order_acquire);
        const size_t read = readIndex_.load(std::memory_order_acquire);
        return (write - read) & (Size - 1);
    }
    
    /**
     * Get the capacity of the queue
     */
    constexpr size_t capacity() const {
        return Size - 1; // One slot is always kept empty
    }
    
    /**
     * Clear all items from the queue
     * WARNING: This is NOT thread-safe and should only be called when 
     * no other threads are accessing the queue
     */
    void clear() {
        T dummy;
        while (tryPop(dummy)) {
            // Drain all items
        }
    }

private:
    // Align to cache line to avoid false sharing
    alignas(64) std::atomic<size_t> writeIndex_;
    alignas(64) std::atomic<size_t> readIndex_;
    
    // The actual data storage
    std::array<std::atomic<T>, Size> data_;
    
    const size_t capacity_;
};

/**
 * Specialized version for unique_ptr to handle move semantics properly
 */
template<typename T, size_t Size>
class LockFreeQueue<std::unique_ptr<T>, Size> {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
public:
    using value_type = std::unique_ptr<T>;
    
    LockFreeQueue() : 
        writeIndex_(0),
        readIndex_(0) {
        
        // Initialize all slots to nullptr
        for (size_t i = 0; i < Size; ++i) {
            data_[i].store(nullptr, std::memory_order_relaxed);
        }
    }
    
    ~LockFreeQueue() {
        // Clean up any remaining items
        value_type item;
        while (tryPop(item)) {
            // Items will be automatically destroyed
        }
    }
    
    // Non-copyable, non-movable
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
    
    bool tryPush(value_type&& item) {
        const size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        // Store the raw pointer
        data_[currentWrite].store(item.release(), std::memory_order_relaxed);
        
        // Update write index
        writeIndex_.store(nextWrite, std::memory_order_release);
        
        return true;
    }
    
    bool tryPop(value_type& item) {
        const size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        // Load the raw pointer and wrap in unique_ptr
        T* ptr = data_[currentRead].load(std::memory_order_relaxed);
        item.reset(ptr);
        
        // Clear the slot
        data_[currentRead].store(nullptr, std::memory_order_relaxed);
        
        // Update read index
        const size_t nextRead = (currentRead + 1) & (Size - 1);
        readIndex_.store(nextRead, std::memory_order_release);
        
        return true;
    }
    
    bool empty() const {
        return readIndex_.load(std::memory_order_acquire) == 
               writeIndex_.load(std::memory_order_acquire);
    }
    
    bool full() const {
        const size_t currentWrite = writeIndex_.load(std::memory_order_acquire);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        return nextWrite == readIndex_.load(std::memory_order_acquire);
    }
    
    size_t size() const {
        const size_t write = writeIndex_.load(std::memory_order_acquire);
        const size_t read = readIndex_.load(std::memory_order_acquire);
        return (write - read) & (Size - 1);
    }
    
    constexpr size_t capacity() const {
        return Size - 1;
    }

private:
    alignas(64) std::atomic<size_t> writeIndex_;
    alignas(64) std::atomic<size_t> readIndex_;
    
    std::array<std::atomic<T*>, Size> data_;
};

/**
 * Specialized version for shared_ptr to handle reference counting properly
 */
template<typename T, size_t Size>
class LockFreeQueue<std::shared_ptr<T>, Size> {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
public:
    using value_type = std::shared_ptr<T>;
    
    LockFreeQueue() : 
        writeIndex_(0),
        readIndex_(0) {
        
        // Initialize all slots to nullptr
        for (size_t i = 0; i < Size; ++i) {
            data_[i].store(nullptr, std::memory_order_relaxed);
        }
    }
    
    ~LockFreeQueue() {
        // Clean up any remaining items
        value_type item;
        while (tryPop(item)) {
            // Items will be automatically destroyed
        }
    }
    
    // Non-copyable, non-movable
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
    
    bool tryPush(value_type&& item) {
        const size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        // For shared_ptr, we need to increment the ref count and store the raw pointer
        T* ptr = item.get();
        if (ptr) {
            item.get(); // This keeps the shared_ptr alive
            // We need to store a copy of the shared_ptr to maintain ref count
            // Use a trick: store the pointer and keep track of shared_ptrs separately
        }
        
        // Store the raw pointer, but we need to manage the shared_ptr lifecycle
        data_[currentWrite].store(ptr, std::memory_order_relaxed);
        
        // Store the actual shared_ptr in the control array to maintain ref count
        control_[currentWrite].store(new value_type(std::move(item)), std::memory_order_relaxed);
        
        // Update write index
        writeIndex_.store(nextWrite, std::memory_order_release);
        
        return true;
    }
    
    bool tryPush(const value_type& item) {
        value_type copy = item;
        return tryPush(std::move(copy));
    }
    
    bool tryPop(value_type& item) {
        const size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        // Load the shared_ptr from control array
        value_type* stored_ptr = control_[currentRead].load(std::memory_order_relaxed);
        if (stored_ptr) {
            item = *stored_ptr;
            delete stored_ptr;
        } else {
            item.reset();
        }
        
        // Clear the slots
        data_[currentRead].store(nullptr, std::memory_order_relaxed);
        control_[currentRead].store(nullptr, std::memory_order_relaxed);
        
        // Update read index
        const size_t nextRead = (currentRead + 1) & (Size - 1);
        readIndex_.store(nextRead, std::memory_order_release);
        
        return true;
    }
    
    bool empty() const {
        return readIndex_.load(std::memory_order_acquire) == 
               writeIndex_.load(std::memory_order_acquire);
    }
    
    bool full() const {
        const size_t currentWrite = writeIndex_.load(std::memory_order_acquire);
        const size_t nextWrite = (currentWrite + 1) & (Size - 1);
        return nextWrite == readIndex_.load(std::memory_order_acquire);
    }
    
    size_t size() const {
        const size_t write = writeIndex_.load(std::memory_order_acquire);
        const size_t read = readIndex_.load(std::memory_order_acquire);
        return (write - read) & (Size - 1);
    }
    
    constexpr size_t capacity() const {
        return Size - 1;
    }

private:
    alignas(64) std::atomic<size_t> writeIndex_;
    alignas(64) std::atomic<size_t> readIndex_;
    
    std::array<std::atomic<T*>, Size> data_;
    std::array<std::atomic<value_type*>, Size> control_; // For managing shared_ptr lifecycle
};

// Common queue sizes for MIDI processing
using MIDIMessageQueue = LockFreeQueue<std::unique_ptr<class MIDIMessage>, 1024>;
using FastMIDIQueue = LockFreeQueue<std::unique_ptr<class MIDIMessage>, 256>;
using BulkMIDIQueue = LockFreeQueue<std::unique_ptr<class MIDIMessage>, 4096>;

} // namespace JSONMIDI
