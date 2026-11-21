#include "MemoryBlock.h"
#include <algorithm>

// MemoryBlock: rule of 5

// contructor
MemoryBlock::MemoryBlock(size_t size)
    : size(size), refCount(0), data(new int[size]()) {}

// destructor
MemoryBlock::~MemoryBlock() { delete[] data; }

// copy constructor
MemoryBlock::MemoryBlock(const MemoryBlock &other)
    : size(other.size), refCount(0), data(new int[other.size]()) {
    std::copy(other.data, other.data + other.size, data);
}

// copy assignment operator
MemoryBlock &MemoryBlock::operator=(const MemoryBlock &other) {
    if (this == &other) {
        return *this;
    }

    int *copiedData = new int[other.size];
    std::copy(other.data, other.data + other.size, copiedData);

    delete[] data;
    data = copiedData;
    size = other.size;
    refCount = 0;

    return *this;
}

// move constructor
MemoryBlock::MemoryBlock(MemoryBlock &&other) noexcept
    : size(other.size), refCount(other.refCount), data(other.data) {
    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
}

// move assignment operator
MemoryBlock &MemoryBlock::operator=(MemoryBlock &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    delete[] data;

    size = other.size;
    refCount = other.refCount;
    data = other.data;

    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;

    return *this;
}

// MemoryBlock other functions
int MemoryBlock::readFromBlock(std::size_t offset) const {
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    data[offset] = value;
}

void MemoryBlock::acquire() { refCount++; }

IdleAction MemoryBlock::release() {
    if (refCount > 0) {
        refCount--;
    }
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}

int MemoryBlock::getRefCount() const { return refCount; }

// KVBlock implementation
KVBlock::KVBlock(std::size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const { return IdleAction::KeepCached; }
MemoryBlock *KVBlock::clone() const { return new KVBlock(*this); }

// EmbeddingWorkspace implementation
EmbeddingWorkspace::EmbeddingWorkspace(std::size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}
MemoryBlock *EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}