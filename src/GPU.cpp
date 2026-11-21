#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <algorithm>
#include <stdexcept>

// delete all blocks from memory
void GPU::freeMemoryBlocks() {
    for (MemoryBlock *block : memory) {
        delete block;
    }
}

// clone blocks from other and acquiring them according to other's ref count
void GPU::cloneBlocksFrom(const GPU &other) {
    for (std::size_t i = 0; i < other.blocksNum; i++) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            // acquire this memory slot the same number of times the other
            // memory slot has been acquired such that the ref counts match
            for (int j = 0; j < other.memory[i]->getRefCount(); j++) {
                memory[i]->acquire();
            }
        }
    }
}

// delete a single block from memory
void GPU::deleteBlock(std::size_t slot) {
    delete memory[slot];
    memory[slot] = nullptr;
    modelInSlot[slot].clear();
    prefixKeyInSlot[slot].clear();
}

// find a slot in memory that is empty or idle and not protected
std::size_t GPU::findNewSlot(const std::vector<std::size_t> &protectedSlots) {
    // find first empty slot
    for (std::size_t i = 0; i < blocksNum; i++) {
        MemoryBlock *block = memory[i];
        if (block == nullptr) {
            return i;
        }
    }

    // find first idle slot
    for (std::size_t i = 0; i < blocksNum; i++) {
        // find if the current slot is inside protectedSlots
        bool isProtected =
            std::find(protectedSlots.begin(), protectedSlots.end(), i) !=
            protectedSlots.end();
        // block i is idle and not protected
        if (memory[i]->getRefCount() == 0 && !isProtected) {
            // empty and delete the block in this slot as well has prefixKey and
            // model
            deleteBlock(i);

            // increment evictions
            evictions++;

            return i;
        }
    }

    // no empty slot or idle slot has been found which is not possible
    throw std::runtime_error("no slot available on the GPU");
}

// returns the slot with this key and model id, if there is none returns
// blocksNum
std::size_t
GPU::findSlotWithKey(const std::string &modelId,
                     const std::vector<std::string> &prefixKey) const {
    for (std::size_t i = 0; i < blocksNum; i++) {
        if (memory[i] != nullptr && !prefixKeyInSlot[i].empty() &&
            modelInSlot[i] == modelId && prefixKeyInSlot[i] == prefixKey) {
            return i;
        }
    }
    // if no matching slot found return blocksNum
    return blocksNum;
}

// rule of 5:

// constructor
GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr),
      modelInSlot(blocksNum), prefixKeyInSlot(blocksNum), reservations(),
      totalReserved(0), cacheHits(0), evictions(0) {}

// destructor
GPU::~GPU() { freeMemoryBlocks(); }

// copy constructor
GPU::GPU(const GPU &other)
    : blocksNum(other.blocksNum), blockSize(other.blockSize),
      memory(other.blocksNum, nullptr), modelInSlot(other.modelInSlot),
      prefixKeyInSlot(other.prefixKeyInSlot), reservations(other.reservations),
      totalReserved(other.totalReserved), cacheHits(other.cacheHits),
      evictions(other.evictions) {
    // clone all blocks from other's memory
    cloneBlocksFrom(other);
}

// copy assigment operator
GPU &GPU::operator=(const GPU &other) {
    if (this == &other) {
        return *this;
    }

    // deletes all blocks on the GPU
    freeMemoryBlocks();

    // assign nullptr to all block in memory
    memory.assign(other.blocksNum, nullptr);

    // clone all blocks from other's memory
    cloneBlocksFrom(other);

    // copy all other fields
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    modelInSlot = other.modelInSlot;
    prefixKeyInSlot = other.prefixKeyInSlot;
    reservations = other.reservations;
    totalReserved = other.totalReserved;
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    return *this;
}

// move constructor
// std::move converts a value to an rvalue making the compiler use the move
// assigement operator or move constructor of vector and map
GPU::GPU(GPU &&other) noexcept
    : blocksNum(other.blocksNum), blockSize(other.blockSize),
      memory(std::move(other.memory)),
      modelInSlot(std::move(other.modelInSlot)),
      prefixKeyInSlot(std::move(other.prefixKeyInSlot)),
      reservations(std::move(other.reservations)),
      totalReserved(other.totalReserved), cacheHits(other.cacheHits),
      evictions(other.evictions) {
    // clearing the memory vector does not free the MemoryBlock pointers
    // so we can still use them in the moved GPU.
    // when we delte other it wont free the MemoryBlocks since they are not in
    // the vector
    other.memory.clear();
    other.reservations.clear();
    other.modelInSlot.clear();
    other.prefixKeyInSlot.clear();
    other.blocksNum = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
}

// move assignment operator
GPU &GPU::operator=(GPU &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    // copy all data
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    totalReserved = other.totalReserved;
    cacheHits = other.cacheHits;
    evictions = other.evictions;

    // deletes all blocks from memory
    freeMemoryBlocks();

    // move other's data using the vector and map move assigment operator or
    // move constructor by converting to an rvalue
    memory = std::move(other.memory);
    reservations = std::move(other.reservations);
    modelInSlot = std::move(other.modelInSlot);
    prefixKeyInSlot = std::move(other.prefixKeyInSlot);

    // clear all others fields
    other.memory.clear();
    other.reservations.clear();
    other.modelInSlot.clear();
    other.prefixKeyInSlot.clear();
    other.blocksNum = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;

    return *this;
}

// return the number of available slots
std::size_t GPU::availableSlots() const {
    // an active block is a block with at least one active request that uses it
    std::size_t activeBlocks = 0;
    for (MemoryBlock *block : memory) {
        if (block != nullptr && block->getRefCount() > 0) {
            activeBlocks++;
        }
    }
    // all the blocks on the gpu minus the active and reserved ones
    return blocksNum - activeBlocks - totalReserved;
}

// blocksNum getter
std::size_t GPU::getBlocksNum() const { return blocksNum; }
// cache hits getter
std::size_t GPU::getCacheHits() const { return cacheHits; }
// evictions getter
std::size_t GPU::getEvictions() const { return evictions; }
// count the number if idle slots from all the slots inside the given vector
std::size_t GPU::countIdle(const std::vector<std::size_t> &slots) const {
    std::size_t counter = 0;
    for (std::size_t slot : slots) {
        if (memory[slot]->getRefCount() == 0) {
            counter++;
        }
    }
    return counter;
}

// block size getter
std::size_t GPU::getBlockSize() const { return blockSize; }

// add cache hit
void GPU::addCacheHits(std::size_t count) { cacheHits += count; }

// how many block needed for given memory
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    // if memory is divisible by blockSize this expression evaluates to memory /
    // blockSize if they are not divisible this expression evaluates to memory /
    // blockSize + 1
    return (memory + blockSize - 1) / blockSize;
}

// reserve n blocks for a request
void GPU::reserve(std::size_t n, const Request &request) {
    // update the number of reserve blocks for this request
    reservations[request.getId()] += n;
    // update the number of the total reserved blocks
    totalReserved += n;
}

// unreserve n blocks for a request
void GPU::unreserve(std::size_t n, const Request &request) {
    // nothing to unreserve
    if (n == 0) {
        return;
    }
    // "pointer" to the request in the map, if it does not exist return
    // reservations.end()
    auto p = reservations.find(request.getId());
    if (p == reservations.end() || p->second < n) {
        // p->second correspond to number of reserved blocks
        throw std::runtime_error(
            "cannot unreserved more then already reserved");
    }
    // update reservations
    p->second -= n;
    totalReserved -= n;
    // erase from map if no blocks are reserved for this request
    if (p->second == 0) {
        reservations.erase(p);
    }
}

// allocate one block of memory for a request
std::size_t GPU::allocateBlock(const Request &request) {
    // "pointer" to the reservation count of request in the map
    // if the request not in map, p = reservation.end()
    auto p = reservations.find(request.getId());
    // NOTE: p->second == 0 might be unnecessary since we erase from the map
    // requests with zero reservations. p->second correspond to the reservation
    // count for this request
    if (p == reservations.end() || p->second == 0) {
        // request did not reserve enough slots
        throw std::runtime_error(
            "request attempt to allocate without reservation");
    }

    // find slot to allocate (no slot is protected)
    std::size_t slot = findNewSlot(std::vector<std::size_t>());
    // allocate the slot and increase the ref count
    request.allocateOneBlock(*this, slot);
    memory[slot]->acquire();

    // decrement reservation count for this request
    p->second--;
    totalReserved--;

    // if ref count is 0 then erase this request from the map
    if (p->second == 0) {
        reservations.erase(p);
    }

    return slot;
}

// create a new KVBlock in this slot
void GPU::addMemoryBlock(const GenerativeRequest &request, std::size_t slot) {
    memory[slot] = new KVBlock(blockSize);
}

// create a new EmbeddingWorkspace block in this slot
void GPU::addMemoryBlock(const EmbeddingRequest &request, std::size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

// increment slot ref count
void GPU::acquire(std::size_t slot) { memory[slot]->acquire(); }

// decrement ref count, if the ref count is now zero the function deletes it
void GPU::release(std::size_t slot) {
    IdleAction action = memory[slot]->release();
    // if the slot is still in use, do nothing
    if (memory[slot]->getRefCount() > 0) {
        return;
    }
    // else, if action is Destroy or the slot has not prefix key delete it
    if (action == IdleAction::Destroy || prefixKeyInSlot[slot].empty()) {
        deleteBlock(slot);
    }
}

void GPU::copyMemoryFromOtherGPU(const GPU &other,
                                 const std::vector<std::size_t> &blocks) {
    // holds the slots we copy into
    std::vector<std::size_t> chain;
    for (std::size_t block : blocks) {
        // fetch model and prefixKey
        const std::string &model = other.modelInSlot[block];
        const std::vector<std::string> &key = other.prefixKeyInSlot[block];

        // finds a slot with the same model and key
        std::size_t slot = findSlotWithKey(model, key);
        if (slot != blocksNum) {
            // the same slot already exist
            // add this slot to the chain
            chain.push_back(slot);
        } else {
            // allocate new slot and clone the other block to the new slot
            std::size_t newSlot = findNewSlot(chain);
            memory[newSlot] = other.memory[block]->clone();
            // set model and prefixKey
            modelInSlot[newSlot] = model;
            prefixKeyInSlot[newSlot] = key;
            // add to the chain
            chain.push_back(newSlot);
        }
    }
}

// attempt to assign a prefix key and model to a given block
void GPU::assignPrefixToBlock(std::size_t block,
                              const std::vector<std::string> &prefixTokens,
                              const std::string &modelId) {
    // if a block with the same model and prefixKey exists do not assign prefix
    if (findSlotWithKey(modelId, prefixTokens) != blocksNum) {
        return;
    }
    // else assign prefix
    modelInSlot[block] = modelId;
    prefixKeyInSlot[block] = prefixTokens;
}

// read data from block
int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

// write data to block
void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    memory[blockId]->writeToBlock(offset, value);
}

PrefixMatch GPU::findLongestCachedPrefix(
    const std::string &modelId, std::size_t embeddingSize,
    const std::vector<std::string> &promptTokens) const {
    PrefixMatch result{0, std::vector<std::size_t>()};
    // how many embedded tokens fit in one block
    std::size_t tokensPerBlock = blockSize / embeddingSize;

    for (std::size_t i = tokensPerBlock; i <= promptTokens.size();
         i += tokensPerBlock) {
        // the first k * tokensPerBlock prompt tokens, where k is the iteration
        // each iteration we add the next tokens that fit in a block
        std::vector<std::string> prefixKey(promptTokens.begin(),
                                           promptTokens.begin() + i);
        // look for a slot with this prefixKey
        std::size_t slot = findSlotWithKey(modelId, prefixKey);
        if (slot == blocksNum) {
            // if no such exist return all the slots that hold a prefix to the
            // tokens of the prompt
            return result;
        }
        // if this prefix already exist add this slot to result
        result.blockIds.push_back(slot);
        // update the number of matched tokens to the length of the largest
        // prefix that already exists
        result.matchedTokens = i;
    }

    return result;
}