#ifndef GPU_H_
#define GPU_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;
class Request;
class GenerativeRequest;
class EmbeddingRequest;

class GPU {
  public:
    // rule of 5
    GPU(std::size_t blocksNum, std::size_t blockSize);
    ~GPU();
    GPU(const GPU &other);
    GPU &operator=(const GPU &other);
    GPU(GPU &&other) noexcept;
    GPU &operator=(GPU &&other) noexcept;

    std::size_t availableSlots() const;
    std::size_t getBlocksNum() const;
    std::size_t getCacheHits() const;
    std::size_t getEvictions() const;
    std::size_t countIdle(const std::vector<std::size_t> &slots) const;
    std::size_t getBlockSize() const;
    void addCacheHits(std::size_t count);
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request &request);
    void unreserve(std::size_t n, const Request &request);
    std::size_t allocateBlock(const Request &request);
    void addMemoryBlock(const GenerativeRequest &request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest &request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU &other,
                                const std::vector<std::size_t> &blocks);
    void assignPrefixToBlock(std::size_t block,
                             const std::vector<std::string> &prefixTokens,
                             const std::string &modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch
    findLongestCachedPrefix(const std::string &modelId,
                            std::size_t embeddingSize,
                            const std::vector<std::string> &promptTokens) const;

  private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock *> memory;

    // slot to model id
    std::vector<std::string> modelInSlot;
    // slot to cached prefix key
    std::vector<std::vector<std::string>> prefixKeyInSlot;

    // request id to the number of blocks that are still reserved
    std::map<int, std::size_t> reservations;
    // total of all reserved blocks
    std::size_t totalReserved;

    std::size_t cacheHits;
    std::size_t evictions;

    std::size_t findNewSlot(const std::vector<std::size_t> &protectedSlots);
    std::size_t
    findSlotWithKey(const std::string &modelId,
                    const std::vector<std::string> &prefixKey) const;
    void deleteBlock(std::size_t slot);
    void freeMemoryBlocks();
    void cloneBlocksFrom(const GPU &other);
};

#endif