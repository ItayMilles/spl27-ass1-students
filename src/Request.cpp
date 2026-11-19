#include "Request.h"
#include "GPU.h"
#include "Model.h"

// Request
// constructor
Request::Request(int id, const Model &model, int admissionCycle,
                 std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), memoryBlocks(),
      reservedBlocks(0), tokens(model.tokenizer(prompt)) {}

// destructor
Request::~Request() {}

// other
int Request::getId() const { return id; }
const Model &Request::getModel() const { return model; }
int Request::getAdmissionCycle() const { return admissionCycle; }
void Request::deallocateMemory(GPU &gpu) {
    std::size_t toUnreserve = getNeedToUnreserve();
    if (toUnreserve > 0) {
        gpu.unreserve(toUnreserve, *this);
    }

    for (std::size_t slot : memoryBlocks) {
        gpu.release(slot);
    }

    memoryBlocks.clear();
}

// GenerativeRequest
// constructor
GenerativeRequest::GenerativeRequest(int id, const Model &model,
                                     int admissionCycle, std::string prompt,
                                     std::string scriptedOutput,
                                     std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt),
      maxOutputTokens(maxOutputTokens), scriptedOutput(scriptedOutput),
      scriptedTokens(model.tokenizer(scriptedOutput)), generatedTokens(),
      processedTokens(0) {}

// other
std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}
std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return (tokens.size() + maxOutputTokens) * model.getEmbeddingSize();
}
std::size_t GenerativeRequest::getNeededMemory(const GPU &gpu) const {
    PrefixMatch match = gpu.findLongestCachedPrefix(
        model.getId(), model.getEmbeddingSize(), tokens);
    return ((tokens.size() + maxOutputTokens) - match.matchedTokens) *
           model.getEmbeddingSize();
}
void GenerativeRequest::reserveMemory(GPU &gpu) {
    PrefixMatch match = gpu.findLongestCachedPrefix(
        model.getId(), model.getEmbeddingSize(), tokens);

    // acquire slots
    for (std::size_t slot : match.blockIds) {
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
    }

    processedTokens = match.matchedTokens;

    // reserve blocks
    reservedBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    if (reservedBlocks > 0) {
        gpu.reserve(reservedBlocks, *this);
    }
}
