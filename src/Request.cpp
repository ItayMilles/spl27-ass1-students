#include "Request.h"

#include "GPU.h"
#include "Model.h"
#include <algorithm>

// Request

// constructors
Request::Request(int id, const Model &model, int admissionCycle,
                 std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle),
      tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0) {}

// copy constructor
Request::Request(const Request &other, const Model &model)
    : id(other.id), model(model), admissionCycle(other.admissionCycle),
      tokens(other.tokens), memoryBlocks(other.memoryBlocks),
      reservedBlocks(other.reservedBlocks) {}

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
        reservedBlocks -= toUnreserve;
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
      scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens),
      scriptedTokens(model.tokenizer(scriptedOutput)), sequence(tokens),
      processedTokens(0) {
    std::size_t outputSize = std::min(scriptedTokens.size(), maxOutputTokens);
    for (std::size_t i = 0; i < outputSize; i++) {
        sequence.push_back(scriptedTokens[i]);
    }
}
// copy constructor
GenerativeRequest::GenerativeRequest(const GenerativeRequest &other,
                                     const Model &model)
    : Request(other, model), scriptedOutput(other.scriptedOutput),
      maxOutputTokens(other.maxOutputTokens),
      scriptedTokens(other.scriptedTokens), sequence(other.sequence),
      processedTokens(other.processedTokens) {}

// tokens getter
std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}
// max number of cells needed to store the both the input tokens and the output
// tokens
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

    // acquire slotsb that already hold the right tokens
    for (std::size_t slot : match.blockIds) {
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
    }

    gpu.addCacheHits(match.blockIds.size());
    processedTokens = match.matchedTokens;

    // reserve blocks in gpu memory for future use
    reservedBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    if (reservedBlocks > 0) {
        gpu.reserve(reservedBlocks, *this);
    }
}

// allocate one block of memory on the gpu (double dispatch)
void GenerativeRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

// clone the request with a possibly a different model
Request *GenerativeRequest::clone(const Model &model) const {
    return new GenerativeRequest(*this, model);
}

// check if the request has finished
bool GenerativeRequest::finished() const {
    return processedTokens == sequence.size();
}

std::string GenerativeRequest::getOutput() const {
    std::string output;
    std::size_t outputStart = tokens.size();
    if (sequence.size() > outputStart) {
        output += sequence[outputStart];
    }
    for (std::size_t i = outputStart + 1; i < sequence.size(); i++) {
        output += " ";
        output += sequence[i];
    }
    return output;
}

// how many blocks have been reserved
std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

// process one token, execute each cycle
void GenerativeRequest::act(GPU &gpu) {
    // if all tokens have been processed do nothing
    if (finished()) {
        return;
    }

    // calcualte position inside the block
    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t tokensPerBlock = gpu.getBlockSize() / embeddingSize;
    std::size_t position = processedTokens % tokensPerBlock;

    // when position is zero a new block needs to be allocated
    if (position == 0) {
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        reservedBlocks--;
    }

    // last slot is the current block we are writing to
    std::size_t currentSlot = memoryBlocks.back();

    // calculate prefix: all the token from the start of sequence up to the
    // current processed token
    std::vector<std::string> prefix(sequence.begin(),
                                    sequence.begin() + processedTokens + 1);

    // generate the embedding vector from that prefix using the model and store
    // it on the gpu
    // NOTE: offset is the position times the size of each embedding
    model.generateTokenEmbedding(prefix, gpu, currentSlot,
                                 position * embeddingSize);

    // if we filled the block attempt to give it a prefix key
    if (position + 1 == tokensPerBlock) {
        gpu.assignPrefixToBlock(currentSlot, prefix, model.getId());
    }

    // increment processedTokens
    processedTokens++;
}

// Embedding Request

// constructor
EmbeddingRequest::EmbeddingRequest(int id, const Model &model,
                                   int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt), processedTokens(0), result() {
}

// copy constructor
EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest &other,
                                   const Model &model)
    : Request(other, model), processedTokens(other.processedTokens),
      result(other.result) {}

// embedded request has no cacheable tokens
std::vector<std::string> EmbeddingRequest::getCacheableTokens() const {
    return std::vector<std::string>();
}

// max amount of memory cells needed to hold the sum and the current embedding
// vector
std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return model.getEmbeddingSize() * 2;
}

// amout known in advanced so its identical to the max memory needed
std::size_t EmbeddingRequest::getNeededMemory(const GPU &) const {
    return getMaxNeededMemory();
}

// reserve the memory needed for this request on the gpu
void EmbeddingRequest::reserveMemory(GPU &gpu) {
    // reserve blocks
    reservedBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    gpu.reserve(reservedBlocks, *this);

    // allocate the blocks and save thier indices
    while (reservedBlocks > 0) {
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        reservedBlocks--;
    }
}

// allocate one block of memory (double dispatch)
void EmbeddingRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

// clone the request with the given model
Request *EmbeddingRequest::clone(const Model &model) const {
    return new EmbeddingRequest(*this, model);
}

// if all tokens have been processed
bool EmbeddingRequest::finished() const {
    return processedTokens == tokens.size();
}

// create a string represeting the result mean vector
std::string EmbeddingRequest::getOutput() const {
    std::string output = "[";
    std::size_t resultSize = result.size();
    if (resultSize > 0) {
        output += std::to_string(result[0]);
    }
    for (std::size_t i = 1; i < resultSize; i++) {
        output += ", ";
        output += std::to_string(result[i]);
    }
    output += "]";
    return output;
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

// process one token, execute each cycle
void EmbeddingRequest::act(GPU &gpu) {
    // exit early if all token have been processed
    if (finished()) {
        return;
    }

    // calcualte the sum slot and the embedded vector slot and offset
    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t blockSize = gpu.getBlockSize();
    // the first slot is always the sum slot
    std::size_t sumSlot = memoryBlocks[0];
    // the slot of the current embedded vector
    std::size_t embeddingSlot = memoryBlocks[embeddingSize / blockSize];
    size_t embeddedOffset = embeddingSize % blockSize;

    // generate and store the embedding of the current token
    std::vector<std::string> currentToken(1, tokens[processedTokens]);
    model.generateTokenEmbedding(currentToken, gpu, embeddingSlot,
                                 embeddedOffset);

    for (std::size_t i = 0; i < embeddingSize; i++) {
        // sum the current entries of the sum vector and the current embedded
        // vector
        int sum = gpu.readFromBlock(sumSlot, i) +
                  gpu.readFromBlock(embeddingSlot, embeddedOffset + i);
        // write the new sum to the sum slot
        gpu.writeToBlock(sumSlot, i, sum);
    }

    // increment the processed tokens count
    processedTokens++;

    // if this is the last token compute the mean vector and store it in result
    if (finished()) {
        int tokensCount = static_cast<int>(tokens.size());
        result.clear();
        for (std::size_t i = 0; i < embeddingSize; i++) {
            result.push_back(gpu.readFromBlock(sumSlot, i) / tokensCount);
        }
    }
}