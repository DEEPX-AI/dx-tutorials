#pragma once

#include <QString>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// C++ port of OpenAI/OpenCLIP's SimpleTokenizer (MIT licensed).
class ClipTokenizer {
public:
    explicit ClipTokenizer(const std::string& bpe_path);

    std::vector<int64_t> tokenize(const QString& text, int context_length = 77);
    int sotTokenId() const { return sot_token_id_; }
    int eotTokenId() const { return eot_token_id_; }

private:
    static std::string utf8ForCodepoint(int codepoint);
    static QString htmlUnescape(QString text);
    static QString cleanText(const QString& text);
    static std::vector<QString> splitTokens(const QString& text);
    static std::string pairKey(const std::string& first, const std::string& second);

    std::vector<std::string> applyBpe(
        const std::string& cache_key,
        std::vector<std::string> symbols);
    std::vector<int64_t> encode(const QString& text);

    std::array<std::string, 256> byte_encoder_{};
    std::unordered_map<std::string, int64_t> encoder_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
    int64_t sot_token_id_ = -1;
    int64_t eot_token_id_ = -1;
};
