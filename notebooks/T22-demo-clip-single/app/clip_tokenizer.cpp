#include "clip_tokenizer.hpp"

#include <QRegularExpression>

#include <zlib.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

constexpr int kMergeCount = 49152 - 256 - 2;
constexpr const char* kStartOfText = "<start_of_text>";
constexpr const char* kEndOfText = "<end_of_text>";

std::string trimLine(std::string line)
{
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

bool startsWithContraction(const QString& text, int offset, QString* token)
{
    static const std::array<QString, 7> contractions = {
        QStringLiteral("'re"), QStringLiteral("'ve"), QStringLiteral("'ll"),
        QStringLiteral("'s"), QStringLiteral("'t"), QStringLiteral("'m"),
        QStringLiteral("'d"),
    };
    for (const QString& candidate : contractions) {
        if (text.midRef(offset, candidate.size()).compare(candidate, Qt::CaseInsensitive) == 0) {
            if (token) {
                *token = text.mid(offset, candidate.size());
            }
            return true;
        }
    }
    return false;
}

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string& bpe_path)
{
    std::vector<int> bytes;
    for (int value = static_cast<int>('!'); value <= static_cast<int>('~'); ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xA1; value <= 0xAC; ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xAE; value <= 0xFF; ++value) {
        bytes.push_back(value);
    }

    std::vector<int> codepoints = bytes;
    std::array<bool, 256> included{};
    for (int value : bytes) {
        included[static_cast<size_t>(value)] = true;
    }
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (!included[static_cast<size_t>(value)]) {
            bytes.push_back(value);
            codepoints.push_back(256 + extra++);
        }
    }
    std::vector<std::string> byte_symbols;
    byte_symbols.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
        const std::string symbol = utf8ForCodepoint(codepoints[i]);
        byte_encoder_[static_cast<size_t>(bytes[i])] = symbol;
        byte_symbols.push_back(symbol);
    }

    gzFile file = gzopen(bpe_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("cannot open CLIP BPE vocabulary: " + bpe_path);
    }

    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(kMergeCount);
    char buffer[4096];
    bool first_line = true;
    while (gzgets(file, buffer, static_cast<int>(sizeof(buffer))) != nullptr &&
           static_cast<int>(merges.size()) < kMergeCount) {
        std::string line = trimLine(buffer);
        if (first_line) {
            first_line = false;
            continue;
        }
        const size_t separator = line.find(' ');
        if (separator == std::string::npos) {
            continue;
        }
        merges.emplace_back(line.substr(0, separator), line.substr(separator + 1));
    }
    const int close_status = gzclose(file);
    if (close_status != Z_OK || static_cast<int>(merges.size()) != kMergeCount) {
        throw std::runtime_error("invalid CLIP BPE vocabulary: expected " +
                                 std::to_string(kMergeCount) + " merges, got " +
                                 std::to_string(merges.size()));
    }

    std::vector<std::string> vocabulary;
    vocabulary.reserve(256 * 2 + merges.size() + 2);
    for (const std::string& symbol : byte_symbols) {
        vocabulary.push_back(symbol);
    }
    for (const std::string& symbol : byte_symbols) {
        vocabulary.push_back(symbol + "</w>");
    }
    for (size_t i = 0; i < merges.size(); ++i) {
        vocabulary.push_back(merges[i].first + merges[i].second);
        bpe_ranks_[pairKey(merges[i].first, merges[i].second)] = static_cast<int>(i);
    }
    vocabulary.emplace_back(kStartOfText);
    vocabulary.emplace_back(kEndOfText);

    for (size_t i = 0; i < vocabulary.size(); ++i) {
        encoder_[vocabulary[i]] = static_cast<int64_t>(i);
    }
    sot_token_id_ = encoder_.at(kStartOfText);
    eot_token_id_ = encoder_.at(kEndOfText);
    bpe_cache_[kStartOfText] = {kStartOfText};
    bpe_cache_[kEndOfText] = {kEndOfText};
}

std::vector<int64_t> ClipTokenizer::tokenize(const QString& text, int context_length)
{
    if (context_length < 2) {
        throw std::runtime_error("CLIP tokenizer context length must be at least 2");
    }
    std::vector<int64_t> result(static_cast<size_t>(context_length), 0);
    std::vector<int64_t> encoded = encode(text);
    result[0] = sot_token_id_;
    const size_t copy_count = std::min(encoded.size(), static_cast<size_t>(context_length - 2));
    std::copy_n(encoded.begin(), copy_count, result.begin() + 1);
    result[copy_count + 1] = eot_token_id_;
    return result;
}

std::vector<int64_t> ClipTokenizer::encode(const QString& text)
{
    std::vector<int64_t> result;
    for (const QString& token : splitTokens(cleanText(text))) {
        const QByteArray utf8 = token.toUtf8();
        std::vector<std::string> symbols;
        symbols.reserve(static_cast<size_t>(utf8.size()));
        std::string cache_key;
        for (char byte : utf8) {
            const std::string& symbol = byte_encoder_[static_cast<unsigned char>(byte)];
            symbols.push_back(symbol);
            cache_key += symbol;
        }
        for (const std::string& piece : applyBpe(cache_key, std::move(symbols))) {
            const auto it = encoder_.find(piece);
            if (it == encoder_.end()) {
                throw std::runtime_error("CLIP BPE produced an unknown token");
            }
            result.push_back(it->second);
        }
    }
    return result;
}

std::vector<std::string> ClipTokenizer::applyBpe(
    const std::string& cache_key,
    std::vector<std::string> symbols)
{
    const auto cached = bpe_cache_.find(cache_key);
    if (cached != bpe_cache_.end()) {
        return cached->second;
    }
    if (symbols.empty()) {
        return {};
    }
    symbols.back() += "</w>";

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        std::string best_first;
        std::string best_second;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto rank = bpe_ranks_.find(pairKey(symbols[i], symbols[i + 1]));
            if (rank != bpe_ranks_.end() && rank->second < best_rank) {
                best_rank = rank->second;
                best_first = symbols[i];
                best_second = symbols[i + 1];
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_first &&
                symbols[i + 1] == best_second) {
                merged.push_back(best_first + best_second);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols = std::move(merged);
    }
    bpe_cache_[cache_key] = symbols;
    return symbols;
}

QString ClipTokenizer::cleanText(const QString& text)
{
    return htmlUnescape(htmlUnescape(text)).simplified().toLower();
}

QString ClipTokenizer::htmlUnescape(QString text)
{
    static const QRegularExpression entity(
        QStringLiteral("&(#x[0-9A-Fa-f]+|#[0-9]+|amp|lt|gt|quot|apos|nbsp);"),
        QRegularExpression::CaseInsensitiveOption);
    int offset = 0;
    while (true) {
        const QRegularExpressionMatch match = entity.match(text, offset);
        if (!match.hasMatch()) {
            break;
        }
        const QString body = match.captured(1);
        const QString normalized_body = body.toLower();
        QString replacement;
        if (body.startsWith(QStringLiteral("#x"), Qt::CaseInsensitive)) {
            bool ok = false;
            const uint codepoint = body.mid(2).toUInt(&ok, 16);
            if (ok) {
                replacement = QString::fromUcs4(&codepoint, 1);
            }
        } else if (body.startsWith(QLatin1Char('#'))) {
            bool ok = false;
            const uint codepoint = body.mid(1).toUInt(&ok, 10);
            if (ok) {
                replacement = QString::fromUcs4(&codepoint, 1);
            }
        } else if (normalized_body == QStringLiteral("amp")) {
            replacement = QStringLiteral("&");
        } else if (normalized_body == QStringLiteral("lt")) {
            replacement = QStringLiteral("<");
        } else if (normalized_body == QStringLiteral("gt")) {
            replacement = QStringLiteral(">");
        } else if (normalized_body == QStringLiteral("quot")) {
            replacement = QStringLiteral("\"");
        } else if (normalized_body == QStringLiteral("apos")) {
            replacement = QStringLiteral("'");
        } else if (normalized_body == QStringLiteral("nbsp")) {
            replacement = QStringLiteral(" ");
        }
        if (replacement.isNull()) {
            offset = match.capturedEnd();
            continue;
        }
        text.replace(match.capturedStart(), match.capturedLength(), replacement);
        offset = match.capturedStart() + replacement.size();
    }
    return text;
}

std::vector<QString> ClipTokenizer::splitTokens(const QString& text)
{
    std::vector<QString> tokens;
    int index = 0;
    while (index < text.size()) {
        const QChar current = text[index];
        if (current.isSpace()) {
            ++index;
            continue;
        }

        QString contraction;
        if (current == QLatin1Char('\'') && startsWithContraction(text, index, &contraction)) {
            tokens.push_back(contraction);
            index += contraction.size();
            continue;
        }
        if (current.isLetter()) {
            const int start = index++;
            while (index < text.size() && text[index].isLetter()) {
                ++index;
            }
            tokens.push_back(text.mid(start, index - start));
            continue;
        }
        if (current.isNumber()) {
            tokens.push_back(QString(current));
            ++index;
            continue;
        }

        const int start = index++;
        while (index < text.size() && !text[index].isSpace() &&
               !text[index].isLetter() && !text[index].isNumber()) {
            QString ignored;
            if (text[index] == QLatin1Char('\'') && startsWithContraction(text, index, &ignored)) {
                break;
            }
            ++index;
        }
        tokens.push_back(text.mid(start, index - start));
    }
    return tokens;
}

std::string ClipTokenizer::pairKey(const std::string& first, const std::string& second)
{
    return first + '\0' + second;
}

std::string ClipTokenizer::utf8ForCodepoint(int codepoint)
{
    std::string result;
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        result.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return result;
}
