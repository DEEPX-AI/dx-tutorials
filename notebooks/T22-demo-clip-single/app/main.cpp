#include "clip_tokenizer.hpp"

#include <dxrt/dxrt_cxx_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSizePolicy>
#include <QTimer>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

Q_DECLARE_METATYPE(cv::Mat)

namespace fs = std::filesystem;

namespace {

constexpr int kPreviewWidth = 1280;
constexpr int kPreviewHeight = 720;
constexpr int kImageSize = 224;
constexpr int kContextLength = 77;
constexpr int kEmbeddingSize = 768;
constexpr int kMaxAsyncJobs = 8;
constexpr float kHighlightMinScore = 0.25F;
constexpr float kHighlightMaxScore = 0.35F;

struct AppOptions {
    std::string text_encoder = "assets/models/ViT-L-14-quickgelu-dfn2b-text.onnx";
    std::string image_encoder = "assets/models/ViT-L-14-quickgelu-dfn2b.dxnn";
    std::string bpe_vocab = "assets/models/bpe_simple_vocab_16e6.txt.gz";
    std::string model_name;
    std::vector<QString> texts;
    std::string input;
    std::string camera = "/dev/video0";
    int width = 1920;
    int height = 1080;
    int fps = 30;
    int skip_frames = 2;
    bool no_normalize = false;
    bool show_exit_button = false;
    bool full_screen = false;
};

std::string requireValue(int& index, int argc, char** argv)
{
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

int parsePositiveInt(const std::string& value, const char* option, bool allow_zero = false)
{
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects an integer");
    }
    if (consumed != value.size() || (allow_zero ? parsed < 0 : parsed <= 0)) {
        throw std::runtime_error(std::string(option) +
                                 (allow_zero ? " expects a non-negative integer"
                                             : " expects a positive integer"));
    }
    return parsed;
}

void printUsage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " --texts TEXT [TEXT ...] [OPTIONS]\n"
        << "      --texts TEXT [TEXT ...]\n"
        << "                              Text queries to compare (required)\n"
        << "      --text-encoder <PATH>   ONNX CLIP text encoder\n"
        << "      --image-encoder <PATH>  DXNN CLIP image encoder\n"
        << "      --model-name <NAME>     OpenCLIP model name used for cache identity\n"
        << "      --bpe-vocab <PATH>      OpenCLIP BPE vocabulary\n"
        << "      --input <SOURCE>        Camera index/device or video file\n"
        << "      --camera <SOURCE>       Camera source when --input is omitted (default: /dev/video0)\n"
        << "      --width <N>             Camera/video output width (default: 1920)\n"
        << "      --height <N>            Camera/video output height (default: 1080)\n"
        << "      --fps <N>               Camera target FPS (default: 30)\n"
        << "      --skip-frames <N>       Infer every N+1 frames (default: 2)\n"
        << "      --no-normalize          Disable L2 feature normalization\n"
        << "      --exit-btn              Show an exit button beside Apply\n"
        << "      --full-screen           Show the GUI in fullscreen mode\n"
        << "  -h, --help                  Show this help\n";
}

std::string extractModelName(const std::string& onnx_path)
{
    const std::string filename = fs::path(onnx_path).filename().string();
    const std::regex pattern(R"((.+?)-[^-]+-text\.onnx)");
    std::smatch match;
    if (std::regex_match(filename, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    return "ViT-L-14-quickgelu";
}

AppOptions parseArgs(int argc, char** argv)
{
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--text-encoder") {
            options.text_encoder = requireValue(i, argc, argv);
        } else if (arg == "--image-encoder") {
            options.image_encoder = requireValue(i, argc, argv);
        } else if (arg == "--bpe-vocab") {
            options.bpe_vocab = requireValue(i, argc, argv);
        } else if (arg == "--model-name") {
            options.model_name = requireValue(i, argc, argv);
        } else if (arg == "--input") {
            options.input = requireValue(i, argc, argv);
        } else if (arg == "--camera") {
            options.camera = requireValue(i, argc, argv);
        } else if (arg == "--width") {
            options.width = parsePositiveInt(requireValue(i, argc, argv), "--width");
        } else if (arg == "--height") {
            options.height = parsePositiveInt(requireValue(i, argc, argv), "--height");
        } else if (arg == "--fps") {
            options.fps = parsePositiveInt(requireValue(i, argc, argv), "--fps");
        } else if (arg == "--skip-frames") {
            options.skip_frames = parsePositiveInt(
                requireValue(i, argc, argv), "--skip-frames", true);
        } else if (arg == "--texts") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                options.texts.push_back(QString::fromUtf8(argv[++i]));
            }
            if (options.texts.empty()) {
                throw std::runtime_error("--texts requires at least one text query");
            }
        } else if (arg == "--no-normalize") {
            options.no_normalize = true;
        } else if (arg == "--exit-btn") {
            options.show_exit_button = true;
        } else if (arg == "--full-screen") {
            options.full_screen = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.texts.empty()) {
        throw std::runtime_error("--texts is required");
    }
    if (options.input.empty()) {
        options.input = options.camera;
    }
    if (options.model_name.empty()) {
        options.model_name = extractModelName(options.text_encoder);
    }
    return options;
}

fs::path resolvePath(const std::string& value)
{
    fs::path path(value);
    if (fs::exists(path)) {
        return fs::absolute(path);
    }
#ifdef PROJECT_ROOT_DIR
    const fs::path rooted = fs::path(PROJECT_ROOT_DIR) / path;
    if (fs::exists(rooted)) {
        return rooted;
    }
#endif
    return path;
}

bool isIntegerString(const std::string& value)
{
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

bool isVideoFile(const std::string& value)
{
    std::error_code error;
    return fs::is_regular_file(fs::path(value), error);
}

bool isNearlyBlack(const cv::Mat& frame)
{
    if (frame.empty()) {
        return true;
    }
    cv::Scalar mean = cv::mean(frame);
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(frame.reshape(1), &minimum, &maximum);
    return (mean[0] + mean[1] + mean[2]) / 3.0 < 1.0 && maximum <= 3.0;
}

cv::Mat cropCenter(const cv::Mat& frame, int width, int height)
{
    if (width >= frame.cols || height >= frame.rows) {
        return frame;
    }
    const int x = (frame.cols - width) / 2;
    const int y = (frame.rows - height) / 2;
    return frame(cv::Rect(x, y, width, height));
}

QImage matToQImage(const cv::Mat& bgr)
{
    if (bgr.empty()) {
        return {};
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                  QImage::Format_RGB888).copy();
}

std::vector<float> preprocessFrame(const cv::Mat& frame)
{
    if (frame.empty()) {
        throw std::runtime_error("cannot preprocess an empty frame");
    }
    const double scale = static_cast<double>(kImageSize) /
                         static_cast<double>(std::min(frame.cols, frame.rows));
    const int resized_width = std::max(kImageSize, static_cast<int>(frame.cols * scale));
    const int resized_height = std::max(kImageSize, static_cast<int>(frame.rows * scale));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_CUBIC);
    const int x = (resized.cols - kImageSize) / 2;
    const int y = (resized.rows - kImageSize) / 2;
    const cv::Mat crop = resized(cv::Rect(x, y, kImageSize, kImageSize));

    constexpr std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> stddev = {0.229F, 0.224F, 0.225F};
    const size_t plane = static_cast<size_t>(kImageSize * kImageSize);
    std::vector<float> tensor(plane * 3);
    for (int row = 0; row < kImageSize; ++row) {
        const cv::Vec3b* pixels = crop.ptr<cv::Vec3b>(row);
        for (int col = 0; col < kImageSize; ++col) {
            const size_t offset = static_cast<size_t>(row * kImageSize + col);
            tensor[offset] = (pixels[col][2] / 255.0F - mean[0]) / stddev[0];
            tensor[plane + offset] = (pixels[col][1] / 255.0F - mean[1]) / stddev[1];
            tensor[plane * 2 + offset] = (pixels[col][0] / 255.0F - mean[2]) / stddev[2];
        }
    }
    return tensor;
}

void normalizeRows(std::vector<float>& features, size_t rows, size_t columns)
{
    if (rows == 0 || columns == 0 || features.size() != rows * columns) {
        throw std::runtime_error("invalid embedding matrix shape");
    }
    for (size_t row = 0; row < rows; ++row) {
        float* values = features.data() + row * columns;
        double sum = 0.0;
        for (size_t col = 0; col < columns; ++col) {
            sum += static_cast<double>(values[col]) * values[col];
        }
        const float norm = static_cast<float>(std::sqrt(sum));
        if (norm <= 1e-12F) {
            continue;
        }
        for (size_t col = 0; col < columns; ++col) {
            values[col] /= norm;
        }
    }
}

int similarityToBarValue(float score)
{
    if (score <= 0.0F) {
        return 0;
    }
    if (score >= kHighlightMaxScore) {
        return 1000;
    }
    if (score <= kHighlightMinScore) {
        return static_cast<int>(std::lround(280.0F * score / kHighlightMinScore));
    }
    if (score <= kHighlightMinScore + 0.05F) {
        const float t = (score - kHighlightMinScore) / 0.05F;
        return static_cast<int>(std::lround(280.0F + 340.0F * t));
    }
    const float t = (score - kHighlightMinScore + 0.05F) / 0.05F;
    return static_cast<int>(std::lround(620.0F + 380.0F * t));
}

class CameraThread : public QThread {
    Q_OBJECT

public:
    CameraThread(std::string input, int width, int height, int fps, QObject* parent = nullptr)
        : QThread(parent), input_(std::move(input)), width_(width), height_(height), fps_(fps)
    {
    }

    ~CameraThread() override { stop(); }

    void stop()
    {
        stop_requested_.store(true);
        if (isRunning()) {
            wait();
        }
    }

signals:
    void frameReady(const cv::Mat& frame);
    void captureError(const QString& message);

protected:
    void run() override
    {
        const bool video_file = isVideoFile(input_);
        cv::VideoCapture capture;
        bool opened = false;
        if (isIntegerString(input_)) {
            opened = capture.open(std::stoi(input_), cv::CAP_V4L2);
            if (!opened) {
                opened = capture.open(std::stoi(input_));
            }
        } else if (input_.rfind("/dev/video", 0) == 0) {
            opened = capture.open(input_, cv::CAP_V4L2);
            if (!opened) {
                opened = capture.open(input_);
            }
        } else {
            opened = capture.open(input_);
        }

        cv::Mat fallback;
        if (!opened) {
            const fs::path fallback_path = resolvePath("assets/images/img-encoder-sample-1.png");
            fallback = cv::imread(fallback_path.string());
            if (fallback.empty()) {
                emit captureError(QString("Input %1 is unavailable and the fallback image is missing")
                                      .arg(QString::fromStdString(input_)));
                return;
            }
            cv::resize(fallback, fallback, cv::Size(width_, height_));
            std::cout << "Input " << input_ << " not available, using test image" << std::endl;
        } else if (!video_file) {
            capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            capture.set(cv::CAP_PROP_FRAME_WIDTH, width_);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
            capture.set(cv::CAP_PROP_FPS, fps_);
        }

        double frame_seconds = 1.0 / std::max(1, fps_);
        if (opened && video_file) {
            const double source_fps = capture.get(cv::CAP_PROP_FPS);
            if (source_fps > 0.0) {
                frame_seconds = 1.0 / source_fps;
            }
        }
        uint64_t frame_count = 0;
        bool saved_black = false;
        bool saved_nonblack = false;
        while (!stop_requested_.load()) {
            const auto started = std::chrono::steady_clock::now();
            cv::Mat frame;
            if (!opened) {
                frame = fallback.clone();
            } else if (!capture.read(frame) || frame.empty()) {
                if (video_file) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                }
                QThread::msleep(2);
                continue;
            }
            if (video_file && (frame.cols != width_ || frame.rows != height_)) {
                cv::resize(frame, frame, cv::Size(width_, height_));
            }
            ++frame_count;
            if (!saved_black && isNearlyBlack(frame)) {
                saved_black = true;
                cv::imwrite("debug_camera_black_frame.png", frame);
            } else if (!saved_nonblack && !isNearlyBlack(frame)) {
                saved_nonblack = true;
                cv::imwrite("debug_camera_nonblack_frame.png", frame);
            }
            emit frameReady(frame.clone());

            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const double remaining = frame_seconds - elapsed - 0.005;
            if (remaining > 0.0) {
                QThread::usleep(static_cast<unsigned long>(remaining * 1'000'000.0));
            }
        }
        capture.release();
    }

private:
    std::string input_;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 0;
    std::atomic_bool stop_requested_{false};
};

class TextEncoder {
public:
    TextEncoder(const std::string& model_path, const std::string& bpe_path)
        : tokenizer_(bpe_path), environment_(ORT_LOGGING_LEVEL_WARNING, "clip_text_encoder")
    {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
        session_ = std::make_unique<Ort::Session>(
            environment_, model_path.c_str(), session_options_);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1) {
            throw std::runtime_error("unexpected CLIP text encoder I/O count");
        }
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    }

    std::vector<float> encode(const std::vector<QString>& texts)
    {
        if (texts.empty()) {
            return {};
        }
        std::vector<float> embeddings;
        embeddings.reserve(texts.size() * kEmbeddingSize);
        const std::array<int64_t, 2> shape = {1, kContextLength};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        for (const QString& text : texts) {
            std::vector<int64_t> tokens = tokenizer_.tokenize(text, kContextLength);
            Ort::Value input = Ort::Value::CreateTensor<int64_t>(
                memory, tokens.data(), tokens.size(), shape.data(), shape.size());
            Ort::RunOptions run_options;
            std::vector<Ort::Value> outputs = session_->Run(
                run_options, input_names, &input, 1, output_names, 1);
            if (outputs.empty() || !outputs.front().IsTensor()) {
                throw std::runtime_error("CLIP text encoder returned no tensor");
            }
            const auto info = outputs.front().GetTensorTypeAndShapeInfo();
            const size_t count = info.GetElementCount();
            const float* data = outputs.front().GetTensorData<float>();
            if (count != kEmbeddingSize) {
                throw std::runtime_error("unexpected CLIP text embedding shape");
            }
            embeddings.insert(embeddings.end(), data, data + count);
        }
        return embeddings;
    }

private:
    ClipTokenizer tokenizer_;
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};

class ImageEncoderAsync {
public:
    struct Result {
        uint64_t job_id = 0;
        std::vector<float> features;
        std::string error;
    };

    explicit ImageEncoderAsync(const std::string& model_path)
    {
        dxrt::InferenceOption option;
        option.bufferCount = kMaxAsyncJobs;
        engine_ = std::make_unique<dxrt::InferenceEngine>(model_path, option);
        const auto inputs = engine_->GetInputs();
        if (inputs.empty() || inputs.front().type() != dxrt::DataType::FLOAT) {
            throw std::runtime_error("CLIP image encoder must have a FLOAT input");
        }
        const std::vector<int64_t> shape = inputs.front().shape();
        if (shape.size() != 4 || shape[1] != 3 || shape[2] != kImageSize ||
            shape[3] != kImageSize) {
            throw std::runtime_error("CLIP image encoder input must be [1,3,224,224]");
        }
        engine_->RegisterCallback([this](dxrt::TensorPtrs& outputs, void* user_arg) -> int {
            return onComplete(outputs, user_arg);
        });
    }

    ~ImageEncoderAsync() { shutdown(); }

    bool trySubmit(std::vector<float> input, uint64_t job_id)
    {
        if (closing_.load()) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (in_flight_ >= kMaxAsyncJobs) {
                return false;
            }
            ++in_flight_;
        }

        std::unique_ptr<Job> job(new Job{job_id, std::move(input)});
        Job* raw = job.release();
        try {
            const int request = engine_->RunAsync(raw->input.data(), raw);
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_request_id_ = request;
            return true;
        } catch (...) {
            delete raw;
            completeOne();
            throw;
        }
    }

    std::vector<Result> drainResults()
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        std::vector<Result> results;
        results.reserve(results_.size());
        while (!results_.empty()) {
            results.push_back(std::move(results_.front()));
            results_.pop_front();
        }
        return results;
    }

    void shutdown()
    {
        if (closing_.exchange(true) || !engine_) {
            return;
        }
        int last_request = -1;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_request = last_request_id_;
        }
        if (last_request >= 0) {
            try {
                engine_->Wait(last_request);
            } catch (...) {
            }
        }
        std::unique_lock<std::mutex> lock(state_mutex_);
        state_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lock.unlock();
        engine_->RegisterCallback({});
        engine_.reset();
    }

private:
    struct Job {
        uint64_t job_id;
        std::vector<float> input;
    };

    int onComplete(dxrt::TensorPtrs& outputs, void* user_arg)
    {
        std::unique_ptr<Job> job(static_cast<Job*>(user_arg));
        Result result;
        result.job_id = job ? job->job_id : 0;
        try {
            if (!job) {
                throw std::runtime_error("DXNN callback received no job context");
            }
            if (outputs.empty() || !outputs.front() ||
                outputs.front()->type() != dxrt::DataType::FLOAT) {
                throw std::runtime_error("DXNN image encoder returned no FLOAT output");
            }
            const size_t count = static_cast<size_t>(outputs.front()->size_in_bytes()) /
                                 sizeof(float);
            const float* data = static_cast<const float*>(outputs.front()->data());
            result.features.assign(data, data + count);
            if (result.features.size() != kEmbeddingSize) {
                throw std::runtime_error("unexpected DXNN image embedding shape");
            }
        } catch (const std::exception& error) {
            result.error = error.what();
        }

        if (!closing_.load()) {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(std::move(result));
        }
        completeOne();
        return 0;
    }

    void completeOne()
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (in_flight_ > 0) {
            --in_flight_;
        }
        state_cv_.notify_all();
    }

    std::unique_ptr<dxrt::InferenceEngine> engine_;
    std::atomic_bool closing_{false};
    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    int in_flight_ = 0;
    int last_request_id_ = -1;
    std::mutex results_mutex_;
    std::deque<Result> results_;
};

class TextFeatureStore {
public:
    explicit TextFeatureStore(const AppOptions& options) : options_(options) {}

    std::vector<float> featuresFor(const std::vector<QString>& texts)
    {
        const fs::path cache_path = cachePath(texts);
        std::vector<float> cached;
        if (load(cache_path, texts.size(), cached)) {
            std::cout << "Loaded cached text features: " << cache_path << std::endl;
            return cached;
        }
        if (!encoder_) {
            encoder_ = std::make_unique<TextEncoder>(options_.text_encoder, options_.bpe_vocab);
        }
        std::vector<float> features = encoder_->encode(texts);
        if (!options_.no_normalize) {
            normalizeRows(features, texts.size(), kEmbeddingSize);
        }
        save(cache_path, texts.size(), features);
        return features;
    }

private:
    fs::path cachePath(const std::vector<QString>& texts) const
    {
        QCryptographicHash hash(QCryptographicHash::Sha256);
        const QFileInfo encoder_info(QString::fromStdString(options_.text_encoder));
        hash.addData(encoder_info.absoluteFilePath().toUtf8());
        hash.addData(QByteArray::number(encoder_info.size()));
        hash.addData(QByteArray::number(encoder_info.lastModified().toMSecsSinceEpoch()));
        hash.addData(QByteArray::fromStdString(options_.model_name));
        hash.addData(options_.no_normalize ? "raw" : "normalized");
        for (const QString& text : texts) {
            hash.addData("\0", 1);
            hash.addData(text.toUtf8());
        }
#ifdef PROJECT_ROOT_DIR
        const fs::path directory = fs::path(PROJECT_ROOT_DIR) / ".cache" / "text_features_cpp";
#else
        const fs::path directory = fs::path(".cache") / "text_features_cpp";
#endif
        return directory / (hash.result().toHex().toStdString() + ".bin");
    }

    static bool load(const fs::path& path, size_t rows, std::vector<float>& features)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return false;
        }
        std::array<char, 8> magic{};
        uint32_t stored_rows = 0;
        uint32_t columns = 0;
        input.read(magic.data(), magic.size());
        input.read(reinterpret_cast<char*>(&stored_rows), sizeof(stored_rows));
        input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
        if (!input || std::string(magic.data(), magic.size()) != "CLIPCPP1" ||
            stored_rows != rows || columns != kEmbeddingSize) {
            return false;
        }
        features.resize(rows * columns);
        input.read(reinterpret_cast<char*>(features.data()),
                   static_cast<std::streamsize>(features.size() * sizeof(float)));
        return input && input.peek() == std::char_traits<char>::eof();
    }

    static void save(const fs::path& path, size_t rows, const std::vector<float>& features)
    {
        std::error_code error;
        fs::create_directories(path.parent_path(), error);
        if (error) {
            return;
        }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            return;
        }
        const uint32_t stored_rows = static_cast<uint32_t>(rows);
        const uint32_t columns = kEmbeddingSize;
        output.write("CLIPCPP1", 8);
        output.write(reinterpret_cast<const char*>(&stored_rows), sizeof(stored_rows));
        output.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
        output.write(reinterpret_cast<const char*>(features.data()),
                     static_cast<std::streamsize>(features.size() * sizeof(float)));
    }

    AppOptions options_;
    std::unique_ptr<TextEncoder> encoder_;
};

class TextRowWidget : public QWidget {
public:
    explicit TextRowWidget(QString text, QWidget* parent = nullptr)
        : QWidget(parent), text_(std::move(text))
    {
        label_ = new QLabel(text_);
        label_->setWordWrap(true);
        score_label_ = new QLabel("0.000");
        progress_ = new QProgressBar();
        progress_->setRange(0, 1000);
        progress_->setTextVisible(false);
        progress_->setFixedHeight(16);

        auto* top_row = new QHBoxLayout();
        top_row->setContentsMargins(0, 0, 0, 0);
        top_row->addWidget(progress_, 1);
        top_row->addWidget(score_label_);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(8, 8, 8, 8);
        layout->setSpacing(6);
        layout->addLayout(top_row);
        layout->addWidget(label_);
        setHighlightRank(-1);
    }

    void updateScore(float score)
    {
        history_.push_back(score);
        if (history_.size() > 32) {
            history_.pop_front();
        }
        progress_->setValue(std::clamp(similarityToBarValue(score), 0, 1000));
        score_label_->setText(QString::number(score, 'f', 3));
    }

    void setHighlightRank(int rank)
    {
        QString chunk = "#4a5568";
        QString border = "#cbd5e1";
        QString weight = "normal";
        QString background = "#f8fafc";
        if (rank == 0) {
            chunk = "#d62828";
            border = "#ff7b7b";
            weight = "700";
            background = "#fff1f1";
        } else if (rank == 1) {
            chunk = "#f59f00";
            border = "#ffd43b";
            weight = "600";
            background = "#fff9db";
        }
        label_->setStyleSheet(QString("font-size:14px;color:#111111;font-weight:%1;").arg(weight));
        score_label_->setStyleSheet(
            QString("font-size:12px;color:#111111;min-width:56px;font-weight:%1;").arg(weight));
        setStyleSheet(QString("background-color:%1;border:1px solid %2;border-radius:8px;")
                          .arg(background, border));
        progress_->setStyleSheet(QString(R"(
            QProgressBar { border:1px solid %1; border-radius:4px; background:#e5e7eb; }
            QProgressBar::chunk { background:%2; border-radius:4px; }
        )").arg(border, chunk));
    }

private:
    QString text_;
    QLabel* label_ = nullptr;
    QLabel* score_label_ = nullptr;
    QProgressBar* progress_ = nullptr;
    std::deque<float> history_;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(AppOptions options, QWidget* parent = nullptr)
        : QMainWindow(parent), options_(std::move(options)), texts_(options_.texts),
          feature_store_(options_), video_input_(isVideoFile(options_.input))
    {
        setWindowTitle("Camera Text Matcher GUI (Async C++)");
        resize(1920, 1080);
        setupUi();
        rebuildTextWidgets();
        setupModels();
        setupCamera();

        result_timer_ = new QTimer(this);
        connect(result_timer_, &QTimer::timeout, this, &MainWindow::pollResults);
        result_timer_->start(30);

        auto* escape = new QShortcut(QKeySequence(Qt::Key_Escape), this);
        escape->setContext(Qt::ApplicationShortcut);
        connect(escape, &QShortcut::activated, this, &MainWindow::close);
        auto* quit = new QShortcut(QKeySequence(Qt::Key_Q), this);
        quit->setContext(Qt::ApplicationShortcut);
        connect(quit, &QShortcut::activated, this, [this] {
            QWidget* focused = QApplication::focusWidget();
            if (focused == add_text_editor_ ||
                (focused && add_text_editor_->isAncestorOf(focused))) {
                return;
            }
            close();
        });
    }

protected:
    void closeEvent(QCloseEvent* event) override
    {
        shutting_down_ = true;
        if (result_timer_) {
            result_timer_->stop();
        }
        if (camera_thread_) {
            disconnect(camera_thread_, nullptr, this, nullptr);
            camera_thread_->stop();
            camera_thread_->deleteLater();
            camera_thread_ = nullptr;
        }
        if (image_encoder_) {
            image_encoder_->shutdown();
            image_encoder_.reset();
        }
        event->accept();
    }

private:
    void setupUi()
    {
        auto* container = new QWidget();
        container->setStyleSheet("background:#111827;");
        setCentralWidget(container);
        auto* main_layout = new QHBoxLayout(container);

        auto* left_layout = new QVBoxLayout();
        video_label_ = new QLabel("Camera feed not ready");
        video_label_->setFixedSize(kPreviewWidth, kPreviewHeight);
        video_label_->setAlignment(Qt::AlignCenter);
        video_label_->setStyleSheet("background:#111;color:#fff;");
        left_layout->addWidget(video_label_);

        auto* match_panel = new QWidget();
        match_panel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        auto* match_layout = new QVBoxLayout(match_panel);
        match_layout->setContentsMargins(0, 0, 0, 0);
        match_layout->setSpacing(8);
        status_label_ = new QLabel("Starting up...");
        status_label_->setWordWrap(true);
        status_label_->setTextFormat(Qt::PlainText);
        status_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        status_label_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        status_label_->setMinimumHeight(96);
        status_label_->setStyleSheet(R"(
            QLabel { font-size:36px;font-weight:700;color:#fff;background:#202632;
                     border:2px solid #4b5563;border-radius:8px;padding:14px 18px; }
        )");
        match_layout->addWidget(status_label_, 2);
        source_info_label_ = new QLabel();
        source_info_label_->setWordWrap(true);
        source_info_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        source_info_label_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        source_info_label_->setMinimumHeight(48);
        source_info_label_->setStyleSheet(R"(
            QLabel { font-size:36px;font-weight:500;color:#94a3b8;background:#1a1f2e;
                     border:1px solid #374151;border-radius:6px;padding:8px 12px; }
        )");
        source_info_label_->setText(
            QString("AI Model: %1")
                .arg(QString::fromStdString(fs::path(options_.image_encoder).filename().string())));
        match_layout->addWidget(source_info_label_, 1);
        left_layout->addWidget(match_panel, 1);
        main_layout->addLayout(left_layout);

        auto* right_layout = new QVBoxLayout();
        auto* add_layout = new QHBoxLayout();
        add_layout->setContentsMargins(0, 0, 0, 0);
        add_layout->setSpacing(8);
        add_text_editor_ = new QPlainTextEdit();
        add_text_editor_->setPlaceholderText("Add 1-2 lines, then Apply");
        add_text_editor_->setFixedHeight(56);
        add_text_editor_->setStyleSheet(R"(
            QPlainTextEdit { background:#fff;color:#111827;border:1px solid #cbd5e1;
                             border-radius:6px;padding:6px 8px;font-size:13px; }
        )");
        add_layout->addWidget(add_text_editor_, 1);
        apply_button_ = new QPushButton("Apply");
        apply_button_->setFixedHeight(56);
        apply_button_->setStyleSheet(R"(
            QPushButton { background:#0f766e;color:#fff;border:none;border-radius:6px;
                          padding:0 14px;font-size:14px;font-weight:700; }
            QPushButton:hover { background:#0d9488; }
            QPushButton:disabled { background:#94a3b8; }
        )");
        connect(apply_button_, &QPushButton::clicked, this, &MainWindow::applyAdditionalTexts);
        add_layout->addWidget(apply_button_);

        if (options_.show_exit_button) {
            exit_button_ = new QPushButton("Exit");
            exit_button_->setFixedSize(56, 56);
            exit_button_->setFocusPolicy(Qt::NoFocus);
            exit_button_->setToolTip("Exit");
            exit_button_->setStyleSheet(R"(
                QPushButton { color:#f4f4f5;background:#252a34;
                              border:1px solid #525866;border-radius:6px;
                              font-size:15px;font-weight:700; }
                QPushButton:hover { color:#fff;background:#dc2626;border-color:#ef4444; }
                QPushButton:pressed { background:#991b1b; }
            )");
            connect(exit_button_, &QPushButton::clicked, this, &MainWindow::close);
            add_layout->addWidget(exit_button_);
        }
        right_layout->addLayout(add_layout);

        text_container_ = new QWidget();
        text_container_->setStyleSheet("background:#f1f5f9;");
        text_layout_ = new QVBoxLayout(text_container_);
        text_layout_->setContentsMargins(0, 0, 0, 0);
        text_layout_->setSpacing(6);
        text_layout_->setAlignment(Qt::AlignTop);
        auto* scroll = new QScrollArea();
        scroll->setWidgetResizable(true);
        scroll->setWidget(text_container_);
        right_layout->addWidget(scroll);
        main_layout->addLayout(right_layout, 1);

    }

    void setupModels()
    {
        try {
            setStatusPlain("Preparing your text list...");
            text_features_ = feature_store_.featuresFor(texts_);
            image_encoder_ = std::make_unique<ImageEncoderAsync>(options_.image_encoder);
            setStatusPlain("Starting live image encoder...");
        } catch (const std::exception& error) {
            showError(QString("Model setup failed: %1").arg(error.what()));
        }
    }

    void setupCamera()
    {
        camera_thread_ = new CameraThread(
            options_.input, options_.width, options_.height, options_.fps, this);
        connect(camera_thread_, &CameraThread::frameReady,
                this, &MainWindow::onFrame, Qt::QueuedConnection);
        connect(camera_thread_, &CameraThread::captureError,
                this, &MainWindow::showError, Qt::QueuedConnection);
        camera_thread_->start();
        if (image_encoder_) {
            setStatusPlain("Camera on — matches will update live below.");
        }
    }

    void rebuildTextWidgets()
    {
        while (QLayoutItem* item = text_layout_->takeAt(0)) {
            if (QWidget* widget = item->widget()) {
                widget->deleteLater();
            }
            delete item;
        }
        text_widgets_.clear();
        for (const QString& text : texts_) {
            auto* widget = new TextRowWidget(text);
            text_layout_->addWidget(widget);
            text_widgets_.push_back(widget);
        }
    }

    void setStatusPlain(const QString& text)
    {
        status_label_->setTextFormat(Qt::PlainText);
        status_label_->setText(text);
    }

    void showError(const QString& message)
    {
        setStatusPlain("Setup issue — " + message);
        std::cerr << message.toStdString() << std::endl;
    }

    void applyAdditionalTexts()
    {
        const QStringList raw_lines = add_text_editor_->toPlainText().split('\n');
        std::vector<QString> appended;
        for (const QString& raw : raw_lines) {
            const QString text = raw.trimmed();
            if (text.isEmpty()) {
                continue;
            }
            const bool already_present = std::find(texts_.begin(), texts_.end(), text) != texts_.end();
            const bool already_added = std::find(appended.begin(), appended.end(), text) != appended.end();
            if (!already_present && !already_added) {
                appended.push_back(text);
            }
        }
        if (appended.empty()) {
            setStatusPlain(raw_lines.join("").trimmed().isEmpty()
                               ? "Enter one or two text lines to add, then press Apply."
                               : "Those text lines are already in the list.");
            add_text_editor_->clear();
            return;
        }

        std::vector<QString> updated = appended;
        updated.insert(updated.end(), texts_.begin(), texts_.end());
        apply_button_->setEnabled(false);
        try {
            std::vector<float> features = feature_store_.featuresFor(updated);
            texts_ = std::move(updated);
            text_features_ = std::move(features);
            rebuildTextWidgets();
            add_text_editor_->clear();
            setStatusPlain(QString("Added %1 text quer%2 to the list.")
                               .arg(appended.size())
                               .arg(appended.size() == 1 ? "y" : "ies"));
        } catch (const std::exception& error) {
            showError(QString("Adding text inputs failed: %1").arg(error.what()));
        }
        apply_button_->setEnabled(true);
    }

    void onFrame(const cv::Mat& frame)
    {
        if (shutting_down_ || frame.empty()) {
            return;
        }
        const cv::Mat preview = video_input_
                                    ? frame
                                    : cropCenter(frame, kPreviewWidth, kPreviewHeight);
        const QImage image = matToQImage(preview);
        if (!image.isNull()) {
            video_label_->setPixmap(QPixmap::fromImage(image).scaled(
                video_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }

        const uint64_t frame_index = frame_index_++;
        if (!image_encoder_ || text_features_.empty()) {
            return;
        }
        const uint64_t interval = static_cast<uint64_t>(options_.skip_frames + 1);
        if (frame_index % interval != 0) {
            return;
        }
        try {
            image_encoder_->trySubmit(preprocessFrame(frame), next_job_id_++);
        } catch (const std::exception& error) {
            std::cerr << "Preprocess/encode_async error: " << error.what() << std::endl;
        }
    }

    void pollResults()
    {
        if (shutting_down_ || !image_encoder_ || text_features_.empty()) {
            return;
        }
        for (ImageEncoderAsync::Result& result : image_encoder_->drainResults()) {
            if (!result.error.empty()) {
                showError(QString::fromStdString(result.error));
                continue;
            }
            if (!options_.no_normalize) {
                normalizeRows(result.features, 1, kEmbeddingSize);
            }
            updateSimilarities(result.features);
        }
    }

    void updateSimilarities(const std::vector<float>& image_features)
    {
        if (image_features.size() != kEmbeddingSize ||
            text_features_.size() != texts_.size() * kEmbeddingSize) {
            return;
        }
        std::vector<float> scores(texts_.size(), 0.0F);
        for (size_t row = 0; row < texts_.size(); ++row) {
            scores[row] = std::inner_product(
                text_features_.begin() + static_cast<std::ptrdiff_t>(row * kEmbeddingSize),
                text_features_.begin() + static_cast<std::ptrdiff_t>((row + 1) * kEmbeddingSize),
                image_features.begin(), 0.0F);
            text_widgets_[row]->updateScore(scores[row]);
        }

        std::vector<size_t> order(scores.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
            return scores[left] > scores[right];
        });
        std::vector<size_t> eligible;
        for (size_t index : order) {
            if (scores[index] >= kHighlightMinScore) {
                eligible.push_back(index);
            }
        }
        for (size_t index = 0; index < text_widgets_.size(); ++index) {
            int rank = -1;
            if (!eligible.empty() && index == eligible[0]) {
                rank = 0;
            } else if (eligible.size() > 1 && index == eligible[1]) {
                rank = 1;
            }
            text_widgets_[index]->setHighlightRank(rank);
        }

        if (!eligible.empty()) {
            const size_t best = eligible[0];
            status_label_->setTextFormat(Qt::RichText);
            status_label_->setText(
                QString("<span style=\"color:#d62828;font-weight:700\">"
                        "Best match: &nbsp;%1 - (%2)</span>")
                    .arg(texts_[best].toHtmlEscaped(), QString::number(scores[best], 'f', 3)));
        } else {
            setStatusPlain("No clear match yet - try a clearer view, better light, or move closer.");
        }
    }

    AppOptions options_;
    std::vector<QString> texts_;
    TextFeatureStore feature_store_;
    bool video_input_ = false;
    bool shutting_down_ = false;
    uint64_t frame_index_ = 0;
    uint64_t next_job_id_ = 0;

    QLabel* video_label_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* source_info_label_ = nullptr;
    QPlainTextEdit* add_text_editor_ = nullptr;
    QPushButton* apply_button_ = nullptr;
    QPushButton* exit_button_ = nullptr;
    QWidget* text_container_ = nullptr;
    QVBoxLayout* text_layout_ = nullptr;
    std::vector<TextRowWidget*> text_widgets_;
    CameraThread* camera_thread_ = nullptr;
    QTimer* result_timer_ = nullptr;
    std::unique_ptr<ImageEncoderAsync> image_encoder_;
    std::vector<float> text_features_;
};

void notifyLauncherReady()
{
    const QByteArray path = qgetenv("DX_LAUNCHER_READY_FILE");
    if (path.isEmpty()) {
        return;
    }
    const QFileInfo info(QString::fromUtf8(path));
    QDir().mkpath(info.absolutePath());
    QFile file(info.absoluteFilePath());
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        file.write("ready\n");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        AppOptions options = parseArgs(argc, argv);
        options.text_encoder = resolvePath(options.text_encoder).string();
        options.image_encoder = resolvePath(options.image_encoder).string();
        options.bpe_vocab = resolvePath(options.bpe_vocab).string();
        if (!isIntegerString(options.input) && options.input.rfind("/dev/video", 0) != 0) {
            const fs::path resolved_input = resolvePath(options.input);
            if (fs::exists(resolved_input)) {
                options.input = resolved_input.string();
            }
        }
        if (!fs::is_regular_file(options.text_encoder)) {
            throw std::runtime_error("text encoder not found: " + options.text_encoder);
        }
        if (!fs::is_regular_file(options.image_encoder)) {
            throw std::runtime_error("image encoder not found: " + options.image_encoder);
        }
        if (!fs::is_regular_file(options.bpe_vocab)) {
            throw std::runtime_error("BPE vocabulary not found: " + options.bpe_vocab);
        }

        qRegisterMetaType<cv::Mat>("cv::Mat");
        QApplication application(argc, argv);
        const bool full_screen = options.full_screen;
        MainWindow window(std::move(options));
        notifyLauncherReady();
        if (full_screen) {
            window.showFullScreen();
        } else {
            window.showMaximized();
        }
        return application.exec();
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << std::endl;
    } catch (const dxrt::Exception& error) {
        std::cerr << "DXRT error: " << error.what() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
    }
    printUsage(argv[0]);
    return 1;
}

#include "main.moc"
