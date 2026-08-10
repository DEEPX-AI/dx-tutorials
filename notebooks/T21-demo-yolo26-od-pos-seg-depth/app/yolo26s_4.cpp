/**
 * @file yolo26s_4.cpp
 * @brief Qt5 2x2 YOLO26-S demo for detection, pose, segmentation, and depth.
 */

#include <dxrt/dxrt_cxx_api.h>

#include <QApplication>
#include <QCloseEvent>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QMetaObject>
#include <QPixmap>
#include <QPushButton>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "extern/cxxopts.hpp"

#include "common/base/i_processor.hpp"
#include "common/utility/model_input.hpp"
#include "common/utility/model_version.hpp"

#include "factory/yolo26s_factory.hpp"
#include "factory/yolo26s_pose_factory.hpp"
#include "factory/yolo26s_seg_factory.hpp"

namespace stdfs = std::filesystem;

namespace {

constexpr int kPanelCount = 4;
constexpr int kOdPanel = 0;
constexpr int kPosePanel = 1;
constexpr int kSegPanel = 2;
constexpr int kDepthPanel = 3;
constexpr int kOdAsyncQueueSize = 2;
constexpr int kPoseAsyncQueueSize = 2;
constexpr int kSegAsyncQueueSize = 3;
constexpr int kDepthAsyncQueueSize = 2;

const char* kPanelTitles[kPanelCount] = {
    "YOLO26-S Object Detection",
    "YOLO26-S Pose Estimation",
    "YOLO26-S Instance Segmentation",
    "YOLO26-S Depth Estimation",
};

const std::vector<cv::Scalar> kDetectionColors = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255},
    {0, 255, 255}, {128, 0, 128}, {255, 165, 0}, {0, 128, 0}, {128, 128, 0},
    {0, 128, 128}, {128, 0, 0}, {192, 192, 192}, {255, 192, 203},
    {255, 215, 0}, {173, 216, 230}, {144, 238, 144}, {255, 218, 185},
    {221, 160, 221}, {255, 240, 245},
};

const std::vector<std::pair<int, int>> kPoseSkeleton = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
    {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {3, 5}, {4, 6},
};

const std::vector<std::pair<int, int>> kBbox3dSkeleton = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

const std::vector<cv::Scalar> kPoseLimbColors = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
    {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0},
    {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0},
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
};

const std::vector<cv::Scalar> kPoseKeypointColors = {
    {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
    {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0},
    {255, 128, 0}, {255, 128, 0}, {51, 153, 255}, {51, 153, 255},
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
};

std::string projectPath(const std::string& rel) {
    stdfs::path root(PROJECT_ROOT_DIR);
    return (root / rel).lexically_normal().string();
}

std::string absolutePath(const std::string& path) {
    stdfs::path p(path);
    if (p.is_relative()) {
        p = stdfs::current_path() / p;
    }
    return stdfs::absolute(p).lexically_normal().string();
}

struct AppArgs {
    std::string model = projectPath("assets/models/yolo26-s_640x640.dxnn");
    std::string model_pose = projectPath("assets/models/yolo26-s-pose_640x640.dxnn");
    std::string model_seg = projectPath("assets/models/yolo26-s-seg_640x640.dxnn");
    std::string model_depth =
        projectPath("assets/models/yolo26-depth-s_768x768_q-lite.dxnn");
    std::string video;
    bool no_loop_video = false;
    std::string device = "/dev/video0";
    int width = 1280;
    int height = 720;
    int fps = 30;
    bool debug_seg_timing = false;
    int debug_timing_interval_ms = 1000;
    bool debug_seg_bbox_only = false;
    int seg_render_width = 640;
    int seg_render_height = 360;
    bool save_video = false;
    std::string output_video;
    bool show_exit_button = false;
};

void printUsage(const char* argv0, const AppArgs& defaults) {
    std::cout
        << "Usage: " << argv0 << " [OPTIONS]\n"
        << "      --model <PATH>          YOLO26 detection .dxnn (default: "
        << defaults.model << ")\n"
        << "      --model-pose <PATH>     YOLO26 pose .dxnn (default: "
        << defaults.model_pose << ")\n"
        << "      --model-seg <PATH>      YOLO26 segmentation .dxnn (default: "
        << defaults.model_seg << ")\n"
        << "      --model-depth <PATH>    YOLO26 depth .dxnn (default: "
        << defaults.model_depth << ")\n"
        << "      --video <PATH>          Video file input path (default: use camera)\n"
        << "      --no-loop-video         Stop at EOF instead of looping video\n"
        << "  -c, --camera <PATH>         V4L2 camera device (default: "
        << defaults.device << ")\n"
        << "      --device <PATH>         Deprecated alias for --camera\n"
        << "      --width <N>             Requested camera width (default: "
        << defaults.width << ")\n"
        << "      --height <N>            Requested camera height (default: "
        << defaults.height << ")\n"
        << "      --fps <N>               Requested camera FPS (default: "
        << defaults.fps << ")\n"
        << "      --debug-seg-timing      Print segmentation worker timing by stage\n"
        << "      --debug-timing-interval-ms <N>\n"
        << "                              Timing report interval in ms (default: "
        << defaults.debug_timing_interval_ms << ")\n"
        << "      --debug-seg-bbox-only   Accepted for compatibility\n"
        << "      --seg-render-width <N>  Segmentation panel width (default: "
        << defaults.seg_render_width << ")\n"
        << "      --seg-render-height <N> Segmentation panel height (default: "
        << defaults.seg_render_height << ")\n"
        << "  -s, --save-video            Save output video (default: output.mp4)\n"
        << "      --output-video <PATH>   Output video path; enables saving\n"
        << "      --exit-btn              Show an exit button in the Pose panel\n"
        << "  -h, --help                  Show this help\n";
}

AppArgs parseArgs(int argc, char* argv[]) {
    AppArgs args;
    std::string legacy_device;
    cxxopts::Options options(
        "yolo26s_4",
        "Qt5 2x2 YOLO26-S demo: object detection, pose, instance segmentation, and depth estimation.");

    options.add_options()
        ("model", "YOLO26 detection .dxnn",
         cxxopts::value<std::string>(args.model)->default_value(args.model))
        ("model-pose", "YOLO26 pose .dxnn",
         cxxopts::value<std::string>(args.model_pose)->default_value(args.model_pose))
        ("model-seg", "YOLO26 segmentation .dxnn",
         cxxopts::value<std::string>(args.model_seg)->default_value(args.model_seg))
        ("model-depth", "YOLO26 depth .dxnn",
         cxxopts::value<std::string>(args.model_depth)->default_value(args.model_depth))
        ("video", "Video file path input. If omitted, USB webcam is used.",
         cxxopts::value<std::string>(args.video)->default_value(""))
        ("no-loop-video", "With --video, stop at EOF instead of looping.",
         cxxopts::value<bool>(args.no_loop_video)->default_value("false"))
        ("c,camera", "V4L2 camera device, e.g. /dev/video0.",
         cxxopts::value<std::string>(args.device)->default_value(args.device))
        ("device", "Deprecated alias for --camera.",
         cxxopts::value<std::string>(legacy_device))
        ("width", "Requested camera width.",
         cxxopts::value<int>(args.width)->default_value(std::to_string(args.width)))
        ("height", "Requested camera height.",
         cxxopts::value<int>(args.height)->default_value(std::to_string(args.height)))
        ("fps", "Requested camera FPS.",
         cxxopts::value<int>(args.fps)->default_value(std::to_string(args.fps)))
        ("debug-seg-timing", "Print segmentation worker timing by stage.",
         cxxopts::value<bool>(args.debug_seg_timing)->default_value("false"))
        ("debug-timing-interval-ms", "Timing report interval in milliseconds.",
         cxxopts::value<int>(args.debug_timing_interval_ms)->default_value(
             std::to_string(args.debug_timing_interval_ms)))
        ("debug-seg-bbox-only", "Accepted for compatibility; segmentation renders mask colors only.",
         cxxopts::value<bool>(args.debug_seg_bbox_only)->default_value("false"))
        ("seg-render-width", "Segmentation panel render width. Capped at 960.",
         cxxopts::value<int>(args.seg_render_width)->default_value(
             std::to_string(args.seg_render_width)))
        ("seg-render-height", "Segmentation panel render height. Capped at 540.",
         cxxopts::value<int>(args.seg_render_height)->default_value(
             std::to_string(args.seg_render_height)))
        ("s,save-video", "Save output video file. If used without --output-video, saves to 'output.mp4'.",
         cxxopts::value<bool>(args.save_video)->default_value("false"))
        ("output-video", "Output video file path (enables saving when specified).",
         cxxopts::value<std::string>(args.output_video)->default_value(""))
        ("exit-btn", "Show an exit button in the Pose Estimation panel header.",
         cxxopts::value<bool>(args.show_exit_button)->default_value("false"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        printUsage(argv[0], args);
        std::exit(0);
    }

    if (result.count("device") > 0 && result.count("camera") == 0) {
        args.device = legacy_device;
    }
    
    // Handle video saving options
    if (!args.output_video.empty()) {
        args.save_video = true;
    } else if (args.save_video && args.output_video.empty()) {
        args.output_video = "output.mp4";
    }
    
    return args;
}

void requireFile(const std::string& label, const std::string& path) {
    if (!stdfs::is_regular_file(path)) {
        throw std::runtime_error("[ERROR] " + label + " not found: " + path);
    }
}

int parseCameraIndex(const std::string& device) {
    if (device.empty()) {
        throw std::runtime_error(
            "Camera device is empty. Use a non-negative index or /dev/videoN.");
    }

    std::string index_text;
    if (std::all_of(device.begin(), device.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        index_text = device;
    } else {
        const std::string prefix = "/dev/video";
        if (device.rfind(prefix, 0) != 0) {
            throw std::runtime_error(
                "Invalid camera device '" + device +
                "'. Use a non-negative index or /dev/videoN.");
        }

        index_text = device.substr(prefix.size());
        if (index_text.empty() ||
            !std::all_of(index_text.begin(), index_text.end(),
                         [](unsigned char ch) { return std::isdigit(ch); })) {
            throw std::runtime_error(
                "Invalid camera device '" + device +
                "'. Use a non-negative index or /dev/videoN.");
        }
    }

    try {
        return std::stoi(index_text);
    } catch (const std::exception&) {
        throw std::runtime_error("Camera index is out of range: " + device);
    }
}

cv::Mat ensureBgr3(const cv::Mat& frame) {
    if (frame.empty()) {
        return {};
    }
    if (frame.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (frame.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (frame.channels() == 3) {
        return frame;
    }
    return {};
}

struct DepthLetterboxContext {
    int original_width{0};
    int original_height{0};
    int top{0};
    int bottom{0};
    int left{0};
    int right{0};
};

class LatestFrameQueue {
public:
    void push(const cv::Mat& frame) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            frame_ = frame.clone();
            has_frame_ = true;
        }
        cv_.notify_one();
    }

    bool waitPop(cv::Mat& frame, const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
            return has_frame_ || !running.load(std::memory_order_relaxed);
        });
        if (!running.load(std::memory_order_relaxed)) {
            return false;
        }
        if (!has_frame_) {
            return false;
        }
        frame = std::move(frame_);
        has_frame_ = false;
        return true;
    }

    void wake() {
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    cv::Mat frame_;
    bool has_frame_{false};
};

class IFrameConsumer {
public:
    virtual ~IFrameConsumer() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void join() = 0;
    virtual void enqueue(const cv::Mat& frame) = 0;
};

class QuadWindow;

double msBetween(const std::chrono::steady_clock::time_point& begin,
                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

class TimingReporter {
public:
    TimingReporter(std::string name, bool enabled, int interval_ms)
        : name_(std::move(name)),
          enabled_(enabled),
          interval_(std::chrono::milliseconds(std::max(1, interval_ms))),
          window_start_(std::chrono::steady_clock::now()) {}

    void add(double preprocess_ms,
             double inference_ms,
             double postprocess_ms,
             double render_ms,
             double total_ms) {
        if (!enabled_) {
            return;
        }

        frames_ += 1;
        preprocess_sum_ += preprocess_ms;
        inference_sum_ += inference_ms;
        postprocess_sum_ += postprocess_ms;
        render_sum_ += render_ms;
        total_sum_ += total_ms;

        auto now = std::chrono::steady_clock::now();
        if (now - window_start_ < interval_) {
            return;
        }

        double wall_ms = msBetween(window_start_, now);
        double fps = wall_ms > 0.0 ? static_cast<double>(frames_) * 1000.0 / wall_ms : 0.0;
        double inv = frames_ > 0 ? 1.0 / static_cast<double>(frames_) : 0.0;

        std::cout << "[TIMING][" << name_ << "] "
                  << "frames=" << frames_
                  << " fps=" << std::fixed << std::setprecision(1) << fps
                  << " avg_ms total=" << std::setprecision(2) << total_sum_ * inv
                  << " preprocess=" << preprocess_sum_ * inv
                  << " inference=" << inference_sum_ * inv
                  << " postprocess=" << postprocess_sum_ * inv
                  << " render=" << render_sum_ * inv
                  << " post+render=" << (postprocess_sum_ + render_sum_) * inv
                  << std::endl;

        frames_ = 0;
        preprocess_sum_ = 0.0;
        inference_sum_ = 0.0;
        postprocess_sum_ = 0.0;
        render_sum_ = 0.0;
        total_sum_ = 0.0;
        window_start_ = now;
    }

private:
    std::string name_;
    bool enabled_{false};
    std::chrono::milliseconds interval_{1000};
    std::chrono::steady_clock::time_point window_start_;
    int frames_{0};
    double preprocess_sum_{0.0};
    double inference_sum_{0.0};
    double postprocess_sum_{0.0};
    double render_sum_{0.0};
    double total_sum_{0.0};
};

struct SegRenderOptions {
    int width{640};
    int height{360};
    int max_width{960};
    int max_height{540};
    bool bbox_only{false};
};

int clampInt(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

cv::Rect rectFromScaledBox(const std::vector<float>& box,
                           float sx,
                           float sy,
                           int width,
                           int height) {
    if (box.size() < 4 || width <= 0 || height <= 0) {
        return {};
    }

    int x1 = clampInt(static_cast<int>(std::floor(box[0] * sx)), 0, width - 1);
    int y1 = clampInt(static_cast<int>(std::floor(box[1] * sy)), 0, height - 1);
    int x2 = clampInt(static_cast<int>(std::ceil(box[2] * sx)), 0, width);
    int y2 = clampInt(static_cast<int>(std::ceil(box[3] * sy)), 0, height);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Mat binaryMaskRoi(const cv::Mat& mask_roi, const cv::Size& target_size) {
    if (mask_roi.empty() || target_size.width <= 0 || target_size.height <= 0) {
        return {};
    }

    cv::Mat resized;
    cv::resize(mask_roi, resized, target_size, 0, 0, cv::INTER_LINEAR);

    cv::Mat binary;
    if (resized.type() == CV_32FC1 || resized.type() == CV_64FC1) {
        cv::threshold(resized, binary, 0.5, 255.0, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1);
    } else {
        resized.convertTo(binary, CV_8UC1);
        cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);
    }
    return binary;
}

void blendMaskIntoRoi(cv::Mat& output,
                      const cv::Rect& render_roi,
                      const cv::Mat& mask_binary,
                      const cv::Vec3b& color,
                      float alpha) {
    if (render_roi.empty() || mask_binary.empty()) {
        return;
    }

    cv::Mat target = output(render_roi);
    cv::Mat color_mat(target.size(), CV_8UC3, cv::Scalar(color[0], color[1], color[2]));
    cv::Mat blended;
    cv::addWeighted(target, 1.0 - alpha, color_mat, alpha, 0, blended);
    blended.copyTo(target, mask_binary);
}

cv::Vec3b segmentationClassColor(int class_id) {
    static const std::vector<cv::Vec3b> kNeonSegmentationColors = {
        cv::Vec3b(180, 0, 255),    // hot pink
        cv::Vec3b(255, 255, 0),    // neon cyan
        cv::Vec3b(20, 255, 80),    // acid green
        cv::Vec3b(0, 255, 255),    // electric yellow
        cv::Vec3b(255, 60, 180),   // electric violet
        cv::Vec3b(0, 95, 255),     // neon orange
        cv::Vec3b(255, 130, 0),    // electric blue
        cv::Vec3b(80, 255, 255),   // lemon glow
        cv::Vec3b(255, 0, 120),    // ultraviolet
        cv::Vec3b(0, 255, 140),    // neon lime
        cv::Vec3b(255, 80, 80),    // aqua blue
        cv::Vec3b(60, 0, 255),     // vivid red-pink
    };

    long long normalized = static_cast<long long>(class_id);
    if (normalized < 0) {
        normalized = -normalized;
    }
    size_t index = static_cast<size_t>(normalized) % kNeonSegmentationColors.size();
    return kNeonSegmentationColors[index];
}

void drawThinTextDetection(cv::Mat& output,
                           const dxapp::DetectionResult& detection,
                           int line_thickness = 2,
                           double font_scale = 0.5) {
    if (detection.box.size() < 4) {
        return;
    }

    float x1 = clampInt(static_cast<int>(std::floor(detection.box[0])), 0, output.cols);
    float y1 = clampInt(static_cast<int>(std::floor(detection.box[1])), 0, output.rows);
    float x2 = clampInt(static_cast<int>(std::ceil(detection.box[2])), 0, output.cols);
    float y2 = clampInt(static_cast<int>(std::ceil(detection.box[3])), 0, output.rows);
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    const size_t color_index = static_cast<size_t>(std::max(0, detection.class_id)) %
                               kDetectionColors.size();
    cv::Scalar color = kDetectionColors[color_index];
    cv::rectangle(output, cv::Point2f(x1, y1), cv::Point2f(x2, y2), color, line_thickness);

    std::string label = detection.class_name + ": " +
        std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         font_scale, 1, &baseline);
    cv::Point label_pt(static_cast<int>(x1),
                       y1 - 10 > 10 ? static_cast<int>(y1 - 10)
                                    : static_cast<int>(y1 + text_size.height + 10));

    cv::rectangle(output,
                  cv::Point(label_pt.x, label_pt.y - text_size.height - 5),
                  cv::Point(label_pt.x + text_size.width, label_pt.y + baseline),
                  color, cv::FILLED);
    cv::putText(output, label, label_pt,
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

cv::Mat renderDetectionsThinText(const cv::Mat& frame,
                                 const std::vector<dxapp::DetectionResult>& results) {
    cv::Mat output = frame.clone();
    for (const auto& detection : results) {
        drawThinTextDetection(output, detection);
    }
    return output;
}

cv::Mat renderPoseThinText(const cv::Mat& frame,
                           const std::vector<dxapp::PoseResult>& results,
                           const dxapp::PreprocessContext& ctx) {
    cv::Mat output = frame.clone();
    float disp_scale = 1.0f;
    if (ctx.original_width > 0 && ctx.original_height > 0 &&
        (ctx.original_width > output.cols || ctx.original_height > output.rows)) {
        disp_scale = std::min(static_cast<float>(output.cols) / ctx.original_width,
                              static_cast<float>(output.rows) / ctx.original_height);
    }
    const float x_off = (ctx.original_width > 0)
        ? (output.cols - ctx.original_width * disp_scale) / 2.0f : 0.0f;
    const float y_off = (ctx.original_height > 0)
        ? (output.rows - ctx.original_height * disp_scale) / 2.0f : 0.0f;

    for (const auto& pose : results) {
        if (pose.box.size() >= 4) {
            cv::Point pt1(static_cast<int>(pose.box[0] * disp_scale + x_off),
                          static_cast<int>(pose.box[1] * disp_scale + y_off));
            cv::Point pt2(static_cast<int>(pose.box[2] * disp_scale + x_off),
                          static_cast<int>(pose.box[3] * disp_scale + y_off));
            cv::rectangle(output, pt1, pt2, cv::Scalar(0, 255, 0), 2);

            std::string conf_text = "Person: " +
                std::to_string(static_cast<int>(pose.confidence * 100)) + "%";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(conf_text, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.5, 1, &baseline);
            cv::Point text_pos(pt1.x, pt1.y - 10);
            cv::rectangle(output,
                          cv::Point(text_pos.x, text_pos.y - text_size.height),
                          cv::Point(text_pos.x + text_size.width, text_pos.y + baseline),
                          cv::Scalar(0, 0, 0), -1);
            cv::putText(output, conf_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        const auto& skeleton = (pose.keypoints.size() <= 8) ? kBbox3dSkeleton : kPoseSkeleton;
        cv::Scalar bbox3d_color(0, 255, 128);
        for (size_t i = 0; i < skeleton.size(); ++i) {
            int idx1 = skeleton[i].first;
            int idx2 = skeleton[i].second;
            if (idx1 >= static_cast<int>(pose.keypoints.size()) ||
                idx2 >= static_cast<int>(pose.keypoints.size())) {
                continue;
            }
            const auto& kp1 = pose.keypoints[idx1];
            const auto& kp2 = pose.keypoints[idx2];
            if (kp1.confidence < 0.3f || kp2.confidence < 0.3f) {
                continue;
            }
            cv::Point pt1(static_cast<int>(kp1.x * disp_scale + x_off),
                          static_cast<int>(kp1.y * disp_scale + y_off));
            cv::Point pt2(static_cast<int>(kp2.x * disp_scale + x_off),
                          static_cast<int>(kp2.y * disp_scale + y_off));
            cv::Scalar color = (pose.keypoints.size() <= 8) ? bbox3d_color
                : (i < kPoseLimbColors.size() ? kPoseLimbColors[i] : bbox3d_color);
            cv::line(output, pt1, pt2, color, 2, cv::LINE_AA);
        }

        for (size_t i = 0; i < pose.keypoints.size(); ++i) {
            const auto& kp = pose.keypoints[i];
            if (kp.confidence < 0.3f) {
                continue;
            }
            cv::Point pt(static_cast<int>(kp.x * disp_scale + x_off),
                         static_cast<int>(kp.y * disp_scale + y_off));
            cv::circle(output, pt, 3,
                       kPoseKeypointColors[i % kPoseKeypointColors.size()], -1, cv::LINE_AA);
        }
    }

    return output;
}

cv::Mat renderSegmentationFast(const cv::Mat& frame,
                               const std::vector<dxapp::InstanceSegmentationResult>& results,
                               const dxapp::PreprocessContext& ctx,
                               const SegRenderOptions& options) {
    int target_w = clampInt(options.width, 1, options.max_width);
    int target_h = clampInt(options.height, 1, options.max_height);

    cv::Mat output;
    cv::resize(frame, output, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    const int original_w = ctx.original_width > 0 ? ctx.original_width : frame.cols;
    const int original_h = ctx.original_height > 0 ? ctx.original_height : frame.rows;
    const float sx = static_cast<float>(output.cols) / static_cast<float>(original_w);
    const float sy = static_cast<float>(output.rows) / static_cast<float>(original_h);

    for (const auto& inst : results) {
        cv::Vec3b color_vec = segmentationClassColor(inst.class_id);

        cv::Rect render_roi = rectFromScaledBox(inst.box, sx, sy, output.cols, output.rows);
        if (render_roi.empty()) {
            continue;
        }

        if (!inst.mask.empty()) {
            const float mx = static_cast<float>(inst.mask.cols) / static_cast<float>(original_w);
            const float my = static_cast<float>(inst.mask.rows) / static_cast<float>(original_h);
            cv::Rect mask_roi = rectFromScaledBox(inst.box, mx, my, inst.mask.cols, inst.mask.rows);
            if (!mask_roi.empty()) {
                cv::Mat mask_binary = binaryMaskRoi(inst.mask(mask_roi), render_roi.size());
                blendMaskIntoRoi(output, render_roi, mask_binary, color_vec, 0.5f);
            }
        }
    }

    return output;
}

template <typename ResultT, typename FactoryT>
class ResultWorker final : public IFrameConsumer {
public:
    ResultWorker(int panel_index,
                 std::string name,
                 std::string model_path,
                 int max_inflight,
                 QuadWindow* window,
                 bool timing_enabled = false,
                 int timing_interval_ms = 1000,
                 SegRenderOptions seg_render_options = {})
        : panel_index_(panel_index),
          name_(std::move(name)),
          model_path_(std::move(model_path)),
          max_inflight_(std::max(1, max_inflight)),
          window_(window),
          timing_enabled_(timing_enabled),
          timing_interval_ms_(timing_interval_ms),
          seg_render_options_(seg_render_options) {}

    void start() override {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&ResultWorker::run, this);
    }

    void stop() override {
        running_.store(false, std::memory_order_relaxed);
        queue_.wake();
        inflight_cv_.notify_all();
    }

    void join() override {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void enqueue(const cv::Mat& frame) override {
        if (running_.load(std::memory_order_relaxed)) {
            queue_.push(frame);
        }
    }

private:
    struct AsyncJob {
        cv::Mat frame;
        dxapp::PreprocessContext ctx;
        cv::Mat input_u8;
        std::vector<float> input_float;
        std::chrono::steady_clock::time_point preprocess_begin;
        std::chrono::steady_clock::time_point preprocess_end;
        std::chrono::steady_clock::time_point submit_time;

        void* inputData() {
            if (!input_float.empty()) {
                return input_float.data();
            }
            return input_u8.data;
        }
    };

    void run();
    bool acquireInflightSlot();
    void releaseInflightSlot();
    void waitForInflightEmpty();

    int panel_index_;
    std::string name_;
    std::string model_path_;
    int max_inflight_{1};
    QuadWindow* window_;
    bool timing_enabled_{false};
    int timing_interval_ms_{1000};
    SegRenderOptions seg_render_options_;
    LatestFrameQueue queue_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex inflight_mutex_;
    std::condition_variable inflight_cv_;
    int inflight_{0};
    std::mutex callback_mutex_;
};

class DepthWorker final : public IFrameConsumer {
public:
    DepthWorker(std::string model_path,
                int max_inflight,
                QuadWindow* window)
        : model_path_(std::move(model_path)),
          max_inflight_(std::max(1, max_inflight)),
          window_(window) {}

    void start() override {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&DepthWorker::run, this);
    }

    void stop() override {
        running_.store(false, std::memory_order_relaxed);
        queue_.wake();
        inflight_cv_.notify_all();
    }

    void join() override {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void enqueue(const cv::Mat& frame) override {
        if (running_.load(std::memory_order_relaxed)) {
            queue_.push(frame);
        }
    }

private:
    struct AsyncJob {
        std::vector<std::uint8_t> input;
        DepthLetterboxContext letterbox;
    };

    void run();
    void preprocess(const cv::Mat& bgr,
                    std::vector<std::uint8_t>& input,
                    DepthLetterboxContext& context);
    static cv::Mat restoreDepth(const dxrt::Tensor* tensor,
                                const DepthLetterboxContext& context,
                                int network_width,
                                int network_height);
    static cv::Mat colorizeDepth(const cv::Mat& depth);
    bool acquireInflightSlot();
    void releaseInflightSlot();
    void waitForInflightEmpty();

    std::string model_path_;
    int max_inflight_{1};
    QuadWindow* window_{nullptr};
    LatestFrameQueue queue_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex inflight_mutex_;
    std::condition_variable inflight_cv_;
    int inflight_{0};
    int network_width_{0};
    int network_height_{0};
    std::size_t input_bytes_{0};
    cv::Mat resized_scratch_;
    cv::Mat padded_scratch_;
    cv::Mat rgb_scratch_;
};

class CaptureThread final {
public:
    CaptureThread(AppArgs args, QuadWindow* window)
        : args_(std::move(args)), window_(window) {}

    void setConsumers(std::vector<IFrameConsumer*> consumers) {
        consumers_ = std::move(consumers);
    }

    void start() {
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread(&CaptureThread::run, this);
    }

    void stop() {
        running_.store(false, std::memory_order_relaxed);
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run();
    void runVideo();
    void runCamera();
    void publishFrame(const cv::Mat& frame);

    AppArgs args_;
    QuadWindow* window_;
    std::vector<IFrameConsumer*> consumers_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

QFrame* makePanel(const QString& title,
                  QLabel** image_label,
                  QLabel** fps_label,
                  bool show_exit_button = false) {
    auto* frame = new QFrame;
    frame->setObjectName("yolo_panel");
    frame->setStyleSheet(
        "QFrame#yolo_panel {"
        "background-color: #13151a;"
        "border: 1px solid #3a404c;"
        "border-radius: 8px;"
        "}");

    auto* vbox = new QVBoxLayout(frame);
    vbox->setContentsMargins(0, 0, 0, 0);
    vbox->setSpacing(0);

    auto* header = new QHBoxLayout;
    header->setContentsMargins(0, 0, 0, 0);
    header->setSpacing(0);

    auto* title_label = new QLabel(title);
    title_label->setObjectName("yolo_panel_title");
    title_label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    title_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    title_label->setFocusPolicy(Qt::NoFocus);
    title_label->setStyleSheet(
        "QLabel#yolo_panel_title {"
        "color: #eceef4;"
        "font-size: 13px;"
        "font-weight: 600;"
        "padding: 8px 12px 6px 12px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "border-top-left-radius: 7px;"
        "}");

    auto* fps = new QLabel("-");
    fps->setObjectName("yolo_panel_fps");
    fps->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    fps->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    fps->setFocusPolicy(Qt::NoFocus);
    fps->setStyleSheet(QString(
        "QLabel#yolo_panel_fps {"
        "color: #9aa3b2;"
        "font-size: 12px;"
        "font-weight: 500;"
        "font-family: monospace;"
        "padding: 8px 12px 6px 8px;"
        "background-color: #1e222b;"
        "border-bottom: 1px solid #353b48;"
        "%1"
        "}").arg(show_exit_button ? "" : "border-top-right-radius: 7px;"));

    header->addWidget(title_label, 1);
    header->addWidget(fps, 0);

    if (show_exit_button) {
        auto* exit_area = new QWidget;
        exit_area->setObjectName("yolo_panel_exit_area");
        exit_area->setFixedWidth(48);
        exit_area->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        exit_area->setStyleSheet(
            "QWidget#yolo_panel_exit_area {"
            "background-color: #1e222b;"
            "border-bottom: 1px solid #353b48;"
            "border-top-right-radius: 7px;"
            "}");

        auto* exit_layout = new QHBoxLayout(exit_area);
        exit_layout->setContentsMargins(8, 0, 8, 0);
        exit_layout->setSpacing(0);

        auto* exit_button = new QPushButton("X");
        exit_button->setFixedSize(32, 28);
        exit_button->setFocusPolicy(Qt::NoFocus);
        exit_button->setToolTip("Exit");
        exit_button->setStyleSheet(
            "QPushButton {"
            "color: #cccccc;"
            "background-color: #2d2d30;"
            "border: 1px solid #3c3c3c;"
            "border-radius: 6px;"
            "font-size: 13px;"
            "font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "background-color: #3a3a3d;"
            "color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            "background-color: #005a9e;"
            "}");
        QObject::connect(exit_button, &QPushButton::clicked, frame, [frame] {
            frame->window()->close();
        });
        exit_layout->addWidget(exit_button, 0, Qt::AlignCenter);
        header->addWidget(exit_area);
    }
    vbox->addLayout(header);

    auto* image = new QLabel;
    image->setAlignment(Qt::AlignCenter);
    image->setScaledContents(true);
    image->setMinimumSize(280, 160);
    image->setFocusPolicy(Qt::NoFocus);
    image->setStyleSheet(
        "background-color: #0c0d10;"
        "border-bottom-left-radius: 7px;"
        "border-bottom-right-radius: 7px;");
    vbox->addWidget(image, 1);

    *image_label = image;
    *fps_label = fps;
    return frame;
}

class QuadWindow final : public QMainWindow {
public:
    explicit QuadWindow(const AppArgs& args)
        : args_(args), capture_(args, this) {
        setWindowTitle("YOLO26-S Multi-Task Demo");

        auto* central = new QWidget;
        central->setFocusPolicy(Qt::StrongFocus);
        setCentralWidget(central);

        auto* grid = new QGridLayout(central);
        grid->setSpacing(0);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setColumnStretch(0, 1);
        grid->setColumnStretch(1, 1);
        grid->setRowStretch(0, 1);
        grid->setRowStretch(1, 1);

        const int cells[kPanelCount][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (int i = 0; i < kPanelCount; ++i) {
            QLabel* image = nullptr;
            QLabel* fps = nullptr;
            QWidget* panel = makePanel(kPanelTitles[i],
                                       &image,
                                       &fps,
                                       args_.show_exit_button && i == kPosePanel);
            grid->addWidget(panel, cells[i][0], cells[i][1]);
            image_labels_.push_back(image);
            fps_labels_.push_back(fps);
            fps_counts_.push_back(0);
        }

        fps_timer_ = new QTimer(this);
        fps_timer_->setInterval(1000);
        QObject::connect(fps_timer_, &QTimer::timeout, this, [this] { tickFps(); });
        fps_timer_->start();

        workers_.push_back(std::make_unique<ResultWorker<dxapp::DetectionResult, dxapp::Yolo26sFactory>>(
            kOdPanel, "Object Detection", args_.model, kOdAsyncQueueSize, this));
        workers_.push_back(std::make_unique<ResultWorker<dxapp::PoseResult, dxapp::Yolo26s_poseFactory>>(
            kPosePanel, "Pose Estimation", args_.model_pose, kPoseAsyncQueueSize, this));
        SegRenderOptions seg_render_options;
        seg_render_options.width = args_.seg_render_width;
        seg_render_options.height = args_.seg_render_height;
        seg_render_options.bbox_only = args_.debug_seg_bbox_only;
        workers_.push_back(std::make_unique<ResultWorker<dxapp::InstanceSegmentationResult, dxapp::Yolo26s_segFactory>>(
            kSegPanel,
            "Instance Segmentation",
            args_.model_seg,
            kSegAsyncQueueSize,
            this,
            args_.debug_seg_timing,
            args_.debug_timing_interval_ms,
            seg_render_options));
        workers_.push_back(std::make_unique<DepthWorker>(
            args_.model_depth, kDepthAsyncQueueSize, this));

        if (args_.save_video && !args_.output_video.empty()) {
            initializeVideoWriters();
        }

        std::vector<IFrameConsumer*> consumers;
        consumers.reserve(workers_.size());
        for (auto& worker : workers_) {
            worker->start();
            consumers.push_back(worker.get());
        }
        capture_.setConsumers(std::move(consumers));
        capture_.start();
    }

    ~QuadWindow() override {
        shutdown();
    }

    void postFrame(int panel_index, const cv::Mat& frame) {
        if (closing_.load(std::memory_order_relaxed)) {
            return;
        }
        cv::Mat safe_frame = frame;
        QMetaObject::invokeMethod(this, [this, panel_index, safe_frame]() mutable {
            if (closing_.load(std::memory_order_relaxed)) {
                return;
            }
            setPanelFrame(panel_index, safe_frame);
            // Save frame to video file
            if (args_.save_video && panel_index < static_cast<int>(video_writers_.size())) {
                saveFrame(panel_index, safe_frame);
            }
        }, Qt::QueuedConnection);
    }

    void postError(const std::string& message) {
        QMetaObject::invokeMethod(this, [this, message]() {
            if (closing_.load(std::memory_order_relaxed)) {
                return;
            }
            showError(message);
        }, Qt::QueuedConnection);
    }

    void postEndOfStream() {
        QMetaObject::invokeMethod(this, [this]() {
            if (closing_.load(std::memory_order_relaxed)) {
                return;
            }
            std::cout << "[INFO] Video ended." << std::endl;
            close();
        }, Qt::QueuedConnection);
    }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Escape || event->key() == Qt::Key_Q) {
            close();
            return;
        }
        QMainWindow::keyPressEvent(event);
    }

    void closeEvent(QCloseEvent* event) override {
        shutdown();
        event->accept();
    }

    void showEvent(QShowEvent* event) override {
        QMainWindow::showEvent(event);
        if (centralWidget() != nullptr) {
            centralWidget()->setFocus(Qt::OtherFocusReason);
        }
    }

private:
    void tickFps() {
        for (int i = 0; i < static_cast<int>(fps_labels_.size()); ++i) {
            int n = fps_counts_[i];
            fps_counts_[i] = 0;
            fps_labels_[i]->setText(n > 0 ? QString("%1 FPS").arg(n) : "-");
        }
    }

    void initializeVideoWriters() {
        try {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            static const char* kOutputSuffixes[kPanelCount] = {
                "_od", "_pose", "_seg", "_depth"
            };

            for (int i = 0; i < kPanelCount; ++i) {
                std::string output_path = args_.output_video;
                const std::size_t dot_pos = output_path.rfind('.');
                if (dot_pos != std::string::npos) {
                    output_path.insert(dot_pos, kOutputSuffixes[i]);
                } else {
                    output_path += kOutputSuffixes[i];
                }

                cv::VideoWriter writer(output_path, fourcc, args_.fps,
                                       cv::Size(args_.width, args_.height), true);
                if (writer.isOpened()) {
                    video_writers_.push_back(writer);
                    std::cout << "[INFO] Video writer initialized: " << output_path
                              << " (" << args_.width << "x" << args_.height
                              << " @ " << args_.fps << " FPS)" << std::endl;
                } else {
                    std::cerr << "[WARN] Failed to open video writer for: "
                              << output_path << std::endl;
                    video_writers_.push_back(cv::VideoWriter());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to initialize video writers: " << e.what() << std::endl;
        }
    }

    void saveFrame(int panel_index, const cv::Mat& frame) {
        if (panel_index < 0 || panel_index >= static_cast<int>(video_writers_.size()) ||
            frame.empty()) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(video_writer_mutex_);
        cv::VideoWriter& writer = video_writers_[panel_index];
        if (writer.isOpened()) {
            try {
                // Resize frame to match writer dimensions if needed
                cv::Mat output_frame = frame;
                if (frame.cols != args_.width || frame.rows != args_.height) {
                    cv::resize(frame, output_frame, cv::Size(args_.width, args_.height));
                }
                writer.write(output_frame);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to write frame to video: " << e.what() << std::endl;
            }
        }
    }

    void setPanelFrame(int panel_index, const cv::Mat& bgr, bool count_fps = true) {
        if (panel_index < 0 || panel_index >= static_cast<int>(image_labels_.size()) || bgr.empty()) {
            return;
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data,
                     rgb.cols,
                     rgb.rows,
                     static_cast<int>(rgb.step),
                     QImage::Format_RGB888);
        image_labels_[panel_index]->setPixmap(QPixmap::fromImage(image.copy()));
        if (count_fps) {
            fps_counts_[panel_index] += 1;
        }
    }

    void showError(const std::string& message) {
        std::cerr << "[ERROR] " << message << std::endl;
        for (auto* label : image_labels_) {
            label->clear();
            label->setText(QString::fromStdString(message));
            label->setStyleSheet(
                "background-color: #0c0d10;"
                "color: #eceef4;"
                "border-bottom-left-radius: 7px;"
                "border-bottom-right-radius: 7px;");
        }
    }

    void shutdown() {
        bool expected = false;
        if (!closing_.compare_exchange_strong(expected, true)) {
            return;
        }

        if (fps_timer_ != nullptr) {
            fps_timer_->stop();
        }

        capture_.stop();
        capture_.join();

        for (auto& worker : workers_) {
            worker->stop();
        }
        for (auto& worker : workers_) {
            worker->join();
        }
        
        // Release video writers
        for (auto& writer : video_writers_) {
            if (writer.isOpened()) {
                writer.release();
            }
        }
    }

    AppArgs args_;
    std::vector<QLabel*> image_labels_;
    std::vector<QLabel*> fps_labels_;
    std::vector<int> fps_counts_;
    QTimer* fps_timer_{nullptr};
    std::vector<std::unique_ptr<IFrameConsumer>> workers_;
    CaptureThread capture_;
    std::atomic<bool> closing_{false};
    std::vector<cv::VideoWriter> video_writers_;
    std::mutex video_writer_mutex_;
};

template <typename ResultT, typename FactoryT>
bool ResultWorker<ResultT, FactoryT>::acquireInflightSlot() {
    std::unique_lock<std::mutex> lock(inflight_mutex_);
    inflight_cv_.wait(lock, [this] {
        return inflight_ < max_inflight_ || !running_.load(std::memory_order_relaxed);
    });
    if (!running_.load(std::memory_order_relaxed)) {
        return false;
    }
    ++inflight_;
    return true;
}

template <typename ResultT, typename FactoryT>
void ResultWorker<ResultT, FactoryT>::releaseInflightSlot() {
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        if (inflight_ > 0) {
            --inflight_;
        }
    }
    inflight_cv_.notify_all();
}

template <typename ResultT, typename FactoryT>
void ResultWorker<ResultT, FactoryT>::waitForInflightEmpty() {
    std::unique_lock<std::mutex> lock(inflight_mutex_);
    inflight_cv_.wait(lock, [this] { return inflight_ == 0; });
}

template <typename ResultT, typename FactoryT>
void ResultWorker<ResultT, FactoryT>::run() {
    try {
        dxrt::InferenceOption io;
        io.bufferCount = max_inflight_;
        dxrt::InferenceEngine ie(model_path_, io);
        if (!dxapp::minversionforRTandCompiler(&ie)) {
            throw std::runtime_error(name_ + " model/runtime version mismatch: " + model_path_);
        }

        auto input_shape = ie.GetInputs().front().shape();
        int input_width = 0;
        int input_height = 0;
        parseInputShape(input_shape, input_width, input_height);
        bool is_float_input = (ie.GetInputs().front().type() == dxrt::DataType::FLOAT);
        bool is_nhwc = isInputNHWC(input_shape);

        FactoryT factory;
        auto preprocessor = factory.createPreprocessor(input_width, input_height);
        std::shared_ptr<dxapp::IPostprocessor<ResultT>> postprocessor(
            factory.createPostprocessor(input_width, input_height, ie.IsOrtConfigured()));

        std::cout << "[INFO] " << name_ << " model: " << model_path_ << std::endl;
        std::cout << "[INFO] " << name_ << " input size (WxH): "
                  << input_width << "x" << input_height << std::endl;
        std::cout << "[INFO] " << name_ << " async queue size: "
                  << max_inflight_ << std::endl;

        auto timing = std::make_shared<TimingReporter>(name_, timing_enabled_, timing_interval_ms_);
        int last_job_id = -1;

        ie.RegisterCallback([this, postprocessor, timing](
                            dxrt::TensorPtrs& outputs, void* user_data) -> int {
            auto job = std::unique_ptr<AsyncJob>(static_cast<AsyncJob*>(user_data));
            if (!job) {
                releaseInflightSlot();
                return -1;
            }

            auto callback_time = std::chrono::steady_clock::now();
            try {
                auto post_begin = std::chrono::steady_clock::now();
                std::vector<ResultT> results;
                cv::Mat rendered;
                std::chrono::steady_clock::time_point post_end;
                std::chrono::steady_clock::time_point render_end;

                {
                    std::lock_guard<std::mutex> lock(callback_mutex_);
                    results = postprocessor->process(outputs, job->ctx);
                    post_end = std::chrono::steady_clock::now();
                    if constexpr (std::is_same_v<ResultT, dxapp::InstanceSegmentationResult>) {
                        rendered = renderSegmentationFast(job->frame, results, job->ctx, seg_render_options_);
                    } else if constexpr (std::is_same_v<ResultT, dxapp::DetectionResult>) {
                        rendered = renderDetectionsThinText(job->frame, results);
                    } else if constexpr (std::is_same_v<ResultT, dxapp::PoseResult>) {
                        rendered = renderPoseThinText(job->frame, results, job->ctx);
                    }
                    render_end = std::chrono::steady_clock::now();
                    timing->add(msBetween(job->preprocess_begin, job->preprocess_end),
                                msBetween(job->submit_time, callback_time),
                                msBetween(post_begin, post_end),
                                msBetween(post_end, render_end),
                                msBetween(job->preprocess_begin, render_end));
                }

                if (running_.load(std::memory_order_relaxed) && !rendered.empty()) {
                    window_->postFrame(panel_index_, rendered);
                }
            } catch (const std::exception& e) {
                running_.store(false, std::memory_order_relaxed);
                queue_.wake();
                window_->postError(name_ + " async callback failed: " + e.what());
            }

            releaseInflightSlot();
            return 0;
        });

        while (running_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!queue_.waitPop(frame, running_)) {
                continue;
            }
            if (frame.empty()) {
                continue;
            }

            auto job = std::make_unique<AsyncJob>();
            job->frame = frame.clone();
            cv::Mat preprocessed;
            job->preprocess_begin = std::chrono::steady_clock::now();
            preprocessor->process(job->frame, preprocessed, job->ctx);
            job->preprocess_end = std::chrono::steady_clock::now();

            if (preprocessed.empty()) {
                continue;
            }
            if (is_float_input) {
                job->input_float = convertToFloatBuffer(preprocessed, is_nhwc);
            } else {
                job->input_u8 = preprocessed.clone();
            }

            if (!acquireInflightSlot()) {
                break;
            }

            job->submit_time = std::chrono::steady_clock::now();
            try {
                last_job_id = ie.RunAsync(job->inputData(), static_cast<void*>(job.get()), nullptr);
                job.release();
            } catch (...) {
                releaseInflightSlot();
                throw;
            }
        }

        if (last_job_id >= 0) {
            ie.Wait(last_job_id);
        }
        waitForInflightEmpty();
    } catch (const std::exception& e) {
        running_.store(false, std::memory_order_relaxed);
        window_->postError(name_ + " worker failed: " + e.what());
        waitForInflightEmpty();
    }
}

bool DepthWorker::acquireInflightSlot() {
    std::unique_lock<std::mutex> lock(inflight_mutex_);
    inflight_cv_.wait(lock, [this] {
        return inflight_ < max_inflight_ ||
               !running_.load(std::memory_order_relaxed);
    });
    if (!running_.load(std::memory_order_relaxed)) {
        return false;
    }
    ++inflight_;
    return true;
}

void DepthWorker::releaseInflightSlot() {
    {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        if (inflight_ > 0) {
            --inflight_;
        }
    }
    inflight_cv_.notify_all();
}

void DepthWorker::waitForInflightEmpty() {
    std::unique_lock<std::mutex> lock(inflight_mutex_);
    inflight_cv_.wait(lock, [this] { return inflight_ == 0; });
}

void DepthWorker::preprocess(const cv::Mat& bgr,
                             std::vector<std::uint8_t>& input,
                             DepthLetterboxContext& context) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("Depth preprocessing expects a non-empty BGR8 frame");
    }

    context.original_width = bgr.cols;
    context.original_height = bgr.rows;
    const double gain = std::min(
        static_cast<double>(network_width_) / context.original_width,
        static_cast<double>(network_height_) / context.original_height);
    const int resized_width =
        std::max(1, cvRound(context.original_width * gain));
    const int resized_height =
        std::max(1, cvRound(context.original_height * gain));
    const double horizontal_padding = (network_width_ - resized_width) / 2.0;
    const double vertical_padding = (network_height_ - resized_height) / 2.0;
    context.left = cvRound(horizontal_padding - 0.1);
    context.right = cvRound(horizontal_padding + 0.1);
    context.top = cvRound(vertical_padding - 0.1);
    context.bottom = cvRound(vertical_padding + 0.1);

    cv::resize(bgr, resized_scratch_, cv::Size(resized_width, resized_height),
               0.0, 0.0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized_scratch_, padded_scratch_,
                       context.top, context.bottom,
                       context.left, context.right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (padded_scratch_.cols != network_width_ ||
        padded_scratch_.rows != network_height_) {
        throw std::runtime_error("Depth letterbox produced an unexpected image size");
    }

    cv::cvtColor(padded_scratch_, rgb_scratch_, cv::COLOR_BGR2RGB);
    if (!rgb_scratch_.isContinuous()) {
        rgb_scratch_ = rgb_scratch_.clone();
    }

    const std::size_t bytes = rgb_scratch_.total() * rgb_scratch_.elemSize();
    if (bytes != input_bytes_) {
        throw std::runtime_error("Depth input tensor size does not match preprocessing output");
    }
    input.resize(bytes);
    std::memcpy(input.data(), rgb_scratch_.data, bytes);
}

cv::Mat DepthWorker::restoreDepth(const dxrt::Tensor* tensor,
                                  const DepthLetterboxContext& context,
                                  int network_width,
                                  int network_height) {
    if (tensor == nullptr || tensor->data() == nullptr ||
        tensor->type() != dxrt::DataType::FLOAT ||
        context.original_width <= 0 || context.original_height <= 0 ||
        network_width <= 0 || network_height <= 0) {
        return {};
    }

    const auto& shape = tensor->shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 ||
        shape[2] <= 0 || shape[3] <= 0) {
        return {};
    }
    const int output_height = static_cast<int>(shape[2]);
    const int output_width = static_cast<int>(shape[3]);
    const std::size_t expected_bytes =
        static_cast<std::size_t>(output_width) * output_height * sizeof(float);
    if (tensor->size_in_bytes() < expected_bytes) {
        return {};
    }

    // The DX-RT output buffer is callback-scoped. OpenCV only reads this view,
    // and cv::resize below copies the selected area before the callback returns.
    cv::Mat raw(output_height, output_width, CV_32F,
                const_cast<void*>(tensor->data()));
    const double scale_x = static_cast<double>(output_width) / network_width;
    const double scale_y = static_cast<double>(output_height) / network_height;
    const int x_begin = std::clamp(
        cvRound(context.left * scale_x), 0, output_width - 1);
    const int x_end = std::clamp(
        output_width - cvRound(context.right * scale_x),
        x_begin + 1, output_width);
    const int y_begin = std::clamp(
        cvRound(context.top * scale_y), 0, output_height - 1);
    const int y_end = std::clamp(
        output_height - cvRound(context.bottom * scale_y),
        y_begin + 1, output_height);

    cv::Mat restored;
    cv::resize(raw(cv::Rect(x_begin, y_begin,
                            x_end - x_begin, y_end - y_begin)),
               restored,
               cv::Size(context.original_width, context.original_height),
               0.0, 0.0, cv::INTER_LINEAR);
    return restored;
}

cv::Mat DepthWorker::colorizeDepth(const cv::Mat& depth) {
    if (depth.empty() || depth.type() != CV_32FC1 || !cv::checkRange(depth)) {
        return {};
    }

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(depth, &minimum, &maximum);
    cv::Mat normalized;
    if (maximum > minimum) {
        depth.convertTo(normalized, CV_8U,
                        255.0 / (maximum - minimum),
                        -255.0 * minimum / (maximum - minimum));
    } else {
        normalized = cv::Mat::zeros(depth.size(), CV_8U);
    }

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_TURBO);
    return colored;
}

void DepthWorker::run() {
    try {
        dxrt::InferenceOption options;
        options.bufferCount = max_inflight_;
        dxrt::InferenceEngine engine(model_path_, options);
        if (!dxapp::minversionforRTandCompiler(&engine)) {
            throw std::runtime_error(
                "Depth model/runtime version mismatch: " + model_path_);
        }

        const dxrt::Tensors inputs = engine.GetInputs();
        const dxrt::Tensors outputs = engine.GetOutputs();
        if (inputs.size() != 1 || outputs.size() != 1) {
            throw std::runtime_error(
                "Depth model must expose exactly one input and one output");
        }

        const auto& input_shape = inputs.front().shape();
        if (inputs.front().type() != dxrt::DataType::UINT8 ||
            input_shape.size() != 4 || input_shape[0] != 1 ||
            input_shape[1] <= 0 || input_shape[2] <= 0 ||
            input_shape[3] != 3) {
            throw std::runtime_error(
                "Depth model expects UINT8 NHWC input [1,H,W,3]");
        }
        network_height_ = static_cast<int>(input_shape[1]);
        network_width_ = static_cast<int>(input_shape[2]);
        input_bytes_ = static_cast<std::size_t>(engine.GetInputSize());
        const std::size_t expected_input_bytes =
            static_cast<std::size_t>(network_width_) * network_height_ * 3;
        if (input_bytes_ != expected_input_bytes) {
            throw std::runtime_error("Unexpected depth model input buffer size");
        }

        const auto& output_shape = outputs.front().shape();
        if (outputs.front().type() != dxrt::DataType::FLOAT ||
            output_shape.size() != 4 || output_shape[0] != 1 ||
            output_shape[1] != 1 || output_shape[2] <= 0 ||
            output_shape[3] <= 0) {
            throw std::runtime_error(
                "Depth model expects FLOAT NCHW output [1,1,H,W]");
        }

        std::cout << "[INFO] Depth Estimation model: " << model_path_ << std::endl;
        std::cout << "[INFO] Depth Estimation input size (WxH): "
                  << network_width_ << "x" << network_height_ << std::endl;
        std::cout << "[INFO] Depth Estimation output size (WxH): "
                  << output_shape[3] << "x" << output_shape[2] << std::endl;
        std::cout << "[INFO] Depth Estimation async queue size: "
                  << max_inflight_ << std::endl;

        engine.RegisterCallback(
            [this](dxrt::TensorPtrs& tensors, void* user_data) -> int {
                auto job = std::unique_ptr<AsyncJob>(
                    static_cast<AsyncJob*>(user_data));
                if (!job) {
                    releaseInflightSlot();
                    return -1;
                }

                try {
                    if (tensors.size() != 1) {
                        throw std::runtime_error(
                            "Depth callback received an unexpected output count");
                    }
                    cv::Mat depth = restoreDepth(
                        tensors.front().get(), job->letterbox,
                        network_width_, network_height_);
                    cv::Mat rendered = colorizeDepth(depth);
                    if (rendered.empty()) {
                        throw std::runtime_error("Depth output tensor is invalid");
                    }
                    if (running_.load(std::memory_order_relaxed)) {
                        window_->postFrame(kDepthPanel, rendered);
                    }
                } catch (const std::exception& error) {
                    running_.store(false, std::memory_order_relaxed);
                    queue_.wake();
                    window_->postError(
                        std::string("Depth async callback failed: ") + error.what());
                }

                releaseInflightSlot();
                return 0;
            });

        int last_job_id = -1;
        while (running_.load(std::memory_order_relaxed)) {
            cv::Mat frame;
            if (!queue_.waitPop(frame, running_) || frame.empty()) {
                continue;
            }

            auto job = std::make_unique<AsyncJob>();
            preprocess(frame, job->input, job->letterbox);
            if (!acquireInflightSlot()) {
                break;
            }

            try {
                last_job_id = engine.RunAsync(
                    job->input.data(), static_cast<void*>(job.get()), nullptr);
                job.release();
            } catch (...) {
                releaseInflightSlot();
                throw;
            }
        }

        if (last_job_id >= 0) {
            engine.Wait(last_job_id);
        }
        waitForInflightEmpty();
    } catch (const std::exception& error) {
        running_.store(false, std::memory_order_relaxed);
        window_->postError(
            std::string("Depth worker failed: ") + error.what());
        waitForInflightEmpty();
    }
}

void CaptureThread::publishFrame(const cv::Mat& frame) {
    cv::Mat bgr = ensureBgr3(frame);
    if (bgr.empty()) {
        return;
    }
    for (auto* consumer : consumers_) {
        consumer->enqueue(bgr);
    }
}

void CaptureThread::runVideo() {
    cv::VideoCapture cap(args_.video);
    if (!cap.isOpened()) {
        window_->postError("Could not open video file: " + args_.video);
        return;
    }

    double source_fps = cap.get(cv::CAP_PROP_FPS);
    if (source_fps <= 1e-3) {
        source_fps = 30.0;
    }
    const auto frame_delay = std::chrono::duration<double>(1.0 / source_fps);

    while (running_.load(std::memory_order_relaxed)) {
        auto loop_start = std::chrono::steady_clock::now();
        cv::Mat frame;
        bool ok = cap.read(frame);
        if (!ok || frame.empty()) {
            if (args_.no_loop_video) {
                window_->postEndOfStream();
                break;
            }

            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            ok = cap.read(frame);
            if (!ok || frame.empty()) {
                cap.release();
                cap.open(args_.video);
                if (!cap.isOpened()) {
                    window_->postError("Video loop failed to reopen file.");
                    break;
                }
                ok = cap.read(frame);
            }
            if (!ok || frame.empty()) {
                window_->postError("Video file has no readable frames.");
                break;
            }
        }

        publishFrame(frame);

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        if (elapsed < frame_delay && running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(frame_delay - elapsed);
        }
    }
}

void CaptureThread::runCamera() {
    cv::VideoCapture cap;
    bool opened = false;

    if (!args_.device.empty() && args_.device[0] == '/') {
        opened = cap.open(args_.device, cv::CAP_V4L2);
    }
    if (!opened) {
        int index = parseCameraIndex(args_.device);
        opened = cap.open(index, cv::CAP_V4L2);
    }
    if (!opened) {
        window_->postError("VideoCapture failed for camera: " + args_.device);
        return;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (args_.width > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, args_.width);
    }
    if (args_.height > 0) {
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, args_.height);
    }
    if (args_.fps > 0) {
        cap.set(cv::CAP_PROP_FPS, args_.fps);
    }

    std::cout << "[INFO] Input: USB webcam " << args_.device << std::endl;
    std::cout << "[INFO] Camera resolution: "
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    while (running_.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            window_->postError("cap.read() failed or stream ended.");
            break;
        }
        publishFrame(frame);
    }
}

void CaptureThread::run() {
    try {
        if (!args_.video.empty()) {
            std::cout << "[INFO] Input: video file " << args_.video << std::endl;
            if (args_.no_loop_video) {
                std::cout << "[INFO] Video will not loop (stop at EOF)" << std::endl;
            }
            runVideo();
        } else {
            runCamera();
        }
    } catch (const std::exception& e) {
        window_->postError(std::string("Capture thread failed: ") + e.what());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        AppArgs args = parseArgs(argc, argv);
        args.model = absolutePath(args.model);
        args.model_pose = absolutePath(args.model_pose);
        args.model_seg = absolutePath(args.model_seg);
        args.model_depth = absolutePath(args.model_depth);
        if (!args.video.empty()) {
            args.video = absolutePath(args.video);
        }

        requireFile("OD model", args.model);
        requireFile("Pose model", args.model_pose);
        requireFile("Seg model", args.model_seg);
        requireFile("Depth model", args.model_depth);
        if (!args.video.empty()) {
            requireFile("Video file", args.video);
        }

        QApplication app(argc, argv);
        QuadWindow window(args);
        window.showFullScreen();
        return app.exec();
    } catch (const dxrt::Exception& e) {
        std::cerr << e.what() << " error-code=" << e.code() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
