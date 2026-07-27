#include <QtCore/QCoreApplication>
#include <QtCore/QMetaObject>
#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtGui/QCloseEvent>
#include <QtGui/QImage>
#include <QtGui/QKeySequence>
#include <QtGui/QPainter>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QShortcut>
#include <QtWidgets/QSlider>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

#include <dxrt/dxrt_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR ".."
#endif

constexpr const char* kDefaultModelPath =
    PROJECT_ROOT_DIR "/assets/models/pidnet_s_cityscapes_val_fixed.dxnn";
constexpr double kDefaultArgmaxScale = 0.4;
constexpr double kMinimumArgmaxScale = 0.1;
constexpr double kMaximumArgmaxScale = 1.0;
constexpr int kArgmaxScaleStepsPerUnit = 20;  // One slider step is 0.05.
constexpr double kDefaultOverlayAlpha = 0.6;
constexpr int kDefaultCameraWidth = 1280;
constexpr int kDefaultCameraHeight = 720;
constexpr int kDefaultCameraFps = 30;
constexpr int kDefaultMaxInflight = 4;

struct Options {
    std::string model_path = kDefaultModelPath;
    bool use_camera = true;
    int camera_index = 0;
    std::string video_path;
    int camera_width = kDefaultCameraWidth;
    int camera_height = kDefaultCameraHeight;
    int camera_fps = kDefaultCameraFps;
    double pidnet_argmax_scale = kDefaultArgmaxScale;
    double overlay_alpha = kDefaultOverlayAlpha;
    int max_inflight = kDefaultMaxInflight;
    bool loop_video = false;
    bool pace_video = true;
    bool fullscreen = true;
};

struct TensorShapeInfo {
    int classes = 0;
    int height = 0;
    int width = 0;
    bool is_nhwc = false;
};

bool file_exists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return static_cast<bool>(input);
}

bool parse_int(const std::string& text, int* value) {
    try {
        std::size_t used = 0;
        const int parsed = std::stoi(text, &used);
        if (used != text.size()) return false;
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& text, double* value) {
    try {
        std::size_t used = 0;
        const double parsed = std::stod(text, &used);
        if (used != text.size()) return false;
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [OPTIONS]\n"
        << "  -m, --model <PATH>                PIDNet .dxnn model path\n"
        << "  -c, --camera [INDEX]              Use a camera (default index: 0)\n"
        << "  -v, --video <PATH>                Use a video file\n"
        << "      --width <N>                   Requested camera width (default: 1280)\n"
        << "      --height <N>                  Requested camera height (default: 720)\n"
        << "      --fps <N>                     Requested camera FPS (default: 30)\n"
        << "      --pidnet-argmax-scale <VALUE> Initial argmax scale in [0.1, 1.0] (default: 0.4)\n"
        << "      --alpha <VALUE>               Mask opacity in [0.0, 1.0] (default: 0.6)\n"
        << "      --inflight <N>                Maximum asynchronous requests (default: 4)\n"
        << "      --loop                        Loop video input\n"
        << "      --no-pace                     Process video as fast as possible\n"
        << "      --windowed                    Start in a window\n"
        << "      --full-screen                 Start in full-screen mode (default)\n"
        << "  -h, --help                        Show this help\n";
}

bool parse_args(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc;) {
        const std::string arg(argv[index++]);
        auto require_value = [&](const char* name) -> const char* {
            if (index >= argc) {
                std::cerr << "Error: missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[index++];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->model_path = value;
        } else if (arg == "-c" || arg == "--camera") {
            options->use_camera = true;
            options->video_path.clear();
            if (index < argc && argv[index][0] != '-') {
                int parsed = 0;
                if (!parse_int(argv[index], &parsed) || parsed < 0) {
                    std::cerr << "Error: " << arg
                              << " expects a non-negative camera index" << std::endl;
                    return false;
                }
                options->camera_index = parsed;
                ++index;
            }
        } else if (arg == "-v" || arg == "--video") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr) return false;
            options->use_camera = false;
            options->video_path = value;
        } else if (arg == "--width") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_width) ||
                options->camera_width <= 0) {
                std::cerr << "Error: --width expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--height") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_height) ||
                options->camera_height <= 0) {
                std::cerr << "Error: --height expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--fps") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->camera_fps) ||
                options->camera_fps <= 0) {
                std::cerr << "Error: --fps expects a positive integer" << std::endl;
                return false;
            }
        } else if (arg == "--pidnet-argmax-scale" || arg == "--argmax-scale") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr ||
                !parse_double(value, &options->pidnet_argmax_scale) ||
                options->pidnet_argmax_scale < kMinimumArgmaxScale ||
                options->pidnet_argmax_scale > kMaximumArgmaxScale) {
                std::cerr << "Error: " << arg << " expects a value in [0.1, 1.0]"
                          << std::endl;
                return false;
            }
        } else if (arg == "--alpha") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_double(value, &options->overlay_alpha) ||
                options->overlay_alpha < 0.0 || options->overlay_alpha > 1.0) {
                std::cerr << "Error: --alpha expects a value in [0.0, 1.0]"
                          << std::endl;
                return false;
            }
        } else if (arg == "--inflight") {
            const char* value = require_value(arg.c_str());
            if (value == nullptr || !parse_int(value, &options->max_inflight) ||
                options->max_inflight <= 0 || options->max_inflight > 6) {
                std::cerr << "Error: --inflight expects an integer in [1, 6]"
                          << std::endl;
                return false;
            }
        } else if (arg == "--loop") {
            options->loop_video = true;
        } else if (arg == "--no-pace") {
            options->pace_video = false;
        } else if (arg == "--windowed") {
            options->fullscreen = false;
        } else if (arg == "--full-screen") {
            options->fullscreen = true;
        } else {
            std::cerr << "Error: unknown option '" << arg << "'" << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if (!options->use_camera && options->video_path.empty()) {
        std::cerr << "Error: --video requires a path" << std::endl;
        return false;
    }
    return true;
}

void parse_model_input_shape(const std::vector<int64_t>& shape,
                             int* width,
                             int* height,
                             int* channels) {
    if (shape.size() != 4) {
        throw std::runtime_error("model input must be a four-dimensional image tensor");
    }
    if (shape[3] == 3) {
        *height = static_cast<int>(shape[1]);
        *width = static_cast<int>(shape[2]);
        *channels = static_cast<int>(shape[3]);
    } else if (shape[1] == 3) {
        *channels = static_cast<int>(shape[1]);
        *height = static_cast<int>(shape[2]);
        *width = static_cast<int>(shape[3]);
    } else {
        throw std::runtime_error("model input must contain three color channels");
    }
}

TensorShapeInfo parse_output_shape(const std::vector<int64_t>& shape) {
    TensorShapeInfo info;
    if (shape.size() == 4) {
        if (shape[1] <= shape[3]) {
            info.classes = static_cast<int>(shape[1]);
            info.height = static_cast<int>(shape[2]);
            info.width = static_cast<int>(shape[3]);
        } else {
            info.height = static_cast<int>(shape[1]);
            info.width = static_cast<int>(shape[2]);
            info.classes = static_cast<int>(shape[3]);
            info.is_nhwc = true;
        }
    } else if (shape.size() == 3) {
        if (shape[0] <= shape[1] && shape[0] <= shape[2]) {
            info.classes = static_cast<int>(shape[0]);
            info.height = static_cast<int>(shape[1]);
            info.width = static_cast<int>(shape[2]);
        } else {
            info.height = static_cast<int>(shape[0]);
            info.width = static_cast<int>(shape[1]);
            info.classes = static_cast<int>(shape[2]);
            info.is_nhwc = true;
        }
    }
    if (info.classes <= 1 || info.height <= 0 || info.width <= 0) {
        throw std::runtime_error("PIDNet output must contain multi-class logits");
    }
    return info;
}

std::vector<std::uint8_t> preprocess_frame(const cv::Mat& frame,
                                           int input_width,
                                           int input_height) {
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(input_width, input_height),
               0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous()) resized = resized.clone();

    const std::size_t byte_count = resized.total() * resized.elemSize();
    std::vector<std::uint8_t> input(byte_count);
    std::memcpy(input.data(), resized.data, byte_count);
    return input;
}

int scaled_argmax_size(int target_size, double scale) {
    int scaled = static_cast<int>(std::round(target_size * scale));
    scaled = std::min(target_size, scaled);
    return std::max(1, scaled);
}

cv::Mat compute_argmax_mask(const float* data,
                            const TensorShapeInfo& info,
                            int target_width,
                            int target_height,
                            double scale) {
    const int output_width = scaled_argmax_size(target_width, scale);
    const int output_height = scaled_argmax_size(target_height, scale);
    cv::Mat scaled_mask(output_height, output_width, CV_32SC1);

    const float scale_x = output_width > 1
        ? static_cast<float>(info.width - 1) / static_cast<float>(output_width - 1)
        : 0.0f;
    const float scale_y = output_height > 1
        ? static_cast<float>(info.height - 1) / static_cast<float>(output_height - 1)
        : 0.0f;

    std::vector<int> source_x0(static_cast<std::size_t>(output_width));
    std::vector<int> source_x1(static_cast<std::size_t>(output_width));
    std::vector<float> weight_x(static_cast<std::size_t>(output_width));
    for (int x = 0; x < output_width; ++x) {
        const float source_x = std::min(
            static_cast<float>(x) * scale_x,
            static_cast<float>(info.width - 1));
        source_x0[static_cast<std::size_t>(x)] =
            static_cast<int>(std::floor(source_x));
        source_x1[static_cast<std::size_t>(x)] =
            std::min(source_x0[static_cast<std::size_t>(x)] + 1,
                     info.width - 1);
        weight_x[static_cast<std::size_t>(x)] =
            source_x - static_cast<float>(source_x0[static_cast<std::size_t>(x)]);
    }

    const std::size_t class_stride =
        static_cast<std::size_t>(info.height) * info.width;
    const std::size_t row_stride =
        static_cast<std::size_t>(info.width) * info.classes;

    for (int y = 0; y < output_height; ++y) {
        int* destination = scaled_mask.ptr<int>(y);
        const float source_y = std::min(
            static_cast<float>(y) * scale_y,
            static_cast<float>(info.height - 1));
        const int y0 = static_cast<int>(std::floor(source_y));
        const int y1 = std::min(y0 + 1, info.height - 1);
        const float weight_y = source_y - static_cast<float>(y0);
        const std::size_t nchw_row0 = static_cast<std::size_t>(y0) * info.width;
        const std::size_t nchw_row1 = static_cast<std::size_t>(y1) * info.width;
        const std::size_t nhwc_row0 = static_cast<std::size_t>(y0) * row_stride;
        const std::size_t nhwc_row1 = static_cast<std::size_t>(y1) * row_stride;

        for (int x = 0; x < output_width; ++x) {
            const int x0 = source_x0[static_cast<std::size_t>(x)];
            const int x1 = source_x1[static_cast<std::size_t>(x)];
            const float wx = weight_x[static_cast<std::size_t>(x)];
            float best_value = -1.0e30f;
            int best_class = 0;

            if (!info.is_nhwc) {
                const std::size_t pixel00 = nchw_row0 + x0;
                const std::size_t pixel01 = nchw_row0 + x1;
                const std::size_t pixel10 = nchw_row1 + x0;
                const std::size_t pixel11 = nchw_row1 + x1;
                for (int class_index = 0; class_index < info.classes; ++class_index) {
                    const float* channel =
                        data + static_cast<std::size_t>(class_index) * class_stride;
                    const float value00 = channel[pixel00];
                    const float value01 = channel[pixel01];
                    const float value10 = channel[pixel10];
                    const float value11 = channel[pixel11];
                    const float top = value00 + (value01 - value00) * wx;
                    const float bottom = value10 + (value11 - value10) * wx;
                    const float value = top + (bottom - top) * weight_y;
                    if (value > best_value) {
                        best_value = value;
                        best_class = class_index;
                    }
                }
            } else {
                const std::size_t pixel00 =
                    nhwc_row0 + static_cast<std::size_t>(x0) * info.classes;
                const std::size_t pixel01 =
                    nhwc_row0 + static_cast<std::size_t>(x1) * info.classes;
                const std::size_t pixel10 =
                    nhwc_row1 + static_cast<std::size_t>(x0) * info.classes;
                const std::size_t pixel11 =
                    nhwc_row1 + static_cast<std::size_t>(x1) * info.classes;
                for (int class_index = 0; class_index < info.classes; ++class_index) {
                    const std::size_t channel = static_cast<std::size_t>(class_index);
                    const float value00 = data[pixel00 + channel];
                    const float value01 = data[pixel01 + channel];
                    const float value10 = data[pixel10 + channel];
                    const float value11 = data[pixel11 + channel];
                    const float top = value00 + (value01 - value00) * wx;
                    const float bottom = value10 + (value11 - value10) * wx;
                    const float value = top + (bottom - top) * weight_y;
                    if (value > best_value) {
                        best_value = value;
                        best_class = class_index;
                    }
                }
            }
            destination[x] = best_class;
        }
    }

    if (scaled_mask.cols == target_width && scaled_mask.rows == target_height) {
        return scaled_mask;
    }
    cv::Mat full_mask;
    cv::resize(scaled_mask, full_mask, cv::Size(target_width, target_height),
               0.0, 0.0, cv::INTER_NEAREST);
    return full_mask;
}

cv::Vec3b rgb(int red, int green, int blue) {
    return cv::Vec3b(static_cast<unsigned char>(blue),
                     static_cast<unsigned char>(green),
                     static_cast<unsigned char>(red));
}

const std::array<cv::Vec3b, 20>& cityscapes_palette() {
    static const std::array<cv::Vec3b, 20> palette = {
        rgb(78, 73, 138), rgb(96, 183, 199), rgb(56, 63, 78),
        rgb(126, 100, 172), rgb(166, 128, 151), rgb(185, 188, 195),
        rgb(255, 204, 77), rgb(255, 149, 89), rgb(45, 170, 116),
        rgb(124, 212, 154), rgb(96, 197, 230), rgb(255, 94, 122),
        rgb(255, 112, 67), rgb(62, 130, 255), rgb(39, 83, 184),
        rgb(0, 166, 217), rgb(34, 107, 148), rgb(178, 97, 255),
        rgb(54, 221, 187), rgb(0, 0, 0),
    };
    return palette;
}

cv::Mat render_segmentation(const cv::Mat& frame,
                            const cv::Mat& class_mask,
                            double alpha) {
    cv::Mat color_mask(class_mask.size(), CV_8UC3);
    const auto& palette = cityscapes_palette();
    for (int y = 0; y < class_mask.rows; ++y) {
        const int* source = class_mask.ptr<int>(y);
        cv::Vec3b* destination = color_mask.ptr<cv::Vec3b>(y);
        for (int x = 0; x < class_mask.cols; ++x) {
            const int class_id = source[x];
            const std::size_t palette_index = class_id >= 0
                ? static_cast<std::size_t>(class_id) % palette.size()
                : palette.size() - 1;
            destination[x] = palette[palette_index];
        }
    }

    cv::Mat output;
    cv::addWeighted(frame, 1.0 - alpha, color_mask, alpha, 0.0, output);
    return output;
}

QImage mat_to_qimage(const cv::Mat& bgr) {
    cv::Mat rgb_image;
    cv::cvtColor(bgr, rgb_image, cv::COLOR_BGR2RGB);
    return QImage(rgb_image.data, rgb_image.cols, rgb_image.rows,
                  static_cast<int>(rgb_image.step), QImage::Format_RGB888).copy();
}

class FrameMailbox {
public:
    void setModelInputShape(int width, int height, int channels) {
        input_width_.store(width);
        input_height_.store(height);
        input_channels_.store(channels);
    }

    bool modelInputShape(int* width, int* height, int* channels) const {
        *width = input_width_.load();
        *height = input_height_.load();
        *channels = input_channels_.load();
        return *width > 0 && *height > 0 && *channels > 0;
    }

    void publish(cv::Mat frame, double fps, std::uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_frame_id_ && frame_id < highest_frame_id_) return;
        highest_frame_id_ = frame_id;
        has_frame_id_ = true;
        frame_ = std::move(frame);
        fps_ = fps;
        ready_ = true;
    }

    bool take(cv::Mat* frame, double* fps) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return false;
        *frame = std::move(frame_);
        *fps = fps_;
        ready_ = false;
        return true;
    }

private:
    std::mutex mutex_;
    cv::Mat frame_;
    double fps_ = 0.0;
    std::uint64_t highest_frame_id_ = 0;
    bool has_frame_id_ = false;
    bool ready_ = false;
    std::atomic<int> input_width_{0};
    std::atomic<int> input_height_{0};
    std::atomic<int> input_channels_{0};
};

class VideoView : public QWidget {
public:
    explicit VideoView(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumSize(640, 360);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    void setFrame(const QImage& frame, double fps) {
        frame_ = frame;
        fps_ = fps;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(4, 7, 11));
        if (frame_.isNull()) return;

        QSize target_size = frame_.size();
        target_size.scale(size(), Qt::KeepAspectRatio);
        const QRect target((width() - target_size.width()) / 2,
                           (height() - target_size.height()) / 2,
                           target_size.width(), target_size.height());
        painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        painter.drawImage(target, frame_);

        const QString fps_text = QString("%1 FPS").arg(fps_, 0, 'f', 1);
        QFont font = painter.font();
        font.setPixelSize(20);
        font.setBold(true);
        painter.setFont(font);
        const QFontMetrics metrics(font);
        const int text_width = metrics.horizontalAdvance(fps_text);
        const QRect overlay(target.left() + 16, target.top() + 16,
                            text_width + 28, metrics.height() + 18);
        painter.setPen(QPen(QColor(90, 103, 120), 1));
        painter.setBrush(QColor(17, 23, 31, 215));
        painter.drawRoundedRect(overlay, 8, 8);
        painter.setPen(QColor(238, 244, 250));
        painter.drawText(overlay, Qt::AlignCenter, fps_text);
    }

private:
    QImage frame_;
    double fps_ = 0.0;
};

class MainWindow : public QWidget {
public:
    MainWindow(FrameMailbox* mailbox,
               std::atomic<double>* argmax_scale,
               double initial_scale,
               QWidget* parent = nullptr)
        : QWidget(parent), mailbox_(mailbox), argmax_scale_(argmax_scale) {
        setWindowTitle("PIDNet Cityscapes Segmentation");
        setStyleSheet(
            "QWidget { background-color: #080d14; color: #eef4fa; }"
            "QLabel#title { font-size: 18px; font-weight: 700; }"
            "QLabel#caption { color: #9eacba; font-size: 13px; }"
            "QLabel#shape { color: #ffb45c; font-size: 14px; font-weight: 700; }"
            "QLabel#value { color: #8acbd8; font-size: 16px; font-weight: 700; }"
            "QPushButton { background: #202a36; border: 1px solid #3b4a5c;"
            " border-radius: 6px; padding: 9px 18px; }"
            "QPushButton:hover { background: #2b3948; }"
            "QSlider::groove:horizontal { height: 4px; background: #273341;"
            " border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #3c9caf; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #e6f0f3; border: 1px solid #5799a7;"
            " width: 14px; margin: -5px 0; border-radius: 7px; }");

        auto* main_layout = new QVBoxLayout(this);
        main_layout->setContentsMargins(0, 0, 0, 0);
        main_layout->setSpacing(0);

        video_view_ = new VideoView(this);
        main_layout->addWidget(video_view_, 1);

        auto* controls = new QWidget(this);
        controls->setFixedHeight(74);
        controls->setStyleSheet("border-top: 1px solid #273341;");
        auto* controls_layout = new QHBoxLayout(controls);
        controls_layout->setContentsMargins(24, 12, 24, 12);
        controls_layout->setSpacing(18);

        auto* heading_layout = new QVBoxLayout();
        auto* title = new QLabel("PIDNet Cityscapes", controls);
        title->setObjectName("title");
        input_shape_label_ = new QLabel("INPUT SHAPE  [loading]", controls);
        input_shape_label_->setObjectName("shape");
        heading_layout->addWidget(title);
        heading_layout->addWidget(input_shape_label_);
        controls_layout->addLayout(heading_layout);
        controls_layout->addSpacing(20);

        auto* scale_label = new QLabel("Argmax scale", controls);
        scale_label->setObjectName("caption");
        controls_layout->addWidget(scale_label);

        slider_ = new QSlider(Qt::Horizontal, controls);
        slider_->setRange(
            static_cast<int>(std::round(
                kMinimumArgmaxScale * kArgmaxScaleStepsPerUnit)),
            static_cast<int>(std::round(
                kMaximumArgmaxScale * kArgmaxScaleStepsPerUnit)));
        slider_->setSingleStep(1);
        slider_->setPageStep(1);
        slider_->setTickPosition(QSlider::NoTicks);
        slider_->setValue(static_cast<int>(
            std::round(initial_scale * kArgmaxScaleStepsPerUnit)));
        slider_->setToolTip("PIDNet argmax scale: 0.10 to 1.00 (step: 0.05)");
        slider_->setMinimumWidth(180);
        slider_->setMaximumWidth(420);
        controls_layout->addWidget(slider_, 1);

        argmax_scale_->store(slider_scale(slider_->value()));

        value_label_ = new QLabel(scale_text(slider_->value()), controls);
        value_label_->setObjectName("value");
        value_label_->setMinimumWidth(44);
        value_label_->setAlignment(Qt::AlignCenter);
        controls_layout->addWidget(value_label_);
        controls_layout->addStretch(1);

        auto* exit_button = new QPushButton("Exit", controls);
        controls_layout->addWidget(exit_button);
        main_layout->addWidget(controls);

        QObject::connect(slider_, &QSlider::valueChanged, [this](int value) {
            const double scale = slider_scale(value);
            argmax_scale_->store(scale);
            value_label_->setText(scale_text(value));
        });
        QObject::connect(exit_button, &QPushButton::clicked,
                         QCoreApplication::instance(), &QCoreApplication::quit);

        auto* escape_shortcut = new QShortcut(QKeySequence(Qt::Key_Escape), this);
        auto* quit_shortcut = new QShortcut(QKeySequence(Qt::Key_Q), this);
        auto* fullscreen_shortcut = new QShortcut(QKeySequence(Qt::Key_F), this);
        QObject::connect(escape_shortcut, &QShortcut::activated,
                         QCoreApplication::instance(), &QCoreApplication::quit);
        QObject::connect(quit_shortcut, &QShortcut::activated,
                         QCoreApplication::instance(), &QCoreApplication::quit);
        QObject::connect(fullscreen_shortcut, &QShortcut::activated, [this]() {
            isFullScreen() ? showNormal() : showFullScreen();
        });

        auto* frame_timer = new QTimer(this);
        frame_timer->setInterval(16);
        QObject::connect(frame_timer, &QTimer::timeout, [this]() {
            int input_width = 0;
            int input_height = 0;
            int input_channels = 0;
            if (mailbox_->modelInputShape(
                    &input_width, &input_height, &input_channels) &&
                (input_width != displayed_input_width_ ||
                 input_height != displayed_input_height_ ||
                 input_channels != displayed_input_channels_)) {
                input_shape_label_->setText(
                    QString("INPUT SHAPE  [1, %1, %2, %3]")
                        .arg(input_height)
                        .arg(input_width)
                        .arg(input_channels));
                displayed_input_width_ = input_width;
                displayed_input_height_ = input_height;
                displayed_input_channels_ = input_channels;
            }
            cv::Mat frame;
            double fps = 0.0;
            if (mailbox_->take(&frame, &fps)) {
                video_view_->setFrame(mat_to_qimage(frame), fps);
            }
        });
        frame_timer->start();
    }

protected:
    void closeEvent(QCloseEvent* event) override {
        QCoreApplication::quit();
        QWidget::closeEvent(event);
    }

private:
    static double slider_scale(int slider_value) {
        return static_cast<double>(slider_value) / kArgmaxScaleStepsPerUnit;
    }

    static QString scale_text(int slider_value) {
        return QString::number(slider_scale(slider_value), 'f', 2);
    }

    FrameMailbox* mailbox_;
    std::atomic<double>* argmax_scale_;
    VideoView* video_view_ = nullptr;
    QSlider* slider_ = nullptr;
    QLabel* value_label_ = nullptr;
    QLabel* input_shape_label_ = nullptr;
    int displayed_input_width_ = 0;
    int displayed_input_height_ = 0;
    int displayed_input_channels_ = 0;
};

class CompletionCounter {
public:
    double update() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_;
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - start_).count();
        return seconds > 1.0e-6 ? static_cast<double>(completed_) / seconds : 0.0;
    }

private:
    std::mutex mutex_;
    std::uint64_t completed_ = 0;
    std::chrono::steady_clock::time_point start_ =
        std::chrono::steady_clock::now();
};

class InflightLimiter {
public:
    explicit InflightLimiter(int limit) : limit_(limit) {}

    bool acquire(const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (count_ >= limit_) {
            if (!running.load()) return false;
            condition_.wait_for(lock, std::chrono::milliseconds(20));
        }
        if (!running.load()) return false;
        ++count_;
        return true;
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (count_ > 0) --count_;
        }
        condition_.notify_all();
    }

    void wait_until_empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return count_ == 0; });
    }

private:
    int limit_;
    int count_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

struct AsyncJob {
    std::vector<std::uint8_t> input;
    cv::Mat frame;
    std::uint64_t frame_id = 0;
};

struct CallbackContext {
    const Options* options = nullptr;
    std::atomic<double>* argmax_scale = nullptr;
    std::atomic<bool>* running = nullptr;
    InflightLimiter* limiter = nullptr;
    CompletionCounter* completion_counter = nullptr;
    FrameMailbox* mailbox = nullptr;
};

void request_quit() {
    QMetaObject::invokeMethod(QCoreApplication::instance(),
                              []() { QCoreApplication::quit(); },
                              Qt::QueuedConnection);
}

void run_inference(const Options& options,
                   FrameMailbox* mailbox,
                   std::atomic<double>* argmax_scale,
                   std::atomic<bool>* running) {
    try {
        dxrt::InferenceEngine engine(options.model_path);
        const auto input_tensor = engine.GetInputs().front();
        int input_width = 0;
        int input_height = 0;
        int input_channels = 0;
        parse_model_input_shape(input_tensor.shape(), &input_width,
                                &input_height, &input_channels);
        if (input_channels != 3 || input_tensor.type() != dxrt::DataType::UINT8) {
            throw std::runtime_error("model input must be a UINT8 image tensor with three channels");
        }
        mailbox->setModelInputShape(input_width, input_height, input_channels);

        cv::VideoCapture capture;
        if (options.use_camera) {
            capture.open(options.camera_index, cv::CAP_V4L2);
            capture.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
            capture.set(cv::CAP_PROP_FPS, options.camera_fps);
        } else {
            capture.open(options.video_path);
        }
        if (!capture.isOpened()) {
            throw std::runtime_error(options.use_camera
                ? "failed to open camera"
                : "failed to open video file");
        }

        const double source_fps = capture.get(cv::CAP_PROP_FPS);
        const bool pace_video =
            options.pace_video && !options.use_camera &&
            source_fps > 1.0 && source_fps < 240.0;
        const auto frame_interval = pace_video
            ? std::chrono::duration<double>(1.0 / source_fps)
            : std::chrono::duration<double>(0.0);
        auto next_frame_time = std::chrono::steady_clock::now();
        InflightLimiter limiter(options.max_inflight);
        CompletionCounter completion_counter;
        CallbackContext callback_context{
            &options,
            argmax_scale,
            running,
            &limiter,
            &completion_counter,
            mailbox,
        };

        engine.RegisterCallback(
            [&callback_context](dxrt::TensorPtrs& outputs, void* user_data) -> int {
                std::unique_ptr<AsyncJob> job(static_cast<AsyncJob*>(user_data));
                try {
                    if (!job) {
                        throw std::runtime_error("asynchronous inference returned no job data");
                    }
                    if (outputs.empty()) {
                        throw std::runtime_error("model returned no output tensors");
                    }
                    const auto& output = outputs.front();
                    if (output->type() != dxrt::DataType::FLOAT ||
                        output->elem_size() != sizeof(float)) {
                        throw std::runtime_error(
                            "PIDNet output must contain float logits");
                    }

                    const TensorShapeInfo output_info =
                        parse_output_shape(output->shape());
                    const double frame_scale = std::clamp(
                        callback_context.argmax_scale->load(),
                        kMinimumArgmaxScale, kMaximumArgmaxScale);
                    const cv::Mat class_mask = compute_argmax_mask(
                        static_cast<const float*>(output->data()), output_info,
                        job->frame.cols, job->frame.rows, frame_scale);
                    cv::Mat rendered = render_segmentation(
                        job->frame, class_mask,
                        callback_context.options->overlay_alpha);
                    const double fps =
                        callback_context.completion_counter->update();
                    callback_context.mailbox->publish(
                        std::move(rendered), fps, job->frame_id);
                } catch (const std::exception& error) {
                    std::cerr << "Post-processing error: " << error.what()
                              << std::endl;
                    callback_context.running->store(false);
                    request_quit();
                }
                callback_context.limiter->release();
                return 0;
            });

        std::cout << "Model input: " << input_width << " x " << input_height
                  << " x " << input_channels << std::endl;
        std::cout << "Initial PIDNet argmax scale: " << std::fixed
                  << std::setprecision(1) << argmax_scale->load() << std::endl;
        std::cout << "Maximum asynchronous requests: "
                  << options.max_inflight << std::endl;

        std::uint64_t frame_id = 0;
        int last_job_id = -1;
        while (running->load()) {
            cv::Mat frame;
            if (!capture.read(frame) || frame.empty()) {
                if (!options.use_camera && options.loop_video) {
                    capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                    next_frame_time = std::chrono::steady_clock::now();
                    continue;
                }
                break;
            }

            if (!limiter.acquire(*running)) break;
            auto job = std::make_unique<AsyncJob>();
            job->input = preprocess_frame(frame, input_width, input_height);
            job->frame = std::move(frame);
            job->frame_id = frame_id++;

            AsyncJob* raw_job = job.release();
            try {
                last_job_id =
                    engine.RunAsync(raw_job->input.data(), raw_job);
            } catch (...) {
                std::unique_ptr<AsyncJob> cleanup(raw_job);
                limiter.release();
                throw;
            }

            if (pace_video) {
                next_frame_time +=
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        frame_interval);
                std::this_thread::sleep_until(next_frame_time);
                if (next_frame_time <
                    std::chrono::steady_clock::now() - std::chrono::seconds(1)) {
                    next_frame_time = std::chrono::steady_clock::now();
                }
            }
        }

        if (last_job_id >= 0) engine.Wait(last_job_id);
        limiter.wait_until_empty();
        if (running->load()) request_quit();
    } catch (const dxrt::Exception& error) {
        std::cerr << "DXRT error: " << error.what() << std::endl;
        request_quit();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        request_quit();
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) return 1;
    if (!file_exists(options.model_path)) {
        std::cerr << "Error: model file not found: " << options.model_path << std::endl;
        return 1;
    }
    if (!options.use_camera && !file_exists(options.video_path)) {
        std::cerr << "Error: video file not found: " << options.video_path << std::endl;
        return 1;
    }

    // Async callbacks provide frame-level CPU parallelism. Keep individual
    // OpenCV operations single-threaded to avoid nested oversubscription on
    // four-core systems.
    cv::setNumThreads(1);

    QApplication application(argc, argv);
    std::atomic<double> argmax_scale(options.pidnet_argmax_scale);
    FrameMailbox mailbox;
    MainWindow window(&mailbox, &argmax_scale, options.pidnet_argmax_scale);
    if (options.fullscreen) {
        window.showFullScreen();
    } else {
        window.resize(1280, 800);
        window.show();
    }

    std::atomic<bool> running(true);
    std::thread worker(run_inference, std::cref(options), &mailbox,
                       &argmax_scale, &running);
    QObject::connect(&application, &QCoreApplication::aboutToQuit,
                     [&running]() { running.store(false); });

    const int result = application.exec();
    running.store(false);
    if (worker.joinable()) worker.join();
    return result;
}
