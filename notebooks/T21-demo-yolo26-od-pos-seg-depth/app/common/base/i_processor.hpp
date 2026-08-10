/**
 * @file i_processor.hpp
 * @brief Processing interfaces and result types required by the YOLO26 tutorial.
 */

#ifndef DXAPP_I_PROCESSOR_HPP
#define DXAPP_I_PROCESSOR_HPP

#include <dxrt/dxrt_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dxapp {

struct PreprocessContext {
    int pad_x{0};
    int pad_y{0};
    float scale{1.0f};
    float scale_x{0.0f};
    float scale_y{0.0f};
    int original_width{0};
    int original_height{0};
    int input_width{0};
    int input_height{0};
    cv::Mat source_image;
};

class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;
    virtual void process(const cv::Mat& input, cv::Mat& output,
                         PreprocessContext& ctx) = 0;
    virtual int getInputWidth() const = 0;
    virtual int getInputHeight() const = 0;
    virtual int getColorConversion() const = 0;
};

struct DetectionResult {
    std::vector<float> box;
    float confidence{0.0f};
    int class_id{0};
    std::string class_name;

    DetectionResult() = default;
    DetectionResult(std::vector<float> value, float score, int id,
                    const std::string& name)
        : box(std::move(value)), confidence(score), class_id(id),
          class_name(name) {}
};

struct Keypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};

    Keypoint() = default;
    Keypoint(float x_value, float y_value, float score = 1.0f)
        : x(x_value), y(y_value), confidence(score) {}
};

struct PoseResult {
    std::vector<float> box;
    float confidence{0.0f};
    std::vector<Keypoint> keypoints;
};

struct InstanceSegmentationResult {
    std::vector<float> box;
    float confidence{0.0f};
    int class_id{0};
    std::string class_name;
    cv::Mat mask;
};

template <typename ResultType>
class IPostprocessor {
public:
    virtual ~IPostprocessor() = default;
    virtual std::vector<ResultType> process(const dxrt::TensorPtrs& outputs,
                                            const PreprocessContext& ctx) = 0;
    virtual std::string getModelName() const = 0;
};

using PreprocessorPtr = std::unique_ptr<IPreprocessor>;

template <typename ResultType>
using PostprocessorPtr = std::unique_ptr<IPostprocessor<ResultType>>;

}  // namespace dxapp

#endif  // DXAPP_I_PROCESSOR_HPP
